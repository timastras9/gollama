package tools

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// NewHTTPClient creates the HTTP request tool
func NewHTTPClient() *Tool {
	return &Tool{
		Name:        "http_request",
		Description: "Make HTTP requests to a URL and inspect the response (headers, body, status)",
		Parameters: []Parameter{
			{Name: "url", Type: "string", Description: "Target URL", Required: true},
			{Name: "method", Type: "string", Description: "HTTP method: GET, POST, HEAD, OPTIONS (default: GET)", Required: false},
			{Name: "headers", Type: "string", Description: "Custom headers as 'Key:Value,Key2:Value2'", Required: false},
			{Name: "body", Type: "string", Description: "Request body for POST/PUT requests", Required: false},
			{Name: "follow_redirects", Type: "string", Description: "Follow redirects: true/false (default: false)", Required: false},
		},
		Execute: executeHTTPRequest,
	}
}

// NewBannerGrab creates the banner grabbing tool
func NewBannerGrab() *Tool {
	return &Tool{
		Name:        "banner_grab",
		Description: "Grab service banners from open ports to identify services and versions",
		Parameters: []Parameter{
			{Name: "target", Type: "string", Description: "Target IP or hostname", Required: true},
			{Name: "port", Type: "string", Description: "Port number", Required: true},
			{Name: "timeout", Type: "string", Description: "Timeout in seconds (default: 5)", Required: false},
		},
		Execute: executeBannerGrab,
	}
}

// NewWhoisLookup creates the WHOIS lookup tool
func NewWhoisLookup() *Tool {
	return &Tool{
		Name:        "whois",
		Description: "Perform WHOIS lookup on a domain to get registration information",
		Parameters: []Parameter{
			{Name: "domain", Type: "string", Description: "Target domain (e.g., example.com)", Required: true},
		},
		Execute: executeWhois,
	}
}

func executeHTTPRequest(args map[string]string) (string, error) {
	url := args["url"]
	if url == "" {
		return "", fmt.Errorf("url is required")
	}

	// Ensure URL has scheme
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		url = "http://" + url
	}

	method := strings.ToUpper(args["method"])
	if method == "" {
		method = "GET"
	}

	followRedirects := args["follow_redirects"] == "true"

	// Create client
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}

	if !followRedirects {
		client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		}
	}

	// Create request
	var bodyReader io.Reader
	if body := args["body"]; body != "" {
		bodyReader = strings.NewReader(body)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	req.Header.Set("User-Agent", "Gollama-Security-Scanner/1.0")
	if headers := args["headers"]; headers != "" {
		for _, h := range strings.Split(headers, ",") {
			parts := strings.SplitN(strings.TrimSpace(h), ":", 2)
			if len(parts) == 2 {
				req.Header.Set(strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]))
			}
		}
	}

	// Execute request
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read body (limit to 10KB)
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 10*1024))

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("HTTP %s %s\n\n", method, url))
	sb.WriteString(fmt.Sprintf("Status: %s\n\n", resp.Status))

	sb.WriteString("Response Headers:\n")
	for key, values := range resp.Header {
		for _, v := range values {
			sb.WriteString(fmt.Sprintf("  %s: %s\n", key, v))
		}
	}

	sb.WriteString(fmt.Sprintf("\nBody (%d bytes):\n", len(body)))
	if len(body) > 0 {
		// Truncate if too long
		bodyStr := string(body)
		if len(bodyStr) > 2000 {
			bodyStr = bodyStr[:2000] + "\n... (truncated)"
		}
		sb.WriteString(bodyStr)
	}

	return sb.String(), nil
}

func executeBannerGrab(args map[string]string) (string, error) {
	target := args["target"]
	if target == "" {
		return "", fmt.Errorf("target is required")
	}

	port := args["port"]
	if port == "" {
		return "", fmt.Errorf("port is required")
	}

	timeout := 5 * time.Second
	if t := args["timeout"]; t != "" {
		if s, err := fmt.Sscanf(t, "%d", &timeout); err == nil {
			timeout = time.Duration(s) * time.Second
		}
	}

	address := fmt.Sprintf("%s:%s", target, port)

	// Try HTTP first for common web ports
	if port == "80" || port == "443" || port == "8080" || port == "8443" {
		scheme := "http"
		if port == "443" || port == "8443" {
			scheme = "https"
		}

		client := &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
		}

		resp, err := client.Head(fmt.Sprintf("%s://%s", scheme, address))
		if err == nil {
			defer resp.Body.Close()

			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("Banner grab for %s:\n\n", address))
			sb.WriteString(fmt.Sprintf("Protocol: HTTP\n"))
			sb.WriteString(fmt.Sprintf("Status: %s\n", resp.Status))
			sb.WriteString("\nServer Headers:\n")

			interesting := []string{"Server", "X-Powered-By", "X-AspNet-Version", "X-AspNetMvc-Version"}
			for _, h := range interesting {
				if v := resp.Header.Get(h); v != "" {
					sb.WriteString(fmt.Sprintf("  %s: %s\n", h, v))
				}
			}

			return sb.String(), nil
		}
	}

	// TCP banner grab
	conn, err := dialWithTimeout("tcp", address, timeout)
	if err != nil {
		return "", fmt.Errorf("connection failed: %w", err)
	}
	defer conn.Close()

	// Set read deadline
	conn.SetReadDeadline(time.Now().Add(timeout))

	// Some services need a probe
	probes := map[string][]byte{
		"21":   []byte(""),                                    // FTP sends banner automatically
		"22":   []byte(""),                                    // SSH sends banner automatically
		"25":   []byte("EHLO scanner\r\n"),                    // SMTP
		"110":  []byte(""),                                    // POP3
		"143":  []byte(""),                                    // IMAP
		"3306": []byte(""),                                    // MySQL
		"6379": []byte("INFO\r\n"),                            // Redis
	}

	if probe, ok := probes[port]; ok && len(probe) > 0 {
		conn.Write(probe)
	}

	// Read response
	buf := make([]byte, 4096)
	n, _ := conn.Read(buf)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Banner grab for %s:\n\n", address))

	if n > 0 {
		banner := string(buf[:n])
		// Clean up non-printable characters
		banner = strings.Map(func(r rune) rune {
			if r < 32 && r != '\n' && r != '\r' && r != '\t' {
				return '.'
			}
			return r
		}, banner)
		sb.WriteString(fmt.Sprintf("Banner:\n%s\n", banner))
	} else {
		sb.WriteString("No banner received (service may require authentication or specific probe)\n")
	}

	return sb.String(), nil
}

func executeWhois(args map[string]string) (string, error) {
	domain := args["domain"]
	if domain == "" {
		return "", fmt.Errorf("domain is required")
	}

	// Connect to WHOIS server
	conn, err := dialWithTimeout("tcp", "whois.iana.org:43", 10*time.Second)
	if err != nil {
		return "", fmt.Errorf("failed to connect to WHOIS server: %w", err)
	}
	defer conn.Close()

	// Send query
	conn.SetDeadline(time.Now().Add(10 * time.Second))
	_, err = conn.Write([]byte(domain + "\r\n"))
	if err != nil {
		return "", fmt.Errorf("failed to send query: %w", err)
	}

	// Read response
	response, err := io.ReadAll(conn)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Check if we need to query a different WHOIS server
	result := string(response)
	if strings.Contains(result, "whois:") {
		// Extract the referred WHOIS server
		lines := strings.Split(result, "\n")
		for _, line := range lines {
			if strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "whois:") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					whoisServer := strings.TrimSpace(parts[1])
					// Query the referred server
					return queryWhoisServer(whoisServer+":43", domain)
				}
			}
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("WHOIS lookup for %s:\n\n", domain))
	sb.WriteString(result)

	return sb.String(), nil
}

func queryWhoisServer(server, domain string) (string, error) {
	conn, err := dialWithTimeout("tcp", server, 10*time.Second)
	if err != nil {
		return "", err
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(10 * time.Second))
	_, err = conn.Write([]byte(domain + "\r\n"))
	if err != nil {
		return "", err
	}

	response, err := io.ReadAll(conn)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("WHOIS lookup for %s:\n\n", domain))
	sb.WriteString(string(response))

	return sb.String(), nil
}

// dialWithTimeout creates a TCP connection with timeout
func dialWithTimeout(network, address string, timeout time.Duration) (net.Conn, error) {
	return (&net.Dialer{Timeout: timeout}).Dial(network, address)
}
