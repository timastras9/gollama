package tools

import (
	"fmt"
	"net"
	"strings"
	"sync"
)

// Common subdomains for enumeration
var commonSubdomains = []string{
	"www", "mail", "remote", "blog", "webmail", "server", "ns1", "ns2",
	"smtp", "secure", "vpn", "m", "shop", "ftp", "mail2", "test",
	"portal", "ns", "ww1", "host", "support", "dev", "web", "bbs",
	"ww42", "mx", "email", "cloud", "1", "mail1", "2", "forum",
	"owa", "www2", "gw", "admin", "store", "mx1", "cdn", "api",
	"exchange", "app", "gov", "2tty", "vps", "govyty", "news",
	"staging", "stage", "beta", "demo", "tech", "ops", "prod",
	"internal", "intranet", "extranet", "cms", "crm", "erp",
	"sso", "auth", "login", "dashboard", "panel", "cp", "git",
	"jenkins", "jira", "confluence", "wiki", "docs", "help",
	"status", "monitor", "grafana", "kibana", "elastic", "redis",
	"db", "database", "mysql", "postgres", "mongo", "sql",
	"img", "images", "static", "assets", "media", "files",
	"download", "downloads", "upload", "uploads", "backup",
}

// NewDNSEnum creates the DNS enumeration tool
func NewDNSEnum() *Tool {
	return &Tool{
		Name:        "dns_enum",
		Description: "Enumerate DNS records for a domain (A, AAAA, MX, NS, TXT, CNAME)",
		Parameters: []Parameter{
			{Name: "domain", Type: "string", Description: "Target domain (e.g., example.com)", Required: true},
			{Name: "type", Type: "string", Description: "Record type: all, A, AAAA, MX, NS, TXT, CNAME (default: all)", Required: false},
		},
		Execute: executeDNSEnum,
	}
}

// NewSubdomainEnum creates the subdomain enumeration tool
func NewSubdomainEnum() *Tool {
	return &Tool{
		Name:        "subdomain_enum",
		Description: "Enumerate subdomains for a domain using DNS brute forcing",
		Parameters: []Parameter{
			{Name: "domain", Type: "string", Description: "Target domain (e.g., example.com)", Required: true},
			{Name: "wordlist", Type: "string", Description: "Comma-separated subdomains (or 'default' for built-in list)", Required: false},
			{Name: "threads", Type: "string", Description: "Number of concurrent threads (default: 50)", Required: false},
		},
		Execute: executeSubdomainEnum,
	}
}

func executeDNSEnum(args map[string]string) (string, error) {
	domain := args["domain"]
	if domain == "" {
		return "", fmt.Errorf("domain is required")
	}

	recordType := strings.ToUpper(args["type"])
	if recordType == "" {
		recordType = "ALL"
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("DNS enumeration for %s:\n\n", domain))

	// A records
	if recordType == "ALL" || recordType == "A" {
		ips, err := net.LookupIP(domain)
		if err == nil {
			sb.WriteString("A/AAAA Records:\n")
			for _, ip := range ips {
				if ipv4 := ip.To4(); ipv4 != nil {
					sb.WriteString(fmt.Sprintf("  A     %s\n", ipv4))
				} else {
					sb.WriteString(fmt.Sprintf("  AAAA  %s\n", ip))
				}
			}
			sb.WriteString("\n")
		}
	}

	// MX records
	if recordType == "ALL" || recordType == "MX" {
		mxs, err := net.LookupMX(domain)
		if err == nil && len(mxs) > 0 {
			sb.WriteString("MX Records:\n")
			for _, mx := range mxs {
				sb.WriteString(fmt.Sprintf("  %s (priority: %d)\n", mx.Host, mx.Pref))
			}
			sb.WriteString("\n")
		}
	}

	// NS records
	if recordType == "ALL" || recordType == "NS" {
		nss, err := net.LookupNS(domain)
		if err == nil && len(nss) > 0 {
			sb.WriteString("NS Records:\n")
			for _, ns := range nss {
				sb.WriteString(fmt.Sprintf("  %s\n", ns.Host))
			}
			sb.WriteString("\n")
		}
	}

	// TXT records
	if recordType == "ALL" || recordType == "TXT" {
		txts, err := net.LookupTXT(domain)
		if err == nil && len(txts) > 0 {
			sb.WriteString("TXT Records:\n")
			for _, txt := range txts {
				sb.WriteString(fmt.Sprintf("  %s\n", txt))
			}
			sb.WriteString("\n")
		}
	}

	// CNAME record
	if recordType == "ALL" || recordType == "CNAME" {
		cname, err := net.LookupCNAME(domain)
		if err == nil && cname != domain+"." {
			sb.WriteString("CNAME Record:\n")
			sb.WriteString(fmt.Sprintf("  %s\n\n", cname))
		}
	}

	result := sb.String()
	if result == fmt.Sprintf("DNS enumeration for %s:\n\n", domain) {
		return fmt.Sprintf("No DNS records found for %s\n", domain), nil
	}

	return result, nil
}

func executeSubdomainEnum(args map[string]string) (string, error) {
	domain := args["domain"]
	if domain == "" {
		return "", fmt.Errorf("domain is required")
	}

	// Parse threads
	threads := 50
	if t, ok := args["threads"]; ok && t != "" {
		var n int
		if _, err := fmt.Sscanf(t, "%d", &n); err == nil {
			threads = n
		}
	}

	// Get wordlist
	wordlist := commonSubdomains
	if w, ok := args["wordlist"]; ok && w != "" && w != "default" {
		wordlist = strings.Split(w, ",")
		for i := range wordlist {
			wordlist[i] = strings.TrimSpace(wordlist[i])
		}
	}

	// Enumerate subdomains concurrently
	results := enumerateSubdomains(domain, wordlist, threads)

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Subdomain enumeration for %s:\n", domain))
	sb.WriteString(fmt.Sprintf("Checked %d subdomains\n\n", len(wordlist)))

	if len(results) == 0 {
		sb.WriteString("No subdomains found.\n")
	} else {
		sb.WriteString("Found subdomains:\n")
		for _, r := range results {
			sb.WriteString(fmt.Sprintf("  %s -> %s\n", r.Subdomain, strings.Join(r.IPs, ", ")))
		}
	}

	return sb.String(), nil
}

type subdomainResult struct {
	Subdomain string
	IPs       []string
}

func enumerateSubdomains(domain string, wordlist []string, threads int) []subdomainResult {
	var results []subdomainResult
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Using default resolver with timeout via context

	// Semaphore for limiting concurrent lookups
	sem := make(chan struct{}, threads)

	for _, sub := range wordlist {
		wg.Add(1)
		sem <- struct{}{} // Acquire

		go func(subdomain string) {
			defer wg.Done()
			defer func() { <-sem }() // Release

			fqdn := subdomain + "." + domain
			ips, err := net.LookupIP(fqdn)
			if err == nil && len(ips) > 0 {
				var ipStrs []string
				for _, ip := range ips {
					ipStrs = append(ipStrs, ip.String())
				}

				mu.Lock()
				results = append(results, subdomainResult{
					Subdomain: fqdn,
					IPs:       ipStrs,
				})
				mu.Unlock()
			}
		}(sub)
	}

	wg.Wait()
	return results
}
