package tools

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Common service ports and their names
var commonPorts = map[int]string{
	21:    "ftp",
	22:    "ssh",
	23:    "telnet",
	25:    "smtp",
	53:    "dns",
	80:    "http",
	110:   "pop3",
	111:   "rpcbind",
	135:   "msrpc",
	139:   "netbios-ssn",
	143:   "imap",
	443:   "https",
	445:   "microsoft-ds",
	993:   "imaps",
	995:   "pop3s",
	1433:  "mssql",
	1521:  "oracle",
	3306:  "mysql",
	3389:  "rdp",
	5432:  "postgresql",
	5900:  "vnc",
	6379:  "redis",
	8080:  "http-proxy",
	8443:  "https-alt",
	27017: "mongodb",
}

// NewPortScanner creates the port scanning tool
func NewPortScanner() *Tool {
	return &Tool{
		Name:        "port_scan",
		Description: "Scan a target host for open TCP ports using concurrent connections",
		Parameters: []Parameter{
			{Name: "target", Type: "string", Description: "Target IP or hostname", Required: true},
			{Name: "ports", Type: "string", Description: "Port range (e.g., '1-1000', '22,80,443', or 'common' for top ports)", Required: false},
			{Name: "timeout", Type: "string", Description: "Connection timeout in milliseconds (default: 500)", Required: false},
			{Name: "threads", Type: "string", Description: "Number of concurrent threads (default: 100)", Required: false},
		},
		Execute: executePortScan,
	}
}

func executePortScan(args map[string]string) (string, error) {
	target := args["target"]
	if target == "" {
		return "", fmt.Errorf("target is required")
	}

	// Parse timeout
	timeout := 500 * time.Millisecond
	if t, ok := args["timeout"]; ok && t != "" {
		if ms, err := strconv.Atoi(t); err == nil {
			timeout = time.Duration(ms) * time.Millisecond
		}
	}

	// Parse threads
	threads := 100
	if t, ok := args["threads"]; ok && t != "" {
		if n, err := strconv.Atoi(t); err == nil {
			threads = n
		}
	}

	// Parse ports
	ports := parsePorts(args["ports"])

	// Scan ports concurrently
	results := scanPorts(target, ports, timeout, threads)

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Port scan results for %s:\n", target))
	sb.WriteString(fmt.Sprintf("Scanned %d ports\n\n", len(ports)))

	if len(results) == 0 {
		sb.WriteString("No open ports found.\n")
	} else {
		sb.WriteString("Open ports:\n")
		// Sort results
		sort.Slice(results, func(i, j int) bool {
			return results[i].Port < results[j].Port
		})
		for _, r := range results {
			sb.WriteString(fmt.Sprintf("  %d/tcp\t%s\t%s\n", r.Port, r.State, r.Service))
		}
	}

	return sb.String(), nil
}

type portResult struct {
	Port    int
	State   string
	Service string
}

func scanPorts(target string, ports []int, timeout time.Duration, threads int) []portResult {
	var results []portResult
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Semaphore for limiting concurrent connections
	sem := make(chan struct{}, threads)

	for _, port := range ports {
		wg.Add(1)
		sem <- struct{}{} // Acquire

		go func(p int) {
			defer wg.Done()
			defer func() { <-sem }() // Release

			address := fmt.Sprintf("%s:%d", target, p)
			conn, err := net.DialTimeout("tcp", address, timeout)
			if err == nil {
				conn.Close()
				service := commonPorts[p]
				if service == "" {
					service = "unknown"
				}
				mu.Lock()
				results = append(results, portResult{
					Port:    p,
					State:   "open",
					Service: service,
				})
				mu.Unlock()
			}
		}(port)
	}

	wg.Wait()
	return results
}

func parsePorts(portStr string) []int {
	if portStr == "" || portStr == "common" {
		// Return common ports
		ports := make([]int, 0, len(commonPorts))
		for p := range commonPorts {
			ports = append(ports, p)
		}
		// Add some additional common ports
		additional := []int{81, 88, 389, 636, 1080, 1443, 2222, 3000, 4443, 5000, 8000, 8008, 8081, 8888, 9000, 9090, 9443}
		ports = append(ports, additional...)
		return ports
	}

	var ports []int

	// Check if it's a range (e.g., "1-1000")
	if strings.Contains(portStr, "-") {
		parts := strings.Split(portStr, "-")
		if len(parts) == 2 {
			start, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			end, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err1 == nil && err2 == nil {
				for p := start; p <= end; p++ {
					ports = append(ports, p)
				}
				return ports
			}
		}
	}

	// Check if it's a comma-separated list
	for _, p := range strings.Split(portStr, ",") {
		port, err := strconv.Atoi(strings.TrimSpace(p))
		if err == nil && port > 0 && port <= 65535 {
			ports = append(ports, port)
		}
	}

	if len(ports) == 0 {
		// Default to common ports
		return parsePorts("common")
	}

	return ports
}
