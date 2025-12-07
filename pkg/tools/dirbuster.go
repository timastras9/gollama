package tools

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Common directories and files to check
var defaultWordlist = []string{
	// Common directories
	"admin", "administrator", "login", "wp-admin", "wp-login.php", "dashboard",
	"panel", "cpanel", "webmail", "mail", "email",
	"api", "api/v1", "api/v2", "graphql", "rest",
	"backup", "backups", "bak", "old", "temp", "tmp", "test",
	"dev", "development", "staging", "stage", "prod", "production",
	"config", "configuration", "settings", "setup", "install",
	"upload", "uploads", "files", "documents", "docs", "media", "images", "img",
	"static", "assets", "css", "js", "javascript",
	"include", "includes", "inc", "lib", "libs", "library",
	"cgi-bin", "bin", "scripts", "script",
	"phpmyadmin", "pma", "mysql", "database", "db",
	"server-status", "server-info", "status", "info", "health",
	".git", ".svn", ".env", ".htaccess", ".htpasswd",
	"robots.txt", "sitemap.xml", "crossdomain.xml", "security.txt",
	"favicon.ico", "web.config", "package.json", "composer.json",
	// Common files
	"index.php", "index.html", "index.htm", "default.aspx", "default.asp",
	"wp-config.php", "config.php", "configuration.php", "settings.php",
	"readme.txt", "readme.md", "README.md", "LICENSE", "CHANGELOG",
	"phpinfo.php", "info.php", "test.php", "shell.php",
	// API endpoints
	"swagger", "swagger-ui", "swagger.json", "openapi.json",
	"health", "healthcheck", "ping", "version",
	"users", "user", "account", "accounts", "profile",
	"auth", "authenticate", "oauth", "token",
	"search", "query", "download", "export",
}

// NewDirBuster creates the directory busting tool
func NewDirBuster() *Tool {
	return &Tool{
		Name:        "dir_bust",
		Description: "Enumerate directories and files on a web server using concurrent requests",
		Parameters: []Parameter{
			{Name: "url", Type: "string", Description: "Target URL (e.g., http://example.com)", Required: true},
			{Name: "wordlist", Type: "string", Description: "Comma-separated paths to check (or 'default' for built-in list)", Required: false},
			{Name: "extensions", Type: "string", Description: "File extensions to append (e.g., 'php,html,txt')", Required: false},
			{Name: "threads", Type: "string", Description: "Number of concurrent threads (default: 20)", Required: false},
			{Name: "timeout", Type: "string", Description: "Request timeout in seconds (default: 10)", Required: false},
		},
		Execute: executeDirBust,
	}
}

func executeDirBust(args map[string]string) (string, error) {
	baseURL := args["url"]
	if baseURL == "" {
		return "", fmt.Errorf("url is required")
	}

	// Ensure URL has scheme
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	baseURL = strings.TrimSuffix(baseURL, "/")

	// Parse threads
	threads := 20
	if t, ok := args["threads"]; ok && t != "" {
		if n, err := strconv.Atoi(t); err == nil {
			threads = n
		}
	}

	// Parse timeout
	timeout := 10 * time.Second
	if t, ok := args["timeout"]; ok && t != "" {
		if s, err := strconv.Atoi(t); err == nil {
			timeout = time.Duration(s) * time.Second
		}
	}

	// Get wordlist
	wordlist := defaultWordlist
	if w, ok := args["wordlist"]; ok && w != "" && w != "default" {
		wordlist = strings.Split(w, ",")
		for i := range wordlist {
			wordlist[i] = strings.TrimSpace(wordlist[i])
		}
	}

	// Get extensions
	var extensions []string
	if ext, ok := args["extensions"]; ok && ext != "" {
		for _, e := range strings.Split(ext, ",") {
			e = strings.TrimSpace(e)
			if !strings.HasPrefix(e, ".") {
				e = "." + e
			}
			extensions = append(extensions, e)
		}
	}

	// Build full path list
	var paths []string
	for _, word := range wordlist {
		paths = append(paths, word)
		for _, ext := range extensions {
			paths = append(paths, word+ext)
		}
	}

	// Scan paths concurrently
	results := scanPaths(baseURL, paths, timeout, threads)

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Directory enumeration for %s:\n", baseURL))
	sb.WriteString(fmt.Sprintf("Checked %d paths\n\n", len(paths)))

	if len(results) == 0 {
		sb.WriteString("No interesting paths found.\n")
	} else {
		sb.WriteString("Found paths:\n")
		for _, r := range results {
			sb.WriteString(fmt.Sprintf("  [%d] %s (%s)\n", r.Status, r.Path, r.Size))
		}
	}

	return sb.String(), nil
}

type pathResult struct {
	Path   string
	Status int
	Size   string
}

func scanPaths(baseURL string, paths []string, timeout time.Duration, threads int) []pathResult {
	var results []pathResult
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create HTTP client with settings
	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			TLSClientConfig:     &tls.Config{InsecureSkipVerify: true},
			MaxIdleConns:        threads,
			MaxIdleConnsPerHost: threads,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse // Don't follow redirects
		},
	}

	// Semaphore for limiting concurrent requests
	sem := make(chan struct{}, threads)

	for _, path := range paths {
		wg.Add(1)
		sem <- struct{}{} // Acquire

		go func(p string) {
			defer wg.Done()
			defer func() { <-sem }() // Release

			url := baseURL + "/" + p
			resp, err := client.Get(url)
			if err != nil {
				return
			}
			defer resp.Body.Close()

			// Only record interesting responses (not 404)
			if resp.StatusCode != 404 && resp.StatusCode != 400 {
				size := resp.Header.Get("Content-Length")
				if size == "" {
					size = "unknown"
				} else {
					size = size + " bytes"
				}

				mu.Lock()
				results = append(results, pathResult{
					Path:   "/" + p,
					Status: resp.StatusCode,
					Size:   size,
				})
				mu.Unlock()
			}
		}(path)
	}

	wg.Wait()
	return results
}
