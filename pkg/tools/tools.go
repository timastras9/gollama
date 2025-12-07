package tools

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Tool represents a callable function
type Tool struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Parameters  []Parameter  `json:"parameters"`
	Execute     func(args map[string]string) (string, error) `json:"-"`
}

// Parameter describes a tool parameter
type Parameter struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

// ToolCall represents a parsed function call from the model
type ToolCall struct {
	Name string            `json:"name"`
	Args map[string]string `json:"args"`
}

// Registry holds all available tools
type Registry struct {
	tools map[string]*Tool
}

// NewRegistry creates a new tool registry with all pentesting tools
func NewRegistry() *Registry {
	r := &Registry{
		tools: make(map[string]*Tool),
	}

	// Register all tools
	r.Register(NewPortScanner())
	r.Register(NewDirBuster())
	r.Register(NewDNSEnum())
	r.Register(NewHTTPClient())
	r.Register(NewSubdomainEnum())
	r.Register(NewBannerGrab())
	r.Register(NewWhoisLookup())

	return r
}

// Register adds a tool to the registry
func (r *Registry) Register(t *Tool) {
	r.tools[t.Name] = t
}

// Get returns a tool by name
func (r *Registry) Get(name string) (*Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// List returns all registered tools
func (r *Registry) List() []*Tool {
	tools := make([]*Tool, 0, len(r.tools))
	for _, t := range r.tools {
		tools = append(tools, t)
	}
	return tools
}

// SystemPrompt generates the system prompt for function calling
func (r *Registry) SystemPrompt() string {
	var sb strings.Builder

	sb.WriteString(`You are a penetration testing assistant with access to security tools.
When you need to use a tool, output a JSON function call in this exact format:

<tool_call>
{"name": "tool_name", "args": {"param1": "value1", "param2": "value2"}}
</tool_call>

Available tools:

`)

	for _, tool := range r.tools {
		sb.WriteString(fmt.Sprintf("## %s\n", tool.Name))
		sb.WriteString(fmt.Sprintf("%s\n", tool.Description))
		sb.WriteString("Parameters:\n")
		for _, p := range tool.Parameters {
			req := ""
			if p.Required {
				req = " (required)"
			}
			sb.WriteString(fmt.Sprintf("  - %s (%s)%s: %s\n", p.Name, p.Type, req, p.Description))
		}
		sb.WriteString("\n")
	}

	sb.WriteString(`
After receiving tool results, analyze them and either:
1. Use another tool if needed
2. Provide a summary of findings

Always explain what you're doing and why. Be thorough but efficient.
`)

	return sb.String()
}

// ParseToolCall extracts a tool call from model output
func ParseToolCall(output string) (*ToolCall, string, bool) {
	startTag := "<tool_call>"
	endTag := "</tool_call>"

	startIdx := strings.Index(output, startTag)
	if startIdx == -1 {
		return nil, output, false
	}

	endIdx := strings.Index(output[startIdx:], endTag)
	if endIdx == -1 {
		return nil, output, false
	}

	jsonStr := output[startIdx+len(startTag) : startIdx+endIdx]
	jsonStr = strings.TrimSpace(jsonStr)

	var call ToolCall
	if err := json.Unmarshal([]byte(jsonStr), &call); err != nil {
		return nil, output, false
	}

	// Return the text before the tool call
	beforeCall := strings.TrimSpace(output[:startIdx])

	return &call, beforeCall, true
}

// Execute runs a tool call and returns the result
func (r *Registry) Execute(call *ToolCall) (string, error) {
	tool, ok := r.Get(call.Name)
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", call.Name)
	}

	return tool.Execute(call.Args)
}
