package api

import "time"

// GenerateRequest is the request for /api/generate
type GenerateRequest struct {
	Model     string         `json:"model"`
	Prompt    string         `json:"prompt"`
	Suffix    string         `json:"suffix,omitempty"`
	System    string         `json:"system,omitempty"`
	Template  string         `json:"template,omitempty"`
	Context   []int          `json:"context,omitempty"`
	Stream    *bool          `json:"stream,omitempty"`
	Raw       bool           `json:"raw,omitempty"`
	Format    any            `json:"format,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
	KeepAlive string         `json:"keep_alive,omitempty"`
}

// GenerateResponse is the response for /api/generate
type GenerateResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Response           string    `json:"response"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason,omitempty"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
}

// ChatRequest is the request for /api/chat
type ChatRequest struct {
	Model     string         `json:"model"`
	Messages  []ChatMessage  `json:"messages"`
	Stream    *bool          `json:"stream,omitempty"`
	Format    any            `json:"format,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
	KeepAlive string         `json:"keep_alive,omitempty"`
	Tools     []Tool         `json:"tools,omitempty"`
}

// ChatMessage represents a message in a chat conversation
type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChatResponse is the response for /api/chat
type ChatResponse struct {
	Model              string      `json:"model"`
	CreatedAt          time.Time   `json:"created_at"`
	Message            ChatMessage `json:"message"`
	Done               bool        `json:"done"`
	DoneReason         string      `json:"done_reason,omitempty"`
	TotalDuration      int64       `json:"total_duration,omitempty"`
	LoadDuration       int64       `json:"load_duration,omitempty"`
	PromptEvalCount    int         `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64       `json:"prompt_eval_duration,omitempty"`
	EvalCount          int         `json:"eval_count,omitempty"`
	EvalDuration       int64       `json:"eval_duration,omitempty"`
}

// Tool represents a function that can be called
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// ToolCall represents a call to a tool
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the function being called
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// EmbeddingRequest is the request for /api/embeddings
type EmbeddingRequest struct {
	Model     string         `json:"model"`
	Prompt    string         `json:"prompt,omitempty"`
	Input     any            `json:"input,omitempty"` // string or []string
	Options   map[string]any `json:"options,omitempty"`
	KeepAlive string         `json:"keep_alive,omitempty"`
}

// EmbeddingResponse is the response for /api/embeddings
type EmbeddingResponse struct {
	Model     string      `json:"model"`
	Embedding []float64   `json:"embedding,omitempty"`
	Embeddings [][]float64 `json:"embeddings,omitempty"`
}

// ListResponse is the response for /api/tags
type ListResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelInfo contains information about a model
type ModelInfo struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	ModifiedAt time.Time    `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

// ModelDetails contains detailed model information
type ModelDetails struct {
	ParentModel       string   `json:"parent_model,omitempty"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families,omitempty"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// ShowRequest is the request for /api/show
type ShowRequest struct {
	Name    string `json:"name"`
	Model   string `json:"model,omitempty"`
	Verbose bool   `json:"verbose,omitempty"`
}

// ShowResponse is the response for /api/show
type ShowResponse struct {
	License    string       `json:"license,omitempty"`
	Modelfile  string       `json:"modelfile,omitempty"`
	Parameters string       `json:"parameters,omitempty"`
	Template   string       `json:"template,omitempty"`
	System     string       `json:"system,omitempty"`
	Details    ModelDetails `json:"details"`
	Messages   []ChatMessage `json:"messages,omitempty"`
	ModifiedAt time.Time    `json:"modified_at"`
}

// DeleteRequest is the request for /api/delete
type DeleteRequest struct {
	Name  string `json:"name"`
	Model string `json:"model,omitempty"`
}

// PullRequest is the request for /api/pull
type PullRequest struct {
	Name     string `json:"name"`
	Model    string `json:"model,omitempty"`
	Insecure bool   `json:"insecure,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
}

// PullResponse is the response for /api/pull
type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// VersionResponse is the response for /api/version
type VersionResponse struct {
	Version string `json:"version"`
}

// Options contains model inference options
type Options struct {
	// Sampling
	Temperature   float32 `json:"temperature,omitempty"`
	TopK          int     `json:"top_k,omitempty"`
	TopP          float32 `json:"top_p,omitempty"`
	MinP          float32 `json:"min_p,omitempty"`
	Seed          int64   `json:"seed,omitempty"`

	// Generation limits
	NumPredict    int     `json:"num_predict,omitempty"`
	NumCtx        int     `json:"num_ctx,omitempty"`
	NumBatch      int     `json:"num_batch,omitempty"`

	// Penalties
	RepeatPenalty    float32 `json:"repeat_penalty,omitempty"`
	RepeatLastN      int     `json:"repeat_last_n,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32 `json:"presence_penalty,omitempty"`

	// Stop sequences
	Stop []string `json:"stop,omitempty"`
}

// DefaultOptions returns default inference options
func DefaultOptions() Options {
	return Options{
		Temperature:   0.8,
		TopK:          40,
		TopP:          0.9,
		NumPredict:    128,
		NumCtx:        2048,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Seed:          -1,
	}
}

// ParseOptions parses options from a map
func ParseOptions(m map[string]any) Options {
	opts := DefaultOptions()

	if m == nil {
		return opts
	}

	if v, ok := m["temperature"].(float64); ok {
		opts.Temperature = float32(v)
	}
	if v, ok := m["top_k"].(float64); ok {
		opts.TopK = int(v)
	}
	if v, ok := m["top_p"].(float64); ok {
		opts.TopP = float32(v)
	}
	if v, ok := m["min_p"].(float64); ok {
		opts.MinP = float32(v)
	}
	if v, ok := m["seed"].(float64); ok {
		opts.Seed = int64(v)
	}
	if v, ok := m["num_predict"].(float64); ok {
		opts.NumPredict = int(v)
	}
	if v, ok := m["num_ctx"].(float64); ok {
		opts.NumCtx = int(v)
	}
	if v, ok := m["repeat_penalty"].(float64); ok {
		opts.RepeatPenalty = float32(v)
	}
	if v, ok := m["repeat_last_n"].(float64); ok {
		opts.RepeatLastN = int(v)
	}
	if v, ok := m["frequency_penalty"].(float64); ok {
		opts.FrequencyPenalty = float32(v)
	}
	if v, ok := m["presence_penalty"].(float64); ok {
		opts.PresencePenalty = float32(v)
	}
	if v, ok := m["stop"].([]any); ok {
		for _, s := range v {
			if str, ok := s.(string); ok {
				opts.Stop = append(opts.Stop, str)
			}
		}
	}

	return opts
}
