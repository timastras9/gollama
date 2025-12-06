package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/timastras9/gollama/pkg/gguf"
	"github.com/timastras9/gollama/pkg/model"
	"github.com/timastras9/gollama/pkg/tokenizer"
)

// Version is the current gollama version
const Version = "0.1.0"

// Server implements the Ollama-compatible API
type Server struct {
	addr      string
	modelsDir string
	models    *ModelManager
	server    *http.Server
}

// NewServer creates a new API server
func NewServer(addr, modelsDir string) *Server {
	s := &Server{
		addr:      addr,
		modelsDir: modelsDir,
		models:    NewModelManager(modelsDir),
	}
	return s
}

// Start starts the API server
func (s *Server) Start() error {
	mux := http.NewServeMux()

	// API routes
	mux.HandleFunc("POST /api/generate", s.handleGenerate)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /api/embed", s.handleEmbeddings)
	mux.HandleFunc("GET /api/tags", s.handleList)
	mux.HandleFunc("POST /api/show", s.handleShow)
	mux.HandleFunc("DELETE /api/delete", s.handleDelete)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("GET /api/version", s.handleVersion)
	mux.HandleFunc("GET /", s.handleRoot)

	// CORS middleware
	handler := corsMiddleware(mux)

	s.server = &http.Server{
		Addr:    s.addr,
		Handler: handler,
	}

	log.Printf("Gollama server starting on %s", s.addr)
	return s.server.ListenAndServe()
}

// Stop gracefully stops the server
func (s *Server) Stop(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

// corsMiddleware adds CORS headers
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// handleRoot handles the root path
func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("Gollama is running"))
}

// handleVersion returns the version
func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(VersionResponse{Version: Version})
}

// handleGenerate handles text generation
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Load model
	m, tok, err := s.models.Load(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to load model: %v", err), http.StatusNotFound)
		return
	}

	// Parse options
	opts := ParseOptions(req.Options)

	// Check streaming mode (default: true)
	stream := req.Stream == nil || *req.Stream

	// Create sampler
	samplerCfg := model.SamplerConfig{
		Temperature: opts.Temperature,
		TopK:        opts.TopK,
		TopP:        opts.TopP,
		MinP:        opts.MinP,
		Seed:        opts.Seed,
	}
	sampler := model.NewSampler(samplerCfg)

	// Tokenize prompt
	tokens := tok.Encode(req.Prompt)
	startTime := time.Now()

	// Reset KV cache
	m.ResetCache()

	if stream {
		s.streamGenerate(w, r.Context(), m, tok, sampler, req.Model, tokens, opts, startTime)
	} else {
		s.blockingGenerate(w, m, tok, sampler, req.Model, tokens, opts, startTime)
	}
}

// streamGenerate handles streaming generation
func (s *Server) streamGenerate(w http.ResponseWriter, ctx context.Context, m *model.Transformer, tok *tokenizer.Tokenizer, sampler *model.Sampler, modelName string, promptTokens []int, opts Options, startTime time.Time) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.Header().Set("Cache-Control", "no-cache")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	promptEvalStart := time.Now()

	// Process prompt (prefill)
	logits := m.Forward(promptTokens, 0)
	lastLogits := logits.Slice(len(promptTokens)-1, len(promptTokens))
	lastLogits = lastLogits.Reshape(m.Config.VocabSize)

	promptEvalDuration := time.Since(promptEvalStart)

	// Generate tokens
	evalStart := time.Now()
	generated := []int{}
	pos := len(promptTokens)

	for i := 0; i < opts.NumPredict; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Sample next token
		nextToken := sampler.Sample(lastLogits)
		generated = append(generated, nextToken)

		// Check for EOS
		if tok.IsEOS(nextToken) {
			break
		}

		// Decode token
		text := tok.DecodeToken(nextToken)

		// Send streaming response
		resp := GenerateResponse{
			Model:     modelName,
			CreatedAt: time.Now(),
			Response:  text,
			Done:      false,
		}

		if err := json.NewEncoder(w).Encode(resp); err != nil {
			return
		}
		flusher.Flush()

		// Check stop sequences
		if containsStop(tok.Decode(generated), opts.Stop) {
			break
		}

		// Run model for next token
		logits = m.Forward([]int{nextToken}, pos)
		lastLogits = logits.Reshape(m.Config.VocabSize)
		pos++
	}

	evalDuration := time.Since(evalStart)

	// Send final response
	finalResp := GenerateResponse{
		Model:              modelName,
		CreatedAt:          time.Now(),
		Response:           "",
		Done:               true,
		DoneReason:         "stop",
		Context:            append(promptTokens, generated...),
		TotalDuration:      time.Since(startTime).Nanoseconds(),
		LoadDuration:       0, // Model was already loaded
		PromptEvalCount:    len(promptTokens),
		PromptEvalDuration: promptEvalDuration.Nanoseconds(),
		EvalCount:          len(generated),
		EvalDuration:       evalDuration.Nanoseconds(),
	}

	json.NewEncoder(w).Encode(finalResp)
	flusher.Flush()
}

// blockingGenerate handles non-streaming generation
func (s *Server) blockingGenerate(w http.ResponseWriter, m *model.Transformer, tok *tokenizer.Tokenizer, sampler *model.Sampler, modelName string, promptTokens []int, opts Options, startTime time.Time) {
	promptEvalStart := time.Now()

	// Process prompt
	logits := m.Forward(promptTokens, 0)
	lastLogits := logits.Slice(len(promptTokens)-1, len(promptTokens))
	lastLogits = lastLogits.Reshape(m.Config.VocabSize)

	promptEvalDuration := time.Since(promptEvalStart)

	// Generate tokens
	evalStart := time.Now()
	generated := []int{}
	pos := len(promptTokens)

	for i := 0; i < opts.NumPredict; i++ {
		nextToken := sampler.Sample(lastLogits)
		generated = append(generated, nextToken)

		if tok.IsEOS(nextToken) {
			break
		}

		// Check stop sequences
		if containsStop(tok.Decode(generated), opts.Stop) {
			break
		}

		logits = m.Forward([]int{nextToken}, pos)
		lastLogits = logits.Reshape(m.Config.VocabSize)
		pos++
	}

	evalDuration := time.Since(evalStart)

	// Decode full response
	responseText := tok.Decode(generated)

	resp := GenerateResponse{
		Model:              modelName,
		CreatedAt:          time.Now(),
		Response:           responseText,
		Done:               true,
		DoneReason:         "stop",
		Context:            append(promptTokens, generated...),
		TotalDuration:      time.Since(startTime).Nanoseconds(),
		PromptEvalCount:    len(promptTokens),
		PromptEvalDuration: promptEvalDuration.Nanoseconds(),
		EvalCount:          len(generated),
		EvalDuration:       evalDuration.Nanoseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleChat handles chat completion
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Format messages into prompt
	prompt := formatChatMessages(req.Messages)

	// Convert to generate request
	stream := req.Stream == nil || *req.Stream
	genReq := GenerateRequest{
		Model:   req.Model,
		Prompt:  prompt,
		Stream:  &stream,
		Options: req.Options,
	}

	// Load model
	m, tok, err := s.models.Load(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to load model: %v", err), http.StatusNotFound)
		return
	}

	opts := ParseOptions(req.Options)
	samplerCfg := model.SamplerConfig{
		Temperature: opts.Temperature,
		TopK:        opts.TopK,
		TopP:        opts.TopP,
		MinP:        opts.MinP,
		Seed:        opts.Seed,
	}
	sampler := model.NewSampler(samplerCfg)

	tokens := tok.Encode(genReq.Prompt)
	startTime := time.Now()
	m.ResetCache()

	if stream {
		s.streamChat(w, r.Context(), m, tok, sampler, req.Model, tokens, opts, startTime)
	} else {
		s.blockingChat(w, m, tok, sampler, req.Model, tokens, opts, startTime)
	}
}

// formatChatMessages formats chat messages into a prompt string
func formatChatMessages(messages []ChatMessage) string {
	var sb strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			sb.WriteString(fmt.Sprintf("System: %s\n\n", msg.Content))
		case "user":
			sb.WriteString(fmt.Sprintf("User: %s\n\n", msg.Content))
		case "assistant":
			sb.WriteString(fmt.Sprintf("Assistant: %s\n\n", msg.Content))
		}
	}

	sb.WriteString("Assistant: ")
	return sb.String()
}

// streamChat handles streaming chat
func (s *Server) streamChat(w http.ResponseWriter, ctx context.Context, m *model.Transformer, tok *tokenizer.Tokenizer, sampler *model.Sampler, modelName string, promptTokens []int, opts Options, startTime time.Time) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	promptEvalStart := time.Now()
	logits := m.Forward(promptTokens, 0)
	lastLogits := logits.Slice(len(promptTokens)-1, len(promptTokens))
	lastLogits = lastLogits.Reshape(m.Config.VocabSize)
	promptEvalDuration := time.Since(promptEvalStart)

	evalStart := time.Now()
	generated := []int{}
	pos := len(promptTokens)
	var fullResponse strings.Builder

	for i := 0; i < opts.NumPredict; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		nextToken := sampler.Sample(lastLogits)
		generated = append(generated, nextToken)

		if tok.IsEOS(nextToken) {
			break
		}

		text := tok.DecodeToken(nextToken)
		fullResponse.WriteString(text)

		resp := ChatResponse{
			Model:     modelName,
			CreatedAt: time.Now(),
			Message:   ChatMessage{Role: "assistant", Content: text},
			Done:      false,
		}

		json.NewEncoder(w).Encode(resp)
		flusher.Flush()

		if containsStop(fullResponse.String(), opts.Stop) {
			break
		}

		logits = m.Forward([]int{nextToken}, pos)
		lastLogits = logits.Reshape(m.Config.VocabSize)
		pos++
	}

	evalDuration := time.Since(evalStart)

	finalResp := ChatResponse{
		Model:     modelName,
		CreatedAt: time.Now(),
		Message:   ChatMessage{Role: "assistant", Content: ""},
		Done:      true,
		DoneReason: "stop",
		TotalDuration:      time.Since(startTime).Nanoseconds(),
		PromptEvalCount:    len(promptTokens),
		PromptEvalDuration: promptEvalDuration.Nanoseconds(),
		EvalCount:          len(generated),
		EvalDuration:       evalDuration.Nanoseconds(),
	}

	json.NewEncoder(w).Encode(finalResp)
	flusher.Flush()
}

// blockingChat handles non-streaming chat
func (s *Server) blockingChat(w http.ResponseWriter, m *model.Transformer, tok *tokenizer.Tokenizer, sampler *model.Sampler, modelName string, promptTokens []int, opts Options, startTime time.Time) {
	promptEvalStart := time.Now()
	logits := m.Forward(promptTokens, 0)
	lastLogits := logits.Slice(len(promptTokens)-1, len(promptTokens))
	lastLogits = lastLogits.Reshape(m.Config.VocabSize)
	promptEvalDuration := time.Since(promptEvalStart)

	evalStart := time.Now()
	generated := []int{}
	pos := len(promptTokens)

	for i := 0; i < opts.NumPredict; i++ {
		nextToken := sampler.Sample(lastLogits)
		generated = append(generated, nextToken)

		if tok.IsEOS(nextToken) {
			break
		}

		if containsStop(tok.Decode(generated), opts.Stop) {
			break
		}

		logits = m.Forward([]int{nextToken}, pos)
		lastLogits = logits.Reshape(m.Config.VocabSize)
		pos++
	}

	evalDuration := time.Since(evalStart)
	responseText := tok.Decode(generated)

	resp := ChatResponse{
		Model:              modelName,
		CreatedAt:          time.Now(),
		Message:            ChatMessage{Role: "assistant", Content: responseText},
		Done:               true,
		DoneReason:         "stop",
		TotalDuration:      time.Since(startTime).Nanoseconds(),
		PromptEvalCount:    len(promptTokens),
		PromptEvalDuration: promptEvalDuration.Nanoseconds(),
		EvalCount:          len(generated),
		EvalDuration:       evalDuration.Nanoseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleEmbeddings handles embedding requests
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	// Embeddings not implemented yet
	http.Error(w, "embeddings not implemented", http.StatusNotImplemented)
}

// handleList lists available models
func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	models := s.models.List()

	resp := ListResponse{Models: models}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleShow shows model information
func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req ShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	name := req.Name
	if name == "" {
		name = req.Model
	}

	info, err := s.models.Show(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// handleDelete deletes a model
func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	name := req.Name
	if name == "" {
		name = req.Model
	}

	if err := s.models.Delete(name); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
}

// handlePull handles model pull requests
func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	// Pull from registry not implemented yet
	http.Error(w, "pull from registry not implemented", http.StatusNotImplemented)
}

// handlePS lists running models
func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	running := s.models.Running()

	type psResponse struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}

	resp := psResponse{}
	for _, name := range running {
		resp.Models = append(resp.Models, struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		}{Name: name, Size: 0})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// containsStop checks if text contains any stop sequence
func containsStop(text string, stops []string) bool {
	for _, stop := range stops {
		if strings.Contains(text, stop) {
			return true
		}
	}
	return false
}

// ModelManager handles model loading and caching
type ModelManager struct {
	modelsDir string
	loaded    map[string]*LoadedModel
	mu        sync.RWMutex
}

// LoadedModel represents a loaded model
type LoadedModel struct {
	Model     *model.Transformer
	Tokenizer *tokenizer.Tokenizer
	Path      string
}

// NewModelManager creates a new model manager
func NewModelManager(modelsDir string) *ModelManager {
	return &ModelManager{
		modelsDir: modelsDir,
		loaded:    make(map[string]*LoadedModel),
	}
}

// Load loads a model by name or path
func (m *ModelManager) Load(name string) (*model.Transformer, *tokenizer.Tokenizer, error) {
	m.mu.RLock()
	if lm, ok := m.loaded[name]; ok {
		m.mu.RUnlock()
		return lm.Model, lm.Tokenizer, nil
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	// Check again after acquiring write lock
	if lm, ok := m.loaded[name]; ok {
		return lm.Model, lm.Tokenizer, nil
	}

	// Determine path
	path := name
	if !strings.HasSuffix(path, ".gguf") {
		path = fmt.Sprintf("%s/%s.gguf", m.modelsDir, name)
	}

	// Open GGUF file
	r, err := gguf.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open model: %w", err)
	}

	// Load model
	mdl, err := model.LoadFromGGUF(r)
	if err != nil {
		r.Close()
		return nil, nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Load tokenizer
	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		r.Close()
		return nil, nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	m.loaded[name] = &LoadedModel{
		Model:     mdl,
		Tokenizer: tok,
		Path:      path,
	}

	log.Printf("Loaded model: %s", name)
	return mdl, tok, nil
}

// List returns available models
func (m *ModelManager) List() []ModelInfo {
	// For now, just return loaded models
	m.mu.RLock()
	defer m.mu.RUnlock()

	var models []ModelInfo
	for name, lm := range m.loaded {
		models = append(models, ModelInfo{
			Name:       name,
			Model:      name,
			ModifiedAt: time.Now(),
			Details: ModelDetails{
				Format: "gguf",
				Family: lm.Model.Config.Architecture,
			},
		})
	}

	return models
}

// Show returns detailed model information
func (m *ModelManager) Show(name string) (*ShowResponse, error) {
	m.mu.RLock()
	lm, ok := m.loaded[name]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("model not found: %s", name)
	}

	return &ShowResponse{
		Details: ModelDetails{
			Format: "gguf",
			Family: lm.Model.Config.Architecture,
		},
		ModifiedAt: time.Now(),
	}, nil
}

// Delete removes a model
func (m *ModelManager) Delete(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.loaded[name]; !ok {
		return fmt.Errorf("model not found: %s", name)
	}

	delete(m.loaded, name)
	return nil
}

// Running returns list of running model names
func (m *ModelManager) Running() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var names []string
	for name := range m.loaded {
		names = append(names, name)
	}
	return names
}
