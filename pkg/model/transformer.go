package model

import (
	"fmt"

	"github.com/timastras9/gollama/pkg/gguf"
	"github.com/timastras9/gollama/pkg/quant"
	"github.com/timastras9/gollama/pkg/tensor"
)

// TransformerBlock represents a single transformer layer
type TransformerBlock struct {
	// Pre-attention normalization
	AttentionNorm *RMSNorm

	// Self-attention
	Attention *Attention

	// Pre-FFN normalization
	FFNNorm *RMSNorm

	// Feed-forward network
	FeedForward *FeedForward
}

// Forward computes one transformer block
// x: [seq_len, hidden_size]
// startPos: position for KV-cache
func (b *TransformerBlock) Forward(x *tensor.Tensor, startPos int) *tensor.Tensor {
	// Pre-norm attention with residual
	h := b.AttentionNorm.Forward(x)
	h = b.Attention.Forward(h, startPos)
	x = tensor.Add(x, h)

	// Pre-norm FFN with residual
	h = b.FFNNorm.Forward(x)
	h = b.FeedForward.Forward(h)
	x = tensor.Add(x, h)

	return x
}

// Transformer is the main model structure
type Transformer struct {
	Config *Config

	// Token embedding table
	TokenEmbed *tensor.Tensor // [vocab_size, hidden_size]

	// Transformer layers
	Layers []*TransformerBlock

	// Output normalization
	OutputNorm *RMSNorm

	// Output projection (often tied to embedding)
	Output *tensor.Tensor // [hidden_size, vocab_size]

	// RoPE (shared across layers)
	RoPE *RoPE
}

// NewTransformer creates a new transformer model from config
func NewTransformer(cfg *Config) *Transformer {
	rope := NewRoPE(cfg.HeadDim, cfg.MaxSeqLen, cfg.RopeTheta)

	layers := make([]*TransformerBlock, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layers[i] = &TransformerBlock{
			AttentionNorm: NewRMSNorm(cfg.HiddenSize, cfg.NormEps),
			Attention:     NewAttention(cfg, rope),
			FFNNorm:       NewRMSNorm(cfg.HiddenSize, cfg.NormEps),
			FeedForward:   NewFeedForward(cfg.HiddenSize, cfg.IntermediateSize),
		}
	}

	return &Transformer{
		Config:     cfg,
		TokenEmbed: tensor.New(cfg.VocabSize, cfg.HiddenSize), // GGUF: [vocab, hidden]
		Layers:     layers,
		OutputNorm: NewRMSNorm(cfg.HiddenSize, cfg.NormEps),
		Output:     tensor.New(cfg.VocabSize, cfg.HiddenSize), // GGUF: [vocab, hidden]
		RoPE:       rope,
	}
}

// Forward runs the transformer on input tokens
// tokens: slice of token IDs
// startPos: position offset for KV-cache
// Returns: logits [seq_len, vocab_size]
func (t *Transformer) Forward(tokens []int, startPos int) *tensor.Tensor {
	seqLen := len(tokens)

	// Get token embeddings
	x := t.embed(tokens)

	// Apply transformer layers
	for _, layer := range t.Layers {
		x = layer.Forward(x, startPos)
	}

	// Final normalization
	x = t.OutputNorm.Forward(x)

	// Output projection to vocabulary
	// Output is [vocab, hidden], x is [seq, hidden]
	// logits = x @ Output^T = [seq, vocab]
	logits := tensor.MatMulTransposeB(x, t.Output)

	return logits.Reshape(seqLen, t.Config.VocabSize)
}

// embed looks up token embeddings
// TokenEmbed is [vocab, hidden], so row `token` contains the embedding
func (t *Transformer) embed(tokens []int) *tensor.Tensor {
	seqLen := len(tokens)
	hiddenSize := t.Config.HiddenSize
	vocabSize := t.Config.VocabSize

	x := tensor.New(seqLen, hiddenSize)

	for i, token := range tokens {
		if token < 0 || token >= vocabSize {
			panic(fmt.Sprintf("token %d out of range [0, %d)", token, vocabSize))
		}

		// TokenEmbed is [vocab, hidden], row `token` is the embedding
		// Copy row: element[h] = TokenEmbed[token * hidden + h]
		dstOffset := i * hiddenSize
		srcOffset := token * hiddenSize
		copy(x.Data[dstOffset:dstOffset+hiddenSize], t.TokenEmbed.Data[srcOffset:srcOffset+hiddenSize])
	}

	return x
}

// Generate generates tokens autoregressively
// prompt: initial token IDs
// maxTokens: maximum number of tokens to generate
// Returns: generated token IDs (including prompt)
func (t *Transformer) Generate(prompt []int, maxTokens int, sampler *Sampler) []int {
	tokens := make([]int, 0, len(prompt)+maxTokens)
	tokens = append(tokens, prompt...)

	// Process prompt (prefill)
	logits := t.Forward(prompt, 0)

	// Get last token's logits for first generation
	lastLogits := logits.Slice(len(prompt)-1, len(prompt))
	lastLogits = lastLogits.Reshape(t.Config.VocabSize)

	nextToken := sampler.Sample(lastLogits)
	tokens = append(tokens, nextToken)

	// Generate remaining tokens
	for i := 1; i < maxTokens; i++ {
		if nextToken == t.Config.EOSTokenID {
			break
		}

		// Run model on just the new token
		logits = t.Forward([]int{nextToken}, len(prompt)+i-1)
		logits = logits.Reshape(t.Config.VocabSize)

		nextToken = sampler.Sample(logits)
		tokens = append(tokens, nextToken)
	}

	return tokens
}

// ResetCache clears all KV-caches
func (t *Transformer) ResetCache() {
	for _, layer := range t.Layers {
		layer.Attention.ResetCache()
	}
}

// LoadFromGGUF loads model weights from a GGUF file
func LoadFromGGUF(r *gguf.Reader) (*Transformer, error) {
	// Get config from metadata
	cfg, err := ConfigFromGGUF(r)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Create model
	model := NewTransformer(cfg)

	// Load weights
	if err := model.loadWeights(r); err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	return model, nil
}

// loadWeights loads all weights from GGUF
func (t *Transformer) loadWeights(r *gguf.Reader) error {
	arch := t.Config.Architecture

	// Load token embeddings
	if err := t.loadTensor(r, "token_embd.weight", t.TokenEmbed); err != nil {
		return err
	}

	// Load each layer
	for i := 0; i < t.Config.NumLayers; i++ {
		layer := t.Layers[i]
		prefix := fmt.Sprintf("blk.%d.", i)

		// Attention norm
		if err := t.loadTensor(r, prefix+"attn_norm.weight", layer.AttentionNorm.Weight); err != nil {
			return err
		}

		// Attention weights
		if err := t.loadTensor(r, prefix+"attn_q.weight", layer.Attention.WQ); err != nil {
			return err
		}
		if err := t.loadTensor(r, prefix+"attn_k.weight", layer.Attention.WK); err != nil {
			return err
		}
		if err := t.loadTensor(r, prefix+"attn_v.weight", layer.Attention.WV); err != nil {
			return err
		}
		if err := t.loadTensor(r, prefix+"attn_output.weight", layer.Attention.WO); err != nil {
			return err
		}

		// FFN norm
		if err := t.loadTensor(r, prefix+"ffn_norm.weight", layer.FFNNorm.Weight); err != nil {
			return err
		}

		// FFN weights - naming varies by architecture
		gateKey := prefix + "ffn_gate.weight"
		upKey := prefix + "ffn_up.weight"
		downKey := prefix + "ffn_down.weight"

		// Some models use different naming
		if _, ok := r.GetTensorInfo(gateKey); !ok {
			// Try alternate naming (e.g., for Phi models)
			gateKey = prefix + "ffn.gate.weight"
			upKey = prefix + "ffn.up.weight"
			downKey = prefix + "ffn.down.weight"
		}

		if err := t.loadTensor(r, gateKey, layer.FeedForward.WGate); err != nil {
			return err
		}
		if err := t.loadTensor(r, upKey, layer.FeedForward.WUp); err != nil {
			return err
		}
		if err := t.loadTensor(r, downKey, layer.FeedForward.WDown); err != nil {
			return err
		}
	}

	// Output norm
	if err := t.loadTensor(r, "output_norm.weight", t.OutputNorm.Weight); err != nil {
		return err
	}

	// Output projection - may be tied to embeddings
	// GGUF stores output as [hidden, vocab], we use MatMulTransposeB so keep as-is
	outputKey := "output.weight"
	if _, ok := r.GetTensorInfo(outputKey); ok {
		if err := t.loadTensor(r, outputKey, t.Output); err != nil {
			return err
		}
	} else {
		// Tie output to embeddings
		t.tieOutputToEmbedding()
	}

	_ = arch // May use arch-specific loading in future

	return nil
}

// loadTensor loads a single tensor from GGUF and dequantizes if needed
func (t *Transformer) loadTensor(r *gguf.Reader, name string, dst *tensor.Tensor) error {
	info, ok := r.GetTensorInfo(name)
	if !ok {
		return fmt.Errorf("tensor not found: %s", name)
	}

	data, err := r.GetTensorData(name)
	if err != nil {
		return err
	}

	// GGUF stores tensors in row-major order matching our layout
	// Weight matrices are [rows, cols] where for linear layers:
	// - Input x is [batch, in_features]
	// - Weight W is [in_features, out_features]
	// - Output y = x @ W is [batch, out_features]
	quant.Dequantize(data, dst.Data, info.Type)

	return nil
}

// loadTensorTransposed loads a tensor and transposes it
func (t *Transformer) loadTensorTransposed(r *gguf.Reader, name string, dst *tensor.Tensor) error {
	info, ok := r.GetTensorInfo(name)
	if !ok {
		return fmt.Errorf("tensor not found: %s", name)
	}

	data, err := r.GetTensorData(name)
	if err != nil {
		return err
	}

	shape := info.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("expected 2D tensor for transpose: %s", name)
	}

	rows, cols := shape[0], shape[1]
	temp := make([]float32, rows*cols)
	quant.Dequantize(data, temp, info.Type)

	// Transpose: [rows, cols] -> [cols, rows]
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dst.Data[j*rows+i] = temp[i*cols+j]
		}
	}

	return nil
}

// tieOutputToEmbedding copies embedding weights to output (both [hidden, vocab])
func (t *Transformer) tieOutputToEmbedding() {
	copy(t.Output.Data, t.TokenEmbed.Data)
}
