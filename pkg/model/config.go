package model

import (
	"fmt"

	"github.com/timastras9/gollama/pkg/gguf"
)

// Config holds the model architecture configuration
type Config struct {
	// Architecture
	Architecture string
	VocabSize    int
	HiddenSize   int // embedding_length
	IntermediateSize int // feed_forward_length
	NumLayers    int // block_count
	NumHeads     int // attention.head_count
	NumKVHeads   int // attention.head_count_kv (for GQA)
	HeadDim      int // hidden_size / num_heads

	// Context
	MaxSeqLen int     // context_length
	RopeTheta float32 // rope.freq_base
	RopeDim   int     // rope.dimension_count (usually head_dim)

	// Normalization
	NormEps float32 // attention.layer_norm_rms_epsilon

	// Tokenizer
	BOSTokenID int
	EOSTokenID int
	PadTokenID int
}

// ConfigFromGGUF extracts model configuration from GGUF metadata
func ConfigFromGGUF(r *gguf.Reader) (*Config, error) {
	arch := r.Architecture()
	if arch == "" {
		return nil, fmt.Errorf("architecture not found in metadata")
	}

	cfg := &Config{
		Architecture: arch,
		NormEps:      1e-5,  // Default
		RopeTheta:    10000, // Default
	}

	// Helper to get arch-prefixed keys
	getUint32 := func(suffix string) (uint32, bool) {
		return r.GetUint32(arch + suffix)
	}
	getFloat32 := func(suffix string) (float32, bool) {
		return r.GetFloat32(arch + suffix)
	}

	// Required parameters
	if v, ok := getUint32(".embedding_length"); ok {
		cfg.HiddenSize = int(v)
	} else {
		return nil, fmt.Errorf("embedding_length not found")
	}

	if v, ok := getUint32(".block_count"); ok {
		cfg.NumLayers = int(v)
	} else {
		return nil, fmt.Errorf("block_count not found")
	}

	if v, ok := getUint32(".attention.head_count"); ok {
		cfg.NumHeads = int(v)
	} else {
		return nil, fmt.Errorf("attention.head_count not found")
	}

	// Optional parameters with defaults
	if v, ok := getUint32(".feed_forward_length"); ok {
		cfg.IntermediateSize = int(v)
	} else {
		// Default: 4 * hidden_size (common for older models)
		// For SwiGLU models it's typically (8/3) * hidden_size, rounded
		cfg.IntermediateSize = 4 * cfg.HiddenSize
	}

	if v, ok := getUint32(".attention.head_count_kv"); ok {
		cfg.NumKVHeads = int(v)
	} else {
		// Default: same as num_heads (no GQA)
		cfg.NumKVHeads = cfg.NumHeads
	}

	if v, ok := getUint32(".context_length"); ok {
		cfg.MaxSeqLen = int(v)
	} else {
		cfg.MaxSeqLen = 2048 // Default fallback
	}

	if v, ok := getFloat32(".attention.layer_norm_rms_epsilon"); ok {
		cfg.NormEps = v
	}

	if v, ok := getFloat32(".rope.freq_base"); ok {
		cfg.RopeTheta = v
	}

	if v, ok := getUint32(".rope.dimension_count"); ok {
		cfg.RopeDim = int(v)
	} else {
		cfg.RopeDim = cfg.HiddenSize / cfg.NumHeads
	}

	// Calculate head dimension
	cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads

	// Tokenizer IDs
	if v, ok := r.GetUint32(gguf.KeyTokenizerBOSID); ok {
		cfg.BOSTokenID = int(v)
	} else {
		cfg.BOSTokenID = 1 // Common default
	}

	if v, ok := r.GetUint32(gguf.KeyTokenizerEOSID); ok {
		cfg.EOSTokenID = int(v)
	} else {
		cfg.EOSTokenID = 2 // Common default
	}

	if v, ok := r.GetUint32(gguf.KeyTokenizerPadID); ok {
		cfg.PadTokenID = int(v)
	} else {
		cfg.PadTokenID = 0
	}

	// Get vocab size from tokenizer tokens array
	if tokens, ok := r.GetStringArray(gguf.KeyTokenizerTokens); ok {
		cfg.VocabSize = len(tokens)
	}

	return cfg, nil
}

// String returns a human-readable representation of the config
func (c *Config) String() string {
	return fmt.Sprintf(`Model Config:
  Architecture: %s
  Vocab Size: %d
  Hidden Size: %d
  Intermediate Size: %d
  Num Layers: %d
  Num Heads: %d
  Num KV Heads: %d
  Head Dim: %d
  Max Seq Len: %d
  RoPE Theta: %.1f
  Norm Eps: %e
  BOS Token: %d
  EOS Token: %d`,
		c.Architecture,
		c.VocabSize,
		c.HiddenSize,
		c.IntermediateSize,
		c.NumLayers,
		c.NumHeads,
		c.NumKVHeads,
		c.HeadDim,
		c.MaxSeqLen,
		c.RopeTheta,
		c.NormEps,
		c.BOSTokenID,
		c.EOSTokenID,
	)
}
