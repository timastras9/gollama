package model

import (
	"math"

	"github.com/timastras9/gollama/pkg/tensor"
)

// Attention implements multi-head attention with support for:
// - Grouped Query Attention (GQA)
// - KV-cache for efficient generation
// - RoPE positional embeddings
type Attention struct {
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	HiddenSize int

	// Weight matrices
	WQ *tensor.Tensor // [hidden_size, num_heads * head_dim]
	WK *tensor.Tensor // [hidden_size, num_kv_heads * head_dim]
	WV *tensor.Tensor // [hidden_size, num_kv_heads * head_dim]
	WO *tensor.Tensor // [num_heads * head_dim, hidden_size]

	// RoPE instance
	RoPE *RoPE

	// KV-Cache
	KeyCache   *tensor.Tensor // [max_seq_len, num_kv_heads * head_dim]
	ValueCache *tensor.Tensor // [max_seq_len, num_kv_heads * head_dim]
}

// NewAttention creates a new attention layer
func NewAttention(cfg *Config, rope *RoPE) *Attention {
	kvDim := cfg.NumKVHeads * cfg.HeadDim

	return &Attention{
		NumHeads:   cfg.NumHeads,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		HiddenSize: cfg.HiddenSize,
		WQ:         tensor.New(cfg.HiddenSize, cfg.NumHeads*cfg.HeadDim),
		WK:         tensor.New(cfg.HiddenSize, kvDim),
		WV:         tensor.New(cfg.HiddenSize, kvDim),
		WO:         tensor.New(cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize),
		RoPE:       rope,
		KeyCache:   tensor.New(cfg.MaxSeqLen, kvDim),
		ValueCache: tensor.New(cfg.MaxSeqLen, kvDim),
	}
}

// Forward computes attention output
// x: [seq_len, hidden_size]
// startPos: position offset for KV-cache
// Returns: [seq_len, hidden_size]
func (a *Attention) Forward(x *tensor.Tensor, startPos int) *tensor.Tensor {
	seqLen := x.Shape[0]

	// Project to Q, K, V
	q := tensor.MatMul(x, a.WQ) // [seq_len, num_heads * head_dim]
	k := tensor.MatMul(x, a.WK) // [seq_len, num_kv_heads * head_dim]
	v := tensor.MatMul(x, a.WV) // [seq_len, num_kv_heads * head_dim]

	// Reshape for multi-head attention
	// Q: [seq_len, num_heads, head_dim]
	// K, V: [seq_len, num_kv_heads, head_dim]
	qReshaped := q.Reshape(seqLen, a.NumHeads, a.HeadDim)
	kReshaped := k.Reshape(seqLen, a.NumKVHeads, a.HeadDim)

	// Apply RoPE to Q and K
	a.RoPE.Apply(qReshaped, kReshaped, startPos)

	// Flatten back for cache operations
	q = qReshaped.Reshape(seqLen, a.NumHeads*a.HeadDim)
	k = kReshaped.Reshape(seqLen, a.NumKVHeads*a.HeadDim)

	// Update KV-cache
	a.updateCache(k, v, startPos, seqLen)

	// Get full K, V from cache
	cacheLen := startPos + seqLen
	fullK := a.KeyCache.Slice(0, cacheLen)
	fullV := a.ValueCache.Slice(0, cacheLen)

	// Compute attention for each head
	output := a.computeAttention(q, fullK, fullV, startPos, seqLen, cacheLen)

	// Output projection
	return tensor.MatMul(output, a.WO)
}

// updateCache updates the KV-cache with new K, V values
func (a *Attention) updateCache(k, v *tensor.Tensor, startPos, seqLen int) {
	kvDim := a.NumKVHeads * a.HeadDim

	for s := 0; s < seqLen; s++ {
		srcOffset := s * kvDim
		dstOffset := (startPos + s) * kvDim

		copy(a.KeyCache.Data[dstOffset:dstOffset+kvDim], k.Data[srcOffset:srcOffset+kvDim])
		copy(a.ValueCache.Data[dstOffset:dstOffset+kvDim], v.Data[srcOffset:srcOffset+kvDim])
	}
}

// computeAttention computes scaled dot-product attention with GQA support
func (a *Attention) computeAttention(q, fullK, fullV *tensor.Tensor, startPos, seqLen, cacheLen int) *tensor.Tensor {
	output := tensor.New(seqLen, a.NumHeads*a.HeadDim)
	scale := float32(1.0 / math.Sqrt(float64(a.HeadDim)))

	// Number of Q heads per KV head (for GQA)
	headsPerKV := a.NumHeads / a.NumKVHeads

	// Process each query position
	for pos := 0; pos < seqLen; pos++ {
		absPos := startPos + pos

		// Process each head
		for h := 0; h < a.NumHeads; h++ {
			// Determine which KV head to use
			kvHead := h / headsPerKV

			// Get query vector for this head
			qOffset := pos*a.NumHeads*a.HeadDim + h*a.HeadDim
			qVec := q.Data[qOffset : qOffset+a.HeadDim]

			// Compute attention scores for all cached positions
			scores := make([]float32, cacheLen)
			maxScore := float32(math.Inf(-1))

			for kPos := 0; kPos <= absPos; kPos++ { // Causal mask: only attend to past and current
				kOffset := kPos*a.NumKVHeads*a.HeadDim + kvHead*a.HeadDim
				kVec := fullK.Data[kOffset : kOffset+a.HeadDim]

				// Dot product
				var dot float32
				for i := 0; i < a.HeadDim; i++ {
					dot += qVec[i] * kVec[i]
				}
				scores[kPos] = dot * scale

				if scores[kPos] > maxScore {
					maxScore = scores[kPos]
				}
			}

			// Softmax (only over valid positions)
			var sumExp float32
			for kPos := 0; kPos <= absPos; kPos++ {
				scores[kPos] = float32(math.Exp(float64(scores[kPos] - maxScore)))
				sumExp += scores[kPos]
			}
			for kPos := 0; kPos <= absPos; kPos++ {
				scores[kPos] /= sumExp
			}

			// Weighted sum of values
			outOffset := pos*a.NumHeads*a.HeadDim + h*a.HeadDim
			for i := 0; i < a.HeadDim; i++ {
				var sum float32
				for vPos := 0; vPos <= absPos; vPos++ {
					vOffset := vPos*a.NumKVHeads*a.HeadDim + kvHead*a.HeadDim
					sum += scores[vPos] * fullV.Data[vOffset+i]
				}
				output.Data[outOffset+i] = sum
			}
		}
	}

	return output
}

// ResetCache clears the KV-cache
func (a *Attention) ResetCache() {
	a.KeyCache.Zero()
	a.ValueCache.Zero()
}
