package model

import (
	"math"

	"github.com/timastras9/gollama/pkg/tensor"
)

// RoPE implements Rotary Position Embeddings
// Reference: https://arxiv.org/abs/2104.09864
type RoPE struct {
	Dim       int     // Dimension to apply rotations (usually head_dim)
	MaxSeqLen int     // Maximum sequence length
	Theta     float32 // Base frequency (usually 10000)

	// Precomputed frequency tables
	FreqsCos *tensor.Tensor // [MaxSeqLen, Dim/2]
	FreqsSin *tensor.Tensor // [MaxSeqLen, Dim/2]
}

// NewRoPE creates a new RoPE instance with precomputed frequencies
func NewRoPE(dim, maxSeqLen int, theta float32) *RoPE {
	rope := &RoPE{
		Dim:       dim,
		MaxSeqLen: maxSeqLen,
		Theta:     theta,
	}
	rope.precomputeFreqs()
	return rope
}

// precomputeFreqs precomputes sin and cos values for all positions
func (r *RoPE) precomputeFreqs() {
	halfDim := r.Dim / 2
	r.FreqsCos = tensor.New(r.MaxSeqLen, halfDim)
	r.FreqsSin = tensor.New(r.MaxSeqLen, halfDim)

	for pos := 0; pos < r.MaxSeqLen; pos++ {
		for i := 0; i < halfDim; i++ {
			// freq = 1 / (theta ^ (2i / dim))
			freq := 1.0 / math.Pow(float64(r.Theta), float64(2*i)/float64(r.Dim))
			angle := float64(pos) * freq

			r.FreqsCos.Set(float32(math.Cos(angle)), pos, i)
			r.FreqsSin.Set(float32(math.Sin(angle)), pos, i)
		}
	}
}

// Apply applies rotary embeddings to query and key tensors
// q, k are [seq_len, num_heads, head_dim] or [seq_len, head_dim]
// startPos is the position offset (for incremental generation)
func (r *RoPE) Apply(q, k *tensor.Tensor, startPos int) {
	if len(q.Shape) == 2 {
		r.apply2D(q, startPos)
		r.apply2D(k, startPos)
	} else if len(q.Shape) == 3 {
		r.apply3D(q, startPos)
		r.apply3D(k, startPos)
	} else {
		panic("RoPE.Apply: unsupported tensor shape")
	}
}

// apply2D applies RoPE to a 2D tensor [seq_len, head_dim]
func (r *RoPE) apply2D(x *tensor.Tensor, startPos int) {
	seqLen := x.Shape[0]
	headDim := x.Shape[1]
	halfDim := headDim / 2

	for pos := 0; pos < seqLen; pos++ {
		absPos := startPos + pos

		for i := 0; i < halfDim; i++ {
			cos := r.FreqsCos.At(absPos, i)
			sin := r.FreqsSin.At(absPos, i)

			// Get the pair of values to rotate
			idx0 := pos*headDim + i*2
			idx1 := pos*headDim + i*2 + 1

			x0 := x.Data[idx0]
			x1 := x.Data[idx1]

			// Apply rotation: [cos, -sin; sin, cos] @ [x0, x1]
			x.Data[idx0] = x0*cos - x1*sin
			x.Data[idx1] = x0*sin + x1*cos
		}
	}
}

// apply3D applies RoPE to a 3D tensor [seq_len, num_heads, head_dim]
func (r *RoPE) apply3D(x *tensor.Tensor, startPos int) {
	seqLen := x.Shape[0]
	numHeads := x.Shape[1]
	headDim := x.Shape[2]
	halfDim := headDim / 2

	for pos := 0; pos < seqLen; pos++ {
		absPos := startPos + pos

		for h := 0; h < numHeads; h++ {
			for i := 0; i < halfDim; i++ {
				cos := r.FreqsCos.At(absPos, i)
				sin := r.FreqsSin.At(absPos, i)

				// Calculate indices
				baseIdx := pos*numHeads*headDim + h*headDim
				idx0 := baseIdx + i*2
				idx1 := baseIdx + i*2 + 1

				x0 := x.Data[idx0]
				x1 := x.Data[idx1]

				// Apply rotation
				x.Data[idx0] = x0*cos - x1*sin
				x.Data[idx1] = x0*sin + x1*cos
			}
		}
	}
}

// ApplyQuery applies RoPE only to query tensor
func (r *RoPE) ApplyQuery(q *tensor.Tensor, startPos int) {
	if len(q.Shape) == 2 {
		r.apply2D(q, startPos)
	} else if len(q.Shape) == 3 {
		r.apply3D(q, startPos)
	}
}

// ApplyKey applies RoPE only to key tensor
func (r *RoPE) ApplyKey(k *tensor.Tensor, startPos int) {
	if len(k.Shape) == 2 {
		r.apply2D(k, startPos)
	} else if len(k.Shape) == 3 {
		r.apply3D(k, startPos)
	}
}
