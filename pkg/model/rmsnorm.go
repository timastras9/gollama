package model

import (
	"math"

	"github.com/timastras9/gollama/pkg/tensor"
)

// RMSNorm implements Root Mean Square Layer Normalization
// Used in LLaMA and similar models instead of LayerNorm
type RMSNorm struct {
	Weight *tensor.Tensor // Learnable scale parameters [hidden_size]
	Eps    float32
}

// NewRMSNorm creates a new RMSNorm layer
func NewRMSNorm(hiddenSize int, eps float32) *RMSNorm {
	return &RMSNorm{
		Weight: tensor.New(hiddenSize),
		Eps:    eps,
	}
}

// Forward applies RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
// Input x is [seq_len, hidden_size]
// Output is same shape as input
func (n *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) == 1 {
		return n.forward1D(x)
	}
	return n.forward2D(x)
}

// forward1D applies RMSNorm to a 1D tensor [hidden_size]
func (n *RMSNorm) forward1D(x *tensor.Tensor) *tensor.Tensor {
	hiddenSize := x.Shape[0]
	out := tensor.New(hiddenSize)

	// Compute RMS = sqrt(mean(x^2) + eps)
	var sumSq float32
	for i := 0; i < hiddenSize; i++ {
		sumSq += x.Data[i] * x.Data[i]
	}
	rms := float32(math.Sqrt(float64(sumSq/float32(hiddenSize) + n.Eps)))
	invRms := 1.0 / rms

	// Normalize and scale
	for i := 0; i < hiddenSize; i++ {
		out.Data[i] = x.Data[i] * invRms * n.Weight.Data[i]
	}

	return out
}

// forward2D applies RMSNorm to a 2D tensor [seq_len, hidden_size]
// Normalizes each row independently
func (n *RMSNorm) forward2D(x *tensor.Tensor) *tensor.Tensor {
	seqLen, hiddenSize := x.Shape[0], x.Shape[1]
	out := tensor.New(seqLen, hiddenSize)

	for s := 0; s < seqLen; s++ {
		rowStart := s * hiddenSize

		// Compute RMS for this row
		var sumSq float32
		for i := 0; i < hiddenSize; i++ {
			v := x.Data[rowStart+i]
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hiddenSize) + n.Eps)))
		invRms := 1.0 / rms

		// Normalize and scale
		for i := 0; i < hiddenSize; i++ {
			out.Data[rowStart+i] = x.Data[rowStart+i] * invRms * n.Weight.Data[i]
		}
	}

	return out
}

// ForwardInplace applies RMSNorm in place
func (n *RMSNorm) ForwardInplace(x *tensor.Tensor) {
	if len(x.Shape) == 1 {
		n.forwardInplace1D(x)
		return
	}
	n.forwardInplace2D(x)
}

func (n *RMSNorm) forwardInplace1D(x *tensor.Tensor) {
	hiddenSize := x.Shape[0]

	var sumSq float32
	for i := 0; i < hiddenSize; i++ {
		sumSq += x.Data[i] * x.Data[i]
	}
	rms := float32(math.Sqrt(float64(sumSq/float32(hiddenSize) + n.Eps)))
	invRms := 1.0 / rms

	for i := 0; i < hiddenSize; i++ {
		x.Data[i] = x.Data[i] * invRms * n.Weight.Data[i]
	}
}

func (n *RMSNorm) forwardInplace2D(x *tensor.Tensor) {
	seqLen, hiddenSize := x.Shape[0], x.Shape[1]

	for s := 0; s < seqLen; s++ {
		rowStart := s * hiddenSize

		var sumSq float32
		for i := 0; i < hiddenSize; i++ {
			v := x.Data[rowStart+i]
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hiddenSize) + n.Eps)))
		invRms := 1.0 / rms

		for i := 0; i < hiddenSize; i++ {
			x.Data[rowStart+i] = x.Data[rowStart+i] * invRms * n.Weight.Data[i]
		}
	}
}
