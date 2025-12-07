package model

import (
	"math"

	"github.com/timastras9/gollama/pkg/tensor"
)

// FeedForward implements the feed-forward network with SwiGLU activation
// Used in LLaMA, Mistral, and similar models
// FFN(x) = down(silu(gate(x)) * up(x))
type FeedForward struct {
	HiddenSize       int
	IntermediateSize int

	// Weight matrices
	WGate *tensor.Tensor // [hidden_size, intermediate_size]
	WUp   *tensor.Tensor // [hidden_size, intermediate_size]
	WDown *tensor.Tensor // [intermediate_size, hidden_size]
}

// NewFeedForward creates a new feed-forward layer
func NewFeedForward(hiddenSize, intermediateSize int) *FeedForward {
	return &FeedForward{
		HiddenSize:       hiddenSize,
		IntermediateSize: intermediateSize,
		// GGUF layout after dimension reversal: [out, in]
		// We use MatMulTransposeB for x @ W^T
		WGate: tensor.New(intermediateSize, hiddenSize), // [intermediate, hidden]
		WUp:   tensor.New(intermediateSize, hiddenSize), // [intermediate, hidden]
		WDown: tensor.New(hiddenSize, intermediateSize), // [hidden, intermediate]
	}
}

// Forward computes the feed-forward output
// x: [seq_len, hidden_size]
// Returns: [seq_len, hidden_size]
func (ff *FeedForward) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Gate projection using transposed weights: [seq_len, intermediate_size]
	gate := tensor.MatMulTransposeB(x, ff.WGate)

	// Up projection using transposed weights: [seq_len, intermediate_size]
	up := tensor.MatMulTransposeB(x, ff.WUp)

	// Apply SiLU to gate and multiply with up
	// hidden = silu(gate) * up
	ff.siluMulInplace(gate, up)

	// Down projection using transposed weights: [seq_len, hidden_size]
	return tensor.MatMulTransposeB(gate, ff.WDown)
}

// siluMulInplace computes gate = silu(gate) * up in place
func (ff *FeedForward) siluMulInplace(gate, up *tensor.Tensor) {
	for i := range gate.Data {
		// SiLU(x) = x * sigmoid(x)
		x := gate.Data[i]
		silu := x * sigmoid(x)
		gate.Data[i] = silu * up.Data[i]
	}
}

// sigmoid computes the sigmoid function
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// ForwardGELU computes FFN with GELU activation (for GPT-style models)
// FFN(x) = down(gelu(gate(x)))
func (ff *FeedForward) ForwardGELU(x *tensor.Tensor) *tensor.Tensor {
	// Project up
	hidden := tensor.MatMul(x, ff.WGate)

	// Apply GELU
	for i := range hidden.Data {
		hidden.Data[i] = gelu(hidden.Data[i])
	}

	// Project down
	return tensor.MatMul(hidden, ff.WDown)
}

// gelu computes the GELU activation (approximate)
func gelu(x float32) float32 {
	// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	return float32(0.5 * float64(x) * (1.0 + math.Tanh(0.7978845608*float64(x)*(1.0+0.044715*float64(x)*float64(x)))))
}
