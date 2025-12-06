package tensor

import "math"

// Softmax computes softmax along the last dimension
// For 2D tensor [M, N], applies softmax to each row
func Softmax(a *Tensor) *Tensor {
	out := NewLike(a)

	switch len(a.Shape) {
	case 1:
		softmax1D(a.Data, out.Data)
	case 2:
		rows, cols := a.Shape[0], a.Shape[1]
		for i := 0; i < rows; i++ {
			start := i * cols
			softmax1D(a.Data[start:start+cols], out.Data[start:start+cols])
		}
	case 3:
		// For 3D tensor [batch, seq, dim], apply softmax along last dim
		batch, seq, dim := a.Shape[0], a.Shape[1], a.Shape[2]
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				start := b*seq*dim + s*dim
				softmax1D(a.Data[start:start+dim], out.Data[start:start+dim])
			}
		}
	default:
		panic("Softmax only supports 1D, 2D, or 3D tensors")
	}

	return out
}

// SoftmaxInplace computes softmax in place
func SoftmaxInplace(a *Tensor) {
	switch len(a.Shape) {
	case 1:
		softmax1D(a.Data, a.Data)
	case 2:
		rows, cols := a.Shape[0], a.Shape[1]
		for i := 0; i < rows; i++ {
			start := i * cols
			softmax1D(a.Data[start:start+cols], a.Data[start:start+cols])
		}
	case 3:
		batch, seq, dim := a.Shape[0], a.Shape[1], a.Shape[2]
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				start := b*seq*dim + s*dim
				softmax1D(a.Data[start:start+dim], a.Data[start:start+dim])
			}
		}
	default:
		panic("SoftmaxInplace only supports 1D, 2D, or 3D tensors")
	}
}

// softmax1D computes softmax on a 1D slice
// Uses the numerically stable version: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
func softmax1D(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}

	// Find max for numerical stability
	maxVal := in[0]
	for i := 1; i < n; i++ {
		if in[i] > maxVal {
			maxVal = in[i]
		}
	}

	// Compute exp(x - max) and sum
	var sum float32
	for i := 0; i < n; i++ {
		out[i] = float32(math.Exp(float64(in[i] - maxVal)))
		sum += out[i]
	}

	// Normalize
	invSum := 1.0 / sum
	for i := 0; i < n; i++ {
		out[i] *= invSum
	}
}

// LogSoftmax computes log(softmax(x)) along the last dimension
// More numerically stable than computing log after softmax
func LogSoftmax(a *Tensor) *Tensor {
	out := NewLike(a)

	switch len(a.Shape) {
	case 1:
		logSoftmax1D(a.Data, out.Data)
	case 2:
		rows, cols := a.Shape[0], a.Shape[1]
		for i := 0; i < rows; i++ {
			start := i * cols
			logSoftmax1D(a.Data[start:start+cols], out.Data[start:start+cols])
		}
	default:
		panic("LogSoftmax only supports 1D or 2D tensors")
	}

	return out
}

// logSoftmax1D computes log softmax on a 1D slice
// log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
func logSoftmax1D(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}

	// Find max
	maxVal := in[0]
	for i := 1; i < n; i++ {
		if in[i] > maxVal {
			maxVal = in[i]
		}
	}

	// Compute sum of exp(x - max)
	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Exp(float64(in[i] - maxVal))
	}

	// Compute log softmax
	logSum := float32(math.Log(sum))
	for i := 0; i < n; i++ {
		out[i] = in[i] - maxVal - logSum
	}
}

// MaskedSoftmax applies softmax with a mask (used for causal attention)
// mask should contain 0 for positions to keep and -inf for positions to ignore
func MaskedSoftmax(a, mask *Tensor) *Tensor {
	if len(a.Shape) != len(mask.Shape) {
		panic("MaskedSoftmax: tensor and mask must have same number of dimensions")
	}

	// Add mask to input
	masked := Add(a, mask)

	// Apply softmax
	return Softmax(masked)
}

// CausalMask creates a causal (lower triangular) attention mask
// Returns tensor of shape [seqLen, seqLen] with 0s in lower triangle and -inf elsewhere
func CausalMask(seqLen int) *Tensor {
	mask := New(seqLen, seqLen)
	negInf := float32(math.Inf(-1))

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask.Data[i*seqLen+j] = negInf
			}
			// else: already 0 from initialization
		}
	}

	return mask
}

// CausalMaskWithOffset creates a causal mask for incremental generation
// startPos is the position of the first token in the current sequence
func CausalMaskWithOffset(seqLen, startPos, totalLen int) *Tensor {
	mask := New(seqLen, totalLen)
	negInf := float32(math.Inf(-1))

	for i := 0; i < seqLen; i++ {
		currentPos := startPos + i
		for j := 0; j < totalLen; j++ {
			if j > currentPos {
				mask.Data[i*totalLen+j] = negInf
			}
		}
	}

	return mask
}

// TopK returns the indices of the top-k largest elements
func TopK(a *Tensor, k int) []int {
	if len(a.Data) == 0 || k <= 0 {
		return nil
	}

	if k > len(a.Data) {
		k = len(a.Data)
	}

	// Create index-value pairs
	type pair struct {
		idx int
		val float32
	}

	pairs := make([]pair, len(a.Data))
	for i, v := range a.Data {
		pairs[i] = pair{i, v}
	}

	// Partial sort to get top-k (simple selection for now)
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].val > pairs[maxIdx].val {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}

	indices := make([]int, k)
	for i := 0; i < k; i++ {
		indices[i] = pairs[i].idx
	}

	return indices
}

// TopP (nucleus sampling) returns indices of elements whose cumulative probability >= p
func TopP(probs *Tensor, p float32) []int {
	if len(probs.Data) == 0 || p <= 0 {
		return nil
	}

	// Create sorted index-probability pairs
	type pair struct {
		idx  int
		prob float32
	}

	pairs := make([]pair, len(probs.Data))
	for i, v := range probs.Data {
		pairs[i] = pair{i, v}
	}

	// Sort by probability (descending)
	for i := 0; i < len(pairs)-1; i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].prob > pairs[maxIdx].prob {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}

	// Select elements until cumulative probability >= p
	var cumProb float32
	var indices []int

	for _, pair := range pairs {
		indices = append(indices, pair.idx)
		cumProb += pair.prob
		if cumProb >= p {
			break
		}
	}

	return indices
}
