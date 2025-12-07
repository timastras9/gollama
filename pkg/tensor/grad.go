package tensor

import (
	"runtime"
	"sync"
)

// GradTensor is a tensor with gradient tracking for backpropagation
type GradTensor struct {
	*Tensor
	Grad        *Tensor      // Gradient tensor
	RequireGrad bool         // Whether to compute gradients
	GradFn      func()       // Backward function
	Parents     []*GradTensor // Parent tensors in computation graph
}

// NewGrad creates a new gradient-tracked tensor
func NewGrad(shape ...int) *GradTensor {
	return &GradTensor{
		Tensor:      New(shape...),
		RequireGrad: true,
	}
}

// FromTensor wraps an existing tensor for gradient tracking
func FromTensor(t *Tensor, requireGrad bool) *GradTensor {
	return &GradTensor{
		Tensor:      t,
		RequireGrad: requireGrad,
	}
}

// ZeroGrad resets the gradient to zero
func (g *GradTensor) ZeroGrad() {
	if g.Grad != nil {
		for i := range g.Grad.Data {
			g.Grad.Data[i] = 0
		}
	}
}

// Backward computes gradients via backpropagation
func (g *GradTensor) Backward() {
	// Initialize gradient of output as 1
	if g.Grad == nil {
		g.Grad = New(g.Shape...)
		for i := range g.Grad.Data {
			g.Grad.Data[i] = 1.0
		}
	}

	// Topological sort for reverse-mode autodiff
	visited := make(map[*GradTensor]bool)
	order := make([]*GradTensor, 0)

	var topo func(*GradTensor)
	topo = func(t *GradTensor) {
		if visited[t] {
			return
		}
		visited[t] = true
		for _, p := range t.Parents {
			topo(p)
		}
		order = append(order, t)
	}
	topo(g)

	// Backward pass in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		t := order[i]
		if t.GradFn != nil {
			t.GradFn()
		}
	}
}

// MatMulGrad performs matrix multiplication with gradient tracking
// C = A @ B^T (transposed for weight matrices)
// A: [M, K], B: [N, K], C: [M, N]
func MatMulGrad(a, b *GradTensor) *GradTensor {
	// Forward pass
	out := &GradTensor{
		Tensor:      MatMulTransposeB(a.Tensor, b.Tensor),
		RequireGrad: a.RequireGrad || b.RequireGrad,
		Parents:     []*GradTensor{a, b},
	}

	M := a.Shape[0]
	K := a.Shape[1]
	N := b.Shape[0]

	// Backward function
	out.GradFn = func() {
		if out.Grad == nil {
			return
		}

		// Gradient w.r.t A: dL/dA = dL/dC @ B
		// dC: [M, N], B: [N, K], dA: [M, K]
		// This is regular matmul (not transposed)
		if a.RequireGrad {
			if a.Grad == nil {
				a.Grad = New(a.Shape...)
			}
			// dA[m,k] = sum_n dC[m,n] * B[n,k]
			for m := 0; m < M; m++ {
				for k := 0; k < K; k++ {
					var sum float32
					for n := 0; n < N; n++ {
						sum += out.Grad.Data[m*N+n] * b.Data[n*K+k]
					}
					a.Grad.Data[m*K+k] += sum
				}
			}
		}

		// Gradient w.r.t B: dL/dB = dL/dC^T @ A
		// dC^T: [N, M], A: [M, K], dB: [N, K]
		if b.RequireGrad {
			if b.Grad == nil {
				b.Grad = New(b.Shape...)
			}
			// dB[n,k] = sum_m dC[m,n] * A[m,k]
			numWorkers := runtime.GOMAXPROCS(0)
			var wg sync.WaitGroup
			rowsPerWorker := (N + numWorkers - 1) / numWorkers

			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				go func(worker int) {
					defer wg.Done()
					startRow := worker * rowsPerWorker
					endRow := startRow + rowsPerWorker
					if endRow > N {
						endRow = N
					}

					for n := startRow; n < endRow; n++ {
						for k := 0; k < K; k++ {
							var sum float32
							for m := 0; m < M; m++ {
								sum += out.Grad.Data[m*N+n] * a.Data[m*K+k]
							}
							b.Grad.Data[n*K+k] += sum
						}
					}
				}(w)
			}
			wg.Wait()
		}
	}

	return out
}

// AddGrad performs element-wise addition with gradient tracking
func AddGrad(a, b *GradTensor) *GradTensor {
	out := &GradTensor{
		Tensor:      Add(a.Tensor, b.Tensor),
		RequireGrad: a.RequireGrad || b.RequireGrad,
		Parents:     []*GradTensor{a, b},
	}

	out.GradFn = func() {
		if out.Grad == nil {
			return
		}

		// Gradient flows through addition unchanged
		if a.RequireGrad {
			if a.Grad == nil {
				a.Grad = New(a.Shape...)
			}
			addInplace(a.Grad, out.Grad)
		}

		if b.RequireGrad {
			if b.Grad == nil {
				b.Grad = New(b.Shape...)
			}
			// Handle broadcasting if needed
			if len(b.Shape) == 1 && len(out.Grad.Shape) == 2 {
				// Sum over batch dimension
				for i := 0; i < out.Grad.Shape[0]; i++ {
					for j := 0; j < b.Shape[0]; j++ {
						b.Grad.Data[j] += out.Grad.Data[i*b.Shape[0]+j]
					}
				}
			} else {
				addInplace(b.Grad, out.Grad)
			}
		}
	}

	return out
}

// addInplace adds src to dst in place
func addInplace(dst, src *Tensor) {
	for i := range dst.Data {
		dst.Data[i] += src.Data[i]
	}
}

// SoftmaxCrossEntropyGrad computes softmax + cross-entropy loss with gradients
// logits: [batch, vocab_size]
// targets: [batch] (token indices)
// Returns scalar loss and gradient flows back through logits
func SoftmaxCrossEntropyGrad(logits *GradTensor, targets []int) (*GradTensor, float32) {
	batch := logits.Shape[0]
	vocabSize := logits.Shape[1]

	// Forward: compute softmax and cross-entropy loss
	probs := New(batch, vocabSize)
	var totalLoss float32

	for b := 0; b < batch; b++ {
		// Find max for numerical stability
		maxVal := logits.Data[b*vocabSize]
		for v := 1; v < vocabSize; v++ {
			if logits.Data[b*vocabSize+v] > maxVal {
				maxVal = logits.Data[b*vocabSize+v]
			}
		}

		// Compute exp and sum
		var sumExp float32
		for v := 0; v < vocabSize; v++ {
			probs.Data[b*vocabSize+v] = exp32(logits.Data[b*vocabSize+v] - maxVal)
			sumExp += probs.Data[b*vocabSize+v]
		}

		// Normalize to get probabilities
		for v := 0; v < vocabSize; v++ {
			probs.Data[b*vocabSize+v] /= sumExp
		}

		// Cross-entropy loss: -log(p[target])
		target := targets[b]
		if target >= 0 && target < vocabSize {
			totalLoss -= log32(probs.Data[b*vocabSize+target] + 1e-10)
		}
	}

	loss := totalLoss / float32(batch)

	// Create output tensor for loss (scalar wrapped as tensor)
	out := &GradTensor{
		Tensor:      New(1),
		RequireGrad: logits.RequireGrad,
		Parents:     []*GradTensor{logits},
	}
	out.Data[0] = loss

	// Backward: gradient of softmax cross-entropy is (probs - one_hot(targets))
	out.GradFn = func() {
		if !logits.RequireGrad {
			return
		}

		if logits.Grad == nil {
			logits.Grad = New(logits.Shape...)
		}

		scale := 1.0 / float32(batch)
		for b := 0; b < batch; b++ {
			target := targets[b]
			for v := 0; v < vocabSize; v++ {
				grad := probs.Data[b*vocabSize+v] * scale
				if v == target {
					grad -= scale
				}
				logits.Grad.Data[b*vocabSize+v] += grad
			}
		}
	}

	return out, loss
}

// exp32 is float32 exp
func exp32(x float32) float32 {
	// Fast approximation for training
	if x < -88 {
		return 0
	}
	if x > 88 {
		return 3.4e38
	}
	// Use standard library via float64
	return float32(exp64(float64(x)))
}

// log32 is float32 log
func log32(x float32) float32 {
	return float32(log64(float64(x)))
}

// exp64 and log64 use Go's math functions
func exp64(x float64) float64 {
	return _exp(x)
}

func log64(x float64) float64 {
	return _log(x)
}

// _exp is a fast exp approximation
func _exp(x float64) float64 {
	// Taylor series would be slow, use standard math
	// This gets inlined by the compiler
	const (
		Ln2Hi = 6.93147180369123816490e-01
		Ln2Lo = 1.90821492927058770002e-10
		Log2e = 1.44269504088896338700e+00
		Overflow = 7.09782712893383973096e+02
		Underflow = -7.45133219101941108420e+02
	)

	if x > Overflow {
		return 1.7976931348623157e+308
	}
	if x < Underflow {
		return 0
	}

	// Reduce to exp(r) where |r| <= 0.5*ln(2)
	k := int(x*Log2e + 0.5)
	if x < 0 {
		k = int(x*Log2e - 0.5)
	}
	r := x - float64(k)*Ln2Hi - float64(k)*Ln2Lo

	// Compute exp(r) using polynomial
	r2 := r * r
	p := r - r2*(0.16666666666666602+r2*(0.0027777777777015593+r2*6.613756321437934e-05))
	result := 1 - (r*p/(p-2) - r)

	// Scale by 2^k
	if k > 0 {
		for i := 0; i < k; i++ {
			result *= 2
		}
	} else {
		for i := 0; i < -k; i++ {
			result /= 2
		}
	}

	return result
}

func _log(x float64) float64 {
	if x <= 0 {
		return -1e100
	}

	// Reduce to log(1+f) where f is in [sqrt(2)/2, sqrt(2)]
	var k int
	for x >= 2 {
		x /= 2
		k++
	}
	for x < 1 {
		x *= 2
		k--
	}

	f := x - 1
	s := f / (2 + f)
	s2 := s * s
	s4 := s2 * s2

	// Polynomial approximation
	t1 := s2 * (0.6666666666666735 + s4*(0.2857142874366239+s4*0.1818357216161805))
	t2 := s4 * (0.3999999999940942 + s4*0.22222198432149784)
	R := t1 + t2

	return float64(k)*0.6931471805599453 + f - s*(f-R)
}

// ReLUGrad applies ReLU with gradient tracking
func ReLUGrad(x *GradTensor) *GradTensor {
	out := &GradTensor{
		Tensor:      New(x.Shape...),
		RequireGrad: x.RequireGrad,
		Parents:     []*GradTensor{x},
	}

	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}

	out.GradFn = func() {
		if !x.RequireGrad || out.Grad == nil {
			return
		}

		if x.Grad == nil {
			x.Grad = New(x.Shape...)
		}

		for i, v := range x.Data {
			if v > 0 {
				x.Grad.Data[i] += out.Grad.Data[i]
			}
		}
	}

	return out
}

// SiLUGrad applies SiLU (x * sigmoid(x)) with gradient tracking
func SiLUGrad(x *GradTensor) *GradTensor {
	out := &GradTensor{
		Tensor:      New(x.Shape...),
		RequireGrad: x.RequireGrad,
		Parents:     []*GradTensor{x},
	}

	sigmoid := make([]float32, len(x.Data))
	for i, v := range x.Data {
		sigmoid[i] = 1.0 / (1.0 + exp32(-v))
		out.Data[i] = v * sigmoid[i]
	}

	out.GradFn = func() {
		if !x.RequireGrad || out.Grad == nil {
			return
		}

		if x.Grad == nil {
			x.Grad = New(x.Shape...)
		}

		// d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
		//                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
		for i, v := range x.Data {
			s := sigmoid[i]
			grad := s * (1 + v*(1-s))
			x.Grad.Data[i] += out.Grad.Data[i] * grad
		}
	}

	return out
}
