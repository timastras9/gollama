package tensor

import (
	"runtime"
	"sync"

	"github.com/timastras9/gollama/pkg/metal"
)

// MatMul performs matrix multiplication: C = A @ B
// A is [M, K], B is [K, N], C is [M, N]
func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul requires 2D tensors")
	}

	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]

	if K != K2 {
		panic("MatMul dimension mismatch: inner dimensions must match")
	}

	out := New(M, N)
	numWorkers := runtime.GOMAXPROCS(0)

	// Parallelize for larger matrices
	if M*N*K > 50000 && M >= numWorkers {
		var wg sync.WaitGroup
		rowsPerWorker := (M + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				startRow := worker * rowsPerWorker
				endRow := startRow + rowsPerWorker
				if endRow > M {
					endRow = M
				}

				for i := startRow; i < endRow; i++ {
					for j := 0; j < N; j++ {
						var sum float32
						for k := 0; k < K; k++ {
							sum += a.Data[i*K+k] * b.Data[k*N+j]
						}
						out.Data[i*N+j] = sum
					}
				}
			}(w)
		}
		wg.Wait()
	} else {
		matmulNaive(a.Data, b.Data, out.Data, M, N, K)
	}
	return out
}

// MatMulInto performs matrix multiplication into an existing tensor
func MatMulInto(a, b, out *Tensor) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 || len(out.Shape) != 2 {
		panic("MatMulInto requires 2D tensors")
	}

	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]

	if K != K2 {
		panic("MatMul dimension mismatch")
	}

	if out.Shape[0] != M || out.Shape[1] != N {
		panic("output tensor has wrong shape")
	}

	matmulNaive(a.Data, b.Data, out.Data, M, N, K)
}

// matmulNaive is the basic O(n^3) matrix multiplication
func matmulNaive(a, b, c []float32, M, N, K int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
}

// MatMulBlocked performs cache-blocked matrix multiplication
// Block size is optimized for typical L1 cache sizes
func MatMulBlocked(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMulBlocked requires 2D tensors")
	}

	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]

	if K != K2 {
		panic("MatMul dimension mismatch")
	}

	out := New(M, N)
	matmulBlocked(a.Data, b.Data, out.Data, M, N, K)
	return out
}

// matmulBlocked performs blocked matrix multiplication for better cache utilization
func matmulBlocked(a, b, c []float32, M, N, K int) {
	const blockSize = 64 // Tuned for L1 cache (typically 32KB)

	// Zero output
	for i := range c {
		c[i] = 0
	}

	for i0 := 0; i0 < M; i0 += blockSize {
		for j0 := 0; j0 < N; j0 += blockSize {
			for k0 := 0; k0 < K; k0 += blockSize {
				// Process block
				iMax := min(i0+blockSize, M)
				jMax := min(j0+blockSize, N)
				kMax := min(k0+blockSize, K)

				for i := i0; i < iMax; i++ {
					for k := k0; k < kMax; k++ {
						aik := a[i*K+k]
						for j := j0; j < jMax; j++ {
							c[i*N+j] += aik * b[k*N+j]
						}
					}
				}
			}
		}
	}
}

// MatMulTransposeB performs matrix multiplication with B transposed: C = A @ B^T
// A is [M, K], B is [N, K], C is [M, N]
// Uses Metal GPU acceleration when available, falls back to parallel CPU
func MatMulTransposeB(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMulTransposeB requires 2D tensors")
	}

	M, K := a.Shape[0], a.Shape[1]
	N, K2 := b.Shape[0], b.Shape[1]

	if K != K2 {
		panic("MatMulTransposeB dimension mismatch: A columns must equal B columns")
	}

	out := New(M, N)

	// Try Metal GPU acceleration for very large batch operations
	// GPU overhead makes it slower for small M (single token inference)
	if metal.IsEnabled() && M >= 16 && M*N*K > 500000 {
		err := metal.MatMulTransposed(a.Data, b.Data, out.Data, M, K, N)
		if err == nil {
			return out
		}
		// Fall back to CPU if Metal fails
	}

	numWorkers := runtime.GOMAXPROCS(0)

	// For single-token inference (M=1), parallelize over N (output neurons)
	if M == 1 && N >= numWorkers {
		var wg sync.WaitGroup
		colsPerWorker := (N + numWorkers - 1) / numWorkers
		aRow := a.Data[:K]

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				startCol := worker * colsPerWorker
				endCol := startCol + colsPerWorker
				if endCol > N {
					endCol = N
				}

				for j := startCol; j < endCol; j++ {
					bRow := b.Data[j*K : j*K+K]
					out.Data[j] = dotProduct(aRow, bRow, K)
				}
			}(w)
		}
		wg.Wait()
		return out
	}

	// For batch inference, parallelize over M (rows)
	if M > 1 && M*N*K > 50000 {
		var wg sync.WaitGroup
		rowsPerWorker := (M + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				startRow := worker * rowsPerWorker
				endRow := startRow + rowsPerWorker
				if endRow > M {
					endRow = M
				}

				for i := startRow; i < endRow; i++ {
					aRow := a.Data[i*K : i*K+K]
					for j := 0; j < N; j++ {
						bRow := b.Data[j*K : j*K+K]
						out.Data[i*N+j] = dotProduct(aRow, bRow, K)
					}
				}
			}(w)
		}
		wg.Wait()
		return out
	}

	// Sequential fallback for tiny matrices
	for i := 0; i < M; i++ {
		aRow := a.Data[i*K : i*K+K]
		for j := 0; j < N; j++ {
			bRow := b.Data[j*K : j*K+K]
			out.Data[i*N+j] = dotProduct(aRow, bRow, K)
		}
	}
	return out
}

// dotProduct computes dot product with 8-way unrolling for better pipelining
func dotProduct(a, b []float32, K int) float32 {
	var sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7 float32
	k := 0

	// 8-way unroll for superscalar execution
	for ; k <= K-8; k += 8 {
		sum0 += a[k] * b[k]
		sum1 += a[k+1] * b[k+1]
		sum2 += a[k+2] * b[k+2]
		sum3 += a[k+3] * b[k+3]
		sum4 += a[k+4] * b[k+4]
		sum5 += a[k+5] * b[k+5]
		sum6 += a[k+6] * b[k+6]
		sum7 += a[k+7] * b[k+7]
	}

	// Handle remainder
	sum := sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
	for ; k < K; k++ {
		sum += a[k] * b[k]
	}
	return sum
}

// MatMulParallel performs parallel matrix multiplication
func MatMulParallel(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMulParallel requires 2D tensors")
	}

	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]

	if K != K2 {
		panic("MatMul dimension mismatch")
	}

	out := New(M, N)

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > M {
		numWorkers = M
	}

	rowsPerWorker := (M + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()

			startRow := worker * rowsPerWorker
			endRow := startRow + rowsPerWorker
			if endRow > M {
				endRow = M
			}

			for i := startRow; i < endRow; i++ {
				for j := 0; j < N; j++ {
					var sum float32
					for k := 0; k < K; k++ {
						sum += a.Data[i*K+k] * b.Data[k*N+j]
					}
					out.Data[i*N+j] = sum
				}
			}
		}(w)
	}

	wg.Wait()
	return out
}

// MatVec performs matrix-vector multiplication: y = A @ x
// A is [M, K], x is [K], y is [M]
func MatVec(a, x *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(x.Shape) != 1 {
		panic("MatVec requires 2D matrix and 1D vector")
	}

	M, K := a.Shape[0], a.Shape[1]
	if x.Shape[0] != K {
		panic("MatVec dimension mismatch")
	}

	out := New(M)

	for i := 0; i < M; i++ {
		var sum float32
		for k := 0; k < K; k++ {
			sum += a.Data[i*K+k] * x.Data[k]
		}
		out.Data[i] = sum
	}

	return out
}

// VecDot computes dot product of two vectors
func VecDot(a, b *Tensor) float32 {
	if len(a.Shape) != 1 || len(b.Shape) != 1 || a.Shape[0] != b.Shape[0] {
		panic("VecDot requires matching 1D tensors")
	}

	var sum float32
	for i := range a.Data {
		sum += a.Data[i] * b.Data[i]
	}
	return sum
}

// BatchMatMul performs batched matrix multiplication
// A is [B, M, K], B is [B, K, N] or [K, N] (broadcast), C is [B, M, N]
func BatchMatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 3 {
		panic("BatchMatMul requires 3D tensor for A")
	}

	batch := a.Shape[0]
	M, K := a.Shape[1], a.Shape[2]

	var N int
	var bBatch bool

	if len(b.Shape) == 3 {
		if b.Shape[0] != batch || b.Shape[1] != K {
			panic("BatchMatMul dimension mismatch")
		}
		N = b.Shape[2]
		bBatch = true
	} else if len(b.Shape) == 2 {
		if b.Shape[0] != K {
			panic("BatchMatMul dimension mismatch")
		}
		N = b.Shape[1]
		bBatch = false
	} else {
		panic("BatchMatMul requires 2D or 3D tensor for B")
	}

	out := New(batch, M, N)

	for bi := 0; bi < batch; bi++ {
		aOffset := bi * M * K
		bOffset := 0
		if bBatch {
			bOffset = bi * K * N
		}
		cOffset := bi * M * N

		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += a.Data[aOffset+i*K+k] * b.Data[bOffset+k*N+j]
				}
				out.Data[cOffset+i*N+j] = sum
			}
		}
	}

	return out
}

// OuterProduct computes outer product of two vectors: C = a ⊗ b
// a is [M], b is [N], C is [M, N]
func OuterProduct(a, b *Tensor) *Tensor {
	if len(a.Shape) != 1 || len(b.Shape) != 1 {
		panic("OuterProduct requires 1D tensors")
	}

	M, N := a.Shape[0], b.Shape[0]
	out := New(M, N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.Data[i*N+j] = a.Data[i] * b.Data[j]
		}
	}

	return out
}
