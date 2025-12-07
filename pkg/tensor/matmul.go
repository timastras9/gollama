package tensor

import (
	"runtime"
	"sync"
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
	matmulNaive(a.Data, b.Data, out.Data, M, N, K)
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

	// C[i,j] = sum_k(A[i,k] * B[j,k])
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a.Data[i*K+k] * b.Data[j*K+k]
			}
			out.Data[i*N+j] = sum
		}
	}

	return out
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
