package tensor

import "math"

// Add performs element-wise addition: out = a + b
func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for Add")
	}

	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// AddInplace performs element-wise addition in place: a += b
func AddInplace(a, b *Tensor) {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for AddInplace")
	}

	for i := range a.Data {
		a.Data[i] += b.Data[i]
	}
}

// Sub performs element-wise subtraction: out = a - b
func Sub(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for Sub")
	}

	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

// Mul performs element-wise multiplication: out = a * b
func Mul(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for Mul")
	}

	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

// MulInplace performs element-wise multiplication in place: a *= b
func MulInplace(a, b *Tensor) {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for MulInplace")
	}

	for i := range a.Data {
		a.Data[i] *= b.Data[i]
	}
}

// Div performs element-wise division: out = a / b
func Div(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch for Div")
	}

	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] / b.Data[i]
	}
	return out
}

// Scale multiplies all elements by a scalar: out = a * scalar
func Scale(a *Tensor, scalar float32) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * scalar
	}
	return out
}

// ScaleInplace multiplies all elements by a scalar in place: a *= scalar
func ScaleInplace(a *Tensor, scalar float32) {
	for i := range a.Data {
		a.Data[i] *= scalar
	}
}

// AddScalar adds a scalar to all elements: out = a + scalar
func AddScalar(a *Tensor, scalar float32) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + scalar
	}
	return out
}

// AddScalarInplace adds a scalar to all elements in place: a += scalar
func AddScalarInplace(a *Tensor, scalar float32) {
	for i := range a.Data {
		a.Data[i] += scalar
	}
}

// Neg negates all elements: out = -a
func Neg(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = -a.Data[i]
	}
	return out
}

// Sqrt computes element-wise square root
func Sqrt(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = float32(math.Sqrt(float64(a.Data[i])))
	}
	return out
}

// Exp computes element-wise exponential
func Exp(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = float32(math.Exp(float64(a.Data[i])))
	}
	return out
}

// Log computes element-wise natural logarithm
func Log(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = float32(math.Log(float64(a.Data[i])))
	}
	return out
}

// Pow computes element-wise power
func Pow(a *Tensor, exp float32) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = float32(math.Pow(float64(a.Data[i]), float64(exp)))
	}
	return out
}

// Abs computes element-wise absolute value
func Abs(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		out.Data[i] = float32(math.Abs(float64(a.Data[i])))
	}
	return out
}

// Clamp clamps all values to [min, max]
func Clamp(a *Tensor, minVal, maxVal float32) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		v := a.Data[i]
		if v < minVal {
			v = minVal
		} else if v > maxVal {
			v = maxVal
		}
		out.Data[i] = v
	}
	return out
}

// SiLU computes the SiLU (Swish) activation: x * sigmoid(x)
func SiLU(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		x := a.Data[i]
		out.Data[i] = x * sigmoid(x)
	}
	return out
}

// SiLUInplace computes SiLU activation in place
func SiLUInplace(a *Tensor) {
	for i := range a.Data {
		x := a.Data[i]
		a.Data[i] = x * sigmoid(x)
	}
}

// sigmoid computes the sigmoid function
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// GELU computes the GELU activation (approximate)
func GELU(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		x := float64(a.Data[i])
		// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		out.Data[i] = float32(0.5 * x * (1.0 + math.Tanh(0.7978845608*x*(1.0+0.044715*x*x))))
	}
	return out
}

// ReLU computes the ReLU activation
func ReLU(a *Tensor) *Tensor {
	out := NewLike(a)
	for i := range a.Data {
		if a.Data[i] > 0 {
			out.Data[i] = a.Data[i]
		} else {
			out.Data[i] = 0
		}
	}
	return out
}

// Transpose2D transposes a 2D tensor
func Transpose2D(a *Tensor) *Tensor {
	if len(a.Shape) != 2 {
		panic("Transpose2D requires 2D tensor")
	}

	rows, cols := a.Shape[0], a.Shape[1]
	out := New(cols, rows)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Data[j*rows+i] = a.Data[i*cols+j]
		}
	}

	return out
}

// Broadcast broadcasts tensor b to match tensor a's shape for element-wise ops
// Currently supports broadcasting a 1D tensor across the last dimension of a 2D tensor
func Broadcast(a, b *Tensor) *Tensor {
	if len(b.Shape) == 1 && len(a.Shape) == 2 && b.Shape[0] == a.Shape[1] {
		// Broadcast [hidden] to [seq, hidden]
		out := NewLike(a)
		rows, cols := a.Shape[0], a.Shape[1]
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out.Data[i*cols+j] = b.Data[j]
			}
		}
		return out
	}
	panic("unsupported broadcast shape")
}

// Copy copies data from src to dst
func Copy(dst, src *Tensor) {
	if len(dst.Data) != len(src.Data) {
		panic("tensor size mismatch for Copy")
	}
	copy(dst.Data, src.Data)
}
