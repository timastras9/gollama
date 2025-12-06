package tensor

import (
	"fmt"
	"math"
)

// Tensor represents an n-dimensional array of float32 values.
// Data is stored in row-major (C-contiguous) order to match GGUF storage.
type Tensor struct {
	Data    []float32 // Flat data storage
	Shape   []int     // Dimensions e.g., [batch, seq, hidden]
	Strides []int     // Strides for indexing
}

// New creates a new tensor with the given shape, initialized to zeros.
func New(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	strides := computeStrides(shape)

	return &Tensor{
		Data:    make([]float32, size),
		Shape:   shape,
		Strides: strides,
	}
}

// NewWithData creates a tensor from existing data with given shape.
func NewWithData(data []float32, shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if len(data) != size {
		panic(fmt.Sprintf("data size %d does not match shape %v (size %d)", len(data), shape, size))
	}

	strides := computeStrides(shape)

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: strides,
	}
}

// NewLike creates a new tensor with the same shape as the given tensor.
func NewLike(t *Tensor) *Tensor {
	return New(t.Shape...)
}

// computeStrides calculates row-major strides for a given shape.
func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1

	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return strides
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	return len(t.Data)
}

// NDim returns the number of dimensions.
func (t *Tensor) NDim() int {
	return len(t.Shape)
}

// At returns the value at the given indices for a 2D tensor.
func (t *Tensor) At(indices ...int) float32 {
	return t.Data[t.index(indices...)]
}

// Set sets the value at the given indices.
func (t *Tensor) Set(value float32, indices ...int) {
	t.Data[t.index(indices...)] = value
}

// index calculates the flat index from multi-dimensional indices.
func (t *Tensor) index(indices ...int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}

	idx := 0
	for i, ind := range indices {
		idx += ind * t.Strides[i]
	}
	return idx
}

// Reshape returns a new tensor with the given shape (data is shared).
func (t *Tensor) Reshape(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != len(t.Data) {
		panic(fmt.Sprintf("cannot reshape size %d to %v (size %d)", len(t.Data), shape, size))
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   shape,
		Strides: computeStrides(shape),
	}
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)

	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)

	return &Tensor{
		Data:    data,
		Shape:   shape,
		Strides: computeStrides(shape),
	}
}

// Zero fills the tensor with zeros.
func (t *Tensor) Zero() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// Fill fills the tensor with the given value.
func (t *Tensor) Fill(value float32) {
	for i := range t.Data {
		t.Data[i] = value
	}
}

// Row returns a slice of the data for the given row (for 2D tensors).
func (t *Tensor) Row(i int) []float32 {
	if len(t.Shape) != 2 {
		panic("Row() only works on 2D tensors")
	}
	start := i * t.Shape[1]
	return t.Data[start : start+t.Shape[1]]
}

// Slice returns a view into a portion of the first dimension.
func (t *Tensor) Slice(start, end int) *Tensor {
	if start < 0 || end > t.Shape[0] || start >= end {
		panic(fmt.Sprintf("invalid slice bounds [%d:%d] for shape %v", start, end, t.Shape))
	}

	// Calculate the size of each slice along first dimension
	sliceSize := t.Strides[0]
	dataStart := start * sliceSize
	dataEnd := end * sliceSize

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[0] = end - start

	return &Tensor{
		Data:    t.Data[dataStart:dataEnd],
		Shape:   newShape,
		Strides: t.Strides,
	}
}

// View returns a view (no copy) of the tensor with a new shape.
func (t *Tensor) View(shape ...int) *Tensor {
	return t.Reshape(shape...)
}

// Contiguous returns a contiguous copy of the tensor.
func (t *Tensor) Contiguous() *Tensor {
	return t.Clone()
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, data=%v...)", t.Shape, t.Data[:min(8, len(t.Data))])
}

// AssertShape panics if the tensor doesn't have the expected shape.
func (t *Tensor) AssertShape(expected ...int) {
	if len(t.Shape) != len(expected) {
		panic(fmt.Sprintf("expected %d dims, got %d", len(expected), len(t.Shape)))
	}
	for i, dim := range expected {
		if dim >= 0 && t.Shape[i] != dim {
			panic(fmt.Sprintf("expected shape %v, got %v", expected, t.Shape))
		}
	}
}

// Max returns the maximum value in the tensor.
func (t *Tensor) Max() float32 {
	if len(t.Data) == 0 {
		return float32(math.Inf(-1))
	}
	max := t.Data[0]
	for _, v := range t.Data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// ArgMax returns the index of the maximum value in the tensor.
func (t *Tensor) ArgMax() int {
	if len(t.Data) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := t.Data[0]
	for i, v := range t.Data[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float32 {
	var sum float32
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() float32 {
	if len(t.Data) == 0 {
		return 0
	}
	return t.Sum() / float32(len(t.Data))
}
