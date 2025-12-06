package quant

import (
	"github.com/timastras9/gollama/pkg/gguf"
)

// Block sizes for different quantization types
const (
	BlockSizeQ4_0 = 32
	BlockSizeQ4_1 = 32
	BlockSizeQ5_0 = 32
	BlockSizeQ5_1 = 32
	BlockSizeQ8_0 = 32
	BlockSizeQ8_1 = 32
)

// BlockQ4_0 represents a Q4_0 quantization block
// 32 elements quantized to 4 bits each with one fp16 scale
// Total: 2 bytes (scale) + 16 bytes (quants) = 18 bytes
type BlockQ4_0 struct {
	Delta  uint16   // FP16 scale factor
	Quants [16]byte // 32 x 4-bit values packed
}

// BlockQ4_1 represents a Q4_1 quantization block
// 32 elements with scale and minimum
// Total: 2 + 2 + 16 = 20 bytes
type BlockQ4_1 struct {
	Delta  uint16   // FP16 scale factor
	Min    uint16   // FP16 minimum value
	Quants [16]byte // 32 x 4-bit values packed
}

// BlockQ8_0 represents a Q8_0 quantization block
// 32 elements quantized to 8 bits with one fp16 scale
// Total: 2 bytes (scale) + 32 bytes (quants) = 34 bytes
type BlockQ8_0 struct {
	Delta  uint16   // FP16 scale factor
	Quants [32]int8 // 32 x 8-bit values
}

// BlockQ8_1 represents a Q8_1 quantization block
// 32 elements with scale and sum
// Total: 2 + 2 + 32 = 36 bytes
type BlockQ8_1 struct {
	Delta  uint16   // FP16 scale factor
	Sum    uint16   // FP16 sum for dot product optimization
	Quants [32]int8 // 32 x 8-bit values
}

// BlockQ5_0 represents a Q5_0 quantization block
// 32 elements quantized to 5 bits
type BlockQ5_0 struct {
	Delta  uint16    // FP16 scale
	QH     [4]byte   // High bits (32 x 1 bit = 4 bytes)
	Quants [16]byte  // Low 4 bits (32 x 4 bits = 16 bytes)
}

// BlockQ5_1 represents a Q5_1 quantization block
// 32 elements quantized to 5 bits with min
type BlockQ5_1 struct {
	Delta  uint16   // FP16 scale
	Min    uint16   // FP16 minimum
	QH     [4]byte  // High bits
	Quants [16]byte // Low 4 bits
}

// Type alias for GGML types
type Type = gguf.GGMLType

// Re-export type constants for convenience
const (
	TYPE_F32  = gguf.GGML_TYPE_F32
	TYPE_F16  = gguf.GGML_TYPE_F16
	TYPE_Q4_0 = gguf.GGML_TYPE_Q4_0
	TYPE_Q4_1 = gguf.GGML_TYPE_Q4_1
	TYPE_Q5_0 = gguf.GGML_TYPE_Q5_0
	TYPE_Q5_1 = gguf.GGML_TYPE_Q5_1
	TYPE_Q8_0 = gguf.GGML_TYPE_Q8_0
	TYPE_Q8_1 = gguf.GGML_TYPE_Q8_1
	TYPE_Q4_K = gguf.GGML_TYPE_Q4_K
	TYPE_Q5_K = gguf.GGML_TYPE_Q5_K
	TYPE_Q6_K = gguf.GGML_TYPE_Q6_K
)

// IsQuantized returns true if the type is a quantized type
func IsQuantized(t Type) bool {
	info := gguf.GetTypeInfo(t)
	return info.IsQuantized
}

// BlockSize returns the block size for a quantization type
func BlockSize(t Type) int {
	info := gguf.GetTypeInfo(t)
	return info.BlockSize
}

// TypeSize returns the byte size of one block for a quantization type
func TypeSize(t Type) int {
	info := gguf.GetTypeInfo(t)
	return info.TypeSize
}

// NumBlocks calculates the number of blocks needed for n elements
func NumBlocks(n int, t Type) int {
	bs := BlockSize(t)
	return (n + bs - 1) / bs
}

// DataSize calculates the total byte size for n elements
func DataSize(n int, t Type) int {
	if !IsQuantized(t) {
		return n * TypeSize(t)
	}
	return NumBlocks(n, t) * TypeSize(t)
}
