package quant

import (
	"encoding/binary"
	"math"
)

// fp16ToFp32 converts a FP16 value (stored as uint16) to float32
func fp16ToFp32(h uint16) float32 {
	// Extract components
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1f)
	mant := uint32(h & 0x3ff)

	var f uint32

	if exp == 0 {
		if mant == 0 {
			// Zero
			f = sign << 31
		} else {
			// Denormalized number
			exp = 1
			for (mant & 0x400) == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3ff
			f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	} else if exp == 31 {
		// Inf or NaN
		f = (sign << 31) | 0x7f800000 | (mant << 13)
	} else {
		// Normalized number
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return math.Float32frombits(f)
}

// bf16ToFp32 converts a BF16 value (stored as uint16) to float32
func bf16ToFp32(h uint16) float32 {
	// BF16 is just the upper 16 bits of float32
	return math.Float32frombits(uint32(h) << 16)
}

// DequantizeF16 dequantizes FP16 data to float32
func DequantizeF16(data []byte, out []float32) {
	n := len(out)
	for i := 0; i < n; i++ {
		h := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = fp16ToFp32(h)
	}
}

// DequantizeBF16 dequantizes BF16 data to float32
func DequantizeBF16(data []byte, out []float32) {
	n := len(out)
	for i := 0; i < n; i++ {
		h := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = bf16ToFp32(h)
	}
}

// DequantizeQ8_0 dequantizes Q8_0 data to float32
// Q8_0: 32 elements per block, 2 bytes scale (fp16) + 32 bytes quants (int8)
func DequantizeQ8_0(data []byte, out []float32) {
	const blockSize = 32
	const bytesPerBlock = 34 // 2 (scale) + 32 (quants)

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read scale (FP16)
		scale := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))

		// Dequantize 32 int8 values
		for i := 0; i < blockSize; i++ {
			q := int8(data[blockOffset+2+i])
			out[b*blockSize+i] = scale * float32(q)
		}
	}
}

// DequantizeQ8_0Block dequantizes a single Q8_0 block
func DequantizeQ8_0Block(data []byte, out []float32) {
	scale := fp16ToFp32(binary.LittleEndian.Uint16(data))
	for i := 0; i < 32; i++ {
		q := int8(data[2+i])
		out[i] = scale * float32(q)
	}
}

// DequantizeQ4_0 dequantizes Q4_0 data to float32
// Q4_0: 32 elements per block, 2 bytes scale (fp16) + 16 bytes quants (4-bit)
func DequantizeQ4_0(data []byte, out []float32) {
	const blockSize = 32
	const bytesPerBlock = 18 // 2 (scale) + 16 (quants)

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read scale (FP16)
		scale := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))

		// Dequantize 32 4-bit values (packed in 16 bytes)
		for i := 0; i < 16; i++ {
			qByte := data[blockOffset+2+i]

			// Low nibble (first of pair)
			q0 := int8(qByte&0x0F) - 8
			out[b*blockSize+i*2] = scale * float32(q0)

			// High nibble (second of pair)
			q1 := int8(qByte>>4) - 8
			out[b*blockSize+i*2+1] = scale * float32(q1)
		}
	}
}

// DequantizeQ4_0Block dequantizes a single Q4_0 block
func DequantizeQ4_0Block(data []byte, out []float32) {
	scale := fp16ToFp32(binary.LittleEndian.Uint16(data))

	for i := 0; i < 16; i++ {
		qByte := data[2+i]
		q0 := int8(qByte&0x0F) - 8
		q1 := int8(qByte>>4) - 8
		out[i*2] = scale * float32(q0)
		out[i*2+1] = scale * float32(q1)
	}
}

// DequantizeQ4_1 dequantizes Q4_1 data to float32
// Q4_1: 32 elements per block, 2 bytes scale + 2 bytes min + 16 bytes quants
func DequantizeQ4_1(data []byte, out []float32) {
	const blockSize = 32
	const bytesPerBlock = 20

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read scale and min (FP16)
		scale := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))
		min := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset+2:]))

		// Dequantize 32 4-bit values
		for i := 0; i < 16; i++ {
			qByte := data[blockOffset+4+i]

			q0 := float32(qByte & 0x0F)
			q1 := float32(qByte >> 4)

			out[b*blockSize+i*2] = scale*q0 + min
			out[b*blockSize+i*2+1] = scale*q1 + min
		}
	}
}

// DequantizeQ5_0 dequantizes Q5_0 data to float32
// Q5_0: 32 elements per block, 2 bytes scale + 4 bytes high bits + 16 bytes low bits
func DequantizeQ5_0(data []byte, out []float32) {
	const blockSize = 32
	const bytesPerBlock = 22

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read scale (FP16)
		scale := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))

		// Read high bits (4 bytes = 32 bits, one per element)
		qh := binary.LittleEndian.Uint32(data[blockOffset+2:])

		// Dequantize 32 5-bit values
		for i := 0; i < 16; i++ {
			qByte := data[blockOffset+6+i]

			// Extract low 4 bits
			q0 := int8(qByte & 0x0F)
			q1 := int8(qByte >> 4)

			// Add high bit
			if (qh>>(i*2))&1 != 0 {
				q0 |= 0x10
			}
			if (qh>>(i*2+1))&1 != 0 {
				q1 |= 0x10
			}

			// Center around zero (subtract 16)
			out[b*blockSize+i*2] = scale * float32(q0-16)
			out[b*blockSize+i*2+1] = scale * float32(q1-16)
		}
	}
}

// DequantizeQ5_1 dequantizes Q5_1 data to float32
// Q5_1: Like Q5_0 but with min value
func DequantizeQ5_1(data []byte, out []float32) {
	const blockSize = 32
	const bytesPerBlock = 24

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read scale and min (FP16)
		scale := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))
		min := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset+2:]))

		// Read high bits
		qh := binary.LittleEndian.Uint32(data[blockOffset+4:])

		// Dequantize 32 5-bit values
		for i := 0; i < 16; i++ {
			qByte := data[blockOffset+8+i]

			q0 := uint8(qByte & 0x0F)
			q1 := uint8(qByte >> 4)

			if (qh>>(i*2))&1 != 0 {
				q0 |= 0x10
			}
			if (qh>>(i*2+1))&1 != 0 {
				q1 |= 0x10
			}

			out[b*blockSize+i*2] = scale*float32(q0) + min
			out[b*blockSize+i*2+1] = scale*float32(q1) + min
		}
	}
}

// DequantizeQ4_K dequantizes Q4_K data to float32
// Q4_K: 256 elements per super-block with multiple scales
// Structure: 2 bytes d (fp16) + 2 bytes dmin (fp16) + 12 bytes scales + 128 bytes quants = 144 bytes
func DequantizeQ4_K(data []byte, out []float32) {
	const blockSize = 256
	const bytesPerBlock = 144

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Read super-block scale factors
		d := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset:]))
		dmin := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset+2:]))

		// Read the 12 bytes of packed scales (6-bit values)
		scales := data[blockOffset+4 : blockOffset+16]

		// Read 128 bytes of 4-bit quantized values
		qs := data[blockOffset+16 : blockOffset+144]

		// Decode scales - 8 scale values packed in 12 bytes (6 bits each)
		sc := make([]uint8, 8)
		m := make([]uint8, 8)

		// First 4 scales
		sc[0] = scales[0] & 63
		sc[1] = scales[1] & 63
		sc[2] = scales[2] & 63
		sc[3] = scales[3] & 63

		// First 4 mins
		m[0] = scales[4] & 63
		m[1] = scales[5] & 63
		m[2] = scales[6] & 63
		m[3] = scales[7] & 63

		// Last 4 scales (using high bits)
		sc[4] = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
		sc[5] = (scales[8] >> 4) | ((scales[1] >> 6) << 4)
		sc[6] = (scales[9] & 0xF) | ((scales[2] >> 6) << 4)
		sc[7] = (scales[9] >> 4) | ((scales[3] >> 6) << 4)

		// Last 4 mins (using high bits)
		m[4] = (scales[10] & 0xF) | ((scales[4] >> 6) << 4)
		m[5] = (scales[10] >> 4) | ((scales[5] >> 6) << 4)
		m[6] = (scales[11] & 0xF) | ((scales[6] >> 6) << 4)
		m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4)

		// Dequantize 256 elements in 8 sub-blocks of 32
		for j := 0; j < 8; j++ {
			scaleVal := d * float32(sc[j])
			minVal := dmin * float32(m[j])

			// Each sub-block has 32 elements packed in 16 bytes
			for i := 0; i < 16; i++ {
				qIdx := j*16 + i
				qByte := qs[qIdx]

				q0 := float32(qByte & 0x0F)
				q1 := float32(qByte >> 4)

				outIdx := b*blockSize + j*32 + i*2
				out[outIdx] = scaleVal*q0 - minVal
				out[outIdx+1] = scaleVal*q1 - minVal
			}
		}
	}
}

// DequantizeQ6_K dequantizes Q6_K data to float32
// Q6_K: 256 elements per super-block
// Structure: 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d = 210 bytes
func DequantizeQ6_K(data []byte, out []float32) {
	const blockSize = 256
	const bytesPerBlock = 210

	numBlocks := len(out) / blockSize

	for b := 0; b < numBlocks; b++ {
		blockOffset := b * bytesPerBlock

		// Layout: ql[128] + qh[64] + scales[16] + d[2]
		ql := data[blockOffset : blockOffset+128]
		qh := data[blockOffset+128 : blockOffset+192]
		scales := data[blockOffset+192 : blockOffset+208]
		d := fp16ToFp32(binary.LittleEndian.Uint16(data[blockOffset+208:]))

		// Dequantize 256 elements in 16 sub-blocks of 16
		for j := 0; j < 16; j++ {
			scaleVal := d * float32(int8(scales[j]))

			// Each sub-block of 16 elements
			for i := 0; i < 16; i++ {
				idx := j*16 + i

				// Get low 4 bits from ql
				qlIdx := idx / 2
				var q uint8
				if idx%2 == 0 {
					q = ql[qlIdx] & 0x0F
				} else {
					q = ql[qlIdx] >> 4
				}

				// Get high 2 bits from qh
				qhIdx := idx / 4
				qhShift := (idx % 4) * 2
				qHigh := (qh[qhIdx] >> qhShift) & 0x03

				// Combine to 6-bit value and center
				q6 := int8(q | (qHigh << 4))
				q6 -= 32

				out[b*blockSize+idx] = scaleVal * float32(q6)
			}
		}
	}
}

// Dequantize is the generic entry point for dequantization
func Dequantize(data []byte, out []float32, t Type) {
	switch t {
	case TYPE_F32:
		// Direct copy
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	case TYPE_F16:
		DequantizeF16(data, out)
	case TYPE_Q4_0:
		DequantizeQ4_0(data, out)
	case TYPE_Q4_1:
		DequantizeQ4_1(data, out)
	case TYPE_Q5_0:
		DequantizeQ5_0(data, out)
	case TYPE_Q5_1:
		DequantizeQ5_1(data, out)
	case TYPE_Q8_0:
		DequantizeQ8_0(data, out)
	case TYPE_Q4_K:
		DequantizeQ4_K(data, out)
	case TYPE_Q6_K:
		DequantizeQ6_K(data, out)
	default:
		panic("unsupported quantization type")
	}
}

// DequantizeRow dequantizes a single row of quantized data
// Useful for computing one output at a time to save memory
func DequantizeRow(data []byte, n int, t Type) []float32 {
	out := make([]float32, n)
	Dequantize(data, out, t)
	return out
}
