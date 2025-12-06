package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Reader provides access to a GGUF file
type Reader struct {
	file       *os.File
	data       []byte // Memory-mapped file data
	header     Header
	metadata   map[string]any
	tensors    []TensorInfo
	tensorMap  map[string]*TensorInfo
	dataOffset uint64 // Where tensor data begins
}

// Open opens a GGUF file and parses its structure
func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	// Memory map the file
	data, err := MmapFile(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to mmap file: %w", err)
	}

	r := &Reader{
		file:      f,
		data:      data,
		metadata:  make(map[string]any),
		tensorMap: make(map[string]*TensorInfo),
	}

	// Parse the file
	if err := r.parse(); err != nil {
		r.Close()
		return nil, err
	}

	return r, nil
}

// Close closes the reader and releases resources
func (r *Reader) Close() error {
	if r.data != nil {
		MunmapFile(r.data)
		r.data = nil
	}
	if r.file != nil {
		return r.file.Close()
	}
	return nil
}

// parse reads and validates the GGUF file structure
func (r *Reader) parse() error {
	if len(r.data) < 24 {
		return fmt.Errorf("file too small to be valid GGUF")
	}

	offset := 0

	// Read header
	r.header.Magic = binary.LittleEndian.Uint32(r.data[offset:])
	offset += 4

	if r.header.Magic != Magic {
		return fmt.Errorf("invalid magic number: expected 0x%X, got 0x%X", Magic, r.header.Magic)
	}

	r.header.Version = binary.LittleEndian.Uint32(r.data[offset:])
	offset += 4

	if r.header.Version < 2 || r.header.Version > 3 {
		return fmt.Errorf("unsupported GGUF version: %d", r.header.Version)
	}

	r.header.TensorCount = binary.LittleEndian.Uint64(r.data[offset:])
	offset += 8

	r.header.MetadataKVCount = binary.LittleEndian.Uint64(r.data[offset:])
	offset += 8

	// Parse metadata
	var err error
	offset, err = r.parseMetadata(offset)
	if err != nil {
		return fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Parse tensor info
	offset, err = r.parseTensorInfo(offset)
	if err != nil {
		return fmt.Errorf("failed to parse tensor info: %w", err)
	}

	// Calculate tensor data offset (aligned)
	alignment := uint64(DefaultAlignment)
	if v, ok := r.metadata["general.alignment"]; ok {
		if a, ok := v.(uint32); ok {
			alignment = uint64(a)
		}
	}

	r.dataOffset = uint64(offset)
	if r.dataOffset%alignment != 0 {
		r.dataOffset += alignment - (r.dataOffset % alignment)
	}

	return nil
}

// parseMetadata reads all metadata key-value pairs
func (r *Reader) parseMetadata(offset int) (int, error) {
	for i := uint64(0); i < r.header.MetadataKVCount; i++ {
		// Read key
		key, newOffset, err := r.readString(offset)
		if err != nil {
			return offset, err
		}
		offset = newOffset

		// Read value type
		valueType := MetadataValueType(binary.LittleEndian.Uint32(r.data[offset:]))
		offset += 4

		// Read value
		value, newOffset, err := r.readValue(offset, valueType)
		if err != nil {
			return offset, err
		}
		offset = newOffset

		r.metadata[key] = value
	}

	return offset, nil
}

// parseTensorInfo reads all tensor information
func (r *Reader) parseTensorInfo(offset int) (int, error) {
	r.tensors = make([]TensorInfo, r.header.TensorCount)

	for i := uint64(0); i < r.header.TensorCount; i++ {
		// Read tensor name
		name, newOffset, err := r.readString(offset)
		if err != nil {
			return offset, err
		}
		offset = newOffset

		// Read number of dimensions
		nDims := binary.LittleEndian.Uint32(r.data[offset:])
		offset += 4

		// Read dimensions
		dims := make([]uint64, nDims)
		for j := uint32(0); j < nDims; j++ {
			dims[j] = binary.LittleEndian.Uint64(r.data[offset:])
			offset += 8
		}

		// Read tensor type
		tensorType := GGMLType(binary.LittleEndian.Uint32(r.data[offset:]))
		offset += 4

		// Read offset
		tensorOffset := binary.LittleEndian.Uint64(r.data[offset:])
		offset += 8

		r.tensors[i] = TensorInfo{
			Name:       name,
			NDims:      nDims,
			Dimensions: dims,
			Type:       tensorType,
			Offset:     tensorOffset,
		}
		r.tensorMap[name] = &r.tensors[i]
	}

	return offset, nil
}

// readString reads a GGUF string (length-prefixed)
func (r *Reader) readString(offset int) (string, int, error) {
	if offset+8 > len(r.data) {
		return "", offset, fmt.Errorf("unexpected EOF reading string length")
	}

	length := binary.LittleEndian.Uint64(r.data[offset:])
	offset += 8

	if offset+int(length) > len(r.data) {
		return "", offset, fmt.Errorf("unexpected EOF reading string data")
	}

	s := string(r.data[offset : offset+int(length)])
	offset += int(length)

	return s, offset, nil
}

// readValue reads a metadata value of the given type
func (r *Reader) readValue(offset int, valueType MetadataValueType) (any, int, error) {
	switch valueType {
	case TypeUint8:
		return r.data[offset], offset + 1, nil

	case TypeInt8:
		return int8(r.data[offset]), offset + 1, nil

	case TypeUint16:
		return binary.LittleEndian.Uint16(r.data[offset:]), offset + 2, nil

	case TypeInt16:
		return int16(binary.LittleEndian.Uint16(r.data[offset:])), offset + 2, nil

	case TypeUint32:
		return binary.LittleEndian.Uint32(r.data[offset:]), offset + 4, nil

	case TypeInt32:
		return int32(binary.LittleEndian.Uint32(r.data[offset:])), offset + 4, nil

	case TypeFloat32:
		bits := binary.LittleEndian.Uint32(r.data[offset:])
		return float32FromBits(bits), offset + 4, nil

	case TypeBool:
		return r.data[offset] != 0, offset + 1, nil

	case TypeString:
		return r.readString(offset)

	case TypeUint64:
		return binary.LittleEndian.Uint64(r.data[offset:]), offset + 8, nil

	case TypeInt64:
		return int64(binary.LittleEndian.Uint64(r.data[offset:])), offset + 8, nil

	case TypeFloat64:
		bits := binary.LittleEndian.Uint64(r.data[offset:])
		return float64FromBits(bits), offset + 8, nil

	case TypeArray:
		// Read array type and length
		elemType := MetadataValueType(binary.LittleEndian.Uint32(r.data[offset:]))
		offset += 4
		length := binary.LittleEndian.Uint64(r.data[offset:])
		offset += 8

		// Read array elements
		arr := make([]any, length)
		var err error
		for i := uint64(0); i < length; i++ {
			arr[i], offset, err = r.readValue(offset, elemType)
			if err != nil {
				return nil, offset, err
			}
		}
		return arr, offset, nil

	default:
		return nil, offset, fmt.Errorf("unknown metadata value type: %d", valueType)
	}
}

// Header returns the file header
func (r *Reader) Header() Header {
	return r.header
}

// Metadata returns all metadata as a map
func (r *Reader) Metadata() map[string]any {
	return r.metadata
}

// GetMetadata returns a specific metadata value
func (r *Reader) GetMetadata(key string) (any, bool) {
	v, ok := r.metadata[key]
	return v, ok
}

// GetString returns a string metadata value
func (r *Reader) GetString(key string) (string, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// GetUint32 returns a uint32 metadata value
func (r *Reader) GetUint32(key string) (uint32, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, false
	}
	u, ok := v.(uint32)
	return u, ok
}

// GetUint64 returns a uint64 metadata value
func (r *Reader) GetUint64(key string) (uint64, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, false
	}
	u, ok := v.(uint64)
	return u, ok
}

// GetFloat32 returns a float32 metadata value
func (r *Reader) GetFloat32(key string) (float32, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, false
	}
	f, ok := v.(float32)
	return f, ok
}

// GetInt32 returns an int32 metadata value
func (r *Reader) GetInt32(key string) (int32, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return 0, false
	}
	i, ok := v.(int32)
	return i, ok
}

// GetStringArray returns a string array metadata value
func (r *Reader) GetStringArray(key string) ([]string, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	result := make([]string, len(arr))
	for i, elem := range arr {
		s, ok := elem.(string)
		if !ok {
			return nil, false
		}
		result[i] = s
	}
	return result, true
}

// GetFloat32Array returns a float32 array metadata value
func (r *Reader) GetFloat32Array(key string) ([]float32, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	result := make([]float32, len(arr))
	for i, elem := range arr {
		f, ok := elem.(float32)
		if !ok {
			return nil, false
		}
		result[i] = f
	}
	return result, true
}

// GetInt32Array returns an int32 array metadata value
func (r *Reader) GetInt32Array(key string) ([]int32, bool) {
	v, ok := r.metadata[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	result := make([]int32, len(arr))
	for i, elem := range arr {
		n, ok := elem.(int32)
		if !ok {
			return nil, false
		}
		result[i] = n
	}
	return result, true
}

// Tensors returns all tensor information
func (r *Reader) Tensors() []TensorInfo {
	return r.tensors
}

// GetTensorInfo returns information about a specific tensor
func (r *Reader) GetTensorInfo(name string) (*TensorInfo, bool) {
	t, ok := r.tensorMap[name]
	return t, ok
}

// GetTensorData returns the raw bytes for a tensor
func (r *Reader) GetTensorData(name string) ([]byte, error) {
	info, ok := r.tensorMap[name]
	if !ok {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}

	start := r.dataOffset + info.Offset
	size := info.SizeBytes()
	end := start + size

	if end > uint64(len(r.data)) {
		return nil, fmt.Errorf("tensor data extends beyond file")
	}

	return r.data[start:end], nil
}

// Architecture returns the model architecture name
func (r *Reader) Architecture() string {
	arch, _ := r.GetString(KeyGeneralArchitecture)
	return arch
}

// Helper functions for float conversion
func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}

func float64FromBits(bits uint64) float64 {
	return math.Float64frombits(bits)
}
