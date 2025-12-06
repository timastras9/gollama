package gguf

// GGUF file format constants
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const (
	// Magic number "GGUF" in little-endian
	Magic uint32 = 0x46554747

	// Current GGUF version
	Version uint32 = 3

	// Default tensor data alignment
	DefaultAlignment uint32 = 32
)

// MetadataValueType represents the type of a metadata value
type MetadataValueType uint32

const (
	TypeUint8   MetadataValueType = 0
	TypeInt8    MetadataValueType = 1
	TypeUint16  MetadataValueType = 2
	TypeInt16   MetadataValueType = 3
	TypeUint32  MetadataValueType = 4
	TypeInt32   MetadataValueType = 5
	TypeFloat32 MetadataValueType = 6
	TypeBool    MetadataValueType = 7
	TypeString  MetadataValueType = 8
	TypeArray   MetadataValueType = 9
	TypeUint64  MetadataValueType = 10
	TypeInt64   MetadataValueType = 11
	TypeFloat64 MetadataValueType = 12
)

// GGMLType represents the data type of tensor elements
type GGMLType uint32

const (
	GGML_TYPE_F32  GGMLType = 0
	GGML_TYPE_F16  GGMLType = 1
	GGML_TYPE_Q4_0 GGMLType = 2
	GGML_TYPE_Q4_1 GGMLType = 3
	GGML_TYPE_Q5_0 GGMLType = 6
	GGML_TYPE_Q5_1 GGMLType = 7
	GGML_TYPE_Q8_0 GGMLType = 8
	GGML_TYPE_Q8_1 GGMLType = 9
	// K-quants
	GGML_TYPE_Q2_K  GGMLType = 10
	GGML_TYPE_Q3_K  GGMLType = 11
	GGML_TYPE_Q4_K  GGMLType = 12
	GGML_TYPE_Q5_K  GGMLType = 13
	GGML_TYPE_Q6_K  GGMLType = 14
	GGML_TYPE_Q8_K  GGMLType = 15
	GGML_TYPE_IQ2_XXS GGMLType = 16
	GGML_TYPE_IQ2_XS  GGMLType = 17
	GGML_TYPE_IQ3_XXS GGMLType = 18
	GGML_TYPE_IQ1_S   GGMLType = 19
	GGML_TYPE_IQ4_NL  GGMLType = 20
	GGML_TYPE_IQ3_S   GGMLType = 21
	GGML_TYPE_IQ2_S   GGMLType = 22
	GGML_TYPE_IQ4_XS  GGMLType = 23
	GGML_TYPE_I8      GGMLType = 24
	GGML_TYPE_I16     GGMLType = 25
	GGML_TYPE_I32     GGMLType = 26
	GGML_TYPE_I64     GGMLType = 27
	GGML_TYPE_F64     GGMLType = 28
	GGML_TYPE_BF16    GGMLType = 30
)

func (t GGMLType) String() string {
	switch t {
	case GGML_TYPE_F32:
		return "F32"
	case GGML_TYPE_F16:
		return "F16"
	case GGML_TYPE_Q4_0:
		return "Q4_0"
	case GGML_TYPE_Q4_1:
		return "Q4_1"
	case GGML_TYPE_Q5_0:
		return "Q5_0"
	case GGML_TYPE_Q5_1:
		return "Q5_1"
	case GGML_TYPE_Q8_0:
		return "Q8_0"
	case GGML_TYPE_Q8_1:
		return "Q8_1"
	case GGML_TYPE_Q2_K:
		return "Q2_K"
	case GGML_TYPE_Q3_K:
		return "Q3_K"
	case GGML_TYPE_Q4_K:
		return "Q4_K"
	case GGML_TYPE_Q5_K:
		return "Q5_K"
	case GGML_TYPE_Q6_K:
		return "Q6_K"
	case GGML_TYPE_Q8_K:
		return "Q8_K"
	case GGML_TYPE_BF16:
		return "BF16"
	default:
		return "UNKNOWN"
	}
}

// TypeInfo contains information about a GGML type
type TypeInfo struct {
	BlockSize  int     // Number of elements per block
	TypeSize   int     // Size of one block in bytes
	IsQuantized bool   // Whether the type is quantized
}

// GetTypeInfo returns type information for a GGML type
func GetTypeInfo(t GGMLType) TypeInfo {
	switch t {
	case GGML_TYPE_F32:
		return TypeInfo{BlockSize: 1, TypeSize: 4, IsQuantized: false}
	case GGML_TYPE_F16:
		return TypeInfo{BlockSize: 1, TypeSize: 2, IsQuantized: false}
	case GGML_TYPE_Q4_0:
		return TypeInfo{BlockSize: 32, TypeSize: 18, IsQuantized: true} // 2 bytes scale + 16 bytes quants
	case GGML_TYPE_Q4_1:
		return TypeInfo{BlockSize: 32, TypeSize: 20, IsQuantized: true} // 2+2 bytes scale/min + 16 bytes quants
	case GGML_TYPE_Q5_0:
		return TypeInfo{BlockSize: 32, TypeSize: 22, IsQuantized: true}
	case GGML_TYPE_Q5_1:
		return TypeInfo{BlockSize: 32, TypeSize: 24, IsQuantized: true}
	case GGML_TYPE_Q8_0:
		return TypeInfo{BlockSize: 32, TypeSize: 34, IsQuantized: true} // 2 bytes scale + 32 bytes quants
	case GGML_TYPE_Q8_1:
		return TypeInfo{BlockSize: 32, TypeSize: 36, IsQuantized: true}
	case GGML_TYPE_Q2_K:
		return TypeInfo{BlockSize: 256, TypeSize: 84, IsQuantized: true}
	case GGML_TYPE_Q3_K:
		return TypeInfo{BlockSize: 256, TypeSize: 110, IsQuantized: true}
	case GGML_TYPE_Q4_K:
		return TypeInfo{BlockSize: 256, TypeSize: 144, IsQuantized: true}
	case GGML_TYPE_Q5_K:
		return TypeInfo{BlockSize: 256, TypeSize: 176, IsQuantized: true}
	case GGML_TYPE_Q6_K:
		return TypeInfo{BlockSize: 256, TypeSize: 210, IsQuantized: true}
	case GGML_TYPE_Q8_K:
		return TypeInfo{BlockSize: 256, TypeSize: 292, IsQuantized: true}
	case GGML_TYPE_BF16:
		return TypeInfo{BlockSize: 1, TypeSize: 2, IsQuantized: false}
	default:
		return TypeInfo{BlockSize: 1, TypeSize: 4, IsQuantized: false}
	}
}

// Header represents the GGUF file header
type Header struct {
	Magic           uint32
	Version         uint32
	TensorCount     uint64
	MetadataKVCount uint64
}

// TensorInfo represents information about a tensor in the file
type TensorInfo struct {
	Name       string
	NDims      uint32
	Dimensions []uint64 // Stored in reverse order in GGUF
	Type       GGMLType
	Offset     uint64   // Offset from start of tensor data section
}

// NumElements returns the total number of elements in the tensor
func (t *TensorInfo) NumElements() uint64 {
	if len(t.Dimensions) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

// SizeBytes returns the size of the tensor data in bytes
func (t *TensorInfo) SizeBytes() uint64 {
	info := GetTypeInfo(t.Type)
	numElements := t.NumElements()
	numBlocks := (numElements + uint64(info.BlockSize) - 1) / uint64(info.BlockSize)
	return numBlocks * uint64(info.TypeSize)
}

// Shape returns the tensor shape as stored in GGUF
// For weight matrices this is [n_out, n_in] / [rows, cols]
func (t *TensorInfo) Shape() []int {
	shape := make([]int, len(t.Dimensions))
	for i, d := range t.Dimensions {
		shape[i] = int(d)
	}
	return shape
}

// Common metadata keys
const (
	KeyGeneralArchitecture      = "general.architecture"
	KeyGeneralName              = "general.name"
	KeyGeneralFileType          = "general.file_type"
	KeyGeneralQuantizationVersion = "general.quantization_version"

	// LLaMA-specific keys (prefix with architecture name)
	KeyContextLength     = ".context_length"
	KeyEmbeddingLength   = ".embedding_length"
	KeyBlockCount        = ".block_count"
	KeyFeedForwardLength = ".feed_forward_length"
	KeyAttentionHeadCount = ".attention.head_count"
	KeyAttentionHeadCountKV = ".attention.head_count_kv"
	KeyAttentionLayerNormRMSEpsilon = ".attention.layer_norm_rms_epsilon"
	KeyRopeFreqBase      = ".rope.freq_base"
	KeyRopeDimensionCount = ".rope.dimension_count"

	// Tokenizer keys
	KeyTokenizerModel    = "tokenizer.ggml.model"
	KeyTokenizerTokens   = "tokenizer.ggml.tokens"
	KeyTokenizerScores   = "tokenizer.ggml.scores"
	KeyTokenizerTokenType = "tokenizer.ggml.token_type"
	KeyTokenizerBOSID    = "tokenizer.ggml.bos_token_id"
	KeyTokenizerEOSID    = "tokenizer.ggml.eos_token_id"
	KeyTokenizerPadID    = "tokenizer.ggml.padding_token_id"
)

// Architecture names
const (
	ArchLlama   = "llama"
	ArchMistral = "mistral"
	ArchPhi     = "phi"
	ArchGemma   = "gemma"
	ArchQwen2   = "qwen2"
)
