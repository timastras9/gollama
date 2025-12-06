package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/timastras9/gollama/pkg/gguf"
)

// TokenType represents the type of a token
type TokenType int32

const (
	TokenNormal    TokenType = 1
	TokenUnknown   TokenType = 2
	TokenControl   TokenType = 3
	TokenUserDef   TokenType = 4
	TokenUnused    TokenType = 5
	TokenByte      TokenType = 6
)

// Tokenizer handles text tokenization using BPE
type Tokenizer struct {
	// Vocabulary
	Vocab       map[string]int // Token string -> ID
	VocabByID   []string       // ID -> Token string
	Scores      []float32      // Token scores (for BPE merges)
	TokenTypes  []TokenType    // Token types

	// Special tokens
	BOSToken   string
	EOSToken   string
	BOSTokenID int
	EOSTokenID int
	PadTokenID int

	// BPE merge rules (derived from scores)
	mergeRanks map[string]int

	// Configuration
	AddBOS bool
	AddEOS bool
}

// New creates a new tokenizer with empty vocabulary
func New() *Tokenizer {
	return &Tokenizer{
		Vocab:      make(map[string]int),
		mergeRanks: make(map[string]int),
		AddBOS:     true,
		AddEOS:     false,
	}
}

// FromGGUF creates a tokenizer from GGUF metadata
func FromGGUF(r *gguf.Reader) (*Tokenizer, error) {
	t := New()

	// Load vocabulary tokens
	tokens, ok := r.GetStringArray(gguf.KeyTokenizerTokens)
	if !ok {
		return nil, fmt.Errorf("tokenizer tokens not found")
	}

	t.VocabByID = tokens
	for i, token := range tokens {
		t.Vocab[token] = i
	}

	// Load scores (used for BPE merge priority)
	if scores, ok := r.GetFloat32Array(gguf.KeyTokenizerScores); ok {
		t.Scores = scores
	} else {
		// Default scores based on position
		t.Scores = make([]float32, len(tokens))
		for i := range t.Scores {
			t.Scores[i] = float32(len(tokens) - i)
		}
	}

	// Load token types
	if types, ok := r.GetInt32Array(gguf.KeyTokenizerTokenType); ok {
		t.TokenTypes = make([]TokenType, len(types))
		for i, typ := range types {
			t.TokenTypes[i] = TokenType(typ)
		}
	}

	// Find special tokens
	t.BOSTokenID = -1
	t.EOSTokenID = -1
	t.PadTokenID = -1

	if id, ok := r.GetUint32(gguf.KeyTokenizerBOSID); ok {
		t.BOSTokenID = int(id)
		if int(id) < len(tokens) {
			t.BOSToken = tokens[id]
		}
	}

	if id, ok := r.GetUint32(gguf.KeyTokenizerEOSID); ok {
		t.EOSTokenID = int(id)
		if int(id) < len(tokens) {
			t.EOSToken = tokens[id]
		}
	}

	if id, ok := r.GetUint32(gguf.KeyTokenizerPadID); ok {
		t.PadTokenID = int(id)
	}

	// Build merge ranks from scores
	t.buildMergeRanks()

	return t, nil
}

// buildMergeRanks creates merge priority map from vocabulary
func (t *Tokenizer) buildMergeRanks() {
	t.mergeRanks = make(map[string]int)

	// Build pairs from vocabulary with scores
	type scoredToken struct {
		token string
		score float32
		id    int
	}

	scored := make([]scoredToken, 0, len(t.VocabByID))
	for i, token := range t.VocabByID {
		if len(token) > 1 && i < len(t.Scores) {
			scored = append(scored, scoredToken{token, t.Scores[i], i})
		}
	}

	// Sort by score (higher = higher priority)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Assign ranks (lower rank = higher priority for merge)
	for rank, st := range scored {
		t.mergeRanks[st.token] = rank
	}
}

// Encode tokenizes text into token IDs
func (t *Tokenizer) Encode(text string) []int {
	tokens := []int{}

	if t.AddBOS && t.BOSTokenID >= 0 {
		tokens = append(tokens, t.BOSTokenID)
	}

	// Tokenize the text
	encoded := t.tokenize(text)
	tokens = append(tokens, encoded...)

	if t.AddEOS && t.EOSTokenID >= 0 {
		tokens = append(tokens, t.EOSTokenID)
	}

	return tokens
}

// tokenize performs BPE tokenization
func (t *Tokenizer) tokenize(text string) []int {
	if text == "" {
		return nil
	}

	// Start with byte-level or character-level tokens
	chars := t.splitToChars(text)
	if len(chars) == 0 {
		return nil
	}

	// Iteratively apply BPE merges
	for len(chars) > 1 {
		// Find the best pair to merge
		bestPair := -1
		bestRank := -1

		for i := 0; i < len(chars)-1; i++ {
			merged := chars[i] + chars[i+1]
			if rank, ok := t.mergeRanks[merged]; ok {
				if bestPair == -1 || rank < bestRank {
					bestPair = i
					bestRank = rank
				}
			}
		}

		if bestPair == -1 {
			break // No more merges possible
		}

		// Apply the merge
		merged := chars[bestPair] + chars[bestPair+1]
		newChars := make([]string, 0, len(chars)-1)
		newChars = append(newChars, chars[:bestPair]...)
		newChars = append(newChars, merged)
		newChars = append(newChars, chars[bestPair+2:]...)
		chars = newChars
	}

	// Convert to token IDs
	ids := make([]int, 0, len(chars))
	for _, tok := range chars {
		if id, ok := t.Vocab[tok]; ok {
			ids = append(ids, id)
		} else {
			// Handle unknown tokens - encode as bytes
			ids = append(ids, t.encodeUnknown(tok)...)
		}
	}

	return ids
}

// splitToChars splits text into initial tokens (character or byte level)
func (t *Tokenizer) splitToChars(text string) []string {
	var result []string

	// Check for SentencePiece-style tokenization (▁ prefix for spaces)
	useSpacePrefix := false
	for token := range t.Vocab {
		if strings.HasPrefix(token, "▁") {
			useSpacePrefix = true
			break
		}
	}

	if useSpacePrefix {
		// SentencePiece style: prepend ▁ to words
		text = strings.ReplaceAll(text, " ", "▁")
		if !strings.HasPrefix(text, "▁") {
			text = "▁" + text
		}
	}

	// Split into UTF-8 characters
	for _, r := range text {
		char := string(r)
		if _, ok := t.Vocab[char]; ok {
			result = append(result, char)
		} else {
			// Fall back to byte encoding
			for _, b := range []byte(char) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				if _, ok := t.Vocab[byteToken]; ok {
					result = append(result, byteToken)
				} else {
					// Last resort: just the character
					result = append(result, char)
				}
			}
		}
	}

	return result
}

// encodeUnknown handles unknown tokens
func (t *Tokenizer) encodeUnknown(s string) []int {
	var ids []int

	// Try to encode as bytes
	for _, b := range []byte(s) {
		byteToken := fmt.Sprintf("<0x%02X>", b)
		if id, ok := t.Vocab[byteToken]; ok {
			ids = append(ids, id)
		}
	}

	return ids
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var builder strings.Builder

	for _, id := range ids {
		if id < 0 || id >= len(t.VocabByID) {
			continue
		}

		// Skip special tokens
		if id == t.BOSTokenID || id == t.EOSTokenID || id == t.PadTokenID {
			continue
		}

		token := t.VocabByID[id]

		// Handle byte tokens
		if strings.HasPrefix(token, "<0x") && strings.HasSuffix(token, ">") {
			var b byte
			fmt.Sscanf(token, "<0x%02X>", &b)
			builder.WriteByte(b)
			continue
		}

		// Handle SentencePiece whitespace
		token = strings.ReplaceAll(token, "▁", " ")

		builder.WriteString(token)
	}

	result := builder.String()

	// Clean up leading space from SentencePiece
	result = strings.TrimPrefix(result, " ")

	return result
}

// DecodeToken decodes a single token ID
func (t *Tokenizer) DecodeToken(id int) string {
	if id < 0 || id >= len(t.VocabByID) {
		return ""
	}

	token := t.VocabByID[id]
	token = strings.ReplaceAll(token, "▁", " ")
	return token
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.VocabByID)
}

// IsSpecialToken checks if an ID is a special token
func (t *Tokenizer) IsSpecialToken(id int) bool {
	return id == t.BOSTokenID || id == t.EOSTokenID || id == t.PadTokenID
}

// IsBOS checks if token is beginning of sequence
func (t *Tokenizer) IsBOS(id int) bool {
	return id == t.BOSTokenID
}

// IsEOS checks if token is end of sequence
func (t *Tokenizer) IsEOS(id int) bool {
	return id == t.EOSTokenID
}

// CountTokens counts the number of tokens in text without allocating IDs
func (t *Tokenizer) CountTokens(text string) int {
	return len(t.Encode(text))
}

// TruncateToFit truncates text to fit within maxTokens
func (t *Tokenizer) TruncateToFit(text string, maxTokens int) string {
	tokens := t.Encode(text)
	if len(tokens) <= maxTokens {
		return text
	}

	// Binary search for the right cutoff
	runes := []rune(text)
	low, high := 0, len(runes)

	for low < high {
		mid := (low + high + 1) / 2
		truncated := string(runes[:mid])
		if len(t.Encode(truncated)) <= maxTokens {
			low = mid
		} else {
			high = mid - 1
		}
	}

	return string(runes[:low])
}

// helper for validating UTF-8
func isValidUTF8(s string) bool {
	return utf8.ValidString(s)
}
