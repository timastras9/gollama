package train

import (
	"bufio"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/timastras9/gollama/pkg/tokenizer"
)

// Dataset represents a training dataset
type Dataset struct {
	Tokens     []int   // All tokens concatenated
	BlockSize  int     // Context length for training
	tokenizer  *tokenizer.Tokenizer
}

// NewDataset creates a new dataset from tokenized text
func NewDataset(tokens []int, blockSize int) *Dataset {
	return &Dataset{
		Tokens:    tokens,
		BlockSize: blockSize,
	}
}

// Len returns the number of training examples
func (d *Dataset) Len() int {
	if len(d.Tokens) <= d.BlockSize {
		return 0
	}
	return len(d.Tokens) - d.BlockSize
}

// GetBatch returns a batch of training examples
// Returns (input, target) where target is input shifted by 1
func (d *Dataset) GetBatch(batchSize int, rng *rand.Rand) ([][]int, [][]int) {
	inputs := make([][]int, batchSize)
	targets := make([][]int, batchSize)

	maxStart := len(d.Tokens) - d.BlockSize - 1
	if maxStart < 0 {
		return nil, nil
	}

	for i := 0; i < batchSize; i++ {
		start := rng.Intn(maxStart)
		inputs[i] = d.Tokens[start : start+d.BlockSize]
		targets[i] = d.Tokens[start+1 : start+d.BlockSize+1]
	}

	return inputs, targets
}

// DataLoader handles batching and shuffling
type DataLoader struct {
	Dataset   *Dataset
	BatchSize int
	Shuffle   bool
	rng       *rand.Rand
}

// NewDataLoader creates a new data loader
func NewDataLoader(dataset *Dataset, batchSize int, shuffle bool, seed int64) *DataLoader {
	return &DataLoader{
		Dataset:   dataset,
		BatchSize: batchSize,
		Shuffle:   shuffle,
		rng:       rand.New(rand.NewSource(seed)),
	}
}

// NextBatch returns the next batch of data
func (dl *DataLoader) NextBatch() ([][]int, [][]int) {
	return dl.Dataset.GetBatch(dl.BatchSize, dl.rng)
}

// LoadTextFile loads and tokenizes a text file
func LoadTextFile(path string, tok *tokenizer.Tokenizer) ([]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	text := string(data)
	tokens := tok.Encode(text)
	return tokens, nil
}

// LoadTextDir loads and tokenizes all .txt files in a directory
func LoadTextDir(dir string, tok *tokenizer.Tokenizer) ([]int, error) {
	var allTokens []int

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".txt" && ext != ".md" && ext != ".json" {
			return nil
		}

		tokens, err := LoadTextFile(path, tok)
		if err != nil {
			return err
		}

		allTokens = append(allTokens, tokens...)
		// Add separator between documents (EOS token ID)
		if tok.EOSTokenID >= 0 {
			allTokens = append(allTokens, tok.EOSTokenID)
		}

		return nil
	})

	return allTokens, err
}

// StreamingDataset reads data on-the-fly without loading all into memory
type StreamingDataset struct {
	files      []string
	tokenizer  *tokenizer.Tokenizer
	blockSize  int
	buffer     []int
	fileIndex  int
	reader     *bufio.Reader
	file       *os.File
}

// NewStreamingDataset creates a dataset that streams from files
func NewStreamingDataset(dir string, tok *tokenizer.Tokenizer, blockSize int) (*StreamingDataset, error) {
	var files []string

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			ext := strings.ToLower(filepath.Ext(path))
			if ext == ".txt" || ext == ".md" || ext == ".json" {
				files = append(files, path)
			}
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	ds := &StreamingDataset{
		files:     files,
		tokenizer: tok,
		blockSize: blockSize,
		buffer:    make([]int, 0, blockSize*10),
	}

	if len(files) > 0 {
		if err := ds.openNextFile(); err != nil {
			return nil, err
		}
	}

	return ds, nil
}

func (ds *StreamingDataset) openNextFile() error {
	if ds.file != nil {
		ds.file.Close()
	}

	if ds.fileIndex >= len(ds.files) {
		ds.fileIndex = 0 // Loop back
	}

	f, err := os.Open(ds.files[ds.fileIndex])
	if err != nil {
		return err
	}

	ds.file = f
	ds.reader = bufio.NewReader(f)
	ds.fileIndex++
	return nil
}

// GetBatch returns a batch from the streaming dataset
func (ds *StreamingDataset) GetBatch(batchSize int) ([][]int, [][]int, error) {
	// Fill buffer if needed
	for len(ds.buffer) < (batchSize+1)*(ds.blockSize+1) {
		line, err := ds.reader.ReadString('\n')
		if err == io.EOF {
			if err := ds.openNextFile(); err != nil {
				return nil, nil, err
			}
			continue
		} else if err != nil {
			return nil, nil, err
		}

		tokens := ds.tokenizer.Encode(line)
		ds.buffer = append(ds.buffer, tokens...)
	}

	inputs := make([][]int, batchSize)
	targets := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		start := i * ds.blockSize
		inputs[i] = make([]int, ds.blockSize)
		targets[i] = make([]int, ds.blockSize)
		copy(inputs[i], ds.buffer[start:start+ds.blockSize])
		copy(targets[i], ds.buffer[start+1:start+ds.blockSize+1])
	}

	// Shift buffer
	consumed := batchSize * ds.blockSize
	ds.buffer = ds.buffer[consumed:]

	return inputs, targets, nil
}

// Close releases resources
func (ds *StreamingDataset) Close() error {
	if ds.file != nil {
		return ds.file.Close()
	}
	return nil
}

// CybersecDataset is specialized for cybersecurity training data
type CybersecDataset struct {
	*Dataset
	categories map[string][]int // Category -> token indices
}

// Cybersecurity data categories
const (
	CategoryExploits     = "exploits"
	CategoryMalware      = "malware"
	CategoryPentesting   = "pentesting"
	CategoryDefense      = "defense"
	CategoryCTF          = "ctf"
	CategoryVulnerability = "vulnerability"
)

// NewCybersecDataset creates a cybersecurity-focused dataset
func NewCybersecDataset(dataDir string, tok *tokenizer.Tokenizer, blockSize int) (*CybersecDataset, error) {
	categories := make(map[string][]int)

	// Load data from category subdirectories
	categoryDirs := []string{
		CategoryExploits,
		CategoryMalware,
		CategoryPentesting,
		CategoryDefense,
		CategoryCTF,
		CategoryVulnerability,
	}

	var allTokens []int

	for _, cat := range categoryDirs {
		catPath := filepath.Join(dataDir, cat)
		if _, err := os.Stat(catPath); os.IsNotExist(err) {
			continue
		}

		tokens, err := LoadTextDir(catPath, tok)
		if err != nil {
			continue // Skip on error
		}

		categories[cat] = tokens
		allTokens = append(allTokens, tokens...)
	}

	// Also load any files in root data directory
	rootTokens, err := LoadTextDir(dataDir, tok)
	if err == nil {
		allTokens = append(allTokens, rootTokens...)
	}

	return &CybersecDataset{
		Dataset:    NewDataset(allTokens, blockSize),
		categories: categories,
	}, nil
}

// GetCategoryBatch returns a batch from a specific category
func (d *CybersecDataset) GetCategoryBatch(category string, batchSize int, rng *rand.Rand) ([][]int, [][]int) {
	tokens, ok := d.categories[category]
	if !ok || len(tokens) <= d.BlockSize {
		return d.GetBatch(batchSize, rng)
	}

	catDataset := NewDataset(tokens, d.BlockSize)
	return catDataset.GetBatch(batchSize, rng)
}
