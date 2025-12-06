package model

import (
	"math"
	"math/rand"
	"sort"

	"github.com/timastras9/gollama/pkg/tensor"
)

// SamplerConfig holds sampling parameters
type SamplerConfig struct {
	Temperature float32 // Temperature for softmax (1.0 = normal, <1 = more deterministic)
	TopK        int     // Only consider top-k tokens (0 = disabled)
	TopP        float32 // Nucleus sampling threshold (1.0 = disabled)
	MinP        float32 // Minimum probability threshold (0.0 = disabled)
	Seed        int64   // Random seed (-1 for random)
}

// DefaultSamplerConfig returns default sampling parameters
func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{
		Temperature: 0.8,
		TopK:        40,
		TopP:        0.95,
		MinP:        0.0,
		Seed:        -1,
	}
}

// Sampler handles token sampling from logits
type Sampler struct {
	Config SamplerConfig
	rng    *rand.Rand
}

// NewSampler creates a new sampler
func NewSampler(cfg SamplerConfig) *Sampler {
	var rng *rand.Rand
	if cfg.Seed < 0 {
		rng = rand.New(rand.NewSource(rand.Int63()))
	} else {
		rng = rand.New(rand.NewSource(cfg.Seed))
	}

	return &Sampler{
		Config: cfg,
		rng:    rng,
	}
}

// Sample samples a token from logits
func (s *Sampler) Sample(logits *tensor.Tensor) int {
	// Apply temperature
	if s.Config.Temperature != 1.0 && s.Config.Temperature > 0 {
		invTemp := 1.0 / s.Config.Temperature
		for i := range logits.Data {
			logits.Data[i] *= invTemp
		}
	}

	// Convert to probabilities (softmax)
	probs := s.softmax(logits.Data)

	// Apply sampling strategies
	candidates := s.getCandidates(probs)

	// Apply top-k
	if s.Config.TopK > 0 && len(candidates) > s.Config.TopK {
		candidates = candidates[:s.Config.TopK]
	}

	// Apply top-p (nucleus sampling)
	if s.Config.TopP < 1.0 && s.Config.TopP > 0 {
		candidates = s.applyTopP(candidates)
	}

	// Apply min-p
	if s.Config.MinP > 0 {
		candidates = s.applyMinP(candidates)
	}

	// Renormalize probabilities
	var sumProb float32
	for _, c := range candidates {
		sumProb += c.prob
	}
	for i := range candidates {
		candidates[i].prob /= sumProb
	}

	// Sample from candidates
	return s.sampleFromCandidates(candidates)
}

// SampleGreedy returns the most likely token
func (s *Sampler) SampleGreedy(logits *tensor.Tensor) int {
	return logits.ArgMax()
}

// candidate represents a token with its probability
type candidate struct {
	token int
	prob  float32
}

// getCandidates creates sorted candidate list from probabilities
func (s *Sampler) getCandidates(probs []float32) []candidate {
	candidates := make([]candidate, len(probs))
	for i, p := range probs {
		candidates[i] = candidate{token: i, prob: p}
	}

	// Sort by probability (descending)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	return candidates
}

// applyTopP implements nucleus sampling
func (s *Sampler) applyTopP(candidates []candidate) []candidate {
	var cumProb float32
	cutoff := 0

	for i, c := range candidates {
		cumProb += c.prob
		cutoff = i + 1
		if cumProb >= s.Config.TopP {
			break
		}
	}

	return candidates[:cutoff]
}

// applyMinP filters candidates below minimum probability
func (s *Sampler) applyMinP(candidates []candidate) []candidate {
	if len(candidates) == 0 {
		return candidates
	}

	// MinP is relative to the top probability
	threshold := candidates[0].prob * s.Config.MinP

	cutoff := len(candidates)
	for i, c := range candidates {
		if c.prob < threshold {
			cutoff = i
			break
		}
	}

	if cutoff == 0 {
		cutoff = 1 // Keep at least one candidate
	}

	return candidates[:cutoff]
}

// sampleFromCandidates randomly samples based on probabilities
func (s *Sampler) sampleFromCandidates(candidates []candidate) int {
	if len(candidates) == 0 {
		return 0
	}

	if len(candidates) == 1 {
		return candidates[0].token
	}

	r := s.rng.Float32()
	var cumProb float32

	for _, c := range candidates {
		cumProb += c.prob
		if r < cumProb {
			return c.token
		}
	}

	return candidates[len(candidates)-1].token
}

// softmax computes softmax probabilities (numerically stable)
func (s *Sampler) softmax(logits []float32) []float32 {
	probs := make([]float32, len(logits))

	// Find max for numerical stability
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}

	// Compute exp and sum
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxLogit)))
		sum += probs[i]
	}

	// Normalize
	invSum := 1.0 / sum
	for i := range probs {
		probs[i] *= invSum
	}

	return probs
}

// SetSeed sets a new random seed
func (s *Sampler) SetSeed(seed int64) {
	if seed < 0 {
		s.rng = rand.New(rand.NewSource(rand.Int63()))
	} else {
		s.rng = rand.New(rand.NewSource(seed))
	}
}

// RepetitionPenalty applies repetition penalty to logits
// penalty > 1.0 reduces probability of repeated tokens
func RepetitionPenalty(logits *tensor.Tensor, previousTokens []int, penalty float32) {
	if penalty == 1.0 {
		return
	}

	for _, token := range previousTokens {
		if token >= 0 && token < logits.Size() {
			if logits.Data[token] > 0 {
				logits.Data[token] /= penalty
			} else {
				logits.Data[token] *= penalty
			}
		}
	}
}

// FrequencyPenalty applies frequency-based penalty
// Reduces probability of tokens based on how often they appear
func FrequencyPenalty(logits *tensor.Tensor, previousTokens []int, penalty float32) {
	if penalty == 0 {
		return
	}

	// Count token frequencies
	freq := make(map[int]int)
	for _, token := range previousTokens {
		freq[token]++
	}

	// Apply penalty
	for token, count := range freq {
		if token >= 0 && token < logits.Size() {
			logits.Data[token] -= penalty * float32(count)
		}
	}
}

// PresencePenalty applies presence-based penalty
// Reduces probability of any token that has appeared before
func PresencePenalty(logits *tensor.Tensor, previousTokens []int, penalty float32) {
	if penalty == 0 {
		return
	}

	// Track which tokens have appeared
	seen := make(map[int]bool)
	for _, token := range previousTokens {
		seen[token] = true
	}

	// Apply penalty
	for token := range seen {
		if token >= 0 && token < logits.Size() {
			logits.Data[token] -= penalty
		}
	}
}
