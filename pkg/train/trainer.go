package train

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/timastras9/gollama/pkg/tensor"
)

// TrainConfig holds training hyperparameters
type TrainConfig struct {
	// Model
	VocabSize    int
	HiddenSize   int
	NumLayers    int
	NumHeads     int
	IntermediateSize int
	MaxSeqLen    int

	// Training
	BatchSize       int
	BlockSize       int     // Context length
	LearningRate    float32
	WeightDecay     float32
	WarmupSteps     int
	MaxSteps        int
	GradClipNorm    float32

	// Logging & Checkpointing
	LogInterval     int
	SaveInterval    int
	SavePath        string

	// Device
	UseGPU bool
}

// DefaultTrainConfig returns sensible defaults for training
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		// Small model suitable for M4 Pro
		VocabSize:        32000,
		HiddenSize:       512,
		NumLayers:        8,
		NumHeads:         8,
		IntermediateSize: 1408, // ~2.75x hidden
		MaxSeqLen:        512,

		// Training
		BatchSize:    4,
		BlockSize:    256,
		LearningRate: 3e-4,
		WeightDecay:  0.01,
		WarmupSteps:  100,
		MaxSteps:     10000,
		GradClipNorm: 1.0,

		// Logging
		LogInterval:  10,
		SaveInterval: 1000,
		SavePath:     "./checkpoints",

		UseGPU: true,
	}
}

// CybersecModelConfig returns config for a cybersecurity-focused model
func CybersecModelConfig() TrainConfig {
	cfg := DefaultTrainConfig()
	cfg.HiddenSize = 768
	cfg.NumLayers = 12
	cfg.NumHeads = 12
	cfg.IntermediateSize = 2048
	cfg.MaxSeqLen = 1024
	cfg.MaxSteps = 50000
	return cfg
}

// TrainableModel is a model that can be trained
type TrainableModel struct {
	Config TrainConfig

	// Embedding
	TokenEmbed *tensor.GradTensor // [vocab_size, hidden_size]
	PosEmbed   *tensor.GradTensor // [max_seq_len, hidden_size]

	// Transformer layers
	Layers []*TrainableLayer

	// Output
	OutputNorm *RMSNormTrainable
	OutputProj *tensor.GradTensor // [vocab_size, hidden_size] (tied with embedding)

	// All trainable parameters
	params []*tensor.GradTensor
}

// TrainableLayer is a transformer layer with gradient tracking
type TrainableLayer struct {
	// Attention
	AttnNorm *RMSNormTrainable
	WQ       *tensor.GradTensor // [hidden_size, hidden_size]
	WK       *tensor.GradTensor // [hidden_size, hidden_size]
	WV       *tensor.GradTensor // [hidden_size, hidden_size]
	WO       *tensor.GradTensor // [hidden_size, hidden_size]

	// FFN
	FFNNorm *RMSNormTrainable
	WGate   *tensor.GradTensor // [intermediate_size, hidden_size]
	WUp     *tensor.GradTensor // [intermediate_size, hidden_size]
	WDown   *tensor.GradTensor // [hidden_size, intermediate_size]

	// Config
	NumHeads int
	HeadDim  int
}

// RMSNormTrainable is RMSNorm with gradient tracking
type RMSNormTrainable struct {
	Weight *tensor.GradTensor
	Eps    float32
}

// NewRMSNormTrainable creates a trainable RMSNorm layer
func NewRMSNormTrainable(size int) *RMSNormTrainable {
	weight := tensor.NewGrad(size)
	for i := range weight.Data {
		weight.Data[i] = 1.0
	}
	return &RMSNormTrainable{
		Weight: weight,
		Eps:    1e-5,
	}
}

// Forward applies RMSNorm with gradient tracking
func (rms *RMSNormTrainable) Forward(x *tensor.GradTensor) *tensor.GradTensor {
	// Compute RMS
	out := &tensor.GradTensor{
		Tensor:      tensor.New(x.Shape...),
		RequireGrad: x.RequireGrad || rms.Weight.RequireGrad,
		Parents:     []*tensor.GradTensor{x, rms.Weight},
	}

	rows := 1
	if len(x.Shape) == 2 {
		rows = x.Shape[0]
	}
	cols := x.Shape[len(x.Shape)-1]

	rmsVals := make([]float32, rows) // Store for backward

	for i := 0; i < rows; i++ {
		var ss float32
		offset := i * cols
		for j := 0; j < cols; j++ {
			ss += x.Data[offset+j] * x.Data[offset+j]
		}
		rmsVals[i] = float32(math.Sqrt(float64(ss)/float64(cols) + float64(rms.Eps)))

		for j := 0; j < cols; j++ {
			out.Data[offset+j] = (x.Data[offset+j] / rmsVals[i]) * rms.Weight.Data[j]
		}
	}

	out.GradFn = func() {
		if out.Grad == nil {
			return
		}

		if x.RequireGrad && x.Grad == nil {
			x.Grad = tensor.New(x.Shape...)
		}
		if rms.Weight.RequireGrad && rms.Weight.Grad == nil {
			rms.Weight.Grad = tensor.New(rms.Weight.Shape...)
		}

		for i := 0; i < rows; i++ {
			offset := i * cols
			rmsVal := rmsVals[i]

			for j := 0; j < cols; j++ {
				dout := out.Grad.Data[offset+j]

				// Gradient w.r.t. weight
				if rms.Weight.RequireGrad {
					rms.Weight.Grad.Data[j] += dout * x.Data[offset+j] / rmsVal
				}

				// Gradient w.r.t. input (simplified)
				if x.RequireGrad {
					x.Grad.Data[offset+j] += dout * rms.Weight.Data[j] / rmsVal
				}
			}
		}
	}

	return out
}

// NewTrainableModel creates a new model for training from scratch
func NewTrainableModel(cfg TrainConfig) *TrainableModel {
	rng := rand.New(rand.NewSource(42))

	model := &TrainableModel{
		Config: cfg,
		params: make([]*tensor.GradTensor, 0),
	}

	// Initialize embeddings
	model.TokenEmbed = initWeight(cfg.VocabSize, cfg.HiddenSize, rng)
	model.PosEmbed = initWeight(cfg.MaxSeqLen, cfg.HiddenSize, rng)
	model.params = append(model.params, model.TokenEmbed, model.PosEmbed)

	// Initialize layers
	model.Layers = make([]*TrainableLayer, cfg.NumLayers)
	headDim := cfg.HiddenSize / cfg.NumHeads

	for l := 0; l < cfg.NumLayers; l++ {
		layer := &TrainableLayer{
			AttnNorm: NewRMSNormTrainable(cfg.HiddenSize),
			WQ:       initWeight(cfg.HiddenSize, cfg.HiddenSize, rng),
			WK:       initWeight(cfg.HiddenSize, cfg.HiddenSize, rng),
			WV:       initWeight(cfg.HiddenSize, cfg.HiddenSize, rng),
			WO:       initWeight(cfg.HiddenSize, cfg.HiddenSize, rng),

			FFNNorm: NewRMSNormTrainable(cfg.HiddenSize),
			WGate:   initWeight(cfg.IntermediateSize, cfg.HiddenSize, rng),
			WUp:     initWeight(cfg.IntermediateSize, cfg.HiddenSize, rng),
			WDown:   initWeight(cfg.HiddenSize, cfg.IntermediateSize, rng),

			NumHeads: cfg.NumHeads,
			HeadDim:  headDim,
		}
		model.Layers[l] = layer

		// Add to params
		model.params = append(model.params,
			layer.AttnNorm.Weight,
			layer.WQ, layer.WK, layer.WV, layer.WO,
			layer.FFNNorm.Weight,
			layer.WGate, layer.WUp, layer.WDown,
		)
	}

	// Output projection (can be tied with embedding)
	model.OutputNorm = NewRMSNormTrainable(cfg.HiddenSize)
	model.OutputProj = model.TokenEmbed // Weight tying
	model.params = append(model.params, model.OutputNorm.Weight)

	return model
}

// initWeight initializes a weight matrix with Xavier/He initialization
func initWeight(rows, cols int, rng *rand.Rand) *tensor.GradTensor {
	t := tensor.NewGrad(rows, cols)
	scale := float32(math.Sqrt(2.0 / float64(rows+cols)))
	for i := range t.Data {
		t.Data[i] = (rng.Float32()*2 - 1) * scale
	}
	return t
}

// Forward runs the forward pass for training
// input: [batch, seq_len] token indices
// Returns logits: [batch, seq_len, vocab_size]
func (m *TrainableModel) Forward(input [][]int) *tensor.GradTensor {
	batchSize := len(input)
	seqLen := len(input[0])

	// Token embedding lookup
	hidden := tensor.NewGrad(batchSize*seqLen, m.Config.HiddenSize)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			tokenID := input[b][s]
			if tokenID < 0 || tokenID >= m.Config.VocabSize {
				tokenID = 0
			}
			offset := (b*seqLen + s) * m.Config.HiddenSize
			embOffset := tokenID * m.Config.HiddenSize
			for h := 0; h < m.Config.HiddenSize; h++ {
				hidden.Data[offset+h] = m.TokenEmbed.Data[embOffset+h] + m.PosEmbed.Data[s*m.Config.HiddenSize+h]
			}
		}
	}
	hidden.RequireGrad = true
	hidden.Parents = []*tensor.GradTensor{m.TokenEmbed, m.PosEmbed}

	// Process through layers
	for _, layer := range m.Layers {
		hidden = m.forwardLayer(layer, hidden, seqLen)
	}

	// Output norm
	hidden = m.OutputNorm.Forward(hidden)

	// Project to vocab
	logits := tensor.MatMulGrad(hidden, m.OutputProj)

	return logits
}

// forwardLayer processes one transformer layer
func (m *TrainableModel) forwardLayer(layer *TrainableLayer, x *tensor.GradTensor, seqLen int) *tensor.GradTensor {
	// Self-attention with residual
	normed := layer.AttnNorm.Forward(x)
	attnOut := m.forwardAttention(layer, normed, seqLen)
	x = tensor.AddGrad(x, attnOut)

	// FFN with residual
	normed = layer.FFNNorm.Forward(x)
	ffnOut := m.forwardFFN(layer, normed)
	x = tensor.AddGrad(x, ffnOut)

	return x
}

// forwardAttention computes self-attention (simplified for training)
func (m *TrainableModel) forwardAttention(layer *TrainableLayer, x *tensor.GradTensor, seqLen int) *tensor.GradTensor {
	// Project Q, K, V
	q := tensor.MatMulGrad(x, layer.WQ)
	k := tensor.MatMulGrad(x, layer.WK)
	v := tensor.MatMulGrad(x, layer.WV)

	// Simplified attention (not fully optimized for training)
	// For real training, would need more efficient implementation
	batchSeq := x.Shape[0]
	hiddenSize := x.Shape[1]

	// Compute attention scores and weighted sum (simplified)
	attn := tensor.NewGrad(batchSeq, hiddenSize)
	attn.RequireGrad = true
	attn.Parents = []*tensor.GradTensor{q, k, v}

	scale := float32(1.0 / math.Sqrt(float64(layer.HeadDim)))

	// Simple scaled dot-product attention per position
	for i := 0; i < batchSeq; i++ {
		posInSeq := i % seqLen
		batchIdx := i / seqLen

		for h := 0; h < layer.NumHeads; h++ {
			headOffset := h * layer.HeadDim

			// Compute attention scores for this head
			scores := make([]float32, posInSeq+1)
			var maxScore float32 = -1e10

			for j := 0; j <= posInSeq; j++ {
				jIdx := batchIdx*seqLen + j
				var dot float32
				for d := 0; d < layer.HeadDim; d++ {
					dot += q.Data[i*hiddenSize+headOffset+d] * k.Data[jIdx*hiddenSize+headOffset+d]
				}
				scores[j] = dot * scale
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}

			// Softmax
			var sumExp float32
			for j := range scores {
				scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
				sumExp += scores[j]
			}
			for j := range scores {
				scores[j] /= sumExp
			}

			// Weighted sum of values
			for d := 0; d < layer.HeadDim; d++ {
				var sum float32
				for j := 0; j <= posInSeq; j++ {
					jIdx := batchIdx*seqLen + j
					sum += scores[j] * v.Data[jIdx*hiddenSize+headOffset+d]
				}
				attn.Data[i*hiddenSize+headOffset+d] = sum
			}
		}
	}

	// Output projection
	return tensor.MatMulGrad(attn, layer.WO)
}

// forwardFFN computes feed-forward network
func (m *TrainableModel) forwardFFN(layer *TrainableLayer, x *tensor.GradTensor) *tensor.GradTensor {
	gate := tensor.MatMulGrad(x, layer.WGate)
	up := tensor.MatMulGrad(x, layer.WUp)

	// SiLU(gate) * up
	gate = tensor.SiLUGrad(gate)

	// Element-wise multiply
	hidden := &tensor.GradTensor{
		Tensor:      tensor.New(gate.Shape...),
		RequireGrad: true,
		Parents:     []*tensor.GradTensor{gate, up},
	}
	for i := range hidden.Data {
		hidden.Data[i] = gate.Data[i] * up.Data[i]
	}
	hidden.GradFn = func() {
		if hidden.Grad == nil {
			return
		}
		if gate.Grad == nil {
			gate.Grad = tensor.New(gate.Shape...)
		}
		if up.Grad == nil {
			up.Grad = tensor.New(up.Shape...)
		}
		for i := range hidden.Grad.Data {
			gate.Grad.Data[i] += hidden.Grad.Data[i] * up.Data[i]
			up.Grad.Data[i] += hidden.Grad.Data[i] * gate.Data[i]
		}
	}

	// Down projection
	return tensor.MatMulGrad(hidden, layer.WDown)
}

// Params returns all trainable parameters
func (m *TrainableModel) Params() []*tensor.GradTensor {
	return m.params
}

// Trainer handles the training loop
type Trainer struct {
	Model     *TrainableModel
	Optimizer Optimizer
	Scheduler *LRScheduler
	Config    TrainConfig

	step       int
	totalLoss  float32
	lossCount  int
	startTime  time.Time
}

// NewTrainer creates a new trainer
func NewTrainer(model *TrainableModel, cfg TrainConfig) *Trainer {
	opt := NewAdamW(cfg.LearningRate)
	opt.WeightDecay = cfg.WeightDecay

	scheduler := NewCosineScheduler(cfg.LearningRate, cfg.WarmupSteps, cfg.MaxSteps)

	return &Trainer{
		Model:     model,
		Optimizer: opt,
		Scheduler: scheduler,
		Config:    cfg,
	}
}

// TrainStep performs one training step
func (t *Trainer) TrainStep(inputs, targets [][]int) float32 {
	t.step++

	// Update learning rate
	lr := t.Scheduler.GetLR(t.step)
	if adamw, ok := t.Optimizer.(*AdamW); ok {
		adamw.LR = lr
	}

	// Zero gradients
	t.Optimizer.ZeroGrad(t.Model.Params())

	// Forward pass
	logits := t.Model.Forward(inputs)

	// Flatten for loss computation
	batchSize := len(inputs)
	seqLen := len(inputs[0])
	flatTargets := make([]int, batchSize*seqLen)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			flatTargets[b*seqLen+s] = targets[b][s]
		}
	}

	// Compute loss
	_, loss := tensor.SoftmaxCrossEntropyGrad(logits, flatTargets)

	// Backward pass
	logits.Backward()

	// Gradient clipping
	gradNorm := GradientClipping(t.Model.Params(), t.Config.GradClipNorm)
	_ = gradNorm

	// Optimizer step
	t.Optimizer.Step(t.Model.Params())

	// Track loss
	t.totalLoss += loss
	t.lossCount++

	return loss
}

// Train runs the full training loop
func (t *Trainer) Train(dataLoader *DataLoader) {
	t.startTime = time.Now()

	// Create checkpoint directory
	os.MkdirAll(t.Config.SavePath, 0755)

	log.Printf("Starting training: %d steps, batch_size=%d, lr=%.2e",
		t.Config.MaxSteps, t.Config.BatchSize, t.Config.LearningRate)
	log.Printf("Model: %d layers, hidden=%d, params=%.1fM",
		t.Config.NumLayers, t.Config.HiddenSize,
		float64(t.countParams())/1e6)

	for t.step < t.Config.MaxSteps {
		inputs, targets := dataLoader.NextBatch()
		if inputs == nil {
			log.Println("Warning: empty batch")
			continue
		}

		loss := t.TrainStep(inputs, targets)

		// Logging
		if t.step%t.Config.LogInterval == 0 {
			avgLoss := t.totalLoss / float32(t.lossCount)
			elapsed := time.Since(t.startTime)
			stepsPerSec := float64(t.step) / elapsed.Seconds()
			lr := t.Scheduler.GetLR(t.step)

			log.Printf("step %d | loss: %.4f (avg: %.4f) | lr: %.2e | %.1f steps/sec",
				t.step, loss, avgLoss, lr, stepsPerSec)

			t.totalLoss = 0
			t.lossCount = 0
		}

		// Checkpointing
		if t.step%t.Config.SaveInterval == 0 {
			t.SaveCheckpoint()
		}
	}

	log.Printf("Training complete! Final step: %d", t.step)
	t.SaveCheckpoint()
}

// SaveCheckpoint saves the model weights
func (t *Trainer) SaveCheckpoint() {
	path := fmt.Sprintf("%s/checkpoint_%d.bin", t.Config.SavePath, t.step)
	log.Printf("Saving checkpoint to %s", path)

	f, err := os.Create(path)
	if err != nil {
		log.Printf("Error saving checkpoint: %v", err)
		return
	}
	defer f.Close()

	// Write config
	fmt.Fprintf(f, "GOLLAMA_CKPT\n")
	fmt.Fprintf(f, "step=%d\n", t.step)
	fmt.Fprintf(f, "vocab_size=%d\n", t.Config.VocabSize)
	fmt.Fprintf(f, "hidden_size=%d\n", t.Config.HiddenSize)
	fmt.Fprintf(f, "num_layers=%d\n", t.Config.NumLayers)
	fmt.Fprintf(f, "num_heads=%d\n", t.Config.NumHeads)
	fmt.Fprintf(f, "intermediate_size=%d\n", t.Config.IntermediateSize)
	fmt.Fprintf(f, "WEIGHTS\n")

	// Write weights as binary
	for _, p := range t.Model.Params() {
		for _, v := range p.Data {
			fmt.Fprintf(f, "%v ", v)
		}
		fmt.Fprintf(f, "\n")
	}

	log.Printf("Checkpoint saved: %d parameters", t.countParams())
}

func (t *Trainer) countParams() int {
	total := 0
	for _, p := range t.Model.Params() {
		total += len(p.Data)
	}
	return total
}
