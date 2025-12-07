package train

import (
	"math"

	"github.com/timastras9/gollama/pkg/tensor"
)

// AdamW implements the AdamW optimizer (Adam with decoupled weight decay)
type AdamW struct {
	LR          float32 // Learning rate
	Beta1       float32 // Exponential decay rate for first moment
	Beta2       float32 // Exponential decay rate for second moment
	Epsilon     float32 // Small constant for numerical stability
	WeightDecay float32 // Weight decay coefficient

	// State
	step int
	m    map[*tensor.Tensor][]float32 // First moment estimates
	v    map[*tensor.Tensor][]float32 // Second moment estimates
}

// NewAdamW creates a new AdamW optimizer with default hyperparameters
func NewAdamW(lr float32) *AdamW {
	return &AdamW{
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
		WeightDecay: 0.01,
		m:           make(map[*tensor.Tensor][]float32),
		v:           make(map[*tensor.Tensor][]float32),
	}
}

// Step performs one optimization step
func (opt *AdamW) Step(params []*tensor.GradTensor) {
	opt.step++

	// Bias correction terms
	bc1 := 1.0 - math.Pow(float64(opt.Beta1), float64(opt.step))
	bc2 := 1.0 - math.Pow(float64(opt.Beta2), float64(opt.step))

	for _, p := range params {
		if p.Grad == nil || !p.RequireGrad {
			continue
		}

		// Initialize momentum buffers if needed
		if _, ok := opt.m[p.Tensor]; !ok {
			opt.m[p.Tensor] = make([]float32, len(p.Data))
			opt.v[p.Tensor] = make([]float32, len(p.Data))
		}

		m := opt.m[p.Tensor]
		v := opt.v[p.Tensor]

		for i := range p.Data {
			g := p.Grad.Data[i]

			// Update biased first moment estimate
			m[i] = opt.Beta1*m[i] + (1-opt.Beta1)*g

			// Update biased second moment estimate
			v[i] = opt.Beta2*v[i] + (1-opt.Beta2)*g*g

			// Compute bias-corrected estimates
			mHat := float64(m[i]) / bc1
			vHat := float64(v[i]) / bc2

			// Update parameters with AdamW (decoupled weight decay)
			update := float32(mHat / (math.Sqrt(vHat) + float64(opt.Epsilon)))
			p.Data[i] -= opt.LR * (update + opt.WeightDecay*p.Data[i])
		}
	}
}

// ZeroGrad resets all gradients to zero
func (opt *AdamW) ZeroGrad(params []*tensor.GradTensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// SGD implements stochastic gradient descent with momentum
type SGD struct {
	LR       float32
	Momentum float32

	velocity map[*tensor.Tensor][]float32
}

// NewSGD creates a new SGD optimizer
func NewSGD(lr float32, momentum float32) *SGD {
	return &SGD{
		LR:       lr,
		Momentum: momentum,
		velocity: make(map[*tensor.Tensor][]float32),
	}
}

// Step performs one optimization step
func (opt *SGD) Step(params []*tensor.GradTensor) {
	for _, p := range params {
		if p.Grad == nil || !p.RequireGrad {
			continue
		}

		if opt.Momentum > 0 {
			if _, ok := opt.velocity[p.Tensor]; !ok {
				opt.velocity[p.Tensor] = make([]float32, len(p.Data))
			}
			v := opt.velocity[p.Tensor]

			for i := range p.Data {
				v[i] = opt.Momentum*v[i] + p.Grad.Data[i]
				p.Data[i] -= opt.LR * v[i]
			}
		} else {
			for i := range p.Data {
				p.Data[i] -= opt.LR * p.Grad.Data[i]
			}
		}
	}
}

// ZeroGrad resets all gradients to zero
func (opt *SGD) ZeroGrad(params []*tensor.GradTensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// Optimizer interface for training
type Optimizer interface {
	Step(params []*tensor.GradTensor)
	ZeroGrad(params []*tensor.GradTensor)
}

// LRScheduler adjusts learning rate during training
type LRScheduler struct {
	InitialLR  float32
	WarmupSteps int
	TotalSteps  int
}

// NewCosineScheduler creates a cosine annealing scheduler with warmup
func NewCosineScheduler(initialLR float32, warmupSteps, totalSteps int) *LRScheduler {
	return &LRScheduler{
		InitialLR:   initialLR,
		WarmupSteps: warmupSteps,
		TotalSteps:  totalSteps,
	}
}

// GetLR returns the learning rate for the current step
func (s *LRScheduler) GetLR(step int) float32 {
	if step < s.WarmupSteps {
		// Linear warmup
		return s.InitialLR * float32(step) / float32(s.WarmupSteps)
	}

	// Cosine annealing
	progress := float64(step-s.WarmupSteps) / float64(s.TotalSteps-s.WarmupSteps)
	return s.InitialLR * float32(0.5*(1+math.Cos(math.Pi*progress)))
}

// GradientClipping clips gradients by global norm
func GradientClipping(params []*tensor.GradTensor, maxNorm float32) float32 {
	// Compute global norm
	var totalNorm float64
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data {
			totalNorm += float64(g) * float64(g)
		}
	}
	totalNorm = math.Sqrt(totalNorm)

	// Clip if necessary
	if float32(totalNorm) > maxNorm {
		scale := float32(float64(maxNorm) / totalNorm)
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= scale
			}
		}
	}

	return float32(totalNorm)
}
