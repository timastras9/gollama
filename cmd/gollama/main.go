package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/timastras9/gollama/pkg/api"
	"github.com/timastras9/gollama/pkg/gguf"
	"github.com/timastras9/gollama/pkg/model"
	"github.com/timastras9/gollama/pkg/tokenizer"
)

func main() {
	// Define subcommands
	serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
	runCmd := flag.NewFlagSet("run", flag.ExitOnError)
	infoCmd := flag.NewFlagSet("info", flag.ExitOnError)

	// Serve flags
	serveAddr := serveCmd.String("addr", "127.0.0.1:11434", "Address to listen on")
	serveModels := serveCmd.String("models", "./models", "Models directory")

	// Run flags
	runModel := runCmd.String("model", "", "Path to GGUF model file")
	runPrompt := runCmd.String("prompt", "", "Prompt to generate from")
	runMaxTokens := runCmd.Int("max-tokens", 100, "Maximum tokens to generate")
	runTemp := runCmd.Float64("temp", 0.8, "Sampling temperature")
	runTopK := runCmd.Int("top-k", 40, "Top-K sampling")
	runTopP := runCmd.Float64("top-p", 0.9, "Top-P (nucleus) sampling")

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd.Parse(os.Args[2:])
		runServe(*serveAddr, *serveModels)

	case "run":
		runCmd.Parse(os.Args[2:])
		if *runModel == "" {
			fmt.Println("Error: --model is required")
			runCmd.Usage()
			os.Exit(1)
		}
		runGenerate(*runModel, *runPrompt, *runMaxTokens, float32(*runTemp), *runTopK, float32(*runTopP))

	case "info":
		infoCmd.Parse(os.Args[2:])
		if len(infoCmd.Args()) < 1 {
			fmt.Println("Error: model path required")
			os.Exit(1)
		}
		showInfo(infoCmd.Args()[0])

	case "version":
		fmt.Printf("gollama version %s\n", api.Version)

	case "help":
		printUsage()

	default:
		// Check if it's a model path
		if _, err := os.Stat(os.Args[1]); err == nil {
			runCmd.Parse(os.Args[2:])
			runGenerate(os.Args[1], *runPrompt, *runMaxTokens, float32(*runTemp), *runTopK, float32(*runTopP))
		} else {
			fmt.Printf("Unknown command: %s\n", os.Args[1])
			printUsage()
			os.Exit(1)
		}
	}
}

func printUsage() {
	fmt.Println(`Gollama - Pure Go LLM Inference Engine

Usage:
  gollama <command> [options]

Commands:
  serve     Start the Ollama-compatible API server
  run       Run inference on a GGUF model
  info      Show information about a GGUF model
  version   Show version information
  help      Show this help message

Examples:
  gollama serve --addr 127.0.0.1:11434 --models ./models
  gollama run --model model.gguf --prompt "Hello, world!"
  gollama info model.gguf

For command-specific help:
  gollama <command> --help`)
}

func runServe(addr, modelsDir string) {
	// Create models directory if it doesn't exist
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}

	absModels, _ := filepath.Abs(modelsDir)
	log.Printf("Models directory: %s", absModels)

	server := api.NewServer(addr, modelsDir)

	// Handle shutdown gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5)
		defer cancel()
		server.Stop(ctx)
	}()

	if err := server.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

func runGenerate(modelPath, prompt string, maxTokens int, temp float32, topK int, topP float32) {
	log.Printf("Loading model from %s...", modelPath)

	// Open GGUF file
	r, err := gguf.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer r.Close()

	// Load model
	mdl, err := model.LoadFromGGUF(r)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	log.Printf("Model loaded: %s", mdl.Config.Architecture)
	log.Printf("  Layers: %d, Hidden: %d, Heads: %d",
		mdl.Config.NumLayers, mdl.Config.HiddenSize, mdl.Config.NumHeads)

	// Load tokenizer
	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	log.Printf("Tokenizer loaded: %d tokens", tok.VocabSize())

	// Get prompt
	if prompt == "" {
		fmt.Print("Enter prompt: ")
		fmt.Scanln(&prompt)
	}

	// Tokenize
	tokens := tok.Encode(prompt)
	log.Printf("Prompt tokens: %d", len(tokens))

	// Create sampler
	sampler := model.NewSampler(model.SamplerConfig{
		Temperature: temp,
		TopK:        topK,
		TopP:        topP,
		Seed:        -1,
	})

	// Generate
	fmt.Println("\n--- Generation ---")
	fmt.Print(prompt)

	mdl.ResetCache()

	// Prefill
	logits := mdl.Forward(tokens, 0)
	lastLogits := logits.Slice(len(tokens)-1, len(tokens))
	lastLogits = lastLogits.Reshape(mdl.Config.VocabSize)

	pos := len(tokens)
	for i := 0; i < maxTokens; i++ {
		nextToken := sampler.Sample(lastLogits)

		if tok.IsEOS(nextToken) {
			break
		}

		text := tok.DecodeToken(nextToken)
		fmt.Print(text)

		logits = mdl.Forward([]int{nextToken}, pos)
		lastLogits = logits.Reshape(mdl.Config.VocabSize)
		pos++
	}

	fmt.Println("\n--- End ---")
}

func showInfo(modelPath string) {
	r, err := gguf.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer r.Close()

	header := r.Header()
	fmt.Printf("GGUF Version: %d\n", header.Version)
	fmt.Printf("Tensor Count: %d\n", header.TensorCount)
	fmt.Printf("Metadata Count: %d\n", header.MetadataKVCount)

	fmt.Println("\nMetadata:")
	for key, value := range r.Metadata() {
		// Skip large arrays
		switch v := value.(type) {
		case []any:
			fmt.Printf("  %s: [array of %d elements]\n", key, len(v))
		case string:
			if len(v) > 100 {
				fmt.Printf("  %s: %s... (truncated)\n", key, v[:100])
			} else {
				fmt.Printf("  %s: %s\n", key, v)
			}
		default:
			fmt.Printf("  %s: %v\n", key, value)
		}
	}

	fmt.Println("\nTensors:")
	for _, t := range r.Tensors() {
		fmt.Printf("  %s: shape=%v type=%s\n", t.Name, t.Shape(), t.Type)
	}

	// Try to load config
	cfg, err := model.ConfigFromGGUF(r)
	if err == nil {
		fmt.Println("\nModel Config:")
		fmt.Print(cfg.String())
	}
}
