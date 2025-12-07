package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/timastras9/gollama/pkg/api"
	"github.com/timastras9/gollama/pkg/gguf"
	"github.com/timastras9/gollama/pkg/metal"
	"github.com/timastras9/gollama/pkg/model"
	"github.com/timastras9/gollama/pkg/tokenizer"
	"github.com/timastras9/gollama/pkg/tools"
)

func main() {
	// Check for global --no-gpu flag before subcommands
	noGPU := false
	for i, arg := range os.Args {
		if arg == "--no-gpu" || arg == "-no-gpu" {
			noGPU = true
			// Remove from args so subcommands don't see it
			os.Args = append(os.Args[:i], os.Args[i+1:]...)
			break
		}
	}

	// Initialize Metal GPU acceleration (unless disabled)
	if noGPU {
		log.Printf("GPU disabled via --no-gpu flag (using CPU only)")
	} else if err := metal.Init(); err != nil {
		log.Printf("Metal GPU not available: %v (using CPU)", err)
	} else {
		log.Printf("Metal GPU enabled: %s", metal.DeviceName())
		defer metal.Cleanup()
	}

	// Define subcommands
	serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
	runCmd := flag.NewFlagSet("run", flag.ExitOnError)
	chatCmd := flag.NewFlagSet("chat", flag.ExitOnError)
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

	// Chat flags
	chatModel := chatCmd.String("model", "", "Path to GGUF model file")
	chatMaxTokens := chatCmd.Int("max-tokens", 256, "Maximum tokens to generate per response")
	chatTemp := chatCmd.Float64("temp", 0.7, "Sampling temperature")
	chatTopK := chatCmd.Int("top-k", 40, "Top-K sampling")
	chatTopP := chatCmd.Float64("top-p", 0.9, "Top-P (nucleus) sampling")

	// Pentest flags
	pentestCmd := flag.NewFlagSet("pentest", flag.ExitOnError)
	pentestModel := pentestCmd.String("model", "", "Path to GGUF model file")
	pentestMaxTokens := pentestCmd.Int("max-tokens", 512, "Maximum tokens to generate per response")
	pentestTemp := pentestCmd.Float64("temp", 0.3, "Sampling temperature (lower = more focused)")
	pentestTopK := pentestCmd.Int("top-k", 40, "Top-K sampling")
	pentestTopP := pentestCmd.Float64("top-p", 0.9, "Top-P (nucleus) sampling")

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

	case "chat":
		chatCmd.Parse(os.Args[2:])
		if *chatModel == "" {
			fmt.Println("Error: --model is required")
			chatCmd.Usage()
			os.Exit(1)
		}
		runChat(*chatModel, *chatMaxTokens, float32(*chatTemp), *chatTopK, float32(*chatTopP))

	case "pentest":
		pentestCmd.Parse(os.Args[2:])
		if *pentestModel == "" {
			fmt.Println("Error: --model is required")
			pentestCmd.Usage()
			os.Exit(1)
		}
		runPentest(*pentestModel, *pentestMaxTokens, float32(*pentestTemp), *pentestTopK, float32(*pentestTopP))

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
  gollama [--no-gpu] <command> [options]

Global Options:
  --no-gpu    Disable Metal GPU acceleration (CPU only)

Commands:
  serve     Start the Ollama-compatible API server
  run       Run inference on a GGUF model
  chat      Interactive chat session with a model
  pentest   Penetration testing assistant with tool calling
  info      Show information about a GGUF model
  version   Show version information
  help      Show this help message

Examples:
  gollama serve --addr 127.0.0.1:11434 --models ./models
  gollama run --model model.gguf --prompt "Hello, world!"
  gollama chat --model model.gguf
  gollama --no-gpu chat --model model.gguf   # CPU only
  gollama pentest --model model.gguf
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

func runChat(modelPath string, maxTokens int, temp float32, topK int, topP float32) {
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

	// Load tokenizer
	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Create sampler
	sampler := model.NewSampler(model.SamplerConfig{
		Temperature: temp,
		TopK:        topK,
		TopP:        topP,
		Seed:        -1,
	})

	fmt.Println("\n╭─────────────────────────────────────────╮")
	fmt.Println("│  Gollama Chat - Type 'exit' to quit     │")
	fmt.Println("╰─────────────────────────────────────────╯")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\nYou: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if strings.ToLower(input) == "exit" || strings.ToLower(input) == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		// Reset cache for each new conversation turn
		mdl.ResetCache()

		// Format with chat template (TinyLlama/Llama style)
		prompt := fmt.Sprintf("<|user|>\n%s</s>\n<|assistant|>\n", input)

		// Tokenize input
		tokens := tok.Encode(prompt)

		// Generate response
		fmt.Print("\nAssistant: ")

		// Prefill
		logits := mdl.Forward(tokens, 0)
		lastLogits := logits.Slice(len(tokens)-1, len(tokens))
		lastLogits = lastLogits.Reshape(mdl.Config.VocabSize)

		pos := len(tokens)
		var printed int // how much we've already printed
		var response strings.Builder
		stopSequences := []string{
			"<|user|>", "<|assistant|>", "</s>", "<|im_end|>",
			"\nUser:", "\nuser:", "\nassistant:", "\nAssistant:",
			"\n\nUser", "\n\nuser", "\n\nassistant", "\n\nAssistant",
		}

		for i := 0; i < maxTokens; i++ {
			nextToken := sampler.Sample(lastLogits)

			if tok.IsEOS(nextToken) {
				break
			}

			text := tok.DecodeToken(nextToken)
			response.WriteString(text)
			resp := response.String()

			// Check for stop sequences
			shouldStop := false
			stopIdx := len(resp)
			for _, stop := range stopSequences {
				if idx := strings.Index(resp, stop); idx != -1 {
					if idx < stopIdx {
						stopIdx = idx
					}
					shouldStop = true
				}
			}

			if shouldStop {
				// Print any remaining content up to the stop sequence
				if stopIdx > printed {
					fmt.Print(resp[printed:stopIdx])
				}
				break
			}

			// Stream output - print new content
			if len(resp) > printed {
				fmt.Print(resp[printed:])
				printed = len(resp)
			}

			logits = mdl.Forward([]int{nextToken}, pos)
			lastLogits = logits.Reshape(mdl.Config.VocabSize)
			pos++
		}
		fmt.Println()
	}
}

func runPentest(modelPath string, maxTokens int, temp float32, topK int, topP float32) {
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

	// Load tokenizer
	tok, err := tokenizer.FromGGUF(r)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Create tool registry
	registry := tools.NewRegistry()

	// Create sampler
	sampler := model.NewSampler(model.SamplerConfig{
		Temperature: temp,
		TopK:        topK,
		TopP:        topP,
		Seed:        -1,
	})

	fmt.Println("\n╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Gollama Pentest Assistant                                ║")
	fmt.Println("║  AI-powered penetration testing with native Go tools      ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════╣")
	fmt.Println("║  Commands:                                                ║")
	fmt.Println("║    /tools    - List available tools                       ║")
	fmt.Println("║    /help     - Show help                                  ║")
	fmt.Println("║    exit      - Quit                                       ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")

	scanner := bufio.NewScanner(os.Stdin)

	// Conversation history for context
	var conversationHistory strings.Builder

	for {
		fmt.Print("\n[pentest]> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Handle special commands
		if strings.ToLower(input) == "exit" || strings.ToLower(input) == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "/tools" {
			fmt.Println("\nAvailable tools:")
			for _, tool := range registry.List() {
				fmt.Printf("  • %s: %s\n", tool.Name, tool.Description)
			}
			continue
		}

		if input == "/help" {
			fmt.Println("\nPentest mode uses AI to assist with security testing.")
			fmt.Println("The AI can use these tools:")
			for _, tool := range registry.List() {
				fmt.Printf("\n  %s\n", tool.Name)
				fmt.Printf("    %s\n", tool.Description)
				fmt.Println("    Parameters:")
				for _, p := range tool.Parameters {
					req := ""
					if p.Required {
						req = " [required]"
					}
					fmt.Printf("      - %s%s: %s\n", p.Name, req, p.Description)
				}
			}
			continue
		}

		// Build prompt with system context and conversation history
		// Using ChatML format for Hermes models
		systemPrompt := registry.SystemPrompt()
		fullPrompt := fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
			systemPrompt,
			conversationHistory.String(),
			input,
		)

		// Add user message to history
		conversationHistory.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n", input))

		// Reset cache and tokenize
		mdl.ResetCache()
		tokens := tok.Encode(fullPrompt)

		// Generate response
		fmt.Print("\n")

		logits := mdl.Forward(tokens, 0)
		lastLogits := logits.Slice(len(tokens)-1, len(tokens))
		lastLogits = lastLogits.Reshape(mdl.Config.VocabSize)

		pos := len(tokens)
		var printed int
		var response strings.Builder
		stopSequences := []string{
			"<|im_end|>", "<|im_start|>", "</s>", "<|user|>", "<|assistant|>", "<|system|>",
		}

		for i := 0; i < maxTokens; i++ {
			nextToken := sampler.Sample(lastLogits)

			if tok.IsEOS(nextToken) {
				break
			}

			text := tok.DecodeToken(nextToken)
			response.WriteString(text)
			resp := response.String()

			// Check for stop sequences
			shouldStop := false
			stopIdx := len(resp)
			for _, stop := range stopSequences {
				if idx := strings.Index(resp, stop); idx != -1 {
					if idx < stopIdx {
						stopIdx = idx
					}
					shouldStop = true
				}
			}

			if shouldStop {
				if stopIdx > printed {
					fmt.Print(resp[printed:stopIdx])
				}
				break
			}

			// Stream output
			if len(resp) > printed {
				fmt.Print(resp[printed:])
				printed = len(resp)
			}

			logits = mdl.Forward([]int{nextToken}, pos)
			lastLogits = logits.Reshape(mdl.Config.VocabSize)
			pos++
		}

		// Get final response
		finalResponse := response.String()
		for _, stop := range stopSequences {
			if idx := strings.Index(finalResponse, stop); idx != -1 {
				finalResponse = finalResponse[:idx]
			}
		}

		// Add assistant response to history
		conversationHistory.WriteString(fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", finalResponse))

		// Check for tool calls in the response
		if call, beforeCall, found := tools.ParseToolCall(finalResponse); found {
			if beforeCall != "" && printed < len(beforeCall) {
				// Already printed above
			}

			fmt.Printf("\n\n[Executing tool: %s]\n", call.Name)
			fmt.Printf("Arguments: %v\n", call.Args)

			result, err := registry.Execute(call)
			if err != nil {
				fmt.Printf("[Tool error: %v]\n", err)
				result = fmt.Sprintf("Error: %v", err)
			} else {
				fmt.Printf("\n%s\n", result)
			}

			// Add tool result to conversation and let AI analyze
			conversationHistory.WriteString(fmt.Sprintf("<|im_start|>tool\n%s<|im_end|>\n", result))

			// Generate follow-up analysis
			fmt.Print("\n[Analysis]: ")
			analysisPrompt := fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n%s<|im_start|>assistant\n",
				systemPrompt,
				conversationHistory.String(),
			)

			mdl.ResetCache()
			analysisTokens := tok.Encode(analysisPrompt)
			logits = mdl.Forward(analysisTokens, 0)
			lastLogits = logits.Slice(len(analysisTokens)-1, len(analysisTokens))
			lastLogits = lastLogits.Reshape(mdl.Config.VocabSize)

			pos = len(analysisTokens)
			printed = 0
			response.Reset()

			for i := 0; i < maxTokens/2; i++ {
				nextToken := sampler.Sample(lastLogits)

				if tok.IsEOS(nextToken) {
					break
				}

				text := tok.DecodeToken(nextToken)
				response.WriteString(text)
				resp := response.String()

				// Check for stop sequences or new tool calls
				shouldStop := false
				stopIdx := len(resp)
				for _, stop := range stopSequences {
					if idx := strings.Index(resp, stop); idx != -1 {
						if idx < stopIdx {
							stopIdx = idx
						}
						shouldStop = true
					}
				}
				// Also stop if trying to make another tool call (keep it simple)
				if strings.Contains(resp, "<tool_call>") {
					if idx := strings.Index(resp, "<tool_call>"); idx < stopIdx {
						stopIdx = idx
					}
					shouldStop = true
				}

				if shouldStop {
					if stopIdx > printed {
						fmt.Print(resp[printed:stopIdx])
					}
					break
				}

				if len(resp) > printed {
					fmt.Print(resp[printed:])
					printed = len(resp)
				}

				logits = mdl.Forward([]int{nextToken}, pos)
				lastLogits = logits.Reshape(mdl.Config.VocabSize)
				pos++
			}
		}

		fmt.Println()
	}
}
