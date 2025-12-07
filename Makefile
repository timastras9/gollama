# Gollama - Pure Go LLM Inference Engine

BINARY := gollama
TOOLTEST := tooltest
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -ldflags "-s -w -X github.com/timastras9/gollama/pkg/api.Version=$(VERSION)"

.PHONY: all build clean test bench install run chat

all: build

# Build the main binary
build:
	@echo "Building $(BINARY)..."
	go build $(LDFLAGS) -o $(BINARY) ./cmd/gollama

# Build with race detector (for debugging)
build-race:
	go build -race $(LDFLAGS) -o $(BINARY) ./cmd/gollama

# Build tooltest utility
tooltest:
	go build -o $(TOOLTEST) ./cmd/tooltest

# Build all binaries
build-all: build tooltest

# Clean build artifacts
clean:
	rm -f $(BINARY) $(TOOLTEST)
	go clean

# Run tests
test:
	go test -v ./...

# Run benchmarks
bench:
	go test -bench=. -benchmem ./pkg/tensor/

# Install to $GOPATH/bin
install:
	go install $(LDFLAGS) ./cmd/gollama

# Quick run with TinyLlama
run:
	@if [ ! -f models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf ]; then \
		echo "Model not found. Download a GGUF model to ./models/"; \
		exit 1; \
	fi
	./$(BINARY) run --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --prompt "Hello!"

# Interactive chat with TinyLlama
chat:
	@if [ ! -f models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf ]; then \
		echo "Model not found. Download a GGUF model to ./models/"; \
		exit 1; \
	fi
	./$(BINARY) chat --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Chat without GPU
chat-cpu:
	./$(BINARY) --no-gpu chat --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Start API server
serve:
	./$(BINARY) serve --addr 127.0.0.1:11434 --models ./models

# Show help
help:
	@echo "Gollama Makefile targets:"
	@echo "  make build      - Build the gollama binary"
	@echo "  make build-all  - Build gollama and tooltest"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make test       - Run tests"
	@echo "  make bench      - Run benchmarks"
	@echo "  make install    - Install to GOPATH/bin"
	@echo "  make run        - Quick inference test"
	@echo "  make chat       - Interactive chat (GPU)"
	@echo "  make chat-cpu   - Interactive chat (CPU only)"
	@echo "  make serve      - Start API server"
