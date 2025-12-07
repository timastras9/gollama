package main

import (
	"fmt"
	"os"

	"github.com/timastras9/gollama/pkg/tools"
)

func main() {
	registry := tools.NewRegistry()

	if len(os.Args) < 2 {
		fmt.Println("Usage: tooltest <tool> [args...]")
		fmt.Println("\nAvailable tools:")
		for _, t := range registry.List() {
			fmt.Printf("  %s - %s\n", t.Name, t.Description)
		}
		return
	}

	toolName := os.Args[1]
	args := make(map[string]string)

	// Parse args as key=value pairs
	for _, arg := range os.Args[2:] {
		for i := 0; i < len(arg); i++ {
			if arg[i] == '=' {
				args[arg[:i]] = arg[i+1:]
				break
			}
		}
	}

	result, err := registry.Execute(&tools.ToolCall{
		Name: toolName,
		Args: args,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(result)
}
