// +build !darwin

package metal

import "fmt"

// Init initializes Metal (stub for non-macOS)
func Init() error {
	return fmt.Errorf("Metal is only available on macOS")
}

// Cleanup releases Metal resources (stub)
func Cleanup() {}

// IsEnabled returns false on non-macOS
func IsEnabled() bool {
	return false
}

// DeviceName returns empty on non-macOS
func DeviceName() string {
	return "Not available"
}

// MatMul stub
func MatMul(a, b, c []float32, M, K, N int) error {
	return fmt.Errorf("Metal is only available on macOS")
}

// MatMulTransposed stub
func MatMulTransposed(a, b, c []float32, M, K, N int) error {
	return fmt.Errorf("Metal is only available on macOS")
}
