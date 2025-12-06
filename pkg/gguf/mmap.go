//go:build unix

package gguf

import (
	"os"
	"syscall"
)

// MmapFile memory-maps a file for reading
func MmapFile(f *os.File) ([]byte, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := int(stat.Size())
	if size == 0 {
		return nil, nil
	}

	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		size,
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		return nil, err
	}

	return data, nil
}

// MunmapFile unmaps a previously mapped file
func MunmapFile(data []byte) error {
	if data == nil {
		return nil
	}
	return syscall.Munmap(data)
}
