//go:build windows

package gguf

import (
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	createFileMappingW = kernel32.NewProc("CreateFileMappingW")
	mapViewOfFile      = kernel32.NewProc("MapViewOfFile")
	unmapViewOfFile    = kernel32.NewProc("UnmapViewOfFile")
	closeHandle        = kernel32.NewProc("CloseHandle")
)

const (
	PAGE_READONLY        = 0x02
	FILE_MAP_READ        = 0x04
)

// MmapFile memory-maps a file for reading on Windows
func MmapFile(f *os.File) ([]byte, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := stat.Size()
	if size == 0 {
		return nil, nil
	}

	// Create file mapping
	sizeLow := uint32(size)
	sizeHigh := uint32(size >> 32)

	handle, _, err := createFileMappingW.Call(
		f.Fd(),
		0,
		PAGE_READONLY,
		uintptr(sizeHigh),
		uintptr(sizeLow),
		0,
	)
	if handle == 0 {
		return nil, err
	}

	// Map view of file
	addr, _, err := mapViewOfFile.Call(
		handle,
		FILE_MAP_READ,
		0,
		0,
		uintptr(size),
	)
	if addr == 0 {
		closeHandle.Call(handle)
		return nil, err
	}

	// Create slice from mapped memory
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(size))

	return data, nil
}

// MunmapFile unmaps a previously mapped file on Windows
func MunmapFile(data []byte) error {
	if data == nil {
		return nil
	}

	ret, _, err := unmapViewOfFile.Call(uintptr(unsafe.Pointer(&data[0])))
	if ret == 0 {
		return err
	}
	return nil
}
