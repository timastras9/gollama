// +build darwin

package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Global Metal device and command queue
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> matmulPipeline = nil;

int metal_init() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return -1;
    }
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        return -2;
    }
    return 0;
}

void metal_cleanup() {
    commandQueue = nil;
    library = nil;
    matmulPipeline = nil;
    device = nil;
}

const char* metal_device_name() {
    if (!device) return "No device";
    return [[device name] UTF8String];
}

// Matrix multiplication using MPS
// A: [M, K], B: [K, N], C: [M, N]
// C = A @ B
int metal_matmul(float* a, float* b, float* c, int M, int K, int N) {
    if (!device || !commandQueue) {
        return -1;
    }

    @autoreleasepool {
        // Create buffers
        id<MTLBuffer> bufferA = [device newBufferWithBytes:a
                                                   length:M * K * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:b
                                                   length:K * N * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        // Create matrix descriptors
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:K * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        // Create matrix multiplication kernel
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:NO
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0
            beta:0.0];

        // Create command buffer and encode
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(c, [bufferC contents], M * N * sizeof(float));
    }

    return 0;
}

// Matrix multiplication with B transposed: C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
int metal_matmul_transposed(float* a, float* b, float* c, int M, int K, int N) {
    if (!device || !commandQueue) {
        return -1;
    }

    @autoreleasepool {
        // Create buffers
        id<MTLBuffer> bufferA = [device newBufferWithBytes:a
                                                   length:M * K * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:b
                                                   length:N * K * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        // B is [N, K] and we want B^T which is [K, N]
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:K * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                           columns:K
                                                                          rowBytes:K * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        // Create matrix multiplication kernel with B transposed
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:YES
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0
            beta:0.0];

        // Create command buffer and encode
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(c, [bufferC contents], M * N * sizeof(float));
    }

    return 0;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var enabled = false

// Init initializes the Metal device
func Init() error {
	result := C.metal_init()
	if result != 0 {
		return fmt.Errorf("failed to initialize Metal: %d", result)
	}
	enabled = true
	return nil
}

// Cleanup releases Metal resources
func Cleanup() {
	C.metal_cleanup()
	enabled = false
}

// IsEnabled returns true if Metal is available and initialized
func IsEnabled() bool {
	return enabled
}

// DeviceName returns the name of the Metal device
func DeviceName() string {
	return C.GoString(C.metal_device_name())
}

// MatMul performs matrix multiplication: C = A @ B
// A is [M, K], B is [K, N], C is [M, N]
func MatMul(a, b, c []float32, M, K, N int) error {
	if !enabled {
		return fmt.Errorf("Metal not initialized")
	}

	result := C.metal_matmul(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&c[0])),
		C.int(M), C.int(K), C.int(N),
	)

	if result != 0 {
		return fmt.Errorf("Metal matmul failed: %d", result)
	}
	return nil
}

// MatMulTransposed performs matrix multiplication with B transposed: C = A @ B^T
// A is [M, K], B is [N, K], C is [M, N]
func MatMulTransposed(a, b, c []float32, M, K, N int) error {
	if !enabled {
		return fmt.Errorf("Metal not initialized")
	}

	result := C.metal_matmul_transposed(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&c[0])),
		C.int(M), C.int(K), C.int(N),
	)

	if result != 0 {
		return fmt.Errorf("Metal matmul transposed failed: %d", result)
	}
	return nil
}
