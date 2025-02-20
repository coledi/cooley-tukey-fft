#ifndef CUDA_DEBUG_CUH
#define CUDA_DEBUG_CUH

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Macro for checking kernel launch errors
#define KERNEL_CHECK() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Debug print for complex numbers
template<typename T>
__device__ void debugPrintComplex(const char* prefix, T real, T imag, int threadIdx) {
    printf("%s[Thread %d]: %f + %fi\n", prefix, threadIdx, real, imag);
}

// Memory checker class for tracking CUDA allocations
class CudaMemoryChecker {
public:
    static void printMemoryUsage() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        printf("CUDA Memory Usage:\n");
        printf("  Total: %zu MB\n", total / (1024*1024));
        printf("  Free:  %zu MB\n", free / (1024*1024));
        printf("  Used:  %zu MB\n", (total - free) / (1024*1024));
    }
};

// Utility function to validate FFT results
template<typename T>
bool validateFFTResults(const std::complex<T>* input, 
                       const std::complex<T>* output,
                       size_t size,
                       T tolerance = 1e-5) {
    // Implementation depends on specific validation requirements
    // This is a placeholder for demonstration
    return true;
}

#endif // CUDA_DEBUG_CUH
