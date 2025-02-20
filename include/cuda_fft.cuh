#ifndef CUDA_FFT_CUH
#define CUDA_FFT_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>

// Constants for FFT implementation
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int MAX_SHARED_MEMORY = 48000; // 48KB for most modern GPUs

class CudaFFT {
public:
    CudaFFT(size_t size);
    ~CudaFFT();

    // Forward FFT
    void forward(const std::complex<float>* input, std::complex<float>* output);
    
    // Inverse FFT
    void inverse(const std::complex<float>* input, std::complex<float>* output);
    
    // Custom implementation using shared memory
    void forwardCustom(const std::complex<float>* input, std::complex<float>* output);
    void inverseCustom(const std::complex<float>* input, std::complex<float>* output);

private:
    size_t signal_size_;
    cufftHandle plan_forward_;
    cufftHandle plan_inverse_;
    
    // Device memory
    std::complex<float>* d_data_;
    std::complex<float>* d_temp_;

    // Helper functions
    void initializePlans();
    void cleanup();
};

// CUDA kernel declarations
__global__ void fftKernel(cuFloatComplex* data, int N, int stage);
__global__ void bitReversalKernel(cuFloatComplex* data, int N);

#endif // CUDA_FFT_CUH
