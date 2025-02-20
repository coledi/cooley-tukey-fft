#include "../include/cuda_fft.cuh"
#include "../include/cuda_debug.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>

// Twiddle factor calculation
__device__ cuFloatComplex twiddle(int k, int N) {
    float angle = -2.0f * M_PI * k / N;
    return make_cuFloatComplex(cosf(angle), sinf(angle));
}

// Butterfly operation
__device__ void butterfly(cuFloatComplex& a, cuFloatComplex& b, cuFloatComplex w) {
    cuFloatComplex temp = cuCmulf(w, b);
    b = cuCsubf(a, temp);
    a = cuCaddf(a, temp);
}

__global__ void fftKernel(cuFloatComplex* data, int N, int stage) {
    extern __shared__ cuFloatComplex shared[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    
    // Load data into shared memory
    int globalIdx = bid * blockSize + tid;
    if (globalIdx < N) {
        shared[tid] = data[globalIdx];
    }
    __syncthreads();
    
    // Perform butterfly operations
    int butterflySize = 1 << stage;
    int halfButterfly = butterflySize >> 1;
    
    if (tid < halfButterfly) {
        int pairIndex = tid + halfButterfly;
        cuFloatComplex w = twiddle(tid, butterflySize);
        butterfly(shared[tid], shared[pairIndex], w);
    }
    __syncthreads();
    
    // Write back to global memory
    if (globalIdx < N) {
        data[globalIdx] = shared[tid];
    }
}

CudaFFT::CudaFFT(size_t size) : signal_size_(size) {
    if (!SignalGenerator::isPowerOfTwo(size)) {
        throw std::invalid_argument("Signal size must be a power of 2");
    }
    
    CUDA_CHECK(cudaMalloc(&d_data_, size * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMalloc(&d_temp_, size * sizeof(std::complex<float>)));
    
    // Print initial memory usage
    CudaMemoryChecker::printMemoryUsage();
    
    initializePlans();
}

void CudaFFT::forward(const std::complex<float>* input, std::complex<float>* output) {
    // Print memory usage before operation
    CudaMemoryChecker::printMemoryUsage();
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_data_, input, signal_size_ * sizeof(std::complex<float>), 
                         cudaMemcpyHostToDevice));
    
    // Execute FFT stages
    int numStages = static_cast<int>(log2(signal_size_));
    for (int stage = 0; stage < numStages; ++stage) {
        int threadsPerBlock = std::min(MAX_THREADS_PER_BLOCK, 
                                     static_cast<int>(signal_size_ / (1 << stage)));
        int numBlocks = (signal_size_ + threadsPerBlock - 1) / threadsPerBlock;
        
        printf("Executing FFT stage %d with %d blocks and %d threads per block\n", 
               stage, numBlocks, threadsPerBlock);
        
        fftKernel<<<numBlocks, threadsPerBlock, 
                    threadsPerBlock * sizeof(cuFloatComplex)>>>
                    (reinterpret_cast<cuFloatComplex*>(d_data_), 
                     signal_size_, stage);
        
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_data_, signal_size_ * sizeof(std::complex<float>), 
                         cudaMemcpyDeviceToHost));
    
    // Print memory usage after operation
    CudaMemoryChecker::printMemoryUsage();
}

// Implementation of other methods would follow...

CudaFFT::~CudaFFT() {
    cleanup();
}
