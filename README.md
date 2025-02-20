# CUDA-Accelerated Fast Fourier Transform (FFT)

A high-performance CUDA implementation of the Fast Fourier Transform algorithm for signal processing.

## Features

- CUDA-accelerated FFT implementation using the Cooley-Tukey algorithm
- CPU (FFTW) implementation for performance comparison
- Signal generation utilities for testing
- Support for both forward FFT and inverse FFT (IFFT)
- Optimized memory access patterns and shared memory usage
- Performance benchmarking tools

## Requirements

- CUDA Toolkit (11.0 or higher)
- CMake (3.18 or higher)
- C++14 compatible compiler
- FFTW3 library (for CPU comparison)

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

```bash
./cuda_fft [signal_size] [num_iterations]
```

- `signal_size`: Size of the input signal (must be a power of 2)
- `num_iterations`: Number of iterations for performance benchmarking

## Implementation Details

The project implements the following optimizations:
- Shared memory usage to reduce global memory access
- Memory coalescing for efficient data transfer
- Bank conflict reduction in shared memory access
- Warp-level optimizations to minimize thread divergence

## Performance

Performance comparisons between CPU and GPU implementations are available in the benchmarking results.
