#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build the project
make -j$(nproc)

# Run the tests with CUDA memory checking
CUDA_VISIBLE_DEVICES=0 cuda-memcheck ./cuda_fft_test

# Print test results summary
echo "Test execution completed."
