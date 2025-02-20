cmake_minimum_required(VERSION 3.18)
project(cuda_fft CUDA CXX)

# Set CUDA architecture based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust this based on your GPU architecture
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)
find_package(FFTW3 REQUIRED)  # For CPU comparison

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${FFTW3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
add_executable(cuda_fft
    src/main.cu
    src/cuda_fft.cu
    src/signal_generator.cpp
    src/cpu_fft.cpp
)

# Link libraries
target_link_libraries(cuda_fft
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${FFTW3_LIBRARIES}
)

# Enable separate compilation for CUDA
set_target_properties(cuda_fft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
