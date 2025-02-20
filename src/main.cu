#include <iostream>
#include <chrono>
#include <vector>
#include "../include/cuda_fft.cuh"
#include "../include/cpu_fft.h"
#include "../include/signal_generator.h"

void runBenchmark(size_t signal_size, int num_iterations) {
    // Create test signal
    auto signal = SignalGenerator::generateSineWave(signal_size, 440.0f, 44100.0f);
    
    // Initialize FFT objects
    CudaFFT cuda_fft(signal_size);
    CpuFFT cpu_fft(signal_size);
    
    std::vector<std::complex<float>> output_gpu(signal_size);
    std::vector<std::complex<float>> output_cpu(signal_size);
    
    // GPU Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        cuda_fft.forward(signal.data(), output_gpu.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CPU Benchmark
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        cpu_fft.forward(signal.data(), output_cpu.data());
    }
    end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Calculate SNR
    float snr = SignalGenerator::calculateSNR(output_cpu, output_gpu);
    
    // Print results
    std::cout << "Benchmark Results:\n";
    std::cout << "Signal size: " << signal_size << "\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    std::cout << "GPU time: " << gpu_time / 1000.0 << " ms\n";
    std::cout << "CPU time: " << cpu_time / 1000.0 << " ms\n";
    std::cout << "Speedup: " << static_cast<float>(cpu_time) / gpu_time << "x\n";
    std::cout << "SNR: " << snr << " dB\n";
}

int main(int argc, char** argv) {
    size_t signal_size = 1 << 20;  // Default: 1M points
    int num_iterations = 100;      // Default: 100 iterations
    
    if (argc > 1) signal_size = std::stoull(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    
    if (!SignalGenerator::isPowerOfTwo(signal_size)) {
        signal_size = SignalGenerator::nextPowerOfTwo(signal_size);
        std::cout << "Adjusting signal size to next power of 2: " << signal_size << "\n";
    }
    
    try {
        runBenchmark(signal_size, num_iterations);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
