#include <gtest/gtest.h>
#include "../include/cuda_fft.cuh"
#include "../include/signal_generator.h"
#include <complex>
#include <vector>
#include <cmath>
#include "../include/cuda_debug.cuh"

class CudaFFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        signal_size = 1024;  // Use small size for testing
        cuda_fft = new CudaFFT(signal_size);
    }

    void TearDown() override {
        delete cuda_fft;
    }

    size_t signal_size;
    CudaFFT* cuda_fft;
};

// Test FFT of a simple sine wave
TEST_F(CudaFFTTest, SineWaveFFT) {
    // Generate a pure sine wave at frequency 440Hz
    auto signal = SignalGenerator::generateSineWave(signal_size, 440.0f, 44100.0f);
    std::vector<std::complex<float>> output(signal_size);
    
    // Perform FFT
    cuda_fft->forward(signal.data(), output.data());
    
    // In frequency domain, we expect a peak at bin corresponding to 440Hz
    int expected_peak_bin = static_cast<int>((440.0f * signal_size) / 44100.0f);
    float max_magnitude = 0;
    int max_magnitude_bin = 0;
    
    for (size_t i = 0; i < signal_size/2; ++i) {
        float magnitude = std::abs(output[i]);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            max_magnitude_bin = i;
        }
    }
    
    EXPECT_NEAR(expected_peak_bin, max_magnitude_bin, 1);
}

// Test FFT followed by IFFT should recover original signal
TEST_F(CudaFFTTest, FFTInverseFFT) {
    auto signal = SignalGenerator::generateSineWave(signal_size, 440.0f, 44100.0f);
    std::vector<std::complex<float>> fft_output(signal_size);
    std::vector<std::complex<float>> ifft_output(signal_size);
    
    // Forward FFT
    cuda_fft->forward(signal.data(), fft_output.data());
    
    // Inverse FFT
    cuda_fft->inverse(fft_output.data(), ifft_output.data());
    
    // Compare original and reconstructed signals
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_NEAR(std::abs(signal[i]), std::abs(ifft_output[i]), 1e-5);
    }
}

// Test FFT of zero signal
TEST_F(CudaFFTTest, ZeroSignal) {
    std::vector<std::complex<float>> signal(signal_size, std::complex<float>(0.0f, 0.0f));
    std::vector<std::complex<float>> output(signal_size);
    
    cuda_fft->forward(signal.data(), output.data());
    
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_NEAR(0.0f, std::abs(output[i]), 1e-5);
    }
}

// Test FFT of impulse signal
TEST_F(CudaFFTTest, ImpulseSignal) {
    std::vector<std::complex<float>> signal(signal_size, std::complex<float>(0.0f, 0.0f));
    signal[0] = std::complex<float>(1.0f, 0.0f);  // Impulse at t=0
    std::vector<std::complex<float>> output(signal_size);
    
    cuda_fft->forward(signal.data(), output.data());
    
    // FFT of impulse should be constant magnitude across all frequencies
    float expected_magnitude = 1.0f / std::sqrt(static_cast<float>(signal_size));
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_NEAR(expected_magnitude, std::abs(output[i]), 1e-5);
    }
}

// Test error handling for invalid signal size
TEST_F(CudaFFTTest, InvalidSize) {
    EXPECT_THROW(CudaFFT(1023), std::invalid_argument);  // Not power of 2
    EXPECT_THROW(CudaFFT(0), std::invalid_argument);     // Zero size
}

// Performance test
TEST_F(CudaFFTTest, Performance) {
    const int num_iterations = 100;
    auto signal = SignalGenerator::generateSineWave(signal_size, 440.0f, 44100.0f);
    std::vector<std::complex<float>> output(signal_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        cuda_fft->forward(signal.data(), output.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float avg_time_ms = duration / (1000.0f * num_iterations);
    std::cout << "Average FFT time: " << avg_time_ms << " ms" << std::endl;
    
    // Ensure performance is within reasonable bounds (adjust threshold as needed)
    EXPECT_LT(avg_time_ms, 10.0);  // Expected to complete within 10ms on modern GPU
}
