#include <gtest/gtest.h>
#include "../include/signal_generator.h"
#include <complex>
#include <vector>
#include <cmath>

class SignalGeneratorTest : public ::testing::Test {
protected:
    const size_t signal_size = 1024;
    const float sample_rate = 44100.0f;
    const float frequency = 440.0f;  // A4 note
};

// Test sine wave generation
TEST_F(SignalGeneratorTest, SineWaveGeneration) {
    auto signal = SignalGenerator::generateSineWave(signal_size, frequency, sample_rate);
    
    // Check signal size
    EXPECT_EQ(signal.size(), signal_size);
    
    // Check signal properties
    float period_samples = sample_rate / frequency;
    float amplitude = 1.0f;
    
    for (size_t i = 0; i < signal_size; ++i) {
        float expected = amplitude * std::sin(2.0f * M_PI * i * frequency / sample_rate);
        EXPECT_NEAR(signal[i].real(), expected, 1e-5);
        EXPECT_NEAR(signal[i].imag(), 0.0f, 1e-5);
    }
}

// Test square wave generation
TEST_F(SignalGeneratorTest, SquareWaveGeneration) {
    auto signal = SignalGenerator::generateSquareWave(signal_size, frequency, sample_rate);
    
    // Check signal size
    EXPECT_EQ(signal.size(), signal_size);
    
    // Check signal properties
    float period_samples = sample_rate / frequency;
    float half_period = period_samples / 2.0f;
    
    for (size_t i = 0; i < signal_size; ++i) {
        float phase = fmod(i, period_samples);
        float expected = (phase < half_period) ? 1.0f : -1.0f;
        EXPECT_NEAR(signal[i].real(), expected, 1e-5);
        EXPECT_NEAR(signal[i].imag(), 0.0f, 1e-5);
    }
}

// Test white noise generation
TEST_F(SignalGeneratorTest, WhiteNoiseGeneration) {
    auto signal = SignalGenerator::generateWhiteNoise(signal_size);
    
    // Check signal size
    EXPECT_EQ(signal.size(), signal_size);
    
    // Calculate mean and variance
    float sum = 0.0f;
    float sum_squared = 0.0f;
    
    for (const auto& sample : signal) {
        sum += sample.real();
        sum_squared += sample.real() * sample.real();
    }
    
    float mean = sum / signal_size;
    float variance = (sum_squared / signal_size) - (mean * mean);
    
    // White noise should have approximately zero mean and unit variance
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(variance, 1.0f, 0.1f);
}

// Test power of two checker
TEST_F(SignalGeneratorTest, PowerOfTwo) {
    EXPECT_TRUE(SignalGenerator::isPowerOfTwo(1));
    EXPECT_TRUE(SignalGenerator::isPowerOfTwo(2));
    EXPECT_TRUE(SignalGenerator::isPowerOfTwo(1024));
    EXPECT_TRUE(SignalGenerator::isPowerOfTwo(4096));
    
    EXPECT_FALSE(SignalGenerator::isPowerOfTwo(0));
    EXPECT_FALSE(SignalGenerator::isPowerOfTwo(3));
    EXPECT_FALSE(SignalGenerator::isPowerOfTwo(1023));
    EXPECT_FALSE(SignalGenerator::isPowerOfTwo(1025));
}

// Test next power of two calculator
TEST_F(SignalGeneratorTest, NextPowerOfTwo) {
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(0), 1);
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(1), 1);
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(2), 2);
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(3), 4);
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(1023), 1024);
    EXPECT_EQ(SignalGenerator::nextPowerOfTwo(1025), 2048);
}

// Test SNR calculation
TEST_F(SignalGeneratorTest, SNRCalculation) {
    // Create two identical signals
    auto signal1 = SignalGenerator::generateSineWave(signal_size, frequency, sample_rate);
    auto signal2 = signal1;  // Perfect copy
    
    // Calculate SNR for identical signals (should be very high)
    float perfect_snr = SignalGenerator::calculateSNR(signal1, signal2);
    EXPECT_GT(perfect_snr, 100.0f);  // Should be effectively infinite
    
    // Add some noise to signal2
    for (auto& sample : signal2) {
        sample += std::complex<float>(0.01f * (rand() / float(RAND_MAX)), 0.0f);
    }
    
    // Calculate SNR for slightly different signals
    float noisy_snr = SignalGenerator::calculateSNR(signal1, signal2);
    EXPECT_LT(noisy_snr, perfect_snr);
    EXPECT_GT(noisy_snr, 20.0f);  // Should still be reasonable
}
