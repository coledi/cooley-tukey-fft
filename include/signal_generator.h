#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <complex>
#include <vector>

class SignalGenerator {
public:
    // Generate test signals
    static std::vector<std::complex<float>> generateSineWave(size_t size, float frequency, float sampleRate);
    static std::vector<std::complex<float>> generateSquareWave(size_t size, float frequency, float sampleRate);
    static std::vector<std::complex<float>> generateWhiteNoise(size_t size);
    
    // Utility functions
    static bool isPowerOfTwo(size_t n);
    static size_t nextPowerOfTwo(size_t n);
    
    // Signal analysis
    static float calculateSNR(const std::vector<std::complex<float>>& signal,
                            const std::vector<std::complex<float>>& processed);
};

#endif // SIGNAL_GENERATOR_H
