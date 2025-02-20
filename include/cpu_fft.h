#ifndef CPU_FFT_H
#define CPU_FFT_H

#include <complex>
#include <vector>
#include <fftw3.h>

class CpuFFT {
public:
    CpuFFT(size_t size);
    ~CpuFFT();

    // Forward FFT using FFTW
    void forward(const std::complex<float>* input, std::complex<float>* output);
    
    // Inverse FFT using FFTW
    void inverse(const std::complex<float>* input, std::complex<float>* output);

private:
    size_t signal_size_;
    fftwf_plan plan_forward_;
    fftwf_plan plan_inverse_;
    fftwf_complex* data_in_;
    fftwf_complex* data_out_;

    void initializePlans();
    void cleanup();
};

#endif // CPU_FFT_H
