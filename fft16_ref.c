#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Reference 16-point FFT (forward, complex-to-complex).
 * Input/Output layout: [Re0, Im0, Re1, Im1, ..., Re15, Im15]
 * Simplicity over speed – O(N^2) DFT, sufficient for correctness checks.
 */
void fft16(float *vec)
{
    float out[32];
    const float TWO_PI_DIV_16 = 2.0f * (float)M_PI / 16.0f;

    for (int k = 0; k < 16; ++k) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        for (int n = 0; n < 16; ++n) {
            float a_r = vec[2*n];
            float a_i = vec[2*n + 1];
            float angle = -TWO_PI_DIV_16 * n * k; // forward FFT uses negative sign
            float c = cosf(angle);
            float s = sinf(angle);
            sum_r += a_r * c - a_i * s;
            sum_i += a_r * s + a_i * c;
        }
        out[2*k]     = sum_r;
        out[2*k + 1] = sum_i;
    }

    /* Copy back to the original array */
    for (int i = 0; i < 32; ++i) {
        vec[i] = out[i];
    }
}