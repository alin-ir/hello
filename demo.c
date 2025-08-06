#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* External FFT implementation we want to test */
extern void fft32(float *vec);

/* Naive reference 32-point DFT (O(N^2)) */
static void dft32_ref(const float *in, float *out)
{
    const float TWO_PI_DIV_32 = 2.0f * (float)M_PI / 32.0f;

    for (int k = 0; k < 32; ++k) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        for (int n = 0; n < 32; ++n) {
            float a_r = in[2*n];
            float a_i = in[2*n + 1];
            float angle = -TWO_PI_DIV_32 * n * k; // forward FFT has negative sign
            float c = cosf(angle);
            float s = sinf(angle);
            sum_r += a_r * c - a_i * s;
            sum_i += a_r * s + a_i * c;
        }
        out[2*k]     = sum_r;
        out[2*k + 1] = sum_i;
    }
}

int main(void)
{
    float input[64];       /* original data */
    float test_vec[64];    /* copy that will be transformed by fft32 */
    float ref_out[64];     /* reference DFT result */

    /* Seed PRNG for reproducibility */
    srand(0);

    /* Generate random complex samples in [-1, 1] */
    for (int i = 0; i < 32; ++i) {
        float re = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        float im = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        input[2*i]       = re;
        input[2*i + 1]   = im;
        test_vec[2*i]    = re;
        test_vec[2*i + 1]= im;
    }

    /* Compute reference DFT */
    dft32_ref(input, ref_out);

    /* Run the fast FFT */
    fft32(test_vec);

    /* Measure maximum absolute error */
    float max_err = 0.0f;
    for (int i = 0; i < 64; ++i) {
        float err = fabsf(test_vec[i] - ref_out[i]);
        if (err > max_err)
            max_err = err;
    }

    /* Report results */
    printf("Maximum absolute error: %e\n", max_err);
    if (max_err < 1e-4f) {
        printf("FFT32 implementation PASSED.\n");
        return 0;
    } else {
        printf("FFT32 implementation FAILED.\n");
        return 1;
    }
}