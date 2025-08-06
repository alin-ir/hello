#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846f

/*
 * Format note
 *
 * The FFT implementation in this file uses an interleaved layout for complex
 * numbers:  { Re0, Im0, Re1, Im1, ... Re(N-1), Im(N-1) } (total 2*N floats).
 * This matches the layout that is typically used in many C DSP libraries and
 * keeps the implementation compact and cache-friendly.
 */

static void fft32(float *vec)
{
    /* Iterative radix-2 decimation-in-time Cooley-Tukey.
     * The algorithm is fully unrolled with respect to the stage loop bounds
     * (N is fixed at 32) while the butterfly itself is implemented with tight
     * scalar code so that modern compilers can auto-vectorise.
     */

    /* Step 1: bit-reverse permutation for N = 32.
     * Pre-computed table so we do not have to compute the permutation at run-time.
     */
    static const unsigned char bitrev[32] = {
        0, 16, 8, 24, 4, 20, 12, 28,
        2, 18, 10, 26, 6, 22, 14, 30,
        1, 17, 9, 25, 5, 21, 13, 29,
        3, 19, 11, 27, 7, 23, 15, 31
    };

    float tmp[64];
    for (int ii = 0; ii < 32; ++ii) {
        tmp[2 * ii]     = vec[2 * bitrev[ii]];
        tmp[2 * ii + 1] = vec[2 * bitrev[ii] + 1];
    }

    for (int ii = 0; ii < 64; ++ii)
        vec[ii] = tmp[ii];

    /* === FFT stages === */

    /* Stage 1: butterflies of size 2 (distance 1) */
    for (int k = 0; k < 64; k += 4) {
        float r0 = vec[k];
        float i0 = vec[k + 1];
        float r1 = vec[k + 2];
        float i1 = vec[k + 3];
        vec[k]     = r0 + r1;
        vec[k + 1] = i0 + i1;
        vec[k + 2] = r0 - r1;
        vec[k + 3] = i0 - i1;
    }

    /* Stage 2: butterflies of size 4 (distance 2) */
    for (int k = 0; k < 64; k += 8) {
        /* twiddle factor for j = 1 : e^{-i 2pi/4 * 1} = -i */
        float wr = 0.0f;
        float wi = -1.0f;

        /* j = 0 (no twiddle) */
        float r0 = vec[k];
        float i0 = vec[k + 1];
        float r1 = vec[k + 4];
        float i1 = vec[k + 5];
        vec[k]     = r0 + r1;
        vec[k + 1] = i0 + i1;
        vec[k + 4] = r0 - r1;
        vec[k + 5] = i0 - i1;

        /* j = 1 (with twiddle) */
        r0 = vec[k + 2];
        i0 = vec[k + 3];
        r1 = vec[k + 6];
        i1 = vec[k + 7];
        /* multiply (r1,i1) by (wr,wi) */
        float tr = wr * r1 - wi * i1;
        float ti = wr * i1 + wi * r1;
        vec[k + 2] = r0 + tr;
        vec[k + 3] = i0 + ti;
        vec[k + 6] = r0 - tr;
        vec[k + 7] = i0 - ti;
    }

    /* Stage 3: butterflies of size 8 (distance 4) */
    for (int k = 0; k < 64; k += 16) {
        for (int j = 0; j < 4; ++j) {
            float angle = -2.0f * PI / 8.0f * j;
            float wr = cosf(angle);
            float wi = sinf(angle);

            int i0 = k + 2 * j;
            int i1 = i0 + 8;

            float r0 = vec[i0];
            float i0v = vec[i0 + 1];
            float r1 = vec[i1];
            float i1v = vec[i1 + 1];

            float tr = wr * r1 - wi * i1v;
            float ti = wr * i1v + wi * r1;

            vec[i0]     = r0 + tr;
            vec[i0 + 1] = i0v + ti;
            vec[i1]     = r0 - tr;
            vec[i1 + 1] = i0v - ti;
        }
    }

    /* Stage 4: butterflies of size 16 (distance 8) */
    for (int k = 0; k < 64; k += 32) {
        for (int j = 0; j < 8; ++j) {
            float angle = -2.0f * PI / 16.0f * j;
            float wr = cosf(angle);
            float wi = sinf(angle);

            int i0 = k + 2 * j;
            int i1 = i0 + 16;

            float r0 = vec[i0];
            float i0v = vec[i0 + 1];
            float r1 = vec[i1];
            float i1v = vec[i1 + 1];

            float tr = wr * r1 - wi * i1v;
            float ti = wr * i1v + wi * r1;

            vec[i0]     = r0 + tr;
            vec[i0 + 1] = i0v + ti;
            vec[i1]     = r0 - tr;
            vec[i1 + 1] = i0v - ti;
        }
    }

    /* Stage 5: butterflies of size 32 (distance 16) */
    for (int j = 0; j < 16; ++j) {
        float angle = -2.0f * PI / 32.0f * j;
        float wr = cosf(angle);
        float wi = sinf(angle);

        int i0 = 2 * j;
        int i1 = i0 + 32;

        float r0 = vec[i0];
        float i0v = vec[i0 + 1];
        float r1 = vec[i1];
        float i1v = vec[i1 + 1];

        float tr = wr * r1 - wi * i1v;
        float ti = wr * i1v + wi * r1;

        vec[i0]     = r0 + tr;
        vec[i0 + 1] = i0v + ti;
        vec[i1]     = r0 - tr;
        vec[i1 + 1] = i0v - ti;
    }
}

/* Slow reference implementation (naive DFT) */
static void dft32(const float *in, float *out)
{
    for (int k = 0; k < 32; ++k) {
        float sumRe = 0.0f, sumIm = 0.0f;
        for (int n = 0; n < 32; ++n) {
            float angle = -2.0f * PI * k * n / 32.0f;
            float wr = cosf(angle);
            float wi = sinf(angle);
            float xr = in[2 * n];
            float xi = in[2 * n + 1];
            sumRe += xr * wr - xi * wi;
            sumIm += xr * wi + xi * wr;
        }
        out[2 * k]     = sumRe;
        out[2 * k + 1] = sumIm;
    }
}

int main(void)
{
    srand((unsigned)time(NULL));

    float input[64];
    for (int i = 0; i < 64; ++i)
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; /* random between -1 and 1 */

    float ref[64];

    dft32(input, ref);

    float data[64];
    for (int i = 0; i < 64; ++i)
        data[i] = input[i];

    fft32(data);

    /* Compare results */
    double maxErr = 0.0;
    for (int i = 0; i < 64; ++i) {
        double diff = fabs((double)data[i] - (double)ref[i]);
        if (diff > maxErr)
            maxErr = diff;
    }

    printf("Maximum absolute error: %g\n", maxErr);
    if (maxErr < 1e-4)
        printf("FFT32 implementation is CORRECT.\n");
    else
        printf("FFT32 implementation FAILED.\n");

    return 0;
}