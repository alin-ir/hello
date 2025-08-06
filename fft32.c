/*
 * Fast 32-point complex FFT based on the fully unrolled 16-point FFT (fft16).
 *
 * The input vector "vec" contains 32 complex samples = 64 floats laid out as
 * [Re0, Im0, Re1, Im1, ... , Re31, Im31].
 * The result is written in-place in the same format.
 *
 * Strategy:
 * 1. Separate the even-indexed and odd-indexed samples (radix-2 DIF).
 * 2. Run the highly-optimized 16-point FFT (fft16) on both halves.
 * 3. Apply the twiddle factors W32^k and recombine the two 16-point spectra.
 *
 * NOTE: This routine relies on the existence of the function
 *   static void fft16( float *vec );
 * present elsewhere in the codebase.
 */
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declaration of the 16-point kernel (defined in another module) */
extern void fft16( float *vec );

void fft32( float *vec )
{
    /* Split input into even and odd indexed samples */
    float even[32];  /* 16 complex numbers */
    float odd [32];

    for (int k = 0; k < 16; ++k) {
        int src_even = 4 * k;       /* 2*(2k)  */
        int src_odd  = src_even + 2;/* 2*(2k+1) */

        even[2*k]     = vec[src_even];     /* real */
        even[2*k + 1] = vec[src_even + 1]; /* imag */

        odd[2*k]      = vec[src_odd];
        odd[2*k + 1]  = vec[src_odd + 1];
    }

    /* Execute the 16-point FFTs */
    fft16(even);
    fft16(odd);

    /* Recombine with twiddle factors */
    const float TWO_PI_DIV_32 = (float)(2.0 * M_PI / 32.0);

    for (int k = 0; k < 16; ++k) {
        /* Compute twiddle W32^k = cos(2*pi*k/32) - i sin(2*pi*k/32) */
        float angle = -TWO_PI_DIV_32 * k;  /* negative for forward FFT */
        float w_r = cosf(angle);
        float w_i = sinf(angle);

        /* Fetch spectra */
        float even_r = even[2*k];
        float even_i = even[2*k + 1];
        float odd_r  = odd[2*k];
        float odd_i  = odd[2*k + 1];

        /* Multiply odd part by twiddle: (odd_r + i*odd_i) * (w_r + i*w_i) */
        float t_r = odd_r * w_r - odd_i * w_i;
        float t_i = odd_r * w_i + odd_i * w_r;

        /* Butterfly recombination */
        vec[2*k]         = even_r + t_r;
        vec[2*k + 1]     = even_i + t_i;
        vec[2*(k + 16)]     = even_r - t_r;
        vec[2*(k + 16) + 1] = even_i - t_i;
    }
}