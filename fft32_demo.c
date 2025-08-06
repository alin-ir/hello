#include <stdio.h>
#include <math.h>
#include <string.h>

// Pre-computed constants for FFT32
#define INV_SQRT2       0.7071067811865476f
#define COS_PI_DIV8     0.9238795325112867f
#define COS_3PI_DIV8    0.3826834323650898f
#define SQRT2PLUS1      2.4142135623730951f
#define SQRT2MINUS1     0.4142135623730950f

// Additional constants for 32-point FFT
#define COS_PI_DIV16    0.9807852804032304f
#define COS_3PI_DIV16   0.8314696123025452f
#define COS_5PI_DIV16   0.5555702330196022f
#define COS_7PI_DIV16   0.1950903220161283f
#define SIN_PI_DIV16    0.1950903220161283f
#define SIN_3PI_DIV16   0.5555702330196022f
#define SIN_5PI_DIV16   0.8314696123025452f
#define SIN_7PI_DIV16   0.9807852804032304f

/* fast implementation, completely unrolled and inlined */
static void fft32( float *vec )
{
    float temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17,
        temp18, temp19, temp110, temp111, temp112, temp113, temp114, temp115;
    float temp20, temp21, temp22, temp23, temp24, temp25, temp26, temp27,
        temp28, temp29, temp210, temp211, temp212, temp213, temp214, temp215;
    float temp30, temp31, temp32, temp33, temp34, temp35, temp36, temp37,
        temp38, temp39, temp310, temp311, temp312, temp313, temp314, temp315;
    float temp40, temp41, temp42, temp43, temp44, temp45, temp46, temp47,
        temp48, temp49, temp410, temp411, temp412, temp413, temp414, temp415;
    
    float vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7,
        vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15,
        vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23,
        vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31;

    /* EVEN PART: First compute the even-indexed DFT */
    /* Stage 1: Load even samples (0,2,4,...,62) and apply first butterfly */
    vec0 = vec[0] + vec[32];
    vec1 = vec[2] + vec[34];
    vec2 = vec[4] + vec[36];
    vec3 = vec[6] + vec[38];
    vec4 = vec[8] + vec[40];
    vec5 = vec[10] + vec[42];
    vec6 = vec[12] + vec[44];
    vec7 = vec[14] + vec[46];
    vec8 = vec[16] + vec[48];
    vec9 = vec[18] + vec[50];
    vec10 = vec[20] + vec[52];
    vec11 = vec[22] + vec[54];
    vec12 = vec[24] + vec[56];
    vec13 = vec[26] + vec[58];
    vec14 = vec[28] + vec[60];
    vec15 = vec[30] + vec[62];

    /* Stage 2: 16-point butterflies */
    temp10 = vec0 + vec8;
    temp12 = vec0 - vec8;
    temp11 = vec1 + vec9;
    temp13 = vec1 - vec9;
    temp14 = vec2 + vec10;
    temp16 = vec2 - vec10;
    temp15 = vec3 + vec11;
    temp17 = vec3 - vec11;
    temp18 = vec4 + vec12;
    temp110 = vec4 - vec12;
    temp19 = vec5 + vec13;
    temp111 = vec5 - vec13;
    temp112 = vec6 + vec14;
    temp114 = vec6 - vec14;
    temp113 = vec7 + vec15;
    temp115 = vec7 - vec15;

    /* Stage 3: 8-point butterflies and twiddle factors */
    temp20 = temp10 + temp18;
    temp24 = temp10 - temp18;
    temp21 = temp11 + temp19;
    temp25 = temp11 - temp19;
    temp28 = temp12 - temp111;
    temp210 = temp12 + temp111;
    temp29 = temp13 + temp110;
    temp211 = temp13 - temp110;
    temp22 = temp14 + temp112;
    temp27 = temp14 - temp112;
    temp23 = temp15 + temp113;
    temp26 = temp113 - temp15;

    temp11 = temp16 + temp114;
    temp12 = temp16 - temp114;
    temp10 = temp17 + temp115;
    temp13 = temp17 - temp115;

    temp212 = ( temp10 + temp12 ) * INV_SQRT2;
    temp214 = ( temp10 - temp12 ) * INV_SQRT2;
    temp213 = ( temp13 - temp11 ) * INV_SQRT2;
    temp215 = ( temp11 + temp13 ) * -INV_SQRT2;

    /* ODD PART: Now compute the odd-indexed DFT */
    /* Stage 1: Load odd samples (1,3,5,...,63) and apply first butterfly */
    vec0 = vec[1] + vec[33];
    vec1 = vec[3] + vec[35];
    vec2 = vec[5] + vec[37];
    vec3 = vec[7] + vec[39];
    vec4 = vec[9] + vec[41];
    vec5 = vec[11] + vec[43];
    vec6 = vec[13] + vec[45];
    vec7 = vec[15] + vec[47];
    vec8 = vec[17] + vec[49];
    vec9 = vec[19] + vec[51];
    vec10 = vec[21] + vec[53];
    vec11 = vec[23] + vec[55];
    vec12 = vec[25] + vec[57];
    vec13 = vec[27] + vec[59];
    vec14 = vec[29] + vec[61];
    vec15 = vec[31] + vec[63];

    /* Apply 32-point twiddle factors */
    temp30 = vec0 * COS_PI_DIV16 - vec8 * SIN_PI_DIV16;
    temp31 = vec1 * COS_3PI_DIV16 - vec9 * SIN_3PI_DIV16;
    temp32 = vec2 * COS_5PI_DIV16 - vec10 * SIN_5PI_DIV16;
    temp33 = vec3 * COS_7PI_DIV16 - vec11 * SIN_7PI_DIV16;
    temp34 = vec4 * INV_SQRT2 - vec12 * INV_SQRT2;
    temp35 = vec5 * COS_5PI_DIV16 - vec13 * SIN_5PI_DIV16;
    temp36 = vec6 * COS_3PI_DIV16 - vec14 * SIN_3PI_DIV16;
    temp37 = vec7 * COS_PI_DIV16 - vec15 * SIN_PI_DIV16;

    temp38 = vec0 * SIN_PI_DIV16 + vec8 * COS_PI_DIV16;
    temp39 = vec1 * SIN_3PI_DIV16 + vec9 * COS_3PI_DIV16;
    temp310 = vec2 * SIN_5PI_DIV16 + vec10 * COS_5PI_DIV16;
    temp311 = vec3 * SIN_7PI_DIV16 + vec11 * COS_7PI_DIV16;
    temp312 = vec4 * INV_SQRT2 + vec12 * INV_SQRT2;
    temp313 = vec5 * SIN_5PI_DIV16 + vec13 * COS_5PI_DIV16;
    temp314 = vec6 * SIN_3PI_DIV16 + vec14 * COS_3PI_DIV16;
    temp315 = vec7 * SIN_PI_DIV16 + vec15 * COS_PI_DIV16;

    /* Stage 2: Process odd part - differences */
    vec0 = vec[1] - vec[33];
    vec1 = vec[3] - vec[35];
    vec2 = vec[5] - vec[37];
    vec3 = vec[7] - vec[39];
    vec4 = vec[9] - vec[41];
    vec5 = vec[11] - vec[43];
    vec6 = vec[13] - vec[45];
    vec7 = vec[15] - vec[47];
    vec8 = vec[17] - vec[49];
    vec9 = vec[19] - vec[51];
    vec10 = vec[21] - vec[53];
    vec11 = vec[23] - vec[55];
    vec12 = vec[25] - vec[57];
    vec13 = vec[27] - vec[59];
    vec14 = vec[29] - vec[61];
    vec15 = vec[31] - vec[63];

    /* Apply more twiddle factors for odd differences */
    temp40 = ( vec2 + vec14 ) * -COS_3PI_DIV8;
    temp41 = ( vec2 - vec14 ) * COS_PI_DIV8;
    temp42 = ( vec3 + vec15 ) * COS_3PI_DIV8;
    temp43 = ( vec3 - vec15 ) * COS_PI_DIV8;
    temp44 = ( vec4 + vec12 ) * -INV_SQRT2;
    temp45 = ( vec4 - vec12 ) * INV_SQRT2;
    temp46 = ( vec5 + vec13 ) * INV_SQRT2;
    temp47 = ( vec5 - vec13 ) * INV_SQRT2;
    temp48 = ( vec6 + vec10 ) * -COS_PI_DIV8;
    temp49 = ( vec6 - vec10 ) * COS_3PI_DIV8;
    temp410 = ( vec7 + vec11 ) * COS_PI_DIV8;
    temp411 = ( vec7 - vec11 ) * COS_3PI_DIV8;

    /* Final butterfly operations and output assignments */
    /* Combine even and odd parts */
    *vec++ = temp20 + temp22 + temp30;
    *vec++ = temp21 + temp23 + temp31;
    *vec++ = temp210 + temp212 + temp32;
    *vec++ = temp211 + temp213 + temp33;
    *vec++ = temp24 - temp26 + temp34;
    *vec++ = temp25 - temp27 + temp35;
    *vec++ = temp28 + temp214 + temp36;
    *vec++ = temp29 + temp215 + temp37;
    *vec++ = temp20 - temp22 + temp38;
    *vec++ = temp21 - temp23 + temp39;
    *vec++ = temp210 - temp212 + temp310;
    *vec++ = temp211 - temp213 + temp311;
    *vec++ = temp24 + temp26 + temp312;
    *vec++ = temp25 + temp27 + temp313;
    *vec++ = temp28 - temp214 + temp314;
    *vec++ = temp29 - temp215 + temp315;

    /* Second half - continue with remaining combinations */
    *vec++ = temp20 + temp22 + temp40;
    *vec++ = temp21 + temp23 + temp41;
    *vec++ = temp210 + temp212 + temp42;
    *vec++ = temp211 + temp213 + temp43;
    *vec++ = temp24 - temp26 + temp44;
    *vec++ = temp25 - temp27 + temp45;
    *vec++ = temp28 + temp214 + temp46;
    *vec++ = temp29 + temp215 + temp47;
    *vec++ = temp20 - temp22 + temp48;
    *vec++ = temp21 - temp23 + temp49;
    *vec++ = temp210 - temp212 + temp410;
    *vec++ = temp211 - temp213 + temp411;
    *vec++ = temp24 + temp26 + vec0;
    *vec++ = temp25 + temp27 + vec1;
    *vec++ = temp28 - temp214 + vec8;
    *vec++ = temp29 - temp215 + vec9;

    return;
}

int main() 
{
    float test_data[64];  // 64 elements for complex data (32 real + 32 imaginary)
    int i;
    
    printf("32-point FFT Fast Implementation Demo\n");
    printf("=====================================\n\n");
    
    // Initialize test data with a simple signal
    // Real part: a simple cosine wave
    for (i = 0; i < 32; i++) {
        test_data[2*i] = cosf(2.0f * M_PI * i / 32.0f);     // Real part
        test_data[2*i + 1] = 0.0f;                          // Imaginary part
    }
    
    printf("Input signal (first 8 complex samples):\n");
    for (i = 0; i < 8; i++) {
        printf("sample[%2d] = %8.4f + %8.4fi\n", i, 
               test_data[2*i], test_data[2*i + 1]);
    }
    
    // Perform FFT
    fft32(test_data);
    
    printf("\nFFT output (first 8 complex samples):\n");
    for (i = 0; i < 8; i++) {
        printf("FFT[%2d]    = %8.4f + %8.4fi\n", i, 
               test_data[2*i], test_data[2*i + 1]);
    }
    
    printf("\nMagnitude spectrum (first 16 bins):\n");
    for (i = 0; i < 16; i++) {
        float magnitude = sqrtf(test_data[2*i] * test_data[2*i] + 
                                test_data[2*i + 1] * test_data[2*i + 1]);
        printf("Bin[%2d]    = %8.4f\n", i, magnitude);
    }
    
    return 0;
}