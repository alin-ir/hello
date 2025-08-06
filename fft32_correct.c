#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

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
    
    float vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7,
        vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15,
        vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23,
        vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31;

    /* EVEN PART - Process even indexed samples (0,2,4,...,30) */
    /* Stage 1: Load and combine even samples */
    vec0 = vec[0] + vec[32];   // Real part of sample 0
    vec1 = vec[1] + vec[33];   // Imag part of sample 0
    vec2 = vec[4] + vec[36];   // Real part of sample 2
    vec3 = vec[5] + vec[37];   // Imag part of sample 2
    vec4 = vec[8] + vec[40];   // Real part of sample 4
    vec5 = vec[9] + vec[41];   // Imag part of sample 4
    vec6 = vec[12] + vec[44];  // Real part of sample 6
    vec7 = vec[13] + vec[45];  // Imag part of sample 6
    vec8 = vec[16] + vec[48];  // Real part of sample 8
    vec9 = vec[17] + vec[49];  // Imag part of sample 8
    vec10 = vec[20] + vec[52]; // Real part of sample 10
    vec11 = vec[21] + vec[53]; // Imag part of sample 10
    vec12 = vec[24] + vec[56]; // Real part of sample 12
    vec13 = vec[25] + vec[57]; // Imag part of sample 12
    vec14 = vec[28] + vec[60]; // Real part of sample 14
    vec15 = vec[29] + vec[61]; // Imag part of sample 14

    /* 16-point butterflies (following FFT16 pattern) */
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

    /* 8-point butterflies and twiddle factors */
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

    /* ODD PART - Process odd indexed samples (1,3,5,...,31) */
    vec0 = vec[2] + vec[34];   // Real part of sample 1
    vec1 = vec[3] + vec[35];   // Imag part of sample 1
    vec2 = vec[6] + vec[38];   // Real part of sample 3
    vec3 = vec[7] + vec[39];   // Imag part of sample 3
    vec4 = vec[10] + vec[42];  // Real part of sample 5
    vec5 = vec[11] + vec[43];  // Imag part of sample 5
    vec6 = vec[14] + vec[46];  // Real part of sample 7
    vec7 = vec[15] + vec[47];  // Imag part of sample 7
    vec8 = vec[18] + vec[50];  // Real part of sample 9
    vec9 = vec[19] + vec[51];  // Imag part of sample 9
    vec10 = vec[22] + vec[54]; // Real part of sample 11
    vec11 = vec[23] + vec[55]; // Imag part of sample 11
    vec12 = vec[26] + vec[58]; // Real part of sample 13
    vec13 = vec[27] + vec[59]; // Imag part of sample 13
    vec14 = vec[30] + vec[62]; // Real part of sample 15
    vec15 = vec[31] + vec[63]; // Imag part of sample 15

    /* Apply twiddle factors for odd part */
    temp10 = ( vec2 + vec14 ) * -COS_3PI_DIV8;
    temp11 = ( vec2 - vec14 ) * COS_PI_DIV8;
    temp12 = ( vec3 + vec15 ) * COS_3PI_DIV8;
    temp13 = ( vec3 - vec15 ) * COS_PI_DIV8;
    temp14 = ( vec4 + vec12 ) * -INV_SQRT2;
    temp15 = ( vec4 - vec12 ) * INV_SQRT2;
    temp16 = ( vec5 + vec13 ) * INV_SQRT2;
    temp17 = ( vec5 - vec13 ) * INV_SQRT2;
    temp18 = ( vec6 + vec10 ) * -COS_PI_DIV8;
    temp19 = ( vec6 - vec10 ) * COS_3PI_DIV8;
    temp110 = ( vec7 + vec11 ) * COS_PI_DIV8;
    temp111 = ( vec7 - vec11 ) * COS_3PI_DIV8;

    /* Core multiplications for odd part */
    vec2 = temp12 * SQRT2PLUS1 - temp110 * SQRT2MINUS1;
    vec3 = temp10 * SQRT2PLUS1 - temp18 * SQRT2MINUS1;
    vec4 = temp11 * SQRT2MINUS1 - temp19 * SQRT2PLUS1;
    vec5 = temp13 * SQRT2MINUS1 - temp111 * SQRT2PLUS1;

    /* Post-additions for odd part */
    temp12 += temp110;
    temp10 += temp18;
    temp11 += temp19;
    temp13 += temp111;

    vec6 = vec0 + temp16;
    vec10 = vec0 - temp16;
    vec7 = vec1 + temp14;
    vec11 = vec1 - temp14;

    vec12 = temp15 - vec9;
    vec14 = temp15 + vec9;
    vec13 = vec8 + temp17;
    vec15 = vec8 - temp17;

    temp18 = vec6 - vec14;
    temp110 = vec6 + vec14;
    temp19 = vec7 + vec15;
    temp111 = vec7 - vec15;
    temp112 = vec10 + vec12;
    temp114 = vec10 - vec12;
    temp113 = vec11 + vec13;
    temp115 = vec11 - vec13;

    vec10 = temp12 + temp11;
    temp11 = temp12 - temp11;
    vec11 = temp10 + temp13;
    temp13 = temp10 - temp13;

    temp116 = vec2 + vec4;
    temp118 = vec2 - vec4;
    temp117 = vec3 + vec5;
    temp119 = vec3 - vec5;

    /* Final output assignments */
    vec[0] = temp20 + temp22;
    vec[1] = temp21 + temp23;
    vec[2] = temp110 + vec10;
    vec[3] = temp19 + vec11;
    vec[4] = temp210 + temp212;
    vec[5] = temp211 + temp213;
    vec[6] = temp18 + temp116;
    vec[7] = temp19 + temp117;
    vec[8] = temp24 - temp26;
    vec[9] = temp25 - temp27;
    vec[10] = temp114 + temp118;
    vec[11] = temp113 + temp119;
    vec[12] = temp28 + temp214;
    vec[13] = temp29 + temp215;
    vec[14] = temp112 + temp11;
    vec[15] = temp113 + temp13;

    vec[16] = temp20 - temp22;
    vec[17] = temp21 - temp23;
    vec[18] = temp110 - vec10;
    vec[19] = temp19 - vec11;
    vec[20] = temp210 - temp212;
    vec[21] = temp211 - temp213;
    vec[22] = temp18 - temp116;
    vec[23] = temp19 - temp117;
    vec[24] = temp24 + temp26;
    vec[25] = temp25 + temp27;
    vec[26] = temp114 - temp118;
    vec[27] = temp113 - temp119;
    vec[28] = temp28 - temp214;
    vec[29] = temp29 - temp215;
    vec[30] = temp112 - temp11;
    vec[31] = temp113 - temp13;

    return;
}

// Reference FFT implementation for comparison
void fft_reference(float *real, float *imag, int n) {
    if (n <= 1) return;
    
    // Divide
    float *even_real = malloc(n/2 * sizeof(float));
    float *even_imag = malloc(n/2 * sizeof(float));
    float *odd_real = malloc(n/2 * sizeof(float));
    float *odd_imag = malloc(n/2 * sizeof(float));
    
    for (int i = 0; i < n/2; i++) {
        even_real[i] = real[2*i];
        even_imag[i] = imag[2*i];
        odd_real[i] = real[2*i + 1];
        odd_imag[i] = imag[2*i + 1];
    }
    
    // Conquer
    fft_reference(even_real, even_imag, n/2);
    fft_reference(odd_real, odd_imag, n/2);
    
    // Combine
    for (int i = 0; i < n/2; i++) {
        float t_real = cosf(-2*M_PI*i/n) * odd_real[i] - sinf(-2*M_PI*i/n) * odd_imag[i];
        float t_imag = sinf(-2*M_PI*i/n) * odd_real[i] + cosf(-2*M_PI*i/n) * odd_imag[i];
        
        real[i] = even_real[i] + t_real;
        imag[i] = even_imag[i] + t_imag;
        real[i + n/2] = even_real[i] - t_real;
        imag[i + n/2] = even_imag[i] - t_imag;
    }
    
    free(even_real);
    free(even_imag);
    free(odd_real);
    free(odd_imag);
}

void print_complex_array(const char* name, float* data, int n) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        printf("  [%2d] = %8.4f + %8.4fi\n", i, data[2*i], data[2*i+1]);
    }
    printf("\n");
}

float calculate_error(float* data1, float* data2, int n) {
    float max_error = 0.0f;
    for (int i = 0; i < 2*n; i++) {
        float error = fabsf(data1[i] - data2[i]);
        if (error > max_error) max_error = error;
    }
    return max_error;
}

int main() {
    printf("32-point FFT 正确性验证\n");
    printf("=========================\n\n");
    
    // Test case 1: 脉冲信号
    printf("测试1: 脉冲信号 (第0个样本为1，其余为0)\n");
    printf("----------------------------------------\n");
    
    float test1[64] = {0}; // 32个复数样本
    float ref1[64] = {0};
    float ref1_real[32] = {0};
    float ref1_imag[32] = {0};
    
    test1[0] = 1.0f; // 脉冲
    ref1[0] = 1.0f;
    ref1_real[0] = 1.0f;
    
    fft32(test1);
    fft_reference(ref1_real, ref1_imag, 32);
    
    // 将参考结果转换为交错格式
    for (int i = 0; i < 32; i++) {
        ref1[2*i] = ref1_real[i];
        ref1[2*i+1] = ref1_imag[i];
    }
    
    float error1 = calculate_error(test1, ref1, 32);
    printf("脉冲信号的FFT结果 (前8个点):\n");
    for (int i = 0; i < 8; i++) {
        printf("  FFT32[%d] = %8.4f + %8.4fi | 参考值 = %8.4f + %8.4fi\n", 
               i, test1[2*i], test1[2*i+1], ref1[2*i], ref1[2*i+1]);
    }
    printf("最大误差: %.6f\n\n", error1);
    
    // Test case 2: 直流信号
    printf("测试2: 直流信号 (所有样本为1)\n");
    printf("-----------------------------\n");
    
    float test2[64] = {0};
    float ref2[64] = {0};
    float ref2_real[32] = {0};
    float ref2_imag[32] = {0};
    
    for (int i = 0; i < 32; i++) {
        test2[2*i] = 1.0f;
        ref2[2*i] = 1.0f;
        ref2_real[i] = 1.0f;
    }
    
    fft32(test2);
    fft_reference(ref2_real, ref2_imag, 32);
    
    for (int i = 0; i < 32; i++) {
        ref2[2*i] = ref2_real[i];
        ref2[2*i+1] = ref2_imag[i];
    }
    
    float error2 = calculate_error(test2, ref2, 32);
    printf("直流信号的FFT结果 (前8个点):\n");
    for (int i = 0; i < 8; i++) {
        printf("  FFT32[%d] = %8.4f + %8.4fi | 参考值 = %8.4f + %8.4fi\n", 
               i, test2[2*i], test2[2*i+1], ref2[2*i], ref2[2*i+1]);
    }
    printf("最大误差: %.6f\n\n", error2);
    
    // Test case 3: 正弦波信号
    printf("测试3: 正弦波信号 (频率 = 2)\n");
    printf("-----------------------------\n");
    
    float test3[64] = {0};
    float ref3[64] = {0};
    float ref3_real[32] = {0};
    float ref3_imag[32] = {0};
    
    for (int i = 0; i < 32; i++) {
        float val = sinf(2.0f * M_PI * 2 * i / 32.0f);
        test3[2*i] = val;
        ref3[2*i] = val;
        ref3_real[i] = val;
    }
    
    fft32(test3);
    fft_reference(ref3_real, ref3_imag, 32);
    
    for (int i = 0; i < 32; i++) {
        ref3[2*i] = ref3_real[i];
        ref3[2*i+1] = ref3_imag[i];
    }
    
    float error3 = calculate_error(test3, ref3, 32);
    printf("正弦波信号的FFT结果 (前8个点):\n");
    for (int i = 0; i < 8; i++) {
        printf("  FFT32[%d] = %8.4f + %8.4fi | 参考值 = %8.4f + %8.4fi\n", 
               i, test3[2*i], test3[2*i+1], ref3[2*i], ref3[2*i+1]);
    }
    printf("最大误差: %.6f\n\n", error3);
    
    // 总结
    printf("验证总结:\n");
    printf("=========\n");
    printf("测试1 (脉冲信号) 最大误差: %.6f\n", error1);
    printf("测试2 (直流信号) 最大误差: %.6f\n", error2);
    printf("测试3 (正弦波) 最大误差: %.6f\n", error3);
    
    float max_overall_error = fmaxf(fmaxf(error1, error2), error3);
    printf("总体最大误差: %.6f\n", max_overall_error);
    
    if (max_overall_error < 1e-5) {
        printf("\n✓ FFT32函数验证通过！误差在可接受范围内。\n");
    } else {
        printf("\n✗ FFT32函数验证失败，误差过大。\n");
    }
    
    return 0;
}