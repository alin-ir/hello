#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Pre-computed constants for FFT16 (from original implementation)
#define INV_SQRT2       0.7071067811865476f
#define COS_PI_DIV8     0.9238795325112867f
#define COS_3PI_DIV8    0.3826834323650898f
#define SQRT2PLUS1      2.4142135623730951f
#define SQRT2MINUS1     0.4142135623730950f

// Additional pre-computed constants for 32-point FFT
#define COS_PI_DIV16    0.9807852804032304f    // cos(π/16)
#define SIN_PI_DIV16    0.1950903220161283f    // sin(π/16)
#define COS_3PI_DIV16   0.8314696123025452f    // cos(3π/16)
#define SIN_3PI_DIV16   0.5555702330196022f    // sin(3π/16)
#define COS_5PI_DIV16   0.5555702330196022f    // cos(5π/16)
#define SIN_5PI_DIV16   0.8314696123025452f    // sin(5π/16)
#define COS_7PI_DIV16   0.1950903220161283f    // cos(7π/16)
#define SIN_7PI_DIV16   0.9807852804032304f    // sin(7π/16)

/* fast implementation, completely unrolled and inlined - 32 point FFT */
static void fft32( float *vec )
{
    // Complete set of temporary variables for full unrolling
    float temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17,
        temp18, temp19, temp110, temp111, temp112, temp113, temp114, temp115,
        temp116, temp117, temp118, temp119, temp120, temp121, temp122, temp123,
        temp124, temp125, temp126, temp127, temp128, temp129, temp130, temp131;
    
    float temp20, temp21, temp22, temp23, temp24, temp25, temp26, temp27,
        temp28, temp29, temp210, temp211, temp212, temp213, temp214, temp215,
        temp216, temp217, temp218, temp219, temp220, temp221, temp222, temp223,
        temp224, temp225, temp226, temp227, temp228, temp229, temp230, temp231;
    
    float temp30, temp31, temp32, temp33, temp34, temp35, temp36, temp37,
        temp38, temp39, temp310, temp311, temp312, temp313, temp314, temp315,
        temp316, temp317, temp318, temp319, temp320, temp321, temp322, temp323,
        temp324, temp325, temp326, temp327, temp328, temp329, temp330, temp331;
    
    float vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7,
        vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15,
        vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23,
        vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31;

    // STAGE 1: 32 -> 16 + 16 (even/odd decomposition)
    /* even part - sum of pairs */
    vec0 = vec[0] + vec[32];
    vec1 = vec[1] + vec[33];
    vec2 = vec[2] + vec[34];
    vec3 = vec[3] + vec[35];
    vec4 = vec[4] + vec[36];
    vec5 = vec[5] + vec[37];
    vec6 = vec[6] + vec[38];
    vec7 = vec[7] + vec[39];
    vec8 = vec[8] + vec[40];
    vec9 = vec[9] + vec[41];
    vec10 = vec[10] + vec[42];
    vec11 = vec[11] + vec[43];
    vec12 = vec[12] + vec[44];
    vec13 = vec[13] + vec[45];
    vec14 = vec[14] + vec[46];
    vec15 = vec[15] + vec[47];
    vec16 = vec[16] + vec[48];
    vec17 = vec[17] + vec[49];
    vec18 = vec[18] + vec[50];
    vec19 = vec[19] + vec[51];
    vec20 = vec[20] + vec[52];
    vec21 = vec[21] + vec[53];
    vec22 = vec[22] + vec[54];
    vec23 = vec[23] + vec[55];
    vec24 = vec[24] + vec[56];
    vec25 = vec[25] + vec[57];
    vec26 = vec[26] + vec[58];
    vec27 = vec[27] + vec[59];
    vec28 = vec[28] + vec[60];
    vec29 = vec[29] + vec[61];
    vec30 = vec[30] + vec[62];
    vec31 = vec[31] + vec[63];

    // STAGE 2: 16 -> 8 + 8 (like FFT16 even stage)
    temp10 = vec0 + vec16;
    temp12 = vec0 - vec16;
    temp11 = vec1 + vec17;
    temp13 = vec1 - vec17;
    temp14 = vec2 + vec18;
    temp16 = vec2 - vec18;
    temp15 = vec3 + vec19;
    temp17 = vec3 - vec19;
    temp18 = vec4 + vec20;
    temp110 = vec4 - vec20;
    temp19 = vec5 + vec21;
    temp111 = vec5 - vec21;
    temp112 = vec6 + vec22;
    temp114 = vec6 - vec22;
    temp113 = vec7 + vec23;
    temp115 = vec7 - vec23;
    temp116 = vec8 + vec24;
    temp118 = vec8 - vec24;
    temp117 = vec9 + vec25;
    temp119 = vec9 - vec25;
    temp120 = vec10 + vec26;
    temp122 = vec10 - vec26;
    temp121 = vec11 + vec27;
    temp123 = vec11 - vec27;
    temp124 = vec12 + vec28;
    temp126 = vec12 - vec28;
    temp125 = vec13 + vec29;
    temp127 = vec13 - vec29;
    temp128 = vec14 + vec30;
    temp130 = vec14 - vec30;
    temp129 = vec15 + vec31;
    temp131 = vec15 - vec31;

    // STAGE 3: 8 -> 4 + 4 (following FFT16 pattern)
    temp20 = temp10 + temp116;
    temp24 = temp10 - temp116;
    temp21 = temp11 + temp117;
    temp25 = temp11 - temp117;
    temp28 = temp12 - temp119;
    temp210 = temp12 + temp119;
    temp29 = temp13 + temp118;
    temp211 = temp13 - temp118;
    temp22 = temp14 + temp120;
    temp27 = temp14 - temp120;
    temp23 = temp15 + temp121;
    temp26 = temp121 - temp15;

    temp11 = temp16 + temp122;
    temp12 = temp16 - temp122;
    temp10 = temp17 + temp123;
    temp13 = temp17 - temp123;

    temp212 = ( temp10 + temp12 ) * INV_SQRT2;
    temp214 = ( temp10 - temp12 ) * INV_SQRT2;
    temp213 = ( temp13 - temp11 ) * INV_SQRT2;
    temp215 = ( temp11 + temp13 ) * -INV_SQRT2;

    // Handle the second 8-point group
    temp216 = temp18 + temp124;
    temp220 = temp18 - temp124;
    temp217 = temp19 + temp125;
    temp221 = temp19 - temp125;
    temp224 = temp110 - temp127;
    temp226 = temp110 + temp127;
    temp225 = temp111 + temp126;
    temp227 = temp111 - temp126;
    temp218 = temp112 + temp128;
    temp223 = temp112 - temp128;
    temp219 = temp113 + temp129;
    temp222 = temp129 - temp113;

    temp312 = temp114 + temp130;
    temp313 = temp114 - temp130;
    temp314 = temp115 + temp131;
    temp315 = temp115 - temp131;

    temp316 = ( temp314 + temp313 ) * INV_SQRT2;
    temp318 = ( temp314 - temp313 ) * INV_SQRT2;
    temp317 = ( temp315 - temp312 ) * INV_SQRT2;
    temp319 = ( temp312 + temp315 ) * -INV_SQRT2;

    // STAGE 4: 4 -> 2 + 2 final stage
    temp30 = temp20 + temp216;
    temp34 = temp20 - temp216;
    temp31 = temp21 + temp217;
    temp35 = temp21 - temp217;
    temp38 = temp24 - temp221;
    temp310 = temp24 + temp221;
    temp39 = temp25 + temp220;
    temp311 = temp25 - temp220;
    temp32 = temp22 + temp218;
    temp37 = temp22 - temp218;
    temp33 = temp23 + temp219;
    temp36 = temp219 - temp23;

    temp320 = temp28 + temp226;
    temp321 = temp28 - temp226;
    temp322 = temp29 + temp227;
    temp323 = temp29 - temp227;
    temp324 = temp210 + temp212;
    temp326 = temp210 - temp212;
    temp325 = temp211 + temp213;
    temp327 = temp211 - temp213;
    temp328 = temp26 + temp222;
    temp330 = temp26 - temp222;
    temp329 = temp27 + temp223;
    temp331 = temp27 - temp223;

    // Process odd part (differences)
    vec0 = vec[0] - vec[32];
    vec1 = vec[1] - vec[33];
    vec2 = vec[2] - vec[34];
    vec3 = vec[3] - vec[35];
    vec4 = vec[4] - vec[36];
    vec5 = vec[5] - vec[37];
    vec6 = vec[6] - vec[38];
    vec7 = vec[7] - vec[39];
    vec8 = vec[8] - vec[40];
    vec9 = vec[9] - vec[41];
    vec10 = vec[10] - vec[42];
    vec11 = vec[11] - vec[43];
    vec12 = vec[12] - vec[44];
    vec13 = vec[13] - vec[45];
    vec14 = vec[14] - vec[46];
    vec15 = vec[15] - vec[47];
    vec16 = vec[16] - vec[48];
    vec17 = vec[17] - vec[49];
    vec18 = vec[18] - vec[50];
    vec19 = vec[19] - vec[51];
    vec20 = vec[20] - vec[52];
    vec21 = vec[21] - vec[53];
    vec22 = vec[22] - vec[54];
    vec23 = vec[23] - vec[55];
    vec24 = vec[24] - vec[56];
    vec25 = vec[25] - vec[57];
    vec26 = vec[26] - vec[58];
    vec27 = vec[27] - vec[59];
    vec28 = vec[28] - vec[60];
    vec29 = vec[29] - vec[61];
    vec30 = vec[30] - vec[62];
    vec31 = vec[31] - vec[63];

    // Apply 32-point twiddle factors (following FFT16 pattern for odd part)
    temp110 = ( vec2 + vec30 ) * -COS_3PI_DIV8;
    temp111 = ( vec2 - vec30 ) * COS_PI_DIV8;
    temp112 = ( vec3 + vec31 ) * COS_3PI_DIV8;
    temp113 = ( vec3 - vec31 ) * COS_PI_DIV8;
    temp114 = ( vec4 + vec28 ) * -INV_SQRT2;
    temp115 = ( vec4 - vec28 ) * INV_SQRT2;
    temp116 = ( vec5 + vec29 ) * INV_SQRT2;
    temp117 = ( vec5 - vec29 ) * INV_SQRT2;
    temp118 = ( vec6 + vec26 ) * -COS_PI_DIV8;
    temp119 = ( vec6 - vec26 ) * COS_3PI_DIV8;
    temp120 = ( vec7 + vec27 ) * COS_PI_DIV8;
    temp121 = ( vec7 - vec27 ) * COS_3PI_DIV8;

    // Apply 16-point specific twiddle factors 
    temp122 = ( vec10 + vec22 ) * -COS_3PI_DIV8;
    temp123 = ( vec10 - vec22 ) * COS_PI_DIV8;
    temp124 = ( vec11 + vec23 ) * COS_3PI_DIV8;
    temp125 = ( vec11 - vec23 ) * COS_PI_DIV8;
    temp126 = ( vec12 + vec20 ) * -INV_SQRT2;
    temp127 = ( vec12 - vec20 ) * INV_SQRT2;
    temp128 = ( vec13 + vec21 ) * INV_SQRT2;
    temp129 = ( vec13 - vec21 ) * INV_SQRT2;
    temp130 = ( vec14 + vec18 ) * -COS_PI_DIV8;
    temp131 = ( vec14 - vec18 ) * COS_3PI_DIV8;
    temp224 = ( vec15 + vec19 ) * COS_PI_DIV8;
    temp225 = ( vec15 - vec19 ) * COS_3PI_DIV8;

    // Core multiplications (like FFT16)
    vec2 = temp112 * SQRT2PLUS1 - temp120 * SQRT2MINUS1;
    vec3 = temp110 * SQRT2PLUS1 - temp118 * SQRT2MINUS1;
    vec4 = temp111 * SQRT2MINUS1 - temp119 * SQRT2PLUS1;
    vec5 = temp113 * SQRT2MINUS1 - temp121 * SQRT2PLUS1;

    vec10 = temp124 * SQRT2PLUS1 - temp224 * SQRT2MINUS1;
    vec11 = temp122 * SQRT2PLUS1 - temp130 * SQRT2MINUS1;
    vec12 = temp123 * SQRT2MINUS1 - temp131 * SQRT2PLUS1;
    vec13 = temp125 * SQRT2MINUS1 - temp225 * SQRT2PLUS1;

    // Post-additions
    temp112 += temp120;
    temp110 += temp118;
    temp111 += temp119;
    temp113 += temp121;

    temp124 += temp224;
    temp122 += temp130;
    temp123 += temp131;
    temp125 += temp225;

    vec6 = vec0 + temp116;
    vec22 = vec0 - temp116;
    vec7 = vec1 + temp114;
    vec23 = vec1 - temp114;

    vec14 = vec8 + temp128;
    vec30 = vec8 - temp128;
    vec15 = vec9 + temp126;
    vec31 = vec9 - temp126;

    vec20 = temp115 - vec25;
    vec28 = temp115 + vec25;
    vec21 = vec24 + temp117;
    vec29 = vec24 - temp117;

    vec24 = temp127 - vec17;
    vec16 = temp127 + vec17;
    vec25 = vec16 + temp129;
    vec17 = vec16 - temp129;

    // Final butterfly operations and output assignment
    *vec++ = temp30 + temp32;
    *vec++ = temp31 + temp33;
    *vec++ = temp324 + temp212 + vec6 + temp112;
    *vec++ = temp325 + temp213 + vec7 + temp110;
    *vec++ = temp326 + temp214 + vec14 + temp124;
    *vec++ = temp327 + temp215 + vec15 + temp122;
    *vec++ = temp34 - temp36 + vec20 + vec2;
    *vec++ = temp35 - temp37 + vec21 + vec3;
    *vec++ = temp320 + temp316 + vec24 + vec10;
    *vec++ = temp321 + temp317 + vec25 + vec11;
    *vec++ = temp322 + temp318 + temp28 + temp123;
    *vec++ = temp323 + temp319 + temp29 + temp125;
    *vec++ = temp310 - temp36 + vec22 + vec4;
    *vec++ = temp311 - temp37 + vec23 + vec5;
    *vec++ = temp38 + temp318 + vec30 + vec12;
    *vec++ = temp39 + temp319 + vec31 + vec13;

    *vec++ = temp30 - temp32;
    *vec++ = temp31 - temp33;
    *vec++ = temp324 - temp212 + vec6 - temp112;
    *vec++ = temp325 - temp213 + vec7 - temp110;
    *vec++ = temp326 - temp214 + vec14 - temp124;
    *vec++ = temp327 - temp215 + vec15 - temp122;
    *vec++ = temp34 + temp36 + vec20 - vec2;
    *vec++ = temp35 + temp37 + vec21 - vec3;
    *vec++ = temp320 - temp316 + vec24 - vec10;
    *vec++ = temp321 - temp317 + vec25 - vec11;
    *vec++ = temp322 - temp318 + temp28 - temp123;
    *vec++ = temp323 - temp319 + temp29 - temp125;
    *vec++ = temp310 + temp36 + vec22 - vec4;
    *vec++ = temp311 + temp37 + vec23 - vec5;
    *vec++ = temp38 - temp318 + vec30 - vec12;
    *vec++ = temp39 - temp319 + vec31 - vec13;

    return;
}

// Reference DFT implementation for verification
void fft_reference_dft(float *vec, int n) {
    float *temp = malloc(2 * n * sizeof(float));
    
    for (int k = 0; k < n; k++) {
        temp[2*k] = 0;      // real part
        temp[2*k+1] = 0;    // imaginary part
        
        for (int n_idx = 0; n_idx < n; n_idx++) {
            float angle = -2.0f * M_PI * k * n_idx / (float)n;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            temp[2*k] += vec[2*n_idx] * cos_val - vec[2*n_idx+1] * sin_val;
            temp[2*k+1] += vec[2*n_idx] * sin_val + vec[2*n_idx+1] * cos_val;
        }
    }
    
    // Copy result back to original array
    for (int i = 0; i < 2*n; i++) {
        vec[i] = temp[i];
    }
    
    free(temp);
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
    printf("完全展开的32点FFT实现演示 (类似FFT16风格)\n");
    printf("==========================================\n\n");
    printf("特点:\n");
    printf("- 完全展开，按照FFT16的实现风格\n");
    printf("- 不调用任何其他FFT函数\n");
    printf("- 所有计算都内联展开\n");
    printf("- 蝶形运算模式与FFT16一致\n\n");
    
    // Test 1: Unit impulse signal
    printf("测试1: 单位脉冲信号\n");
    printf("-------------------\n");
    float test1[64] = {0};
    float ref1[64] = {0};
    
    test1[0] = 1.0f;  // impulse
    ref1[0] = 1.0f;
    
    printf("输入: δ[0] = 1, 其余为0\n");
    printf("期望: 所有FFT输出都应该是1\n\n");
    
    fft32(test1);
    fft_reference_dft(ref1, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %8.4f+%8.4fi | 参考: %8.4f+%8.4fi\n", 
               i, test1[2*i], test1[2*i+1], ref1[2*i], ref1[2*i+1]);
    }
    float error1 = calculate_error(test1, ref1, 32);
    printf("最大误差: %.8f\n\n", error1);
    
    // Test 2: DC signal
    printf("测试2: 直流信号\n");
    printf("---------------\n");
    float test2[64] = {0};
    float ref2[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        test2[2*i] = 1.0f;
        ref2[2*i] = 1.0f;
    }
    
    printf("输入: 所有样本实部 = 1\n");
    printf("期望: FFT[0] = 32, 其余为0\n\n");
    
    fft32(test2);
    fft_reference_dft(ref2, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %8.4f+%8.4fi | 参考: %8.4f+%8.4fi\n", 
               i, test2[2*i], test2[2*i+1], ref2[2*i], ref2[2*i+1]);
    }
    float error2 = calculate_error(test2, ref2, 32);
    printf("最大误差: %.8f\n\n", error2);
    
    // Test 3: Simple sine wave  
    printf("测试3: 简单正弦波 (频率bin=1)\n");
    printf("-----------------------------\n");
    float test3[64] = {0};
    float ref3[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        float angle = 2.0f * M_PI * 1 * i / 32.0f;
        test3[2*i] = cosf(angle);      // real part
        test3[2*i+1] = sinf(angle);    // imaginary part
        ref3[2*i] = cosf(angle);
        ref3[2*i+1] = sinf(angle);
    }
    
    printf("输入: e^(j*2π*1*n/32)\n");
    printf("期望: FFT[1] = 32, 其余为0\n\n");
    
    fft32(test3);
    fft_reference_dft(ref3, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %8.4f+%8.4fi | 参考: %8.4f+%8.4fi\n", 
               i, test3[2*i], test3[2*i+1], ref3[2*i], ref3[2*i+1]);
    }
    float error3 = calculate_error(test3, ref3, 32);
    printf("最大误差: %.8f\n\n", error3);
    
    // Verification summary
    printf("验证总结:\n");
    printf("=========\n");
    printf("单位脉冲信号误差:  %.8f\n", error1);
    printf("直流信号误差:      %.8f\n", error2);
    printf("正弦波信号误差:    %.8f\n", error3);
    
    float max_error = fmaxf(fmaxf(error1, error2), error3);
    printf("最大总误差:        %.8f\n\n", max_error);
    
    if (max_error < 1e-3) {
        printf("✓ 完全展开的FFT32实现基本正确！\n");
        printf("  按照FFT16风格实现，算法结构合理。\n");
        printf("\n实现特点:\n");
        printf("  ✓ 完全展开，无任何循环或函数调用\n");
        printf("  ✓ 类似FFT16的实现风格和结构\n");
        printf("  ✓ 所有旋转因子和运算完全内联\n");
        printf("  ✓ 单体实现，适合高性能要求\n");
        printf("  ✓ 蝶形运算模式与FFT16一致\n");
    } else {
        printf("⚠ FFT32实现需要进一步优化。\n");
        printf("  当前误差: %.8f\n", max_error);
        printf("  建议检查蝶形运算和旋转因子的实现。\n");
        printf("  这是一个复杂的完全展开实现，可能需要更精细的调试。\n");
    }
    
    return 0;
}