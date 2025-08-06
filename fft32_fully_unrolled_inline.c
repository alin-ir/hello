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
    // Massive set of temporary variables for complete unrolling
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
    
    float temp40, temp41, temp42, temp43, temp44, temp45, temp46, temp47,
        temp48, temp49, temp410, temp411, temp412, temp413, temp414, temp415,
        temp416, temp417, temp418, temp419, temp420, temp421, temp422, temp423,
        temp424, temp425, temp426, temp427, temp428, temp429, temp430, temp431;
    
    float vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7,
        vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15,
        vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23,
        vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31;

    // First stage: 32 -> 16 + 16 (first half of even/odd split)
    /* even part - first 16 points */
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

    // Second stage: 16 -> 8 + 8 for even part
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

    // Third stage: 8 -> 4 + 4 for even part
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

    temp212 = temp16 + temp122;
    temp213 = temp16 - temp122;
    temp214 = temp17 + temp123;
    temp215 = temp17 - temp123;
    temp216 = temp18 + temp124;
    temp218 = temp18 - temp124;
    temp217 = temp19 + temp125;
    temp219 = temp19 - temp125;
    temp220 = temp110 + temp126;
    temp222 = temp110 - temp126;
    temp221 = temp111 + temp127;
    temp223 = temp111 - temp127;
    temp224 = temp112 + temp128;
    temp226 = temp112 - temp128;
    temp225 = temp113 + temp129;
    temp227 = temp113 - temp129;
    temp228 = temp114 + temp130;
    temp230 = temp114 - temp130;
    temp229 = temp115 + temp131;
    temp231 = temp115 - temp131;

    // Apply twiddle factors for 8-point transforms (stage 3)
    temp11 = temp212 + temp228;
    temp12 = temp212 - temp228;
    temp10 = temp214 + temp229;
    temp13 = temp214 - temp229;

    temp30 = ( temp10 + temp12 ) * INV_SQRT2;
    temp32 = ( temp10 - temp12 ) * INV_SQRT2;
    temp31 = ( temp13 - temp11 ) * INV_SQRT2;
    temp33 = ( temp11 + temp13 ) * -INV_SQRT2;

    // Additional 8-point twiddle factors
    temp34 = temp216 + temp224;
    temp38 = temp216 - temp224;
    temp35 = temp217 + temp225;
    temp39 = temp217 - temp225;
    temp310 = temp218 - temp221;
    temp312 = temp218 + temp221;
    temp311 = temp219 + temp220;
    temp313 = temp219 - temp220;
    temp36 = temp220 + temp226;
    temp314 = temp220 - temp226;
    temp37 = temp221 + temp227;
    temp315 = temp227 - temp221;

    temp315 = temp222 + temp230;
    temp316 = temp222 - temp230;
    temp314 = temp223 + temp231;
    temp317 = temp223 - temp231;

    temp318 = ( temp314 + temp316 ) * INV_SQRT2;
    temp320 = ( temp314 - temp316 ) * INV_SQRT2;
    temp319 = ( temp317 - temp315 ) * INV_SQRT2;
    temp321 = ( temp315 + temp317 ) * -INV_SQRT2;

    // Fourth stage: 4 -> 2 + 2 for even part
    temp40 = temp20 + temp34;
    temp44 = temp20 - temp34;
    temp41 = temp21 + temp35;
    temp45 = temp21 - temp35;
    temp48 = temp24 - temp39;
    temp410 = temp24 + temp39;
    temp49 = temp25 + temp38;
    temp411 = temp25 - temp38;
    temp42 = temp22 + temp36;
    temp47 = temp22 - temp36;
    temp43 = temp23 + temp37;
    temp46 = temp37 - temp23;

    temp412 = temp28 + temp312;
    temp413 = temp28 - temp312;
    temp414 = temp29 + temp313;
    temp415 = temp29 - temp313;
    temp416 = temp210 + temp30;
    temp418 = temp210 - temp30;
    temp417 = temp211 + temp31;
    temp419 = temp211 - temp31;
    temp420 = temp26 + temp314;
    temp422 = temp26 - temp314;
    temp421 = temp27 + temp315;
    temp423 = temp27 - temp315;

    // Final butterfly for first part of output
    vec[0] = temp40 + temp42;
    vec[1] = temp41 + temp43;
    vec[2] = temp416 + temp30;
    vec[3] = temp417 + temp31;
    vec[4] = temp44 - temp46;
    vec[5] = temp45 - temp47;
    vec[6] = temp418 + temp32;
    vec[7] = temp419 + temp33;
    vec[8] = temp48 + temp318;
    vec[9] = temp49 + temp319;
    vec[10] = temp412 + temp320;
    vec[11] = temp413 + temp321;
    vec[12] = temp410 - temp314;
    vec[13] = temp411 - temp315;
    vec[14] = temp414 + temp316;
    vec[15] = temp415 + temp317;

    // Handle odd part of the original 32-point transform
    /* odd part - differences */
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

    // Apply 32-point twiddle factors to odd part
    temp10 = vec1 * COS_PI_DIV16 + vec1 * SIN_PI_DIV16;
    temp11 = vec1 * SIN_PI_DIV16 - vec1 * COS_PI_DIV16;
    temp12 = vec3 * COS_3PI_DIV16 + vec3 * SIN_3PI_DIV16;
    temp13 = vec3 * SIN_3PI_DIV16 - vec3 * COS_3PI_DIV16;
    temp14 = vec5 * COS_5PI_DIV16 + vec5 * SIN_5PI_DIV16;
    temp15 = vec5 * SIN_5PI_DIV16 - vec5 * COS_5PI_DIV16;
    temp16 = vec7 * COS_7PI_DIV16 + vec7 * SIN_7PI_DIV16;
    temp17 = vec7 * SIN_7PI_DIV16 - vec7 * COS_7PI_DIV16;
    temp18 = vec9 * INV_SQRT2 + vec9 * INV_SQRT2;
    temp19 = vec9 * INV_SQRT2 - vec9 * INV_SQRT2;
    temp110 = vec11 * COS_5PI_DIV16 + vec11 * SIN_5PI_DIV16;
    temp111 = vec11 * SIN_5PI_DIV16 - vec11 * COS_5PI_DIV16;
    temp112 = vec13 * COS_3PI_DIV16 + vec13 * SIN_3PI_DIV16;
    temp113 = vec13 * SIN_3PI_DIV16 - vec13 * COS_3PI_DIV16;
    temp114 = vec15 * COS_PI_DIV16 + vec15 * SIN_PI_DIV16;
    temp115 = vec15 * SIN_PI_DIV16 - vec15 * COS_PI_DIV16;

    // More complex twiddle factor applications
    temp116 = ( vec2 + vec14 ) * -COS_3PI_DIV8;
    temp117 = ( vec2 - vec14 ) * COS_PI_DIV8;
    temp118 = ( vec3 + vec15 ) * COS_3PI_DIV8;
    temp119 = ( vec3 - vec15 ) * COS_PI_DIV8;
    temp120 = ( vec4 + vec12 ) * -INV_SQRT2;
    temp121 = ( vec4 - vec12 ) * INV_SQRT2;
    temp122 = ( vec5 + vec13 ) * INV_SQRT2;
    temp123 = ( vec5 - vec13 ) * INV_SQRT2;
    temp124 = ( vec6 + vec10 ) * -COS_PI_DIV8;
    temp125 = ( vec6 - vec10 ) * COS_3PI_DIV8;
    temp126 = ( vec7 + vec11 ) * COS_PI_DIV8;
    temp127 = ( vec7 - vec11 ) * COS_3PI_DIV8;

    // Core multiplications for odd part
    vec2 = temp118 * SQRT2PLUS1 - temp126 * SQRT2MINUS1;
    vec3 = temp116 * SQRT2PLUS1 - temp124 * SQRT2MINUS1;
    vec4 = temp117 * SQRT2MINUS1 - temp125 * SQRT2PLUS1;
    vec5 = temp119 * SQRT2MINUS1 - temp127 * SQRT2PLUS1;

    // Post-additions for odd part
    temp118 += temp126;
    temp116 += temp124;
    temp117 += temp125;
    temp119 += temp127;

    vec6 = vec0 + temp122;
    vec10 = vec0 - temp122;
    vec7 = vec1 + temp120;
    vec11 = vec1 - temp120;

    vec12 = temp121 - vec9;
    vec14 = temp121 + vec9;
    vec13 = vec8 + temp123;
    vec15 = vec8 - temp123;

    temp128 = vec6 - vec14;
    temp129 = vec6 + vec14;
    temp130 = vec7 + vec15;
    temp131 = vec7 - vec15;
    temp20 = vec10 + vec12;
    temp21 = vec10 - vec12;
    temp22 = vec11 + vec13;
    temp23 = vec11 - vec13;

    vec10 = temp118 + temp117;
    temp117 = temp118 - temp117;
    vec11 = temp116 + temp119;
    temp119 = temp116 - temp119;

    temp24 = vec2 + vec4;
    temp25 = vec2 - vec4;
    temp26 = vec3 + vec5;
    temp27 = vec3 - vec5;

    // Final assignments for second half
    vec[16] = temp40 - temp42;
    vec[17] = temp41 - temp43;
    vec[18] = temp129 + vec10;
    vec[19] = temp130 + vec11;
    vec[20] = temp416 - temp30;
    vec[21] = temp417 - temp31;
    vec[22] = temp128 + temp24;
    vec[23] = temp131 + temp26;
    vec[24] = temp44 + temp46;
    vec[25] = temp45 + temp47;
    vec[26] = temp21 + temp25;
    vec[27] = temp23 + temp27;
    vec[28] = temp418 - temp32;
    vec[29] = temp419 - temp33;
    vec[30] = temp20 + temp117;
    vec[31] = temp22 + temp119;
    vec[32] = temp48 - temp318;
    vec[33] = temp49 - temp319;
    vec[34] = temp129 - vec10;
    vec[35] = temp130 - vec11;
    vec[36] = temp412 - temp320;
    vec[37] = temp413 - temp321;
    vec[38] = temp128 - temp24;
    vec[39] = temp131 - temp26;
    vec[40] = temp410 + temp314;
    vec[41] = temp411 + temp315;
    vec[42] = temp21 - temp25;
    vec[43] = temp23 - temp27;
    vec[44] = temp414 - temp316;
    vec[45] = temp415 - temp317;
    vec[46] = temp20 - temp117;
    vec[47] = temp22 - temp119;

    // Additional outputs for upper half
    vec[48] = temp410 + temp314;
    vec[49] = temp411 + temp315;
    vec[50] = temp414 + temp316;
    vec[51] = temp415 + temp317;
    vec[52] = temp412 + temp320;
    vec[53] = temp413 + temp321;
    vec[54] = temp48 + temp318;
    vec[55] = temp49 + temp319;
    vec[56] = temp44 + temp46;
    vec[57] = temp45 + temp47;
    vec[58] = temp40 + temp42;
    vec[59] = temp41 + temp43;
    vec[60] = temp416 + temp30;
    vec[61] = temp417 + temp31;
    vec[62] = temp418 + temp32;
    vec[63] = temp419 + temp33;

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
    printf("- 最高性能的单体实现\n\n");
    
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
    
    // Test 3: Complex exponential signal
    printf("测试3: 复数指数信号 (频率bin=2)\n");
    printf("---------------------------------\n");
    float test3[64] = {0};
    float ref3[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        float angle = 2.0f * M_PI * 2 * i / 32.0f;
        test3[2*i] = cosf(angle);      // real part
        test3[2*i+1] = sinf(angle);    // imaginary part
        ref3[2*i] = cosf(angle);
        ref3[2*i+1] = sinf(angle);
    }
    
    printf("输入: e^(j*2π*2*n/32)\n");
    printf("期望: FFT[2] = 32, 其余为0\n\n");
    
    fft32(test3);
    fft_reference_dft(ref3, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %8.4f+%8.4fi | 参考: %8.4f+%8.4fi\n", 
               i, test3[2*i], test3[2*i+1], ref3[2*i], ref3[2*i+1]);
    }
    float error3 = calculate_error(test3, ref3, 32);
    printf("最大误差: %.8f\n\n", error3);
    
    // Test 4: Random signal for robustness
    printf("测试4: 随机信号 (鲁棒性测试)\n");
    printf("---------------------------\n");
    float test4[64] = {0};
    float ref4[64] = {0};
    
    srand(12345);  // Fixed seed for reproducibility
    for (int i = 0; i < 64; i++) {
        test4[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        ref4[i] = test4[i];
    }
    
    printf("输入: 随机复数信号\n");
    printf("期望: FFT结果与参考DFT一致\n\n");
    
    fft32(test4);
    fft_reference_dft(ref4, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %8.4f+%8.4fi | 参考: %8.4f+%8.4fi\n", 
               i, test4[2*i], test4[2*i+1], ref4[2*i], ref4[2*i+1]);
    }
    float error4 = calculate_error(test4, ref4, 32);
    printf("最大误差: %.8f\n\n", error4);
    
    // Verification summary
    printf("验证总结:\n");
    printf("=========\n");
    printf("单位脉冲信号误差:  %.8f\n", error1);
    printf("直流信号误差:      %.8f\n", error2);
    printf("复数指数信号误差:  %.8f\n", error3);
    printf("随机信号误差:      %.8f\n", error4);
    
    float max_error = fmaxf(fmaxf(error1, error2), fmaxf(error3, error4));
    printf("最大总误差:        %.8f\n\n", max_error);
    
    if (max_error < 1e-4) {
        printf("✓ 完全展开的FFT32实现验证成功！\n");
        printf("  算法正确，按照FFT16风格实现，性能最优。\n");
        printf("\n实现特点:\n");
        printf("  ✓ 完全展开，无任何循环或函数调用\n");
        printf("  ✓ 类似FFT16的实现风格\n");
        printf("  ✓ 所有旋转因子和运算完全内联\n");
        printf("  ✓ 单体实现，适合极高性能要求\n");
        printf("  ✓ 编译器友好，易于优化\n");
    } else if (max_error < 1e-3) {
        printf("⚠ FFT32实现基本正确，存在轻微数值误差。\n");
        printf("  当前误差: %.8f (在单精度浮点可接受范围内)\n", max_error);
        printf("  建议进一步优化数值稳定性。\n");
    } else {
        printf("✗ FFT32实现需要进一步调试。\n");
        printf("  当前误差: %.8f (超出预期范围)\n", max_error);
        printf("  请检查蝶形运算和旋转因子的实现。\n");
    }
    
    return 0;
}