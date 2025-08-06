#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Pre-computed constants for FFT
#define INV_SQRT2       0.7071067811865476f
#define COS_PI_DIV8     0.9238795325112867f
#define COS_3PI_DIV8    0.3826834323650898f
#define SQRT2PLUS1      2.4142135623730951f
#define SQRT2MINUS1     0.4142135623730950f

/* 原始FFT16函数 - 完全展开和内联 */
static void fft16( float *vec )
{
    float temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17,
        temp18, temp19, temp110, temp111, temp112, temp113, temp114, temp115;
    float temp20, temp21, temp22, temp23, temp24, temp25, temp26, temp27,
        temp28, temp29, temp210, temp211, temp212, temp213, temp214, temp215;
    float vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7,
        vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

    /* even */
    vec0 = vec[0] + vec[16];
    vec1 = vec[1] + vec[17];
    vec2 = vec[2] + vec[18];
    vec3 = vec[3] + vec[19];
    vec4 = vec[4] + vec[20];
    vec5 = vec[5] + vec[21];
    vec6 = vec[6] + vec[22];
    vec7 = vec[7] + vec[23];
    vec8 = vec[8] + vec[24];
    vec9 = vec[9] + vec[25];
    vec10 = vec[10] + vec[26];
    vec11 = vec[11] + vec[27];
    vec12 = vec[12] + vec[28];
    vec13 = vec[13] + vec[29];
    vec14 = vec[14] + vec[30];
    vec15 = vec[15] + vec[31];

    /* Pre-additions */
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

    /* Pre-additions and core multiplications */
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

    /* odd */
    vec0 = vec[0] - vec[16];
    vec1 = vec[1] - vec[17];
    vec2 = vec[2] - vec[18];
    vec3 = vec[3] - vec[19];
    vec4 = vec[4] - vec[20];
    vec5 = vec[5] - vec[21];
    vec6 = vec[6] - vec[22];
    vec7 = vec[7] - vec[23];
    vec8 = vec[8] - vec[24];
    vec9 = vec[9] - vec[25];
    vec10 = vec[10] - vec[26];
    vec11 = vec[11] - vec[27];
    vec12 = vec[12] - vec[28];
    vec13 = vec[13] - vec[29];
    vec14 = vec[14] - vec[30];
    vec15 = vec[15] - vec[31];

    /* Pre-additions and core multiplications */
    temp19 = ( vec2 + vec14 ) * -COS_3PI_DIV8;
    temp110 = ( vec2 - vec14 ) * COS_PI_DIV8;
    temp18 = ( vec3 + vec15 ) * COS_3PI_DIV8;
    temp111 = ( vec3 - vec15 ) * COS_PI_DIV8;
    temp15 = ( vec4 + vec12 ) * -INV_SQRT2;
    temp16 = ( vec4 - vec12 ) * INV_SQRT2;
    temp14 = ( vec5 + vec13 ) * INV_SQRT2;
    temp17 = ( vec5 - vec13 ) * INV_SQRT2;
    temp113 = ( vec6 + vec10 ) * -COS_PI_DIV8;
    temp114 = ( vec6 - vec10 ) * COS_3PI_DIV8;
    temp112 = ( vec7 + vec11 ) * COS_PI_DIV8;
    temp115 = ( vec7 - vec11 ) * COS_3PI_DIV8;

    /* Core multiplications */
    vec2 = temp18 * SQRT2PLUS1 - temp112 * SQRT2MINUS1;
    vec3 = temp19 * SQRT2PLUS1 - temp113 * SQRT2MINUS1;
    vec4 = temp110 * SQRT2MINUS1 - temp114 * SQRT2PLUS1;
    vec5 = temp111 * SQRT2MINUS1 - temp115 * SQRT2PLUS1;

    /* Post-additions */
    temp18 += temp112;
    temp19 += temp113;
    temp110 += temp114;
    temp111 += temp115;

    vec6 = vec0 + temp14;
    vec10 = vec0 - temp14;
    vec7 = vec1 + temp15;
    vec11 = vec1 - temp15;

    vec12 = temp16 - vec9;
    vec14 = temp16 + vec9;
    vec13 = vec8 + temp17;
    vec15 = vec8 - temp17;

    temp10 = vec6 - vec14;
    temp12 = vec6 + vec14;
    temp11 = vec7 + vec15;
    temp13 = vec7 - vec15;
    temp14 = vec10 + vec12;
    temp16 = vec10 - vec12;
    temp15 = vec11 + vec13;
    temp17 = vec11 - vec13;

    vec10 = temp18 + temp110;
    temp110 = temp18 - temp110;
    vec11 = temp19 + temp111;
    temp111 = temp19 - temp111;

    temp112 = vec2 + vec4;
    temp114 = vec2 - vec4;
    temp113 = vec3 + vec5;
    temp115 = vec3 - vec5;

    /* Post-additions */
    *vec++ = temp20 + temp22;
    *vec++ = temp21 + temp23;
    *vec++ = temp12 + vec10;
    *vec++ = temp13 + vec11;
    *vec++ = temp210 + temp212;
    *vec++ = temp211 + temp213;
    *vec++ = temp10 + temp112;
    *vec++ = temp11 + temp113;
    *vec++ = temp24 - temp26;
    *vec++ = temp25 - temp27;
    *vec++ = temp16 + temp114;
    *vec++ = temp17 + temp115;
    *vec++ = temp28 + temp214;
    *vec++ = temp29 + temp215;
    *vec++ = temp14 + temp110;
    *vec++ = temp15 + temp111;
    *vec++ = temp20 - temp22;
    *vec++ = temp21 - temp23;
    *vec++ = temp12 - vec10;
    *vec++ = temp13 - vec11;
    *vec++ = temp210 - temp212;
    *vec++ = temp211 - temp213;
    *vec++ = temp10 - temp112;
    *vec++ = temp11 - temp113;
    *vec++ = temp24 + temp26;
    *vec++ = temp25 + temp27;
    *vec++ = temp16 - temp114;
    *vec++ = temp17 - temp115;
    *vec++ = temp28 - temp214;
    *vec++ = temp29 - temp215;
    *vec++ = temp14 - temp110;
    *vec++ = temp15 - temp111;

    return;
}

/* 完全展开的32点FFT实现 - 最高效率版本 */
static void fft32( float *vec )
{
    // 临时存储数组
    float even_data[32];  // 偶数索引样本 (0,2,4,...,30)
    float odd_data[32];   // 奇数索引样本 (1,3,5,...,31)
    
    // 完全展开的数据重排 - 偶数索引样本
    even_data[0] = vec[0];    even_data[1] = vec[1];      // 样本0
    even_data[2] = vec[4];    even_data[3] = vec[5];      // 样本2  
    even_data[4] = vec[8];    even_data[5] = vec[9];      // 样本4
    even_data[6] = vec[12];   even_data[7] = vec[13];     // 样本6
    even_data[8] = vec[16];   even_data[9] = vec[17];     // 样本8
    even_data[10] = vec[20];  even_data[11] = vec[21];    // 样本10
    even_data[12] = vec[24];  even_data[13] = vec[25];    // 样本12
    even_data[14] = vec[28];  even_data[15] = vec[29];    // 样本14
    even_data[16] = vec[32];  even_data[17] = vec[33];    // 样本16
    even_data[18] = vec[36];  even_data[19] = vec[37];    // 样本18
    even_data[20] = vec[40];  even_data[21] = vec[41];    // 样本20
    even_data[22] = vec[44];  even_data[23] = vec[45];    // 样本22
    even_data[24] = vec[48];  even_data[25] = vec[49];    // 样本24
    even_data[26] = vec[52];  even_data[27] = vec[53];    // 样本26
    even_data[28] = vec[56];  even_data[29] = vec[57];    // 样本28
    even_data[30] = vec[60];  even_data[31] = vec[61];    // 样本30

    // 完全展开的数据重排 - 奇数索引样本
    odd_data[0] = vec[2];     odd_data[1] = vec[3];       // 样本1
    odd_data[2] = vec[6];     odd_data[3] = vec[7];       // 样本3
    odd_data[4] = vec[10];    odd_data[5] = vec[11];      // 样本5
    odd_data[6] = vec[14];    odd_data[7] = vec[15];      // 样本7
    odd_data[8] = vec[18];    odd_data[9] = vec[19];      // 样本9
    odd_data[10] = vec[22];   odd_data[11] = vec[23];     // 样本11
    odd_data[12] = vec[26];   odd_data[13] = vec[27];     // 样本13
    odd_data[14] = vec[30];   odd_data[15] = vec[31];     // 样本15
    odd_data[16] = vec[34];   odd_data[17] = vec[35];     // 样本17
    odd_data[18] = vec[38];   odd_data[19] = vec[39];     // 样本19
    odd_data[20] = vec[42];   odd_data[21] = vec[43];     // 样本21
    odd_data[22] = vec[46];   odd_data[23] = vec[47];     // 样本23
    odd_data[24] = vec[50];   odd_data[25] = vec[51];     // 样本25
    odd_data[26] = vec[54];   odd_data[27] = vec[55];     // 样本27
    odd_data[28] = vec[58];   odd_data[29] = vec[59];     // 样本29
    odd_data[30] = vec[62];   odd_data[31] = vec[63];     // 样本31

    // 对偶数和奇数部分分别执行16点FFT
    fft16(even_data);
    fft16(odd_data);

    // 完全展开的蝶形运算 - 所有旋转因子预计算
    float temp_real, temp_imag;

    // k=0: W32^0 = 1.0 + 0.0j
    temp_real = odd_data[0];
    temp_imag = odd_data[1];
    vec[0] = even_data[0] + temp_real;
    vec[1] = even_data[1] + temp_imag;
    vec[32] = even_data[0] - temp_real;
    vec[33] = even_data[1] - temp_imag;

    // k=1: W32^1 = cos(π/16) - j*sin(π/16) = 0.9807852804 - j*0.1950903220
    temp_real = odd_data[2] * 0.9807852804032304f - odd_data[3] * (-0.1950903220161283f);
    temp_imag = odd_data[2] * (-0.1950903220161283f) + odd_data[3] * 0.9807852804032304f;
    vec[2] = even_data[2] + temp_real;
    vec[3] = even_data[3] + temp_imag;
    vec[34] = even_data[2] - temp_real;
    vec[35] = even_data[3] - temp_imag;

    // k=2: W32^2 = cos(π/8) - j*sin(π/8) = 0.9238795325 - j*0.3826834324
    temp_real = odd_data[4] * 0.9238795325112867f - odd_data[5] * (-0.3826834323650898f);
    temp_imag = odd_data[4] * (-0.3826834323650898f) + odd_data[5] * 0.9238795325112867f;
    vec[4] = even_data[4] + temp_real;
    vec[5] = even_data[5] + temp_imag;
    vec[36] = even_data[4] - temp_real;
    vec[37] = even_data[5] - temp_imag;

    // k=3: W32^3 = cos(3π/16) - j*sin(3π/16) = 0.8314696123 - j*0.5555702330
    temp_real = odd_data[6] * 0.8314696123025452f - odd_data[7] * (-0.5555702330196022f);
    temp_imag = odd_data[6] * (-0.5555702330196022f) + odd_data[7] * 0.8314696123025452f;
    vec[6] = even_data[6] + temp_real;
    vec[7] = even_data[7] + temp_imag;
    vec[38] = even_data[6] - temp_real;
    vec[39] = even_data[7] - temp_imag;

    // k=4: W32^4 = cos(π/4) - j*sin(π/4) = 1/√2 - j/√2 = 0.7071067812 - j*0.7071067812
    temp_real = odd_data[8] * 0.7071067811865476f - odd_data[9] * (-0.7071067811865476f);
    temp_imag = odd_data[8] * (-0.7071067811865476f) + odd_data[9] * 0.7071067811865476f;
    vec[8] = even_data[8] + temp_real;
    vec[9] = even_data[9] + temp_imag;
    vec[40] = even_data[8] - temp_real;
    vec[41] = even_data[9] - temp_imag;

    // k=5: W32^5 = cos(5π/16) - j*sin(5π/16) = 0.5555702330 - j*0.8314696123
    temp_real = odd_data[10] * 0.5555702330196022f - odd_data[11] * (-0.8314696123025452f);
    temp_imag = odd_data[10] * (-0.8314696123025452f) + odd_data[11] * 0.5555702330196022f;
    vec[10] = even_data[10] + temp_real;
    vec[11] = even_data[11] + temp_imag;
    vec[42] = even_data[10] - temp_real;
    vec[43] = even_data[11] - temp_imag;

    // k=6: W32^6 = cos(3π/8) - j*sin(3π/8) = 0.3826834324 - j*0.9238795325
    temp_real = odd_data[12] * 0.3826834323650898f - odd_data[13] * (-0.9238795325112867f);
    temp_imag = odd_data[12] * (-0.9238795325112867f) + odd_data[13] * 0.3826834323650898f;
    vec[12] = even_data[12] + temp_real;
    vec[13] = even_data[13] + temp_imag;
    vec[44] = even_data[12] - temp_real;
    vec[45] = even_data[13] - temp_imag;

    // k=7: W32^7 = cos(7π/16) - j*sin(7π/16) = 0.1950903220 - j*0.9807852804
    temp_real = odd_data[14] * 0.1950903220161283f - odd_data[15] * (-0.9807852804032304f);
    temp_imag = odd_data[14] * (-0.9807852804032304f) + odd_data[15] * 0.1950903220161283f;
    vec[14] = even_data[14] + temp_real;
    vec[15] = even_data[15] + temp_imag;
    vec[46] = even_data[14] - temp_real;
    vec[47] = even_data[15] - temp_imag;

    // k=8: W32^8 = cos(π/2) - j*sin(π/2) = 0.0 - j*1.0
    temp_real = odd_data[16] * 0.0f - odd_data[17] * (-1.0f);
    temp_imag = odd_data[16] * (-1.0f) + odd_data[17] * 0.0f;
    vec[16] = even_data[16] + temp_real;
    vec[17] = even_data[17] + temp_imag;
    vec[48] = even_data[16] - temp_real;
    vec[49] = even_data[17] - temp_imag;

    // k=9: W32^9 = cos(9π/16) - j*sin(9π/16) = -0.1950903220 - j*0.9807852804
    temp_real = odd_data[18] * (-0.1950903220161283f) - odd_data[19] * (-0.9807852804032304f);
    temp_imag = odd_data[18] * (-0.9807852804032304f) + odd_data[19] * (-0.1950903220161283f);
    vec[18] = even_data[18] + temp_real;
    vec[19] = even_data[19] + temp_imag;
    vec[50] = even_data[18] - temp_real;
    vec[51] = even_data[19] - temp_imag;

    // k=10: W32^10 = cos(5π/8) - j*sin(5π/8) = -0.3826834324 - j*0.9238795325
    temp_real = odd_data[20] * (-0.3826834323650898f) - odd_data[21] * (-0.9238795325112867f);
    temp_imag = odd_data[20] * (-0.9238795325112867f) + odd_data[21] * (-0.3826834323650898f);
    vec[20] = even_data[20] + temp_real;
    vec[21] = even_data[21] + temp_imag;
    vec[52] = even_data[20] - temp_real;
    vec[53] = even_data[21] - temp_imag;

    // k=11: W32^11 = cos(11π/16) - j*sin(11π/16) = -0.5555702330 - j*0.8314696123
    temp_real = odd_data[22] * (-0.5555702330196022f) - odd_data[23] * (-0.8314696123025452f);
    temp_imag = odd_data[22] * (-0.8314696123025452f) + odd_data[23] * (-0.5555702330196022f);
    vec[22] = even_data[22] + temp_real;
    vec[23] = even_data[23] + temp_imag;
    vec[54] = even_data[22] - temp_real;
    vec[55] = even_data[23] - temp_imag;

    // k=12: W32^12 = cos(3π/4) - j*sin(3π/4) = -1/√2 - j/√2 = -0.7071067812 - j*0.7071067812
    temp_real = odd_data[24] * (-0.7071067811865476f) - odd_data[25] * (-0.7071067811865476f);
    temp_imag = odd_data[24] * (-0.7071067811865476f) + odd_data[25] * (-0.7071067811865476f);
    vec[24] = even_data[24] + temp_real;
    vec[25] = even_data[25] + temp_imag;
    vec[56] = even_data[24] - temp_real;
    vec[57] = even_data[25] - temp_imag;

    // k=13: W32^13 = cos(13π/16) - j*sin(13π/16) = -0.8314696123 - j*0.5555702330
    temp_real = odd_data[26] * (-0.8314696123025452f) - odd_data[27] * (-0.5555702330196022f);
    temp_imag = odd_data[26] * (-0.5555702330196022f) + odd_data[27] * (-0.8314696123025452f);
    vec[26] = even_data[26] + temp_real;
    vec[27] = even_data[27] + temp_imag;
    vec[58] = even_data[26] - temp_real;
    vec[59] = even_data[27] - temp_imag;

    // k=14: W32^14 = cos(7π/8) - j*sin(7π/8) = -0.9238795325 - j*0.3826834324
    temp_real = odd_data[28] * (-0.9238795325112867f) - odd_data[29] * (-0.3826834323650898f);
    temp_imag = odd_data[28] * (-0.3826834323650898f) + odd_data[29] * (-0.9238795325112867f);
    vec[28] = even_data[28] + temp_real;
    vec[29] = even_data[29] + temp_imag;
    vec[60] = even_data[28] - temp_real;
    vec[61] = even_data[29] - temp_imag;

    // k=15: W32^15 = cos(15π/16) - j*sin(15π/16) = -0.9807852804 - j*0.1950903220
    temp_real = odd_data[30] * (-0.9807852804032304f) - odd_data[31] * (-0.1950903220161283f);
    temp_imag = odd_data[30] * (-0.1950903220161283f) + odd_data[31] * (-0.9807852804032304f);
    vec[30] = even_data[30] + temp_real;
    vec[31] = even_data[31] + temp_imag;
    vec[62] = even_data[30] - temp_real;
    vec[63] = even_data[31] - temp_imag;
}

// 参考FFT实现用于验证
void fft_reference_dft(float *vec, int n) {
    float *temp = malloc(2 * n * sizeof(float));
    
    for (int k = 0; k < n; k++) {
        temp[2*k] = 0;      // 实部
        temp[2*k+1] = 0;    // 虚部
        
        for (int n_idx = 0; n_idx < n; n_idx++) {
            float angle = -2.0f * M_PI * k * n_idx / (float)n;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            temp[2*k] += vec[2*n_idx] * cos_val - vec[2*n_idx+1] * sin_val;
            temp[2*k+1] += vec[2*n_idx] * sin_val + vec[2*n_idx+1] * cos_val;
        }
    }
    
    // 复制结果回原数组
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
    printf("完全展开的32点FFT实现演示\n");
    printf("===========================\n\n");
    printf("特点:\n");
    printf("- 完全展开，无任何循环\n");
    printf("- 所有旋转因子预计算并硬编码\n");
    printf("- 基于已验证的FFT16函数\n");
    printf("- 最高性能，适合对速度要求极高的场合\n\n");
    
    // 测试1: 脉冲信号
    printf("测试1: 单位脉冲信号\n");
    printf("-------------------\n");
    float test1[64] = {0};
    float ref1[64] = {0};
    
    test1[0] = 1.0f;  // 脉冲
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
    
    // 测试2: 直流信号
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
    
    // 测试3: 复数指数信号
    printf("测试3: 复数指数信号 (频率bin=2)\n");
    printf("---------------------------------\n");
    float test3[64] = {0};
    float ref3[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        float angle = 2.0f * M_PI * 2 * i / 32.0f;
        test3[2*i] = cosf(angle);      // 实部
        test3[2*i+1] = sinf(angle);    // 虚部
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
    
    // 验证总结
    printf("验证总结:\n");
    printf("=========\n");
    printf("脉冲信号误差:      %.8f\n", error1);
    printf("直流信号误差:      %.8f\n", error2);
    printf("复数指数信号误差:  %.8f\n", error3);
    
    float max_error = fmaxf(fmaxf(error1, error2), error3);
    printf("最大总误差:        %.8f\n\n", max_error);
    
    if (max_error < 1e-4) {
        printf("✓ 完全展开的FFT32实现验证成功！\n");
        printf("  算法正确，性能最优，适合对速度要求极高的场合。\n");
        printf("\n性能特点:\n");
        printf("  ✓ 完全展开，无循环开销\n");
        printf("  ✓ 所有旋转因子预计算\n");
        printf("  ✓ 直接内存访问模式\n");
        printf("  ✓ 基于已验证的FFT16函数\n");
        printf("  ✓ 编译器友好，易于优化\n");
    } else if (max_error < 1e-3) {
        printf("⚠ FFT32实现基本正确，存在轻微数值误差。\n");
        printf("  当前误差: %.8f (在单精度浮点可接受范围内)\n", max_error);
    } else {
        printf("✗ FFT32实现需要进一步调试。\n");
        printf("  当前误差: %.8f\n", max_error);
    }
    
    return 0;
}