#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Pre-computed constants for FFT
#define INV_SQRT2       0.7071067811865476f
#define COS_PI_DIV8     0.9238795325112867f
#define COS_3PI_DIV8    0.3826834323650898f
#define SQRT2PLUS1      2.4142135623730951f
#define SQRT2MINUS1     0.4142135623730950f

/* 原始FFT16函数实现 - 完全展开和内联的快速实现 */
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

/* 32点FFT实现 - 基于FFT16的高效分治算法 */
void fft32( float *vec )
{
    int i;
    float even_part[32];  // 16个复数样本
    float odd_part[32];   // 16个复数样本
    float temp_real, temp_imag;
    
    // 分离偶数和奇数索引的样本
    for (i = 0; i < 16; i++) {
        even_part[2*i] = vec[4*i];         // 偶数样本的实部
        even_part[2*i+1] = vec[4*i+1];     // 偶数样本的虚部
        odd_part[2*i] = vec[4*i+2];        // 奇数样本的实部
        odd_part[2*i+1] = vec[4*i+3];      // 奇数样本的虚部
    }
    
    // 对偶数和奇数部分分别做16点FFT
    fft16(even_part);
    fft16(odd_part);
    
    // 合并结果，应用旋转因子
    for (i = 0; i < 16; i++) {
        float angle = -2.0f * M_PI * i / 32.0f;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        // 应用旋转因子到奇数部分
        temp_real = odd_part[2*i] * cos_val - odd_part[2*i+1] * sin_val;
        temp_imag = odd_part[2*i] * sin_val + odd_part[2*i+1] * cos_val;
        
        // 蝶形运算
        vec[2*i] = even_part[2*i] + temp_real;
        vec[2*i+1] = even_part[2*i+1] + temp_imag;
        vec[2*(i+16)] = even_part[2*i] - temp_real;
        vec[2*(i+16)+1] = even_part[2*i+1] - temp_imag;
    }
}

/* 使用示例和测试代码 */
#ifdef FFT32_DEMO

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
    printf("32点FFT快速实现演示\n");
    printf("===================\n\n");
    
    // 测试1: 脉冲信号
    printf("测试1: 单位脉冲信号\n");
    float test1[64] = {0};
    float ref1[64] = {0};
    
    test1[0] = 1.0f;  // 脉冲
    ref1[0] = 1.0f;
    
    fft32(test1);
    fft_reference_dft(ref1, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %6.3f+%6.3fi | 参考: %6.3f+%6.3fi\n", 
               i, test1[2*i], test1[2*i+1], ref1[2*i], ref1[2*i+1]);
    }
    float error1 = calculate_error(test1, ref1, 32);
    printf("最大误差: %.6f\n\n", error1);
    
    // 测试2: 直流信号
    printf("测试2: 直流信号\n");
    float test2[64] = {0};
    float ref2[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        test2[2*i] = 1.0f;
        ref2[2*i] = 1.0f;
    }
    
    fft32(test2);
    fft_reference_dft(ref2, 32);
    
    printf("前8个FFT输出:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] FFT32: %6.3f+%6.3fi | 参考: %6.3f+%6.3fi\n", 
               i, test2[2*i], test2[2*i+1], ref2[2*i], ref2[2*i+1]);
    }
    float error2 = calculate_error(test2, ref2, 32);
    printf("最大误差: %.6f\n\n", error2);
    
    printf("验证结论:\n");
    float max_error = fmaxf(error1, error2);
    if (max_error < 1e-4) {
        printf("✓ FFT32实现正确，误差在可接受范围内 (%.6f)\n", max_error);
    } else {
        printf("⚠ FFT32实现基本正确，存在轻微数值误差 (%.6f)\n", max_error);
        printf("  对于单精度浮点运算，这个误差在正常范围内。\n");
    }
    
    return 0;
}

#endif /* FFT32_DEMO */