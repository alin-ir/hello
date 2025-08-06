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

/* 修正后的32点FFT实现 - 基于分治思想 */
static void fft32_corrected( float *vec )
{
    int i;
    float even_part[32];  // 16个复数
    float odd_part[32];   // 16个复数
    float temp_real, temp_imag;
    
    // 分离偶数和奇数索引的样本
    for (i = 0; i < 16; i++) {
        even_part[2*i] = vec[4*i];         // 偶数样本的实部
        even_part[2*i+1] = vec[4*i+1];     // 偶数样本的虚部
        odd_part[2*i] = vec[4*i+2];        // 奇数样本的实部
        odd_part[2*i+1] = vec[4*i+3];      // 奇数样本的虚部
    }
    
    // 递归调用16点FFT（这里我们使用简化版）
    // 实际实现中应该调用真正的16点FFT
    
    // 简化的16点FFT处理（仅用于演示）
    // 这里我们使用DFT来替代，虽然效率较低但结果正确
    float temp_even[32], temp_odd[32];
    
    // 对偶数部分做DFT
    for (int k = 0; k < 16; k++) {
        temp_even[2*k] = 0;     // 实部
        temp_even[2*k+1] = 0;   // 虚部
        for (int n = 0; n < 16; n++) {
            float angle = -2.0f * M_PI * k * n / 16.0f;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            temp_even[2*k] += even_part[2*n] * cos_val - even_part[2*n+1] * sin_val;
            temp_even[2*k+1] += even_part[2*n] * sin_val + even_part[2*n+1] * cos_val;
        }
    }
    
    // 对奇数部分做DFT
    for (int k = 0; k < 16; k++) {
        temp_odd[2*k] = 0;      // 实部
        temp_odd[2*k+1] = 0;    // 虚部
        for (int n = 0; n < 16; n++) {
            float angle = -2.0f * M_PI * k * n / 16.0f;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            temp_odd[2*k] += odd_part[2*n] * cos_val - odd_part[2*n+1] * sin_val;
            temp_odd[2*k+1] += odd_part[2*n] * sin_val + odd_part[2*n+1] * cos_val;
        }
    }
    
    // 合并结果，应用旋转因子
    for (i = 0; i < 16; i++) {
        float angle = -2.0f * M_PI * i / 32.0f;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        // 应用旋转因子到奇数部分
        temp_real = temp_odd[2*i] * cos_val - temp_odd[2*i+1] * sin_val;
        temp_imag = temp_odd[2*i] * sin_val + temp_odd[2*i+1] * cos_val;
        
        // 蝶形运算
        vec[2*i] = temp_even[2*i] + temp_real;
        vec[2*i+1] = temp_even[2*i+1] + temp_imag;
        vec[2*(i+16)] = temp_even[2*i] - temp_real;
        vec[2*(i+16)+1] = temp_even[2*i+1] - temp_imag;
    }
}

// 标准参考FFT实现（DFT）
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

void print_comparison(const char* test_name, float* test_result, float* ref_result, int show_points) {
    printf("\n%s 结果对比:\n", test_name);
    printf("========================================\n");
    for (int i = 0; i < show_points; i++) {
        printf("  [%2d] 测试: %8.4f + %8.4fi | 参考: %8.4f + %8.4fi\n", 
               i, test_result[2*i], test_result[2*i+1], 
               ref_result[2*i], ref_result[2*i+1]);
    }
    float error = calculate_error(test_result, ref_result, 32);
    printf("最大误差: %.6f\n\n", error);
}

int main() {
    printf("32点FFT正确性验证程序\n");
    printf("========================\n");
    printf("本程序验证修正后的32点FFT实现的正确性\n\n");
    
    // 测试1: 单位脉冲
    printf("测试1: 单位脉冲信号\n");
    printf("-------------------\n");
    
    float test1[64] = {0};  // 32个复数样本，交错存储
    float ref1[64] = {0};
    
    test1[0] = 1.0f;  // 设置脉冲
    ref1[0] = 1.0f;
    
    printf("输入信号: δ[0] = 1, 其余为0\n");
    printf("理论结果: 所有FFT输出应该都是1\n");
    
    fft32_corrected(test1);
    fft_reference_dft(ref1, 32);
    
    print_comparison("单位脉冲", test1, ref1, 8);
    
    // 测试2: 直流信号
    printf("测试2: 直流信号\n");
    printf("---------------\n");
    
    float test2[64] = {0};
    float ref2[64] = {0};
    
    for (int i = 0; i < 32; i++) {
        test2[2*i] = 1.0f;  // 所有样本实部为1
        ref2[2*i] = 1.0f;
    }
    
    printf("输入信号: 所有样本实部 = 1\n");
    printf("理论结果: FFT[0] = 32, 其余为0\n");
    
    fft32_corrected(test2);
    fft_reference_dft(ref2, 32);
    
    print_comparison("直流信号", test2, ref2, 8);
    
    // 测试3: 复数指数信号
    printf("测试3: 复数指数信号 e^(j*2πk/32), k=3\n");
    printf("-------------------------------------\n");
    
    float test3[64] = {0};
    float ref3[64] = {0};
    
    int freq_bin = 3;  // 频率bin
    for (int i = 0; i < 32; i++) {
        float angle = 2.0f * M_PI * freq_bin * i / 32.0f;
        test3[2*i] = cosf(angle);      // 实部
        test3[2*i+1] = sinf(angle);    // 虚部
        ref3[2*i] = cosf(angle);
        ref3[2*i+1] = sinf(angle);
    }
    
    printf("输入信号: e^(j*2π*3*n/32)\n");
    printf("理论结果: FFT[3] = 32, 其余为0\n");
    
    fft32_corrected(test3);
    fft_reference_dft(ref3, 32);
    
    print_comparison("复数指数信号", test3, ref3, 8);
    
    // 测试4: 随机信号
    printf("测试4: 随机复数信号\n");
    printf("-------------------\n");
    
    float test4[64];
    float ref4[64];
    
    // 生成随机信号
    srand(12345);  // 固定种子以便结果可重现
    for (int i = 0; i < 32; i++) {
        test4[2*i] = (float)rand() / RAND_MAX - 0.5f;      // 实部
        test4[2*i+1] = (float)rand() / RAND_MAX - 0.5f;    // 虚部
        ref4[2*i] = test4[2*i];
        ref4[2*i+1] = test4[2*i+1];
    }
    
    printf("输入信号: 随机复数信号\n");
    printf("验证目的: 测试算法的通用正确性\n");
    
    fft32_corrected(test4);
    fft_reference_dft(ref4, 32);
    
    print_comparison("随机信号", test4, ref4, 8);
    
    // 计算总体误差统计
    float error1 = calculate_error(test1, ref1, 32);
    float error2 = calculate_error(test2, ref2, 32);
    float error3 = calculate_error(test3, ref3, 32);
    float error4 = calculate_error(test4, ref4, 32);
    float max_error = fmaxf(fmaxf(error1, error2), fmaxf(error3, error4));
    
    printf("验证总结:\n");
    printf("=========\n");
    printf("单位脉冲信号误差:    %.6f\n", error1);
    printf("直流信号误差:        %.6f\n", error2);
    printf("复数指数信号误差:    %.6f\n", error3);
    printf("随机信号误差:        %.6f\n", error4);
    printf("最大误差:            %.6f\n", max_error);
    
    if (max_error < 1e-5) {
        printf("\n✓ FFT32函数验证通过！所有测试误差在可接受范围内。\n");
        printf("  算法实现正确，可以用于实际应用。\n");
    } else if (max_error < 1e-3) {
        printf("\n⚠ FFT32函数基本正确，但存在一定的数值误差。\n");
        printf("  建议检查实现细节或使用更高精度。\n");
    } else {
        printf("\n✗ FFT32函数验证失败，误差过大。\n");
        printf("  需要重新检查算法实现。\n");
    }
    
    return 0;
}