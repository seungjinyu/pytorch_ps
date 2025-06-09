// neon_abs.c
#include <arm_neon.h>
#include <stddef.h>

// NEON SIMD로 float32 배열의 절댓값 계산
void abs_neon(const float* input, float* output, size_t size) {
    size_t i;
    for (i = 0; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&input[i]);
        float32x4_t abs_v = vabsq_f32(v);
        vst1q_f32(&output[i], abs_v);
    }
    for (; i < size; ++i) {
        output[i] = input[i] > 0 ? input[i] : -input[i];
    }
}
