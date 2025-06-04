// neon_add.c
#include <arm_neon.h>
#include <stdio.h>

int main() {
    float32x4_t a = {1.0, 2.0, 3.0, 4.0};
    float32x4_t b = {10.0, 20.0, 30.0, 40.0};
    float32x4_t result = vaddq_f32(a, b);

    float res_array[4];
    vst1q_f32(res_array, result);

    printf("Result: ");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ", res_array[i]);
    }
    printf("\n");
    return 0;
}
