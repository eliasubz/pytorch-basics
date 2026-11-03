#include <metal_stdlib>
using namespace metal;

kernel void matVecMulKernel(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* C [[buffer(2)]],
                            constant uint& size [[buffer(3)]],
                            uint row [[thread_position_in_grid]]) {
    float sum = 0;
    if (row < size) {
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i];
        }
        C[row] = sum;
    }
}