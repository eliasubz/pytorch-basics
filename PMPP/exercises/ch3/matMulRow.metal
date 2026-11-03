#include <metal_stdlib>
using namespace metal;

kernel void matrixMulRowKernel(device const float* M [[buffer(0)]],
                               device const float* N [[buffer(1)]],
                               device float* P [[buffer(2)]],
                               constant uint& size [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
    // metal is col-major -> col * height + row
    if (row < size) {
        for (uint col=0; col < size; col++){
            float sum = 0;
            for (uint j=0;j<size;j++){
                sum += M[row * size + j] * N[j*size+col];
            }
            P[row * size + col] = sum;
        }
    }
}






// __global__
// void matrixMulRowKernel(float* M, float* N, float* P, int size){
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < size){
//         //do this for every element in the row:
//         for (int col=0; col<size; ++col){
//             float sum = 0;
//             for (int j=0; j<size; ++j){
//                 sum += M[row * size + j] * N[j * size + col];
//             }
//             P[row * size + col] = sum;
//         }
//     }
// }