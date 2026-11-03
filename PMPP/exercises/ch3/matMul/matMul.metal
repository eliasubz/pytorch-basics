#include <metal_stdlib>
using namespace metal;

kernel void matMulKernel(device const float* x [[buffer(0)]],
                   device const float* y [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint& m [[buffer(3)]],
                   constant uint& n [[buffer(4)]],
                   constant uint& o [[buffer(5)]],
                   uint i [[thread_position_in_grid]]) {
    
    // this is an untiled matrix Multiplication kernel
    // QUestion how do I solve these issues with blocks and grids (hieght width)
    // we have M: (mxn); N(n x o) and A(rows, cols)
    uint out_col = i % o; // len(o) is number of cols of c -> length of row
    uint out_row = i / o; // divide by len(row) gives in which row we are

    if (out_row < m && out_col < o) {
        float sum = 0;
        for (uint i = 0; i < n; i++) {
            // add up vector product of row in x and col in y
            sum += x[out_row * n + i] * y[i * o + out_col];
        }
        c[out_row * o + out_col] = sum;
    }
}