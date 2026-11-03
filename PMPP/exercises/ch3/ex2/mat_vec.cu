#include <cuda_runtime.h>
#include <stdio.h>

void vetMatMulHost(float* aH, float* bH, float* cH, int n) {
    // Lets assume aH is the matrix and bH is the vector
    // (m,n) @ (n,1) -> (m,1) but we have m = n
    for (int i = 0; i < n; i++) {       // row
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += aH[i * n + k] * bH[k];
        }
        cH[i] = sum;
    }
}

__global__ void vetMatMulKernel (float* A, float* B, float* C, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float out = 0;
    for (int i = 0; i < n; i++){
        if (i < n) {
            out += A[row * n + i] * B[i];
        }
    }
    C[row] = out;
}

void vetMatMulDevice(float* aH, float* bH, float* cH, int n) {
    int matSize = n*n*sizeof(float);
    int vecSize = n*sizeof(float);
    float *ad, *bd, *cd;

    // allocate memory on device
    cudaMalloc((void**)&ad, matSize);
    cudaMalloc((void**)&bd, vecSize);

    // sent data from host to device
    cudaMemcpy(ad, aH, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, bH, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, cH, vecSize, cudaMemcpyHostToDevice);

    // launch kernel
    vecMatMulKernel<<<ceil(n/256), 256>>>(ad, bd, cd, n);

    cudaMemcpy(cH, cd, vecSize, cudaMemcpyDeviceToHost);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}