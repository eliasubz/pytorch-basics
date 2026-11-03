#include <cuda_runtime.h>
#include <stdio.h>

void vecMulHost (float* aH, float* bH, float* cH, int n){
    for (int i = 0; i < n; i++){
        cH[i] = aH[i] * bH[i];
    }
}


__global__ vecMulKernel (float* A_H, float* B_H, float* C_H, int n){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C_H[i] = A_H[i] * B_H[i];
    }
}


void vecMulDevice(float* aH, float* bH, float* cH, int n) {
    int size = n * size(float)
    float *A_d, *B_d, *C_d

    // Allocate memory on the device
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // sent data to the device
    cudaMemcpy(A_d, aH, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, bH, size, cudaMemcpyHostToDevice);

    // launch kernel
    vecMulKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy C_d from device to host
    cudaMemcpy(ch, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}



int main() {
    int n = 123397730;  // Example array size
    float *A, *B, *C;

    // Allocate host memory
    A = (float*)malloc(n * sizeof(float));
    B = (float*)malloc(n * sizeof(float));
    C = (float*)malloc(n * sizeof(float));

    // Initialize A and B with values
    for (int i = 0; i < n; i++) {
        A[i] = i;  // Just an example value
        B[i] = i;  // Just an example value
    }

    // Timing for vecMulHost
    clock_t start_host = clock();
    vecMulHost(A, B, C, n);
    clock_t end_host = clock();
    double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC;

    // Timing for vecMulDevice
    clock_t start_device = clock();
    vecMulDevice(A, B, C, n);
    clock_t end_device = clock();
    double time_device = (double)(end_device - start_device) / CLOCKS_PER_SEC;

    printf("Time taken for vecMulHost: %f seconds\n", time_host);
    printf("Time taken for vecMulDevice: %f seconds\n", time_device);

    return 0;
}