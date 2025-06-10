#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void matrixMulGPU(int *a, int *b, int *c) {
int val = 0;
int row = threadIdx.x + blockIdx.x * blockDim.x;
int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < N && col < N) {
        for (int k = 0; k < N; ++k)
            val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
  }
}

void matrixMulCPU(int *a, int *b, int *c) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            int val = 0;
            for (int k = 0; k < N; ++k)
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}


int main() {
  int *a, *b, *c_cpu *c_gpu;
  int size = N * N * sizeof(int);

// Allocate memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);

  // Initialize memory
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col) {
      a[row * N + col] = row;
      b[row * N + col] = col + 2;
      c_cpu[row * N + col] = 0;
      c_gpu[row * N + col] = 0;
    }

dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                   (N + threads_per_block.y - 1) / threads_per_block.y, 1);

cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu);
    cudaEventRecord(stop);

    cudaCheckError();
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

 // Call the CPU version to check our work
    cudaEvent_t cpuStart, cpuStop;
    cudaEventCreate(&cpuStart);
    cudaEventCreate(&cpuStop);

    cudaEventRecord(cpuStart);
    matrixMulCPU(a, b, c_cpu);
    cudaEventRecord(cpuStop);
    cudaEventSynchronize(cpuStop);

    float cpuMilliseconds = 0;
    cudaEventElapsedTime(&cpuMilliseconds, cpuStart, cpuStop);
    double cpuTime = cpuMilliseconds / 1000.0;

// Compare the results
    bool error = false;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (c_cpu[row * N + col] != c_gpu[row * N + col]) {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }

    if (!error)
        printf("Success!\n");

float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double gpuTime = milliseconds / 1000.0;
    
    printf("CPU Matrix Multiplication Time: %f seconds\n", cpuTime);
    printf("GPU Matrix Multiplication Time: %f seconds\n", gpuTime);

    // Calculate and print speedup
    calculateSpeedup(cpuTime, gpuTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(cpuStart);
    cudaEventDestroy(cpuStop);

    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);

    return 0;

}

