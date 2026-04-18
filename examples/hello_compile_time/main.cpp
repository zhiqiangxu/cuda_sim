// saxpy on CPU via cuda_sim (compile-time mode)
#include <cstdio>
#include <cuda_runtime.h>
#include "kernel_cpu.h"

int main() {
    const int N = 8;
    float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float y[] = {10, 20, 30, 40, 50, 60, 70, 80};

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy_launch(N, 2.0f, d_x, d_y, dim3(1), dim3(N));  // y = 2*x + y

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < N; i++) printf("%.0f ", y[i]);
    printf("\n");
    // Output: 12 24 36 48 60 72 84 96
}
