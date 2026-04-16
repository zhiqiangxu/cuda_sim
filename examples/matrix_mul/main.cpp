#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

void matMul_launch(const void*, const void*, const void*, uint32_t,
                   dim3, dim3);

int main() {
    // Small matrix for testing (TILE_SIZE=16, so use 16x16)
    const int N = 16;
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // A = identity, B = sequential values
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (i == j) ? 1.0f : 0.0f;
            h_B[i * N + j] = (float)(i * N + j);
        }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // 2D grid: one block of 16x16 threads for a 16x16 matrix
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    matMul_launch(d_A, d_B, d_C, N, grid, block);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Identity * B = B
    int errors = 0;
    for (int i = 0; i < N * N; i++) {
        if (fabsf(h_C[i] - h_B[i]) > 0.001f) {
            printf("FAIL at [%d][%d]: got %f, expected %f\n",
                   i / N, i % N, h_C[i], h_B[i]);
            if (++errors >= 10) break;
        }
    }
    if (errors == 0) {
        printf("PASS: matrix multiply correct (%dx%d)\n", N, N);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return errors > 0 ? 1 : 0;
}
