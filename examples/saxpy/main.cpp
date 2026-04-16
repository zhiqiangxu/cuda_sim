#include <cstdio>
#include <cstdint>
#include <cmath>
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

// Use the void* convenience overload for pointer params
// float param stays as float
void saxpy_launch(float, const void*, const void*, uint32_t,
                  cuda_sim::dim3, cuda_sim::dim3);

int main() {
    const int N = 512;
    float h_x[N], h_y[N];

    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(i * 2);
    }

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    float a = 2.0f;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    saxpy_launch(a, d_x, d_y, N,
                 cuda_sim::dim3(numBlocks), cuda_sim::dim3(blockSize));

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = a * (float)i + (float)(i * 2);
        if (fabsf(h_y[i] - expected) > 0.001f) {
            printf("FAIL saxpy at %d: got %f, expected %f\n", i, h_y[i], expected);
            if (++errors >= 5) break;
        }
    }
    if (errors == 0) {
        printf("PASS: saxpy correct (%d elements)\n", N);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return errors > 0 ? 1 : 0;
}
