#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "kernel_cpu.h"  // auto-generated launch declarations

int main() {
    const int N = 1024;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Clean: no (uint64_t) casts needed
    vectorAdd_launch(d_a, d_b, d_c, N,
                     dim3(numBlocks), dim3(blockSize));

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            printf("FAIL at %d: got %f, expected %f\n", i, h_c[i], expected);
            if (++errors >= 10) break;
        }
    }
    if (errors == 0) {
        printf("PASS: all %d elements correct\n", N);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return errors > 0 ? 1 : 0;
}
