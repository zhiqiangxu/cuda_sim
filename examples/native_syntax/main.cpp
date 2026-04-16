// This file uses real CUDA <<<>>> syntax.
// cuda_sim_add_sources() auto-preprocesses it before compilation.
#include <cstdio>
#include <cuda_runtime.h>
#include "kernel_cpu.h"

int main() {
    const int N = 256;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 10);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Real CUDA syntax — auto-preprocessed to vectorAdd_launch(...)
    vectorAdd<<<1, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("FAIL at %d\n", i);
            errors++;
        }
    }
    printf(errors == 0 ? "PASS: native <<<>>> syntax works\n" : "FAIL\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return errors > 0 ? 1 : 0;
}
