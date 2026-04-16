#include <cstdio>
#include <cstdint>
#include <cmath>
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

void reduce_launch(const void*, const void*, uint32_t,
                   cuda_sim::dim3, cuda_sim::dim3);

int main() {
    // Use small N for testing (blockSize threads per block)
    const int N = 64;
    const int blockSize = 32;  // Keep small — each block spawns real threads
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float h_input[N];
    float h_output[numBlocks];
    float expected_sum = 0.0f;

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple: all 1s, sum should be N
        expected_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_launch(d_input, d_output, N,
                  cuda_sim::dim3(numBlocks), cuda_sim::dim3(blockSize));

    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum partial results from each block
    float total = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        total += h_output[i];
    }

    if (fabsf(total - expected_sum) < 0.001f) {
        printf("PASS: reduction correct, sum = %.1f (expected %.1f)\n", total, expected_sum);
    } else {
        printf("FAIL: sum = %.1f, expected %.1f\n", total, expected_sum);
        for (int i = 0; i < numBlocks; i++) {
            printf("  block[%d] = %.1f\n", i, h_output[i]);
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return (fabsf(total - expected_sum) < 0.001f) ? 0 : 1;
}
