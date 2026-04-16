#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

void warpReduce_launch(const void*, const void*, uint32_t,
                       dim3, dim3);

int main() {
    // 32 elements = 1 warp, simplest test
    const int N = 32;
    const int blockSize = 32;
    const int numBlocks = 1;
    const int numWarps = blockSize / 32;

    float h_input[N];
    float h_output[numBlocks * numWarps];

    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i + 1);  // 1, 2, 3, ..., 32
        expected += h_input[i];         // sum = 528
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * numWarps * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, numBlocks * numWarps * sizeof(float));

    warpReduce_launch(d_input, d_output, N,
                      dim3(numBlocks), dim3(blockSize));

    cudaMemcpy(h_output, d_output, numBlocks * numWarps * sizeof(float),
               cudaMemcpyDeviceToHost);

    float total = h_output[0];

    if (fabsf(total - expected) < 0.001f) {
        printf("PASS: warp reduction correct, sum = %.1f (expected %.1f)\n",
               total, expected);
    } else {
        printf("FAIL: sum = %.1f, expected %.1f\n", total, expected);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return (fabsf(total - expected) < 0.001f) ? 0 : 1;
}
