#include <cstdio>
#include <cstdint>
#include <cstring>
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

extern "C"
void histogram_launch(uint64_t, uint64_t, uint32_t,
                      cuda_sim::dim3, cuda_sim::dim3);

int main() {
    const int N = 1000;
    const int NUM_BINS = 10;

    // Each element is a value 0..9
    unsigned int h_data[N];
    unsigned int h_bins[NUM_BINS];
    unsigned int expected[NUM_BINS] = {};

    for (int i = 0; i < N; i++) {
        h_data[i] = i % NUM_BINS;
        expected[h_data[i]]++;
    }

    unsigned int *d_data, *d_bins;
    cudaMalloc((void**)&d_data, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bins, NUM_BINS * sizeof(unsigned int));

    cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    histogram_launch((uint64_t)d_data, (uint64_t)d_bins, N,
                     cuda_sim::dim3(numBlocks), cuda_sim::dim3(blockSize));

    cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_bins[i] != expected[i]) {
            printf("FAIL bin[%d]: got %u, expected %u\n", i, h_bins[i], expected[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS: histogram correct, all %d bins match\n", NUM_BINS);
    }

    cudaFree(d_data);
    cudaFree(d_bins);
    return errors > 0 ? 1 : 0;
}
