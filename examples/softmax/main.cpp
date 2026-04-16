#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include "kernel_cpu.h"

int main() {
    const int rows = 2;
    const int cols = 8;
    float h_input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,  // row 0
        5.0f, 6.0f, 7.0f, 8.0f, 5.0f, 6.0f, 7.0f, 8.0f,  // row 1
    };
    float h_output[rows * cols];

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // One block per row, 8 threads per block (= cols)
    softmax_launch(d_input, d_output, rows, cols,
                   dim3(rows), dim3(cols));

    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: each row should sum to 1.0
    int errors = 0;
    for (int i = 0; i < rows; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_sum += h_output[i * cols + j];
            if (h_output[i * cols + j] < 0.0f || h_output[i * cols + j] > 1.0f) {
                printf("FAIL: row %d col %d = %f (out of [0,1])\n", i, j, h_output[i * cols + j]);
                errors++;
            }
        }
        if (fabsf(row_sum - 1.0f) > 0.001f) {
            printf("FAIL: row %d sum = %f (expected 1.0)\n", i, row_sum);
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS: softmax correct (%dx%d)\n", rows, cols);
        printf("  row 0: ");
        for (int j = 0; j < cols; j++) printf("%.4f ", h_output[j]);
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return errors > 0 ? 1 : 0;
}
