#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include "kernel_cpu.h"

int main() {
    const int N = 8;
    float h_data[] = {-3, -2, -1, 0, 1, 2, 3, 4};
    float h_result[N];

    float *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float));

    // Test relu
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    relu_launch(d_data, N, dim3(1), dim3(N));
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("relu:       ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_result[i]);
    printf("\n");

    float expected_relu[] = {0, 0, 0, 0, 1, 2, 3, 4};
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_result[i] != expected_relu[i]) errors++;

    // Test leaky_relu
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    leaky_relu_launch(d_data, 0.1f, N, dim3(1), dim3(N));
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("leaky_relu: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_result[i]);
    printf("\n");

    float expected_leaky[] = {-0.3f, -0.2f, -0.1f, 0, 1, 2, 3, 4};
    for (int i = 0; i < N; i++)
        if (fabsf(h_result[i] - expected_leaky[i]) > 0.01f) errors++;

    // Test float_to_int
    float h_floats[] = {-1.7f, -0.5f, 0.0f, 0.5f, 1.3f, 2.8f, 3.5f, 4.1f};
    int h_ints[N];
    float *d_floats;
    int *d_ints;
    cudaMalloc((void**)&d_floats, N * sizeof(float));
    cudaMalloc((void**)&d_ints, N * sizeof(int));

    cudaMemcpy(d_floats, h_floats, N * sizeof(float), cudaMemcpyHostToDevice);
    float_to_int_launch(d_floats, d_ints, N, dim3(1), dim3(N));
    cudaMemcpy(h_ints, d_ints, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("f2i_rn:     ");
    for (int i = 0; i < N; i++) printf("%d ", h_ints[i]);
    printf("\n");

    // __float2int_rn uses round-to-nearest-even (banker's rounding)
    // -1.7→-2, -0.5→0, 0→0, 0.5→0, 1.3→1, 2.8→3, 3.5→4, 4.1→4
    int expected_ints[] = {-2, 0, 0, 0, 1, 3, 4, 4};
    for (int i = 0; i < N; i++)
        if (h_ints[i] != expected_ints[i]) errors++;

    printf(errors == 0 ? "PASS: all activation kernels correct\n" : "FAIL\n");

    cudaFree(d_data);
    cudaFree(d_floats);
    cudaFree(d_ints);
    return errors > 0 ? 1 : 0;
}
