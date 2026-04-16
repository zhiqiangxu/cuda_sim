// Tests for memory error detection
#include <cstdio>
#include "cuda_sim/cuda_runtime_api.h"

void test_leak() {
    printf("=== Test: memory leak ===\n");
    void* ptr;
    cudaMalloc(&ptr, 1024);
    // Intentionally NOT calling cudaFree(ptr)
    // Leak should be reported at program exit
    printf("(leak report will appear at exit)\n\n");
}

void test_double_free() {
    printf("=== Test: double free ===\n");
    void* ptr;
    cudaMalloc(&ptr, 512);
    cudaFree(ptr);
    cudaError_t err = cudaFree(ptr);  // should report error
    printf("double free returned: %s\n\n", cudaGetErrorString(err));
}

void test_invalid_free() {
    printf("=== Test: invalid pointer free ===\n");
    int stack_var = 42;
    cudaError_t err = cudaFree(&stack_var);  // never allocated by cudaMalloc
    printf("invalid free returned: %s\n\n", cudaGetErrorString(err));
}

void test_normal() {
    printf("=== Test: normal usage (no errors) ===\n");
    void* ptr;
    cudaMalloc(&ptr, 256);
    cudaMemset(ptr, 0, 256);
    cudaFree(ptr);
    printf("OK\n\n");
}

int main() {
    test_normal();
    test_double_free();
    test_invalid_free();
    test_leak();  // leak reported at exit
    return 0;
}
