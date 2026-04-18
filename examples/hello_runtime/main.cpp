// saxpy on CPU via cuda_sim (runtime JIT mode)
// No pre-compiled PTX needed — compiles kernel source at runtime.
// Requires: nvcc (for NVRTC), python3, g++ at runtime.
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

const char* kernel_src = R"(
extern "C" __global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
)";

int main() {
    const int N = 8;
    float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float y[] = {10, 20, 30, 40, 50, 60, 70, 80};

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Compile kernel source → PTX → JIT
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernel_src, "saxpy.cu", 0, NULL, NULL);
    nvrtcCompileProgram(prog, 0, NULL);
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    CUfunction func;
    cuModuleGetFunction(&func, module, "saxpy");
    delete[] ptx;

    // Launch: y = 2*x + y
    int n = N;
    float a = 2.0f;
    float* px = d_x;
    float* py = d_y;
    void* args[] = {&n, &a, &px, &py};
    cuLaunchKernel(func, 1,1,1, N,1,1, 0, NULL, args, NULL);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < N; i++) printf("%.0f ", y[i]);
    printf("\n");
    // Output: 12 24 36 48 60 72 84 96
}
