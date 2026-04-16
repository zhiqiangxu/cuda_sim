extern "C"
__global__ void histogram(const unsigned int* data, unsigned int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bins[data[i]], 1);
    }
}
