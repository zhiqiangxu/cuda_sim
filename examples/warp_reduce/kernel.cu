extern "C"
__global__ void warpReduce(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? input[i] : 0.0f;

    // Warp-level reduction using shfl_down
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    // Lane 0 of each warp writes the result
    if (threadIdx.x % 32 == 0) {
        output[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = val;
    }
}
