extern "C"
__global__ void softmax(const float* input, float* output, int rows, int cols) {
    __shared__ float sdata[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Step 1: find max in row (for numerical stability)
    float max_val = -1e30f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float val = input[row * cols + j];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Parallel reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Step 2: compute exp(x - max) and sum
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float val = expf(input[row * cols + j] - max_val);
        output[row * cols + j] = val;
        sum += val;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // Step 3: normalize
    for (int j = tid; j < cols; j += blockDim.x) {
        output[row * cols + j] /= sum;
    }
}
