extern "C"
__global__ void relu(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = fmaxf(data[i], 0.0f);
    }
}

extern "C"
__global__ void leaky_relu(float* data, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = (x > 0.0f) ? x : alpha * x;
    }
}

extern "C"
__global__ void float_to_int(const float* input, int* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __float2int_rn(input[i]);
    }
}
