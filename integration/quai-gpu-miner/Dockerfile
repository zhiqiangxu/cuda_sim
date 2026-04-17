# Use the NVIDIA CUDA image that's already cached locally
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies (CUDA is already in the base image)
RUN apt update && apt install -y \
    build-essential cmake git python3 perl \
    mesa-common-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Fix Hunter/Boost build issues on Ubuntu 22.04
ENV CXXFLAGS="-Wno-nonnull -Wno-error -Wno-unused-function -DPTHREAD_STACK_MIN=16384"

WORKDIR /build

# Copy cuda_sim
COPY . /build/cuda_sim

# Clone quai-gpu-miner and init submodules
RUN git clone --depth 1 https://github.com/dominant-strategies/quai-gpu-miner.git \
    && cd quai-gpu-miner \
    && git submodule update --init --recursive

# Apply cuda_sim integration patches
RUN cp /build/cuda_sim/integration/quai-gpu-miner/patches/CMakeLists.cuda_sim.txt \
       /build/quai-gpu-miner/libethash-cuda/CMakeLists.txt \
    && cp /build/cuda_sim/integration/quai-gpu-miner/patches/CUDAMiner_cuda_sim.cpp \
       /build/quai-gpu-miner/libethash-cuda/CUDAMiner_cuda_sim.cpp \
    && cp /build/cuda_sim/integration/quai-gpu-miner/patches/test_verify.cpp \
       /build/quai-gpu-miner/libethash-cuda/test_verify.cpp \
    && cp /build/cuda_sim/integration/quai-gpu-miner/patches/test_jit_progpow.cpp \
       /build/quai-gpu-miner/libethash-cuda/test_jit_progpow.cpp

# Build quai-gpu-miner + verification tests
RUN cd /build/quai-gpu-miner \
    && mkdir build && cd build \
    && cmake .. \
        -DETHASHCUDA=ON \
        -DETHASHCL=OFF \
        -DCUDA_SIM_ROOT=/build/cuda_sim \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . -- -j$(nproc)

# Run verification tests
RUN cd /build/quai-gpu-miner/build \
    && echo "=== Running verification tests ===" \
    && ./libethash-cuda/test_verify \
    && echo "" \
    && echo "=== Running JIT ProgPow test ===" \
    && rm -rf /tmp/cuda_sim_jit \
    && CUDA_SIM_ROOT=/build/cuda_sim ./libethash-cuda/test_jit_progpow ; \
    echo "JIT test exit code: $?" ; \
    echo "=== JIT cache contents ===" ; \
    ls -la /tmp/cuda_sim_jit/ 2>/dev/null ; \
    echo "=== ptx2cpp output ===" ; \
    cat /tmp/cuda_sim_jit/*_cpu.cpp 2>/dev/null | head -50 ; \
    echo "=== g++ errors ===" ; \
    cd /tmp/cuda_sim_jit && g++ -std=c++17 -O2 -fPIC -shared \
      -I/build/cuda_sim/include/compat -I/build/cuda_sim/include \
      *_cpu.cpp -o /dev/null 2>&1 | head -30 ; \
    true
