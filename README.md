# cuda_sim

[![CI](https://github.com/zhiqiangxu/cuda_sim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhiqiangxu/cuda_sim/actions/workflows/ci.yml)

Run CUDA kernels on CPU without GPU. Supports both compile-time PTX translation and runtime JIT (NVRTC + Driver API).

## How it works

**Mode 1: Compile-time** (static kernels)
```
kernel.cu → nvcc -ptx → kernel.ptx → ptx2cpp.py → kernel_cpu.cpp → g++ → CPU binary
```

**Mode 2: Runtime JIT** (dynamic kernels, e.g. ProgPow mining)
```
Program → nvrtcCompileProgram(src) → PTX → cuModuleLoadDataEx → ptx2cpp.py → g++ → .so → dlopen
       → cuLaunchKernel(func, grid, block, void** args) → execute on CPU
```

1. **nvcc** compiles `.cu` to PTX (CUDA's intermediate representation)
2. **ptx2cpp.py** translates each PTX instruction to C++ (64+ instructions supported, including `.func` calls)
3. **g++** compiles to native binary or shared library

No GPU needed at runtime. nvcc runs inside Docker for cross-platform support.

## Quick start

```bash
# Generate PTX using Docker (one-time image pull ~3.5GB)
docker run --rm -v $(pwd)/examples/vector_add:/src \
    nvidia/cuda:12.6.0-devel-ubuntu22.04 \
    nvcc -ptx /src/kernel.cu -o /src/kernel.ptx

# Translate PTX to C++ (also generates kernel_cpu.h with declarations)
python3 tools/ptx2cpp.py examples/vector_add/kernel.ptx \
    -o examples/vector_add/kernel_cpu.cpp \
    -H examples/vector_add/kernel_cpu.h

# Compile and run
g++ -std=c++17 -O2 -Iinclude/compat -Iinclude \
    examples/vector_add/main.cpp \
    examples/vector_add/kernel_cpu.cpp \
    -o vector_add
./vector_add
# PASS: all 1024 elements correct
```

## Using CMake

```bash
mkdir build && cd build
cmake ..
make
ctest    # 10 tests, all passing
```

### Integrating with your project

```cmake
# Simple: single kernel, single executable
cuda_sim_add_executable(my_app kernel.ptx main.cpp)

# Flexible: add kernels to an existing target with multiple sources
add_executable(my_app src/main.cpp src/utils.cpp src/renderer.cpp)
cuda_sim_add_kernel(my_app kernels/physics.ptx)
cuda_sim_add_kernel(my_app kernels/sort.ptx)
target_link_libraries(my_app PRIVATE some_lib)

# Using real CUDA <<<>>> syntax (auto-preprocessed)
add_executable(my_app)
cuda_sim_add_kernel(my_app kernel.ptx)
cuda_sim_add_sources(my_app src/main.cpp src/utils.cpp)  # <<<>>> auto-converted
```

### Runtime .cu compilation (new)

```cmake
# cuda_sim_add_library: drop-in replacement for cuda_add_library
# .cu files → nvcc -ptx → ptx2cpp.py → g++ (automatic)
set(CUDA_SIM_ROOT /path/to/cuda_sim)
include(${CUDA_SIM_ROOT}/cmake/CudaSimCompile.cmake)
cuda_sim_add_library(mylib STATIC kernel.cu host.cpp)
```

### Dual GPU/CPU support

```cmake
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND USE_GPU)
    enable_language(CUDA)
    add_executable(my_app main.cpp kernel.cu)
    target_link_libraries(my_app CUDA::cudart)
else()
    add_executable(my_app)
    cuda_sim_add_kernel(my_app kernel.ptx)
    cuda_sim_add_sources(my_app main.cpp)
endif()
```

## Host code compatibility

Your `main.cpp` uses standard CUDA API and syntax — no changes needed:

```cpp
#include <cuda_runtime.h>
#include "kernel_cpu.h"      // auto-generated launch declarations

int main() {
    float *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    // Real CUDA syntax works (auto-preprocessed by cuda_sim_add_sources)
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    cudaFree(d_a);
}
```

Compat headers intercept `<cuda_runtime.h>`, `<cuda.h>`, `<nvrtc.h>`, `<device_launch_parameters.h>`, `<cuda_fp16.h>`, etc.

## Runtime JIT (CUDA Driver API + NVRTC)

For projects that compile kernels at runtime (e.g., GPU miners):

```cpp
#include <nvrtc.h>
#include <cuda.h>

// Compile kernel source to PTX
nvrtcCreateProgram(&prog, kernel_src, "kernel.cu", 0, NULL, NULL);
nvrtcCompileProgram(prog, num_opts, opts);  // calls nvcc -ptx
nvrtcGetPTX(prog, ptx);

// Load PTX module (JIT: ptx2cpp.py → g++ → .so → dlopen)
cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
cuModuleGetFunction(&func, module, "my_kernel");

// Launch with void** args (type-safe unpacking via _launch_generic)
void* args[] = {&nonce, &header, &target, &dag_ptr, &output_ptr};
cuLaunchKernel(func, gridX, 1, 1, blockX, 1, 1, 0, stream, args, NULL);
```

Verified with quai-gpu-miner's ProgPow kernel: 56KB PTX, 1116 instructions, `.func` calls, shared memory, warp shuffles.

## Examples

| Example | Features tested |
|---------|----------------|
| vector_add | Basic 1D grid, global memory |
| histogram | `atomicAdd` |
| reduction | Shared memory, `__syncthreads()` |
| matrix_mul | 2D shared memory tiles, 2D grid/block |
| saxpy | Multiple kernels, `float` params |
| warp_reduce | `__shfl_down_sync` warp-level reduction |
| softmax | Shared memory reduction, `expf`, complex control flow |
| relu | 3 kernels (relu, leaky_relu, float_to_int), type conversions |
| native_syntax | Real `<<<>>>` syntax, auto-preprocessed |

All examples verified with real nvcc-generated PTX (CUDA 12.6).

## Supported PTX instructions (64+)

**Arithmetic**: add, sub, mul, mad, div, rem, fma, abs, neg, min, max, sad

**Bitwise**: and, or, xor, not, shl, shr, shf (funnel shift l/r with wrap/clamp)

**Memory**: ld (param/global/shared/const, v4 vector), st (global/shared/param), cvta, prefetch, red

**Control flow**: mov, setp, selp, slct, bra, ret, exit, bar, trap, brkpt

**Function calls**: .func definitions, call.uni sequences, struct params, return values

**Type conversion**: cvt (with rounding modes: rn/rz/rm/rp), copysign, prmt

**Math**: sin, cos, sqrt, rsqrt, rcp, lg2, ex2, testp

**Atomics**: atom (add/cas/exch/min/max/inc/dec)

**Warp**: shfl (down/up/idx/xor), vote (ballot/any/all), match (any/all), activemask

**Bit ops**: popc, clz, bfind, brev, bfe, bfi, ffs

**Misc**: isspacep, nop, membar, fence

## Memory error detection

Built-in runtime checks — always active, no flags needed:

```
[cuda_sim] ERROR: cudaFree(0x...) — double free! (originally 512 bytes)
[cuda_sim] ERROR: cudaFree(0x...) — pointer was never allocated
[cuda_sim] ERROR: buffer overflow at 0x... (back redzone corrupted)
[cuda_sim] ERROR: buffer underflow at 0x... (front redzone corrupted)
[cuda_sim] LEAK: 0x... (1024 bytes) never freed
```

Detects: memory leaks, double free, invalid free, buffer overflow, buffer underflow, use-after-free (poisoned memory).

## Real-world integration: quai-gpu-miner

cuda_sim has been integrated with [quai-gpu-miner](https://github.com/dominant-strategies/quai-gpu-miner) — a ProgPow GPU miner using NVRTC + CUDA Driver API.

```bash
# Build in Docker (includes CUDA toolkit for NVRTC)
cd cuda_sim
./integration/quai-gpu-miner/build.sh
```

**Verified**: full binary compiles, ProgPow JIT kernel executes with 16 threads, shared memory, warp shuffles, deterministic hash output.

## Requirements

- **Docker** — for running nvcc (generates PTX from `.cu` files)
- **Python 3** — for ptx2cpp.py and cuda_preprocess.py
- **C++17 compiler** — g++ or clang++
- No GPU, no CUDA Toolkit installation needed on the host

## License

MIT
