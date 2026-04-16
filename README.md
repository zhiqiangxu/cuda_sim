# cuda_sim

Run CUDA kernels on CPU without GPU. Translates PTX to C++ at compile time via Docker nvcc.

## How it works

```
kernel.cu → Docker nvcc -ptx → kernel.ptx → ptx2cpp.py → kernel_cpu.cpp → g++ → CPU binary
             (generate PTX)                  (translate)                    (compile)
```

1. **nvcc** compiles your `.cu` kernel to PTX (CUDA's intermediate representation)
2. **ptx2cpp.py** translates each PTX instruction to one line of C++
3. **g++** compiles the generated C++ into a native CPU binary

No GPU needed at runtime. nvcc runs inside Docker for cross-platform support.

## Quick start

```bash
# Generate PTX using Docker (one-time image pull ~3.5GB)
docker run --rm -v $(pwd)/examples/vector_add:/src \
    nvidia/cuda:12.6.0-devel-ubuntu22.04 \
    nvcc -ptx /src/kernel.cu -o /src/kernel.ptx

# Translate PTX to C++
python3 tools/ptx2cpp.py examples/vector_add/kernel.ptx \
    -o examples/vector_add/kernel_cpu.cpp

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
ctest
```

CMake automatically runs ptx2cpp.py on `.ptx` files. To add your own kernel:

```cmake
# In your CMakeLists.txt
cuda_sim_add_executable(my_app kernel.ptx main.cpp)
```

## Host code compatibility

Your `main.cpp` uses standard CUDA API — no special includes needed:

```cpp
#include <cuda_runtime.h>   // works with both real CUDA and cuda_sim

int main() {
    float *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    // Only difference: call xxx_launch() instead of xxx<<<grid, block>>>()
    vectorAdd_launch(d_a, d_b, d_c, N, dim3(numBlocks), dim3(blockSize));

    cudaFree(d_a);
}
```

Compile with `-Iinclude/compat -Iinclude` and our headers intercept `<cuda_runtime.h>`.

## Examples

| Example | Features tested |
|---------|----------------|
| vector_add | Basic 1D grid, global memory |
| histogram | `atomicAdd` |
| reduction | Shared memory, `__syncthreads()` |
| matrix_mul | 2D shared memory tiles, 2D grid/block |
| saxpy | Multiple kernels, `float` params |
| warp_reduce | `__shfl_down_sync` warp-level reduction |

All examples verified with real nvcc-generated PTX.

## Supported PTX instructions (56)

**Arithmetic**: add, sub, mul, mad, div, rem, fma, abs, neg, min, max, sad

**Bitwise**: and, or, xor, not, shl, shr

**Memory**: ld (param/global/shared), st (global/shared), cvta, prefetch, red

**Control flow**: mov, setp, selp, slct, bra, ret, exit, bar (syncthreads), trap, brkpt

**Type conversion**: cvt, copysign, prmt

**Math**: sin, cos, sqrt, rsqrt, rcp, lg2, ex2, testp

**Atomics**: atom (add/cas/exch/min/max)

**Warp**: shfl (down/up/idx/xor), vote (ballot/any/all), match (any/all), activemask

**Bit ops**: popc, clz, bfind, brev, bfe, bfi, ffs

**Misc**: isspacep, nop, membar, fence

## Memory error detection

Built-in runtime checks (no extra flags needed):

```
[cuda_sim] ERROR: cudaFree(0x...) — double free! (originally 512 bytes)
[cuda_sim] ERROR: cudaFree(0x...) — pointer was never allocated
[cuda_sim] LEAK: 0x... (1024 bytes) never freed
[cuda_sim] SUMMARY: 1 leak(s), 1024 bytes lost
```

## Requirements

- **Docker** — for running nvcc (generates PTX from `.cu` files)
- **Python 3** — for ptx2cpp.py
- **C++17 compiler** — g++ or clang++
- No GPU, no CUDA Toolkit installation needed on the host

## Limitations

- Kernel launch uses `xxx_launch()` instead of `<<<>>>` syntax
- ~56 of ~200 PTX instructions supported (covers most common kernels)
- Unsupported instructions produce a warning and `// UNSUPPORTED` comment
- Use `--strict` flag to fail on unsupported instructions
- No texture/surface memory, no CUDA streams, no dynamic parallelism
