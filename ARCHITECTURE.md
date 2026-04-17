# cuda_sim - Architecture

## Goal

Run CUDA kernels on CPU without a GPU. User provides `.cu` source code, our toolchain compiles it into a native CPU executable.

**Use cases**: debugging kernels without GPU, running CUDA tests in CI, teaching CUDA programming model.

---

## Approach: PTX-level Translation

We translate at the **PTX level** (CUDA's intermediate representation), not at the source level.

Why PTX, not source-level?
- Source-level simulation requires reimplementing hundreds of CUDA SDK functions
- nvcc compiles all SDK functions down to ~200 PTX instructions
- Translating PTX is mechanical and complete — no coverage gaps

```
CUDA's normal flow:
  .cu → nvcc → PTX → SASS → GPU execution

Our flow:
  .cu → nvcc -ptx → PTX → ptx2cpp.py → C++ → g++ → CPU execution
                          ↑
                    our translator
```

---

## Compilation Pipeline

```
Step 1             Step 2                Step 3             Step 4
  .cu      →    kernel.ptx     →    kernel_cpu.cpp   →   executable
         nvcc -ptx          ptx2cpp.py             g++

(NVIDIA)        (text file,       (generated C++,      (native CPU
                 readable)         one line per          binary)
                                   PTX instruction)
```

### Cross-platform: Docker for PTX generation

Since nvcc is not available on macOS (NVIDIA dropped support in 2020), we use Docker to run nvcc. Only the PTX generation step needs Docker — the rest runs natively.

```
Any platform (Mac / Linux / Windows)
├── Docker (nvidia/cuda image) → nvcc -ptx → kernel.ptx   ← only this needs Docker
├── ptx2cpp.py                 → kernel_cpu.cpp            ← pure Python, native
└── g++ / clang++              → executable                ← native compiler
```

### Normal CUDA compilation (requires GPU)

```bash
nvcc main.cpp kernel.cu -lcudart -o app
./app    # runs kernel on GPU
```

### Our approach (no GPU needed)

```bash
# Step 1: Generate PTX via Docker
docker run --rm -v $(pwd):/src nvidia/cuda:12.6.0-devel-ubuntu22.04 \
    nvcc -ptx /src/kernel.cu -o /src/kernel.ptx

# Step 2: Translate PTX to C++
python3 tools/ptx2cpp.py kernel.ptx -o kernel_cpu.cpp

# Step 3: Compile and run
g++ -std=c++17 -O2 -Iinclude/compat -Iinclude main.cpp kernel_cpu.cpp -o app
./app
```

---

## How PTX Translation Works

### PTX format

Every PTX instruction follows a regular pattern:

```
opcode.type  dest, src1, src2;
```

### Translation table

Each PTX instruction maps to one C++ statement:

```
PTX                                 C++
──────────────────────────          ───────────────────────────────────
mov.u32     %r1, %tid.x            r[1] = tid_x;
mad.lo.s32  %r4, %r2, %r3, %r1    r[4] = (int32_t)r[2] * (int32_t)r[3] + (int32_t)r[1];
setp.ge.s32  %p1, %r4, %r5         p[1] = ((int32_t)r[4] >= (int32_t)r[5]);
@%p1 bra     $L__BB0_2             if (p[1]) goto L__BB0_2;
ld.global.f32 %f1, [%rd4]          f[1] = *(float*)(rd[4]);
add.f32       %f3, %f1, %f2        f[3] = f[1] + f[2];
st.global.f32 [%rd8], %f3          *(float*)(rd[8]) = f[3];
atom.global.add.s32 %r1,[%rd2],%r3 r[1] = atomic_add((int*)rd[2], r[3]);
shfl.sync.down.b32 %r1, %r2, 16   r[1] = warp_ctx->shfl_down(lane_id, r[2], 16);
bar.sync 0                         barrier->arrive_and_wait();
```

PTX registers (`%r`, `%rd`, `%f`, `%p`) become C++ arrays (`r[]`, `rd[]`, `f[]`, `p[]`). Types come from PTX register declarations (`.reg .b32 %r<6>` → `uint32_t r[6]`).

---

## Execution Models

ptx2cpp.py generates different launch wrappers depending on kernel features:

### Sequential (no shared memory, no warp ops)

```cpp
for each block (bx, by, bz):
    for each thread (tx, ty, tz):
        kernel_thread(params, tx, ty, tz, bx, by, bz, ...);
```

Simple function calls. Fast, deterministic, easy to debug with gdb.

### Multi-threaded (shared memory and/or warp primitives)

```cpp
for each block (bx, by, bz):
    uint8_t shared_mem[SIZE] = {};
    SimpleBarrier barrier(num_threads);
    WarpContext warp_ctxs[num_warps];       // only if warp ops used
    vector<thread> threads;
    for each thread (tx, ty, tz):
        threads.emplace_back(kernel_thread, params,
            tx, ty, tz, bx, by, bz, ...,
            shared_mem, &barrier,           // shared memory support
            &warp_ctxs[tid/32], tid%32);    // warp support
    for (auto& t : threads) t.join();
```

Real OS threads per block. `__syncthreads()` becomes a real barrier. Warp primitives use per-warp shared buffers for register exchange.

### Why multi-threading is needed

```cpp
// Thread 0 writes, Thread 1 reads — requires concurrent execution
sdata[tid] = input[i];
__syncthreads();          // all threads must finish writing before any reads
sum = sdata[tid] + sdata[tid + s];
```

In sequential mode, Thread 0 would try to read Thread 1's value before Thread 1 has executed. The barrier only works when threads run concurrently.

---

## Host Code Compatibility

Users write standard CUDA host code with `#include <cuda_runtime.h>`:

```cpp
#include <cuda_runtime.h>   // intercepted by our compat header

int main() {
    float *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    vectorAdd_launch(d_a, d_b, d_c, N, dim3(numBlocks), dim3(blockSize));
    cudaFree(d_a);
}
```

`include/compat/cuda_runtime.h` intercepts the standard include and provides our implementations. Compile with `-Iinclude/compat -Iinclude`.

The only API difference from real CUDA: `kernel<<<grid, block>>>(args)` becomes `kernel_launch(args, grid, block)`.

---

## Memory Error Detection

Built into `cuda_runtime_api.h`, always active:

| Error | Detection method |
|-------|-----------------|
| Memory leak | Track all cudaMalloc/cudaFree, report at exit |
| Double free | Check if pointer already freed |
| Invalid free | Check if pointer was ever allocated |
| Use-after-free | Poison freed memory with 0xDE |
| Buffer overflow | 16-byte redzone after each allocation, checked on free |
| Buffer underflow | 16-byte redzone before each allocation, checked on free |

---

## CUDA SDK Compatibility

CUDA is only needed at **compile time** to generate PTX. The final binary is a **pure CPU program**.

| | Compile time | Runtime |
|---|---|---|
| CUDA Toolkit (nvcc) | Required (via Docker) | NOT needed |
| GPU Driver | NOT needed | NOT needed |
| GPU Hardware | NOT needed | NOT needed |
| libcudart.so | NOT needed | NOT needed |

---

## Project Structure

```
cuda_sim/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
├── CMakeLists.txt
│
├── cmake/
│   ├── CudaSimConfig.cmake     # Compile-time PTX integration
│   └── CudaSimCompile.cmake    # cuda_sim_add_library() — .cu → PTX → C++ pipeline
│
├── docker/
│   ├── Dockerfile
│   └── generate_ptx.sh
│
├── tools/
│   ├── ptx2cpp.py              # Core: PTX → C++ translator
│   └── cuda_preprocess.py      # <<<>>> syntax → _launch() calls
│
├── include/
│   ├── compat/                 # Drop-in replacements for CUDA headers
│   │   ├── cuda_runtime.h      # vector types (uint2/4), device qualifiers, intrinsics
│   │   ├── cuda_runtime_api.h
│   │   ├── cuda.h              # includes Driver API (driver_api.h)
│   │   ├── nvrtc.h             # NVRTC runtime compilation (nvcc -ptx backend)
│   │   ├── device_functions.h  # CPU-compatible CUDA intrinsics
│   │   ├── cuda_fp16.h
│   │   └── device_launch_parameters.h
│   └── cuda_sim/
│       ├── cuda_runtime_api.h  # cudaMalloc/Free/Memcpy + streams + error detection
│       ├── runtime.h           # dim3
│       ├── driver_api.h        # CUDA Driver API + JIT engine (PTX → .so → dlopen)
│       ├── device_atomic.h     # atomic_add, atomic_cas, atomic_inc/dec, etc.
│       ├── barrier.h           # SimpleBarrier for __syncthreads()
│       └── warp.h              # WarpContext + bit ops (popc, clz, etc.)
│
├── examples/
│   ├── vector_add/             # 1D grid, global memory
│   ├── histogram/              # atomicAdd
│   ├── reduction/              # shared memory + syncthreads
│   ├── matrix_mul/             # 2D shared tiles, 2D grid/block
│   ├── saxpy/                  # multiple kernels, float params
│   ├── warp_reduce/            # __shfl_down_sync
│   ├── softmax/                # shared memory reduction, expf
│   ├── relu/                   # 3 kernels, type conversions with rounding
│   └── native_syntax/          # real <<<>>> syntax, auto-preprocessed
│
├── integration/
│   └── quai-gpu-miner/         # Real-world integration example
│       ├── Dockerfile           # Docker build (full binary + tests)
│       ├── build.sh
│       └── patches/             # CMakeLists, CPU DAG gen, verification tests
│
└── tests/
    ├── test_ptx_parser.py      # Parser + translator unit tests
    └── test_error_detection.cpp # Memory error detection tests
```

---

## Supported PTX Instructions (64+)

**Arithmetic**: add, sub, mul, mad, div, rem, fma, abs, neg, min, max, sad

**Bitwise**: and, or, xor, not, shl, shr, shf (funnel shift l/r with wrap/clamp)

**Memory**: ld (param/global/shared/const, v4 vector load), st (global/shared/param), cvta, prefetch, red

**Control flow**: mov, setp, selp, slct, bra, ret, exit, bar, trap, brkpt

**Function calls**: .func definitions, call.uni sequences, st.param (return values)

**Type conversion**: cvt (with rounding modes: rn/rz/rm/rp), copysign, prmt

**Math**: sin, cos, sqrt, rsqrt, rcp, lg2, ex2, testp

**Atomics**: atom (add/cas/exch/min/max/inc/dec)

**Warp**: shfl (down/up/idx/xor), vote (ballot/any/all), match (any/all), activemask

**Bit ops**: popc, clz, bfind, brev, bfe, bfi, ffs

**Misc**: isspacep, nop, membar, fence

---

## Runtime JIT: CUDA Driver API + NVRTC Support

cuda_sim supports two modes:

### Mode 1: Compile-time (existing)
```
.cu → nvcc -ptx → ptx2cpp.py → kernel_cpu.cpp → g++ → binary
```
For projects with static kernels. No runtime overhead.

### Mode 2: Runtime JIT (new)
```
Program runs → cuModuleLoadDataEx(ptx_text)
                    → ptx2cpp.py (translate)
                    → g++ -shared (compile to .so)
                    → dlopen (load)
               cuLaunchKernel(func, grid, block, args)
                    → dlsym (find generic entry)
                    → call with void** args unpacking
```
For projects using NVRTC or Driver API (e.g., quai-gpu-miner).

### How cuLaunchKernel works with void** args

cuLaunchKernel passes parameters as `void** args` — an array of pointers
with no type information. We solve this by generating an additional
"generic entry" function at translate time:

```
PTX declares:                    ptx2cpp.py generates:
  .param .u64 start_nonce          void kernel_launch_generic(void** args, ...) {
  .param .u32 n                        uint64_t p0 = *(uint64_t*)args[0];
  .param .f32 alpha                    uint32_t p1 = *(uint32_t*)args[1];
                                       float p2 = *(float*)args[2];
                                       kernel_launch(p0, p1, p2, grid, block);
                                   }
```

Type information comes from PTX `.param` declarations — the same info
ptx2cpp.py already uses for the typed launch function.

### __constant__ variable support

PTX `__constant__` variables become global variables in the generated .so.
A symbol lookup function is also generated:

```cpp
extern "C" void* __cuda_sim_get_symbol(const char* name) {
    if (strcmp(name, "d_header") == 0) return &d_header;
    // ...
}
```

`cudaMemcpyToSymbol(d_header, &val, size)` → finds address via
symbol lookup → memcpy.

### PTX .func support (device-side function calls)

Complex kernels (e.g., ProgPow) generate helper functions as PTX `.func` definitions.
ptx2cpp.py translates these to C++ functions and converts `call.uni` sequences:

```
PTX:                                C++:
.func (.param .b64 retval)          static uint64_t keccak_f800(
  keccak_f800(                          const uint8_t* param_0,
    .param .align 4 .b8 p0[32],         uint64_t param_1,
    .param .b64 p1,                      const uint8_t* param_2)
    .param .align 4 .b8 p2[32])     {
{ ... st.param [retval], %rd8; }        ... return __func_retval;
                                    }

{ // callseq                        {
  .param .align 4 .b8 param0[32];       uint8_t param0[32];
  st.param [param0+0], %r1;             *(uint32_t*)(param0+0) = r[1];
  .param .b64 retval0;                  ...
  call.uni (retval0), keccak_f800;      rd[3] = keccak_f800(param0, ...);
  ld.param %rd3, [retval0];         }
}
```

### NVRTC flow

```
nvrtcCreateProgram(src)     → store source text
nvrtcCompileProgram(opts)   → call nvcc -ptx (Docker or local)
nvrtcGetPTX()               → return generated PTX
cuModuleLoadDataEx(ptx)     → JIT compile to .so (see above)
cuLaunchKernel(func, args)  → call generic entry
```

---

## Real-World Integration: quai-gpu-miner

cuda_sim has been integrated with [quai-gpu-miner](https://github.com/dominant-strategies/quai-gpu-miner), a ProgPow GPU miner that uses NVRTC + CUDA Driver API.

### What works

- **Full binary compiles** with g++ (no nvcc for host code)
- **ProgPow JIT kernel** executes end-to-end: `getKern → NVRTC → PTX (56KB, 1116 instructions) → ptx2cpp.py → g++ → .so → cuLaunchKernel`
- **16 threads** with shared memory (16KB), warp shuffles, barriers
- **Deterministic results** verified across multiple runs
- **Non-trivial hash output** (`c2c1b28d0f12ef33...`)

### Integration approach

```
quai-gpu-miner libethash-cuda/
├── CUDAMiner.cpp           → compiles directly with cuda_sim compat headers
├── CUDAMiner_cuda_sim.cpp  → replaces CUDAMiner_cuda.cu (CPU DAG generation)
└── CUDAMiner_kernel.cu     → embedded as string, compiled by NVRTC at runtime
```

Docker build: `integration/quai-gpu-miner/Dockerfile`

---

## Verified Examples

All examples tested with **real nvcc-generated PTX** (CUDA 12.6, Docker).
10 tests passing on Ubuntu and macOS (GitHub Actions CI).

| Example | Features | Execution mode |
|---------|----------|----------------|
| vector_add | 1D, global memory | sequential |
| histogram | atomicAdd | sequential |
| reduction | shared memory, syncthreads | multi-threaded |
| matrix_mul | 2D shared tiles, 2D grid | multi-threaded |
| saxpy | 2 kernels, float params | sequential |
| warp_reduce | shfl_down_sync | multi-threaded + warp |
| softmax | shared memory reduction, expf, complex flow | multi-threaded |
| relu | 3 kernels, type conversions with rounding | sequential |
| native_syntax | real `<<<>>>` syntax, auto-preprocessed | sequential |
