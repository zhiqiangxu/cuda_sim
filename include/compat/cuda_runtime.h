// Compatibility header: drop-in replacement for <cuda_runtime.h>
// Users can #include <cuda_runtime.h> without modification.
// Compile with: g++ -Iinclude/compat -Iinclude ...
#pragma once
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

// Bring dim3 into global scope (like real CUDA)
using cuda_sim::dim3;

// ===========================================================================
// Part 1: CUDA vector types — universally needed
// ===========================================================================

#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__

struct uint2   { unsigned int x, y; };
struct uint3   { unsigned int x, y, z; };
struct uint4   { unsigned int x, y, z, w; };
struct int2    { int x, y; };
struct int3    { int x, y, z; };
struct int4    { int x, y, z, w; };
struct float2  { float x, y; };
struct float4  { float x, y, z, w; };
struct ulong2  { unsigned long long x, y; };
struct uchar4  { unsigned char x, y, z, w; };
struct ushort2 { unsigned short x, y; };
struct ulonglong2 { unsigned long long x, y; };
struct longlong2  { long long x, y; };

inline uint2  make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
inline uint4  make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x, y, z, w}; }
inline int2   make_int2(int x, int y) { return {x, y}; }
inline int4   make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline ulong2 make_ulong2(unsigned long long x, unsigned long long y) { return {x, y}; }
inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return {x, y, z, w}; }

#endif // __VECTOR_TYPES_H__

// ===========================================================================
// Part 2: CUDA compiler macros and function qualifiers
// ===========================================================================

// Simulate CUDA 12.x compiler version
#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 12
#endif

// NOTE: __CUDA_ARCH__ is intentionally NOT defined.
// In real CUDA, it's only set for device code. Host code uses
// #ifndef __CUDA_ARCH__ guards to select CPU-compatible paths.

// Device function annotations (no-op on CPU)
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef DEV_INLINE
#define DEV_INLINE inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

// ===========================================================================
// Part 3: CUDA device intrinsics (CPU-compatible implementations)
// ===========================================================================

// __byte_perm — CUDA byte permute intrinsic
inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    union { unsigned int val[2]; unsigned char b[8]; } src;
    src.val[0] = x;
    src.val[1] = y;
    unsigned int result = 0;
    for (int i = 0; i < 4; i++) {
        unsigned int sel = (s >> (i * 4)) & 0xF;
        result |= ((unsigned int)src.b[sel & 0x7]) << (i * 8);
    }
    return result;
}

// __funnelshift_l / __funnelshift_r
inline unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
    unsigned long long concat = ((unsigned long long)hi << 32) | lo;
    shift &= 31;
    return (unsigned int)(concat << shift >> 32);
}

inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
    unsigned long long concat = ((unsigned long long)hi << 32) | lo;
    shift &= 31;
    return (unsigned int)(concat >> shift);
}

// __ldg — on CPU just dereference (no cache hints)
#ifndef __ldg
#define __ldg(ptr) (*(ptr))
#endif

// Warp-level primitives (CPU stubs — real warp ops go through PTX translation)
#ifndef __shfl_sync
#define __shfl_sync(mask, val, srcLane, ...) (val)
#endif
#ifndef __activemask
#define __activemask() 0xFFFFFFFF
#endif

// Synchronization (no-op stubs)
inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_block() {}

// Atomic operations (single-threaded host stubs)
inline unsigned int atomicExch(unsigned int* addr, unsigned int val) {
    unsigned int old = *addr; *addr = val; return old;
}
inline unsigned int atomicAdd(unsigned int* addr, unsigned int val) {
    unsigned int old = *addr; *addr += val; return old;
}
inline int atomicAdd(int* addr, int val) {
    int old = *addr; *addr += val; return old;
}
inline unsigned long long atomicAdd(unsigned long long* addr, unsigned long long val) {
    unsigned long long old = *addr; *addr += val; return old;
}

// ===========================================================================
// Part 4: Host-side stubs for compiling .cu files directly with g++
// Opt-in: define CUDA_SIM_HOST_DEVICE_STUBS before including this header
// to enable threadIdx/blockIdx/blockDim globals and PTX asm neutralization.
// ===========================================================================

#ifdef CUDA_SIM_HOST_DEVICE_STUBS

// Thread/block built-in variables — extern declarations.
// Link with cuda_sim_builtins.cpp or define these in one translation unit.
#ifndef CUDA_SIM_BUILTIN_VARS_DEFINED
extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const dim3  blockDim;
extern const dim3  gridDim;
#endif

// Neutralize PTX inline asm (uses NVIDIA-specific constraints like =l that
// g++/clang don't understand). Functions using it are dead code on CPU.
#ifdef asm
#undef asm
#endif
#define asm(...) /* PTX asm neutralized by cuda_sim */

// Force plain-C fallback paths for XOR helper functions
#undef USE_XOR_ASM_OPTS
#define USE_XOR_ASM_OPTS 0

#endif // CUDA_SIM_HOST_DEVICE_STUBS
