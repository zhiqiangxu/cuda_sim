#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <mutex>

// Error types
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInsufficientDriver = 35,
    cudaErrorNoDevice = 100,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

// ---------------------------------------------------------------------------
// Allocation tracker — leak detection, double-free, use-after-free
// ---------------------------------------------------------------------------
namespace cuda_sim {
namespace detail {

constexpr size_t REDZONE_SIZE = 16;
constexpr uint8_t REDZONE_FILL = 0xAB;

struct AllocInfo {
    void* real_ptr;   // actual malloc'd pointer (before front redzone)
    size_t size;      // user-requested size
    bool freed;
};

struct AllocTracker {
    std::unordered_map<void*, AllocInfo> allocs;
    std::mutex mtx;
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t peak_usage = 0;
    size_t current_usage = 0;

    void track_alloc(void* real_ptr, void* user_ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        allocs[user_ptr] = {real_ptr, size, false};
        total_allocated += size;
        current_usage += size;
        if (current_usage > peak_usage)
            peak_usage = current_usage;
    }

    cudaError_t track_free(void* ptr) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = allocs.find(ptr);
        if (it == allocs.end()) {
            fprintf(stderr, "[cuda_sim] ERROR: cudaFree(%p) — pointer was never allocated\n", ptr);
            return cudaErrorInvalidDevicePointer;
        }
        if (it->second.freed) {
            fprintf(stderr, "[cuda_sim] ERROR: cudaFree(%p) — double free! (originally %zu bytes)\n",
                    ptr, it->second.size);
            return cudaErrorInvalidDevicePointer;
        }
        it->second.freed = true;
        total_freed += it->second.size;
        current_usage -= it->second.size;
        return cudaSuccess;
    }

    bool is_valid(void* ptr) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = allocs.find(ptr);
        return it != allocs.end() && !it->second.freed;
    }

    bool check_redzone(void* user_ptr, const AllocInfo& info) {
        auto* front = static_cast<uint8_t*>(info.real_ptr);
        auto* back = static_cast<uint8_t*>(user_ptr) + info.size;
        bool ok = true;
        for (size_t i = 0; i < REDZONE_SIZE; i++) {
            if (front[i] != REDZONE_FILL) {
                fprintf(stderr, "[cuda_sim] ERROR: buffer underflow at %p (front redzone corrupted at byte %zu)\n",
                        user_ptr, i);
                ok = false;
                break;
            }
        }
        for (size_t i = 0; i < REDZONE_SIZE; i++) {
            if (back[i] != REDZONE_FILL) {
                fprintf(stderr, "[cuda_sim] ERROR: buffer overflow at %p (%zu bytes, back redzone corrupted at byte %zu)\n",
                        user_ptr, info.size, i);
                ok = false;
                break;
            }
        }
        return ok;
    }

    void report() {
        std::lock_guard<std::mutex> lock(mtx);
        int leaks = 0;
        size_t leaked_bytes = 0;
        for (auto it = allocs.begin(); it != allocs.end(); ++it) {
            auto ptr = it->first;
            auto& info = it->second;
            if (!info.freed) {
                // Check redzone on leaked memory too
                check_redzone(ptr, info);
                leaks++;
                leaked_bytes += info.size;
                fprintf(stderr, "[cuda_sim] LEAK: %p (%zu bytes) never freed\n",
                        ptr, info.size);
            }
        }
        if (leaks > 0) {
            fprintf(stderr, "[cuda_sim] SUMMARY: %d leak(s), %zu bytes lost\n",
                    leaks, leaked_bytes);
            fprintf(stderr, "[cuda_sim]   total allocated: %zu bytes, peak: %zu bytes\n",
                    total_allocated, peak_usage);
        }
    }

    ~AllocTracker() {
        report();
    }
};

inline AllocTracker& tracker() {
    static AllocTracker t;
    return t;
}

} // namespace detail
} // namespace cuda_sim

// ---------------------------------------------------------------------------
// CUDA Runtime API
// ---------------------------------------------------------------------------

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    // Allocate: [front redzone | user data | back redzone]
    size_t total = cuda_sim::detail::REDZONE_SIZE + size + cuda_sim::detail::REDZONE_SIZE;
    void* real_ptr = std::malloc(total);
    if (!real_ptr) { *ptr = nullptr; return cudaErrorMemoryAllocation; }

    auto* base = static_cast<uint8_t*>(real_ptr);
    // Fill redzones with sentinel
    std::memset(base, cuda_sim::detail::REDZONE_FILL, cuda_sim::detail::REDZONE_SIZE);
    std::memset(base + cuda_sim::detail::REDZONE_SIZE + size,
                cuda_sim::detail::REDZONE_FILL, cuda_sim::detail::REDZONE_SIZE);
    // Zero user region
    std::memset(base + cuda_sim::detail::REDZONE_SIZE, 0, size);

    *ptr = base + cuda_sim::detail::REDZONE_SIZE;
    cuda_sim::detail::tracker().track_alloc(real_ptr, *ptr, size);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* ptr) {
    if (!ptr) return cudaSuccess;
    auto& t = cuda_sim::detail::tracker();

    // Check redzone before freeing
    {
        std::lock_guard<std::mutex> lock(t.mtx);
        auto it = t.allocs.find(ptr);
        if (it != t.allocs.end() && !it->second.freed) {
            t.check_redzone(ptr, it->second);
        }
    }

    cudaError_t err = t.track_free(ptr);
    if (err != cudaSuccess) return err;

    // Poison user memory to catch use-after-free
    void* real_ptr;
    size_t size;
    {
        std::lock_guard<std::mutex> lock(t.mtx);
        auto it = t.allocs.find(ptr);
        real_ptr = it->second.real_ptr;
        size = it->second.size;
    }
    std::memset(ptr, 0xDE, size);
    std::free(real_ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src,
                               size_t count, cudaMemcpyKind kind) {
    (void)kind;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaPeekAtLastError() {
    return cudaSuccess;
}

inline const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
        default: return "cudaErrorUnknown";
    }
}

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        default: return "unknown error";
    }
}

// ---------------------------------------------------------------------------
// Stream API (synchronous simulation — no actual async)
// ---------------------------------------------------------------------------

typedef void* cudaStream_t;

// Stream flags (ignored in simulation)
#define cudaStreamDefault        0x00
#define cudaStreamNonBlocking    0x01

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    static int dummy_stream = 0;
    *stream = &dummy_stream;
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(stream);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Pinned memory (CPU simulation — just use regular malloc/free)
// ---------------------------------------------------------------------------

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

// Overload for volatile pointers (e.g., volatile Search_results**)
template<typename T>
inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    void* p = std::malloc(size);
    *ptr = static_cast<T*>(p);
    return p ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

struct cudaDeviceProp {
    char name[256] = "cuda_sim CPU Device";
    size_t totalGlobalMem = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    size_t sharedMemPerBlock = 49152;   // 48 KB
    int maxThreadsPerBlock = 1024;
    int maxThreadsDim[3] = {1024, 1024, 64};
    int maxGridSize[3] = {2147483647, 65535, 65535};
    int warpSize = 32;
    int multiProcessorCount = 1;
    int major = 7;
    int minor = 5;
    int pciBusID = 0;
    int pciDeviceID = 0;
    int pciDomainID = 0;
    size_t totalConstMem = 65536;
    int clockRate = 1500000;  // kHz
    size_t memPitch = 2147483647;
};

inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 1;
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    (void)device;
    *prop = cudaDeviceProp{};
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {
    (void)device;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceReset() {
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int* driverVersion) {
    *driverVersion = 12060;  // CUDA 12.6
    return cudaSuccess;
}

inline cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    *runtimeVersion = 12060;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Memory info
// ---------------------------------------------------------------------------

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    // Report simulated GPU memory — must be large enough for DAG (~4GB for epoch 0)
    if (total) *total = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    if (free)  *free  = 7ULL * 1024 * 1024 * 1024;  // 7 GB free
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// cudaMemcpyToSymbol / cudaMemcpyFromSymbol — __constant__ variables
// ---------------------------------------------------------------------------
// NOTE: Requires driver_api.h to be included for symbol lookup.
// Usage: cudaMemcpyToSymbol(symbol_name, src, size)
// The macro converts the symbol to a string for runtime lookup.

// Forward declaration — actual implementation needs JIT engine access
namespace cuda_sim { namespace jit { struct JitEngine; } }

// These are helpers that do the actual memcpy via symbol lookup.
// Must be called after driver_api.h is included.
#define cudaMemcpyToSymbol(symbol, src, count, ...) \
    cudaMemcpyToSymbol_impl(#symbol, src, count)

#define cudaMemcpyFromSymbol(dst, symbol, count, ...) \
    cudaMemcpyFromSymbol_impl(dst, #symbol, count)
