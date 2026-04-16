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

struct AllocInfo {
    size_t size;
    bool freed;  // true = already freed (kept for double-free detection)
};

struct AllocTracker {
    std::unordered_map<void*, AllocInfo> allocs;
    std::mutex mtx;
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t peak_usage = 0;
    size_t current_usage = 0;

    void track_alloc(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        allocs[ptr] = {size, false};
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

    void report() {
        std::lock_guard<std::mutex> lock(mtx);
        int leaks = 0;
        size_t leaked_bytes = 0;
        for (auto& [ptr, info] : allocs) {
            if (!info.freed) {
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
    *ptr = std::malloc(size);
    if (!*ptr) return cudaErrorMemoryAllocation;
    cuda_sim::detail::tracker().track_alloc(*ptr, size);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* ptr) {
    if (!ptr) return cudaSuccess;
    cudaError_t err = cuda_sim::detail::tracker().track_free(ptr);
    if (err != cudaSuccess) return err;
    // Poison freed memory to catch use-after-free
    auto& t = cuda_sim::detail::tracker();
    {
        std::lock_guard<std::mutex> lock(t.mtx);
        auto it = t.allocs.find(ptr);
        if (it != t.allocs.end()) {
            std::memset(ptr, 0xDE, it->second.size);
        }
    }
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src,
                               size_t count, cudaMemcpyKind kind) {
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

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        default: return "unknown error";
    }
}
