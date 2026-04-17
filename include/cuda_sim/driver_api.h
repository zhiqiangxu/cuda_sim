#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <mutex>
#include <memory>
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

// ---------------------------------------------------------------------------
// CUDA Driver API types
// ---------------------------------------------------------------------------

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;

// CUresult codes
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_FOUND 500

// JIT options (ignored in simulation)
typedef enum {
    CU_JIT_INFO_LOG_BUFFER = 0,
    CU_JIT_ERROR_LOG_BUFFER = 1,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 2,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 3,
    CU_JIT_LOG_VERBOSE = 4,
    CU_JIT_GENERATE_LINE_INFO = 5,
    CU_JIT_MAX_REGISTERS = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
} CUjit_option;

// Context flags (ignored)
#define CU_CTX_SCHED_BLOCKING_SYNC 0x04

// ---------------------------------------------------------------------------
// Runtime PTX JIT: PTX text → .cpp → .so → dlopen → function pointers
// ---------------------------------------------------------------------------

namespace cuda_sim {
namespace jit {

// Generic launch function signature: void(void**, gx, gy, gz, bx, by, bz, shared)
using GenericLaunchFn = void(*)(void**,
    uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    uint32_t);

// Symbol lookup function: void*(const char*)
using SymbolLookupFn = void*(*)(const char*);

struct Module {
    void* dl_handle = nullptr;  // dlopen handle
    std::string so_path;
    std::unordered_map<std::string, GenericLaunchFn> functions;  // kernel_name → generic entry
    SymbolLookupFn symbol_lookup = nullptr;

    ~Module() {
        if (dl_handle) dlclose(dl_handle);
    }
};

struct JitEngine {
    std::vector<std::unique_ptr<Module>> modules;
    std::mutex mtx;
    std::string ptx2cpp_path;
    std::string include_path;
    std::string include_compat_path;
    std::string cache_dir;
    bool initialized = false;

    void init() {
        if (initialized) return;
        // Find ptx2cpp.py relative to this header, or from env
        const char* env = std::getenv("CUDA_SIM_ROOT");
        std::string root;
        if (env) {
            root = env;
        } else {
            // Try common locations
            root = ".";
        }
        ptx2cpp_path = root + "/tools/ptx2cpp.py";
        include_path = root + "/include";
        include_compat_path = root + "/include/compat";
        cache_dir = "/tmp/cuda_sim_jit";
        std::string cmd = "mkdir -p " + cache_dir;
        (void)system(cmd.c_str());
        initialized = true;
    }

    Module* load_ptx(const char* ptx_text, size_t ptx_size) {
        std::lock_guard<std::mutex> lock(mtx);
        init();

        // Hash PTX for caching
        size_t hash = std::hash<std::string>{}(std::string(ptx_text, ptx_size));
        std::string base = cache_dir + "/mod_" + std::to_string(hash);
        std::string ptx_path = base + ".ptx";
        std::string cpp_path = base + "_cpu.cpp";
        std::string hdr_path = base + "_cpu.h";
#ifdef __APPLE__
        std::string so_path = base + ".dylib";
        std::string so_flag = "-dynamiclib";
#else
        std::string so_path = base + ".so";
        std::string so_flag = "-shared";
#endif

        // Check cache
        if (FILE* f = fopen(so_path.c_str(), "r")) {
            fclose(f);
        } else {
            // Write PTX
            {
                std::ofstream out(ptx_path);
                out.write(ptx_text, ptx_size);
            }

            // Translate PTX → C++
            std::string cmd = "python3 " + ptx2cpp_path +
                " " + ptx_path +
                " -o " + cpp_path +
                " -H " + hdr_path + " 2>&1";
            int ret = system(cmd.c_str());
            if (ret != 0) {
                fprintf(stderr, "[cuda_sim] JIT: ptx2cpp failed\n");
                return nullptr;
            }

            // Compile C++ → .so
            cmd = "g++ -std=c++17 -O0 -g -fPIC " + so_flag +
                " -I" + include_compat_path +
                " -I" + include_path +
                " " + cpp_path +
                " -o " + so_path + " 2>&1";
            ret = system(cmd.c_str());
            if (ret != 0) {
                fprintf(stderr, "[cuda_sim] JIT: g++ compilation failed\n");
                return nullptr;
            }
        }

        // dlopen
        void* handle = dlopen(so_path.c_str(), RTLD_NOW);
        if (!handle) {
            fprintf(stderr, "[cuda_sim] JIT: dlopen failed: %s\n", dlerror());
            return nullptr;
        }

        std::unique_ptr<Module> mod(new Module());
        mod->dl_handle = handle;
        mod->so_path = so_path;

        // Look for __constant__ symbol lookup function
        auto* sym_fn = (SymbolLookupFn)dlsym(handle, "__cuda_sim_get_symbol");
        mod->symbol_lookup = sym_fn;

        Module* ptr = mod.get();
        modules.push_back(std::move(mod));
        return ptr;
    }

    GenericLaunchFn get_function(Module* mod, const char* name) {
        if (!mod) return nullptr;

        // Check cache
        auto it = mod->functions.find(name);
        if (it != mod->functions.end()) return it->second;

        // Try _launch_generic suffix (preferred for cuLaunchKernel)
        std::string generic_name = std::string(name) + "_launch_generic";
        auto* func = (GenericLaunchFn)dlsym(mod->dl_handle, generic_name.c_str());
        if (func) {
            mod->functions[name] = func;
            return func;
        }

        fprintf(stderr, "[cuda_sim] JIT: function '%s' not found (tried %s)\n",
                name, generic_name.c_str());
        return nullptr;
    }

    // Find which module a function pointer belongs to
    Module* find_module_for_function(GenericLaunchFn fn) {
        for (auto& mod : modules) {
            for (auto it = mod->functions.begin(); it != mod->functions.end(); ++it) {
                if (it->second == fn) return mod.get();
            }
        }
        return nullptr;
    }

    // Get the most recently loaded module (for cudaMemcpyToSymbol)
    Module* current_module() {
        if (modules.empty()) return nullptr;
        return modules.back().get();
    }
};

inline JitEngine& engine() {
    static JitEngine e;
    return e;
}

} // namespace jit
} // namespace cuda_sim

// ---------------------------------------------------------------------------
// CUDA Driver API functions
// ---------------------------------------------------------------------------

// Error name/string
inline CUresult cuGetErrorName(CUresult error, const char** pStr) {
    switch (error) {
        case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
        case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
        case CUDA_ERROR_NOT_FOUND: *pStr = "CUDA_ERROR_NOT_FOUND"; break;
        default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

inline CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return cuGetErrorName(error, pStr);
}

// Device management
inline CUresult cuInit(unsigned int flags) {
    (void)flags;
    return CUDA_SUCCESS;
}

inline CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    *device = ordinal;
    return CUDA_SUCCESS;
}

inline CUresult cuDeviceGetCount(int* count) {
    *count = 1;  // simulate 1 device
    return CUDA_SUCCESS;
}

inline CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    (void)dev;
    snprintf(name, len, "cuda_sim CPU Device");
    return CUDA_SUCCESS;
}

// Context management (all no-ops — single CPU context)
inline CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice dev) {
    (void)dev;
    static int dummy_ctx = 0;
    *ctx = &dummy_ctx;
    return CUDA_SUCCESS;
}

inline CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

inline CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    (void)dev; (void)flags;
    return CUDA_SUCCESS;
}

inline CUresult cuCtxSetCurrent(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

inline CUresult cuCtxGetCurrent(CUcontext* ctx) {
    static int dummy_ctx = 0;
    *ctx = &dummy_ctx;
    return CUDA_SUCCESS;
}

inline CUresult cuCtxCreate(CUcontext* ctx, unsigned int flags, CUdevice dev) {
    (void)flags; (void)dev;
    static int dummy_ctx = 0;
    *ctx = &dummy_ctx;
    return CUDA_SUCCESS;
}

inline CUresult cuCtxDestroy(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

// Module management — the important part: JIT compilation
inline CUresult cuModuleLoadDataEx(CUmodule* module, const void* ptx,
                                    unsigned int numOptions,
                                    CUjit_option* options,
                                    void** optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    const char* ptx_text = static_cast<const char*>(ptx);
    auto* mod = cuda_sim::jit::engine().load_ptx(ptx_text, strlen(ptx_text));
    *module = mod;
    return mod ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

inline CUresult cuModuleLoadData(CUmodule* module, const void* ptx) {
    return cuModuleLoadDataEx(module, ptx, 0, nullptr, nullptr);
}

inline CUresult cuModuleGetFunction(CUfunction* func, CUmodule module,
                                     const char* name) {
    auto* mod = static_cast<cuda_sim::jit::Module*>(module);
    auto f = cuda_sim::jit::engine().get_function(mod, name);
    *func = reinterpret_cast<void*>(f);
    return f ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

inline CUresult cuModuleUnload(CUmodule module) {
    (void)module;
    return CUDA_SUCCESS;
}

// Kernel launch — calls the _launch_generic entry with void** args
inline CUresult cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream stream,
                                void** kernelParams,
                                void** extra) {
    (void)stream; (void)extra;

    if (!f) return CUDA_ERROR_INVALID_VALUE;

    // f is a GenericLaunchFn: void(void**, gx, gy, gz, bx, by, bz, shared)
    auto launch_fn = reinterpret_cast<cuda_sim::jit::GenericLaunchFn>(f);
    launch_fn(kernelParams,
              gridDimX, gridDimY, gridDimZ,
              blockDimX, blockDimY, blockDimZ,
              sharedMemBytes);
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// cudaMemcpyToSymbol implementation (needs JIT engine for symbol lookup)
// ---------------------------------------------------------------------------

inline cudaError_t cudaMemcpyToSymbol_impl(const char* symbol_name,
                                            const void* src, size_t count) {
    auto* mod = cuda_sim::jit::engine().current_module();
    if (!mod || !mod->symbol_lookup) {
        fprintf(stderr, "[cuda_sim] cudaMemcpyToSymbol: no module loaded or no symbol lookup\n");
        return cudaErrorInvalidDevicePointer;
    }
    void* addr = mod->symbol_lookup(symbol_name);
    if (!addr) {
        fprintf(stderr, "[cuda_sim] cudaMemcpyToSymbol: symbol '%s' not found\n", symbol_name);
        return cudaErrorInvalidDevicePointer;
    }
    std::memcpy(addr, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyFromSymbol_impl(void* dst,
                                              const char* symbol_name, size_t count) {
    auto* mod = cuda_sim::jit::engine().current_module();
    if (!mod || !mod->symbol_lookup) {
        fprintf(stderr, "[cuda_sim] cudaMemcpyFromSymbol: no module loaded or no symbol lookup\n");
        return cudaErrorInvalidDevicePointer;
    }
    void* addr = mod->symbol_lookup(symbol_name);
    if (!addr) {
        fprintf(stderr, "[cuda_sim] cudaMemcpyFromSymbol: symbol '%s' not found\n", symbol_name);
        return cudaErrorInvalidDevicePointer;
    }
    std::memcpy(dst, addr, count);
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// cuStreamCreate (Driver API stream — same as runtime, synchronous no-op)
// ---------------------------------------------------------------------------

inline CUresult cuStreamCreate(CUstream* stream, unsigned int flags) {
    (void)flags;
    static int dummy = 0;
    *stream = &dummy;
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// cuDeviceGetAttribute — return simulated device attributes
// ---------------------------------------------------------------------------

typedef enum {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY = 100,
} CUdevice_attribute;

inline CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attrib, CUdevice dev) {
    (void)dev;
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *value = 7; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *value = 5; break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *value = 1024; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *value = 1; break;
        default: *value = 0; break;
    }
    return CUDA_SUCCESS;
}

inline CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    (void)dev;
    *bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
    return CUDA_SUCCESS;
}
