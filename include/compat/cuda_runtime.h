// Compatibility header: drop-in replacement for <cuda_runtime.h>
// Users can #include <cuda_runtime.h> without modification.
// Compile with: g++ -Iinclude/compat -Iinclude ...
#pragma once
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"

// Bring dim3 into global scope (like real CUDA)
using cuda_sim::dim3;
