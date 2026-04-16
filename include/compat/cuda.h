// Compatibility header: drop-in for <cuda.h> (CUDA Driver API)
// Most users only need the runtime API, but some files include this.
#pragma once
#include "cuda_sim/cuda_runtime_api.h"
#include "cuda_sim/runtime.h"
using cuda_sim::dim3;
