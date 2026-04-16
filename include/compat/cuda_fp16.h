// Compatibility header: drop-in for <cuda_fp16.h>
// Provides minimal half-precision type support.
#pragma once
#include <cstdint>

struct __half {
    uint16_t x;
};

inline __half __float2half(float f) {
    // Simplified float→half conversion (no proper rounding)
    union { float f; uint32_t u; } v = {f};
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (v.u >> 13) & 0x3FF;
    if (exp <= 0) { exp = 0; mant = 0; }
    if (exp >= 31) { exp = 31; mant = 0; }
    __half h;
    h.x = (uint16_t)(sign | (exp << 10) | mant);
    return h;
}

inline float __half2float(__half h) {
    uint32_t sign = (h.x & 0x8000) << 16;
    uint32_t exp = (h.x >> 10) & 0x1F;
    uint32_t mant = h.x & 0x3FF;
    if (exp == 0) { exp = 0; mant = 0; }
    else if (exp == 31) { exp = 0xFF; }
    else { exp = exp - 15 + 127; }
    union { uint32_t u; float f; } v;
    v.u = sign | (exp << 23) | (mant << 13);
    return v.f;
}
