// Compatibility header: drop-in for <device_functions.h>
// Provides CPU-compatible replacements for CUDA device intrinsics
// that normally use PTX inline assembly.
#pragma once

#include <cstdint>

// MAKE_ULONGLONG — CPU replacement for PTX asm version
inline uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI) {
    return ((uint64_t)HI << 32) | (uint64_t)LO;
}

// xor1 — CPU replacement for PTX xor.b64 asm
inline uint64_t xor1(uint64_t a, uint64_t b) {
    return a ^ b;
}

// xor8 — CPU replacement for PTX xor chain
inline uint64_t xor8(uint64_t a, uint64_t b, uint64_t c,
                     uint64_t d, uint64_t e, uint64_t f,
                     uint64_t g, uint64_t h) {
    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}
