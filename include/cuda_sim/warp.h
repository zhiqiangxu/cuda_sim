#pragma once
#include <cstdint>
#include <cstring>
#include "cuda_sim/barrier.h"

namespace cuda_sim {

constexpr uint32_t WARP_SIZE = 32;

/// Per-warp shared context for warp-level primitives.
/// All 32 lanes in a warp share one WarpContext.
struct WarpContext {
    uint32_t values[WARP_SIZE] = {};   // shfl exchange buffer
    uint32_t ballot_result = 0;        // ballot output
    SimpleBarrier barrier{WARP_SIZE};

    // __shfl_sync: read value from src_lane
    uint32_t shfl_idx(uint32_t lane_id, uint32_t my_val, uint32_t src_lane) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        uint32_t result = values[src_lane % WARP_SIZE];
        barrier.arrive_and_wait();
        return result;
    }

    // __shfl_down_sync: read value from lane_id + delta
    uint32_t shfl_down(uint32_t lane_id, uint32_t my_val, uint32_t delta) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        uint32_t src = lane_id + delta;
        uint32_t result = (src < WARP_SIZE) ? values[src] : my_val;
        barrier.arrive_and_wait();
        return result;
    }

    // __shfl_up_sync: read value from lane_id - delta
    uint32_t shfl_up(uint32_t lane_id, uint32_t my_val, uint32_t delta) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        uint32_t result = (lane_id >= delta) ? values[lane_id - delta] : my_val;
        barrier.arrive_and_wait();
        return result;
    }

    // __shfl_xor_sync: read value from lane_id ^ mask
    uint32_t shfl_xor(uint32_t lane_id, uint32_t my_val, uint32_t lane_mask) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        uint32_t result = values[(lane_id ^ lane_mask) % WARP_SIZE];
        barrier.arrive_and_wait();
        return result;
    }

    // __ballot_sync: each lane contributes a bit
    uint32_t ballot(uint32_t lane_id, bool predicate) {
        values[lane_id] = predicate ? 1 : 0;
        barrier.arrive_and_wait();
        uint32_t result = 0;
        for (uint32_t i = 0; i < WARP_SIZE; i++) {
            if (values[i]) result |= (1u << i);
        }
        barrier.arrive_and_wait();
        return result;
    }

    // __any_sync: true if any lane's predicate is true
    uint32_t any(uint32_t lane_id, bool predicate) {
        return ballot(lane_id, predicate) != 0 ? 1 : 0;
    }

    // __all_sync: true if all lanes' predicates are true
    uint32_t all(uint32_t lane_id, bool predicate) {
        return ballot(lane_id, predicate) == 0xFFFFFFFF ? 1 : 0;
    }

    // __match_any_sync: returns bitmask of lanes with same value as this lane
    uint32_t match_any(uint32_t lane_id, uint32_t my_val) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        uint32_t result = 0;
        for (uint32_t i = 0; i < WARP_SIZE; i++) {
            if (values[i] == my_val) result |= (1u << i);
        }
        barrier.arrive_and_wait();
        return result;
    }

    // __match_all_sync: returns value and sets pred=true if all lanes have same value
    uint32_t match_all(uint32_t lane_id, uint32_t my_val, bool& pred) {
        values[lane_id] = my_val;
        barrier.arrive_and_wait();
        bool all_same = true;
        for (uint32_t i = 1; i < WARP_SIZE; i++) {
            if (values[i] != values[0]) { all_same = false; break; }
        }
        pred = all_same;
        uint32_t result = all_same ? 0xFFFFFFFF : 0;
        barrier.arrive_and_wait();
        return result;
    }

    // activemask: return mask of all active lanes (we assume all 32 active)
    static uint32_t activemask() {
        return 0xFFFFFFFF;
    }
};

// ---------------------------------------------------------------------------
// Per-thread bit operations (not warp-level, but commonly used with warp code)
// ---------------------------------------------------------------------------

// popc: population count (number of 1-bits)
inline uint32_t device_popc(uint32_t x) {
    return __builtin_popcount(x);
}
inline uint32_t device_popc(uint64_t x) {
    return __builtin_popcountll(x);
}

// clz: count leading zeros
inline uint32_t device_clz(uint32_t x) {
    return x == 0 ? 32 : __builtin_clz(x);
}
inline uint32_t device_clz(uint64_t x) {
    return x == 0 ? 64 : __builtin_clzll(x);
}

// bfind: find most significant bit (returns bit position, or 0xFFFFFFFF if 0)
inline uint32_t device_bfind(uint32_t x) {
    return x == 0 ? 0xFFFFFFFF : (31 - __builtin_clz(x));
}
inline uint32_t device_bfind(uint64_t x) {
    return x == 0 ? 0xFFFFFFFF : (63 - __builtin_clzll(x));
}

// brev: bit reverse
inline uint32_t device_brev(uint32_t x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    return (x << 16) | (x >> 16);
}

// bfe: bit field extract — extract 'len' bits starting at 'start'
inline uint32_t device_bfe(uint32_t x, uint32_t start, uint32_t len) {
    if (len == 0) return 0;
    return (x >> start) & ((1u << len) - 1);
}
inline uint32_t device_bfe_signed(int32_t x, uint32_t start, uint32_t len) {
    if (len == 0) return 0;
    uint32_t field = ((uint32_t)x >> start) & ((1u << len) - 1);
    // Sign extend
    if (field & (1u << (len - 1)))
        field |= ~((1u << len) - 1);
    return field;
}

// bfi: bit field insert — insert 'len' bits from 'src' into 'base' at 'start'
inline uint32_t device_bfi(uint32_t src, uint32_t base, uint32_t start, uint32_t len) {
    if (len == 0) return base;
    uint32_t mask = ((1u << len) - 1) << start;
    return (base & ~mask) | ((src << start) & mask);
}

// ffs: find first set bit (returns position 1-32, or 0 if no bits set)
inline uint32_t device_ffs(uint32_t x) {
    return x == 0 ? 0 : __builtin_ffs(x);
}

} // namespace cuda_sim
