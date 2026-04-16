#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>

namespace cuda_sim {

template<typename T>
T atomic_add(T* address, T val) {
    auto* a = reinterpret_cast<std::atomic<T>*>(address);
    return a->fetch_add(val, std::memory_order_relaxed);
}

template<typename T>
T atomic_cas(T* address, T compare, T val) {
    auto* a = reinterpret_cast<std::atomic<T>*>(address);
    a->compare_exchange_strong(compare, val, std::memory_order_relaxed);
    return compare;
}

template<typename T>
T atomic_exch(T* address, T val) {
    auto* a = reinterpret_cast<std::atomic<T>*>(address);
    return a->exchange(val, std::memory_order_relaxed);
}

template<typename T>
T atomic_min(T* address, T val) {
    auto* a = reinterpret_cast<std::atomic<T>*>(address);
    T old = a->load(std::memory_order_relaxed);
    while (old > val &&
           !a->compare_exchange_weak(old, val, std::memory_order_relaxed));
    return old;
}

template<typename T>
T atomic_max(T* address, T val) {
    auto* a = reinterpret_cast<std::atomic<T>*>(address);
    T old = a->load(std::memory_order_relaxed);
    while (old < val &&
           !a->compare_exchange_weak(old, val, std::memory_order_relaxed));
    return old;
}

// Float atomic add via CAS loop
inline float atomic_add(float* address, float val) {
    auto* a = reinterpret_cast<std::atomic<uint32_t>*>(address);
    uint32_t old_bits, new_bits;
    float old_float;
    do {
        old_bits = a->load(std::memory_order_relaxed);
        std::memcpy(&old_float, &old_bits, sizeof(float));
        float new_float = old_float + val;
        std::memcpy(&new_bits, &new_float, sizeof(float));
    } while (!a->compare_exchange_weak(old_bits, new_bits,
                                        std::memory_order_relaxed));
    return old_float;
}

} // namespace cuda_sim
