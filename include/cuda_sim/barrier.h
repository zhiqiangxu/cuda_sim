#pragma once
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace cuda_sim {

/// Simple reusable barrier for C++17 (std::barrier is C++20).
/// All N threads call arrive_and_wait(); once all arrive, all are released.
/// Reusable: can be called multiple times (__syncthreads() may appear in a loop).
class SimpleBarrier {
public:
    explicit SimpleBarrier(uint32_t count = 1)
        : threshold_(count), count_(count), generation_(0) {}

    /// Reset the barrier for a new thread count (must not be called while threads are waiting)
    void reset(uint32_t count) {
        threshold_ = count;
        count_ = count;
        generation_ = 0;
    }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        uint32_t gen = generation_;
        if (--count_ == 0) {
            // Last thread to arrive: reset and wake everyone
            count_ = threshold_;
            ++generation_;
            cv_.notify_all();
        } else {
            // Wait until generation changes
            cv_.wait(lock, [this, gen] { return generation_ != gen; });
        }
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    uint32_t threshold_;
    uint32_t count_;
    uint32_t generation_;
};

} // namespace cuda_sim
