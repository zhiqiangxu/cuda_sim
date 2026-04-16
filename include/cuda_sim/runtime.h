#pragma once
#include <cstdint>

namespace cuda_sim {

struct dim3 {
    uint32_t x, y, z;
    dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

} // namespace cuda_sim
