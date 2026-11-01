#include "icp.hpp"
#include <algorithm>
#include <stdexcept>
#include <time.h>

namespace icp {

double now_ms() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        throw std::runtime_error("clock_gettime failed");
    }
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

double rotation_angle(const Pose& p) {
    const double c = std::clamp((p.R[0] + p.R[4] + p.R[8] - 1) * 0.5, -1.0, 1.0);
    return std::acos(c);
}

double translation_norm(const Pose& p) {
    return std::sqrt(p.t[0] * p.t[0] + p.t[1] * p.t[1] + p.t[2] * p.t[2]);
}
} // namespace icp
