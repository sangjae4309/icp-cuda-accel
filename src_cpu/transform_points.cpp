#include "icp/icp.hpp"

namespace icp {

Point transform_point(const Pose& pose, Point point) {
    // GPU와 동일하게 점 변환은 float로 계산합니다.
    float r[9];
    float t[3];
    for (int k = 0; k < 9; ++k) {
        r[k] = static_cast<float>(pose.R[k]);
    }
    for (int k = 0; k < 3; ++k) {
        t[k] = static_cast<float>(pose.t[k]);
    }

    return {
        r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0],
        r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1],
        r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]
    };
}

// 1단계: 원본 source에 현재 pose를 적용합니다.
void transform_points(const Cloud& source, const Pose& pose, Cloud& current) {
    current.resize(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        current[i] = transform_point(pose, source[i]);
    }
}

} // namespace icp
