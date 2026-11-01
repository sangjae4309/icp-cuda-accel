#include "icp/icp.hpp"
#include <cfloat>

namespace icp {

// 2단계: 각 source 점에서 가장 가까운 target 점을 찾습니다.
void find_correspondences(const Cloud& current, const Cloud& target,
                          float max_squared_distance, Matches& matches) {
    matches.index.resize(current.size());
    matches.squared_distance.resize(current.size());

    for (std::size_t i = 0; i < current.size(); ++i) {
        float best_distance = FLT_MAX;
        int best_index = -1;

        for (std::size_t j = 0; j < target.size(); ++j) {
            float dx = current[i].x - target[j].x;
            float dy = current[i].y - target[j].y;
            float dz = current[i].z - target[j].z;
            float distance = dx * dx + dy * dy + dz * dz;

            // 같은 거리라면 먼저 찾은 작은 인덱스를 유지합니다.
            if (distance < best_distance) {
                best_distance = distance;
                best_index = int(j);
            }
        }

        if (best_distance > max_squared_distance) {
            best_index = -1;
        }
        matches.index[i] = best_index;
        matches.squared_distance[i] = best_distance;
    }
}

} // namespace icp
