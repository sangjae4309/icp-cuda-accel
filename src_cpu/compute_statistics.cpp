#include "icp/icp.hpp"

namespace icp {

// 3단계: 대응점 개수, 좌표 합, 외적 합, 거리 제곱 합을 double로 누적합니다.
Stats serial_stats(const Cloud& current, const Cloud& target, const Matches& matches) {
    Stats stats;

    for (std::size_t i = 0; i < current.size(); ++i) {
        int j = matches.index[i];
        if (j < 0) {
            continue;
        }

        double source_point[3] = {current[i].x, current[i].y, current[i].z};
        double target_point[3] = {target[j].x, target[j].y, target[j].z};

        stats.v[0] += 1;
        for (int row = 0; row < 3; ++row) {
            stats.v[1 + row] += source_point[row];
            stats.v[4 + row] += target_point[row];
            for (int column = 0; column < 3; ++column) {
                stats.v[7 + 3 * row + column] += source_point[row] * target_point[column];
            }
        }
        stats.v[16] += matches.squared_distance[i];
    }

    return stats;
}

} // namespace icp
