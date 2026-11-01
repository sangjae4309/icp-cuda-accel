#include "icp/icp.hpp"
#include <cfloat>

namespace icp {

// Step 2: Find the nearest target point for each source point.
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

            // Keep the smaller index when distances are equal.
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
