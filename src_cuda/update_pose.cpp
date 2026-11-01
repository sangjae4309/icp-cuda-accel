#include "icp.hpp"

namespace icp {

// Step 5: T_next = delta * T_current (left composition).
Pose compose(const Pose& delta, const Pose& pose) {
    Pose updated;

    for (int row = 0; row < 3; ++row) {
        updated.t[row] = delta.t[row];
        for (int k = 0; k < 3; ++k) {
            updated.t[row] += delta.R[3 * row + k] * pose.t[k];
        }

        for (int column = 0; column < 3; ++column) {
            updated.R[3 * row + column] = 0;
            for (int k = 0; k < 3; ++k) {
                updated.R[3 * row + column] += delta.R[3 * row + k] * pose.R[3 * k + column];
            }
        }
    }

    return updated;
}

} // namespace icp
