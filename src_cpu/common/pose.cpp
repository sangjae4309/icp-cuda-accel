#include "point_cloud.h"
#include <algorithm>
#include <cmath>

void identityPose(Pose* p)
{
    *p = {};
    p->R[0] = p->R[4] = p->R[8] = 1;
}

IcpStatus composePoseCPU(
    const Pose* delta_h, const Pose* current_h, Pose* next_h)
{
    Pose next;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            next.R[3 * row + col] = 0;
            for (int k = 0; k < 3; ++k) {
                next.R[3 * row + col] +=
                    delta_h->R[3 * row + k] * current_h->R[3 * k + col];
            }
        }
        next.t[row] = delta_h->t[row];
        for (int k = 0; k < 3; ++k) {
            next.t[row] += delta_h->R[3 * row + k] * current_h->t[k];
        }
    }
    *next_h = next;
    return ICP_OK;
}

IcpStatus checkConvergenceCPU(
    const Pose* delta_h, const IcpOptions* options, bool* converged)
{
    const double translation =
        std::hypot(delta_h->t[0], delta_h->t[1], delta_h->t[2]);
    const double* R = delta_h->R;
    const double sine = 0.5 * std::hypot(R[7] - R[5], R[2] - R[6], R[3] - R[1]);
    const double cosine =
        std::max(-1.0, std::min(1.0, 0.5 * (R[0] + R[4] + R[8] - 1.0)));
    const double angle = std::atan2(sine, cosine);
    *converged = translation <= options->translation_tolerance &&
        angle <= options->rotation_tolerance;
    return ICP_OK;
}
