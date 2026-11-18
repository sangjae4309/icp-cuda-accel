#include "icp.h"

IcpStatus updatePoseCPU(const Pose* delta, const Pose* current,
    const IcpOptions* icp_opt, Pose* next, bool* converged)
{
    composePoseCPU(delta, current, next);
    return checkConvergenceCPU(delta, icp_opt, converged);
}
