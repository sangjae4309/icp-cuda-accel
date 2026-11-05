#include "icp.h"
__global__ void transformPointsKernel(const Point* source_d, Point* transformed_d, int n, const Pose* pose_d) {
    // TODO[TRANSFORM]: Device original source[n] + accumulated pose -> transformed[n].
    // One thread per source, guard i<n; x=R*p+t. Do not mutate source_d.
    (void)source_d; (void)transformed_d; (void)n; (void)pose_d;
}
IcpStatus transformPointsCUDA(DeviceWorkspace* w) {
    // TODO[TRANSFORM]: Device source_d/pose_d -> transformed_d; launch and check errors.
    // Do not launch the empty kernel before implementing its contract.
    (void)w; return ICP_NOT_IMPLEMENTED;
}
