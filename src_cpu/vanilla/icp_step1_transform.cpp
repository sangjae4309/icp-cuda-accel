#include "icp.h"

IcpStatus transformPointsCPU(
    const Point* source_h, Point* transformed_h, int n, const Pose* pose_h)
{
    // Host input original source[n], accumulated pose; output transformed[n].
    // x = R*p+t recomputed from the ORIGINAL source every iteration; source is
    // never written.
    if (!source_h || !transformed_h || !pose_h || n <= 0)
        return ICP_INVALID_INPUT;
    const double* R = pose_h->R; // Row-major 3x3
    const double* t = pose_h->t;
    for (int i = 0; i < n; ++i) {
        // Copy first so an in-place call (transformed_h == source_h) still
        // reads the original.
        const Point p = source_h[i];
        const double x = double(p.x), y = double(p.y), z = double(p.z);
        Point out;
        // Accumulate in double and round to float exactly once, matching
        // transformPointsKernel operand for operand so the device kernel agrees
        // bit-for-bit. See CMakeLists.txt.
        out.x = float(R[0] * x + R[1] * y + R[2] * z + t[0]);
        out.y = float(R[3] * x + R[4] * y + R[5] * z + t[1]);
        out.z = float(R[6] * x + R[7] * y + R[8] * z + t[2]);
        transformed_h[i] = out;
    }
    return ICP_OK;
}
