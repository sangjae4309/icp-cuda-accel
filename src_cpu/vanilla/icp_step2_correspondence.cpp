#include "icp.h"
#include <cfloat>

// Squared distance in float. findCorrespondencesNaiveKernel carries a
// byte-identical copy of this body: dist2 accumulates in its own output type,
// so any difference in operand order or multiply-add contraction changes the
// stored float and flips exact ties. Both projects build with contraction
// pinned off; see CMakeLists.txt.
static inline float squaredDistance(Point a, Point b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

IcpStatus findCorrespondencesCPU(const Point* transformed_h,
    const Point* target_h, int n, int m, float max_distance2,
    int* match_index_h, float* match_dist2_h)
{
    // Host input transformed[n], target[m], squared cutoff.
    // Output index[n], squared distance[n]. Brute force O(N*M), no distance
    // matrix. Many-to-one allowed; exact float distance ties choose smallest
    // target index. Reject distance > cutoff with index=-1; retain nearest
    // squared distance.
    if (!transformed_h || !target_h || !match_index_h || !match_dist2_h)
        return ICP_INVALID_INPUT;
    if (n <= 0 || m <= 0)
        return ICP_INVALID_INPUT;
    if (!(max_distance2 >= 0))
        return ICP_INVALID_INPUT; // also rejects NaN
    for (int i = 0; i < n; ++i) {
        const Point p = transformed_h[i];
        float best = FLT_MAX;
        int best_j = -1;
        // Ascending j with a strict < keeps the FIRST of several equal
        // distances, which is the smallest target index. Using <= here would
        // keep the largest instead.
        for (int j = 0; j < m; ++j) {
            const float d = squaredDistance(p, target_h[j]);
            if (d < best) {
                best = d;
                best_j = j;
            }
        }
        // The nearest squared distance is reported even when the pair is
        // rejected. best <= cutoff keeps a point sitting exactly on the cutoff
        // as a match.
        match_dist2_h[i] = best;
        match_index_h[i] = (best <= max_distance2) ? best_j : -1;
    }
    return ICP_OK;
}
