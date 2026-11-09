#include "icp.h"
#include <cstdlib>
#include <cstdint>
#include <cfloat>

IcpStatus preallocHostWorkspace(HostWorkspace* w, int n) {
    if (!w || n <= 0 || w->transformed_h || w->match_index_h || w->match_dist2_h)
        return ICP_INVALID_INPUT;
    if (size_t(n) > SIZE_MAX / sizeof(Point)) return ICP_ALLOCATION_ERROR;
    w->transformed_h = static_cast<Point*>(std::malloc(size_t(n) * sizeof(Point)));
    w->match_index_h = static_cast<int*>(std::malloc(size_t(n) * sizeof(int)));
    w->match_dist2_h = static_cast<float*>(std::malloc(size_t(n) * sizeof(float)));
    if (!w->transformed_h || !w->match_index_h || !w->match_dist2_h) {
        deallocHostWorkspace(w); return ICP_ALLOCATION_ERROR;
    }
    return ICP_OK;
}
void deallocHostWorkspace(HostWorkspace* w) {
    if (!w) return;
    std::free(w->transformed_h); std::free(w->match_index_h); std::free(w->match_dist2_h);
    *w = {};
}
IcpStatus transformPointsCPU(const Point* source_h, Point* transformed_h, int n, const Pose* pose_h) {
    // Host input original source[n], accumulated pose; output transformed[n].
    // x = R*p+t recomputed from the ORIGINAL source every iteration; source is never written.
    if (!source_h || !transformed_h || !pose_h || n <= 0) return ICP_INVALID_INPUT;
    const double* R = pose_h->R; // Row-major 3x3
    const double* t = pose_h->t;
    for (int i = 0; i < n; ++i) {
        // Copy first so an in-place call (transformed_h == source_h) still reads the original.
        const Point p = source_h[i];
        const double x = double(p.x), y = double(p.y), z = double(p.z);
        Point out;
        // Accumulate in double and round to float exactly once, matching transformPointsKernel
        // operand for operand so the device kernel agrees bit-for-bit. See CMakeLists.txt.
        out.x = float(R[0] * x + R[1] * y + R[2] * z + t[0]);
        out.y = float(R[3] * x + R[4] * y + R[5] * z + t[1]);
        out.z = float(R[6] * x + R[7] * y + R[8] * z + t[2]);
        transformed_h[i] = out;
    }
    return ICP_OK;
}
// Squared distance in float. findCorrespondencesNaiveKernel carries a byte-identical
// copy of this body: dist2 accumulates in its own output type, so any difference in
// operand order or multiply-add contraction changes the stored float and flips exact
// ties. Both projects build with contraction pinned off; see CMakeLists.txt.
static inline float squaredDistance(Point a, Point b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}
IcpStatus findCorrespondencesCPU(const Point* transformed_h, const Point* target_h,
    int n, int m, float max_distance2, int* match_index_h, float* match_dist2_h) {
    // Host input transformed[n], target[m], squared cutoff.
    // Output index[n], squared distance[n]. Brute force O(N*M), no distance matrix.
    // Many-to-one allowed; exact float distance ties choose smallest target index.
    // Reject distance > cutoff with index=-1; retain nearest squared distance.
    if (!transformed_h || !target_h || !match_index_h || !match_dist2_h) return ICP_INVALID_INPUT;
    if (n <= 0 || m <= 0) return ICP_INVALID_INPUT;
    if (!(max_distance2 >= 0)) return ICP_INVALID_INPUT; // also rejects NaN
    for (int i = 0; i < n; ++i) {
        const Point p = transformed_h[i];
        float best = FLT_MAX;
        int best_j = -1;
        // Ascending j with a strict < keeps the FIRST of several equal distances, which
        // is the smallest target index. Using <= here would keep the largest instead.
        for (int j = 0; j < m; ++j) {
            const float d = squaredDistance(p, target_h[j]);
            if (d < best) { best = d; best_j = j; }
        }
        // The nearest squared distance is reported even when the pair is rejected.
        // best <= cutoff keeps a point sitting exactly on the cutoff as a match.
        match_dist2_h[i] = best;
        match_index_h[i] = (best <= max_distance2) ? best_j : -1;
    }
    return ICP_OK;
}
IcpStatus computeStatisticsCPU(const Point* transformed_h, const Point* target_h,
    const int* match_index_h, const float* match_dist2_h, int n, int m, IcpStatistics* stats_h) {
    // TODO[STATISTICS]: Host inputs transformed[n], target[m], matches[n], distances[n].
    // Output count, double SSE, centroids and centered H[9]. Only valid matched pairs.
    // Count duplicate target selections separately; never use the full-target centroid.
    // Pass A: sums/count/SSE and centroids. Pass B: sum((x-mu_x)*(q-mu_q)^T).
    (void)transformed_h; (void)target_h; (void)match_index_h; (void)match_dist2_h;
    (void)n; (void)m; (void)stats_h;
    return ICP_NOT_IMPLEMENTED;
}
