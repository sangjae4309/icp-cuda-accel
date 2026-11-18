#include "icp.h"

IcpStatus computeStatisticsCPU(const Point* transformed_h,
    const Point* target_h, const int* match_index_h, const float* match_dist2_h,
    int n, IcpStatistics* stats_h)
{
    IcpStatistics stats{};
    // First pass: matched-pair sums, including repeated target selections.
    for (int i = 0; i < n; ++i) {
        const int j = match_index_h[i];
        if (j == -1) {
            continue;
        }
        const Point x = transformed_h[i];
        const Point q = target_h[j];
        ++stats.count;
        stats.sse += double(match_dist2_h[i]);
        stats.mu_source[0] += double(x.x);
        stats.mu_source[1] += double(x.y);
        stats.mu_source[2] += double(x.z);
        stats.mu_target[0] += double(q.x);
        stats.mu_target[1] += double(q.y);
        stats.mu_target[2] += double(q.z);
    }
    if (stats.count == 0) {
        *stats_h = stats;
        return ICP_OK;
    }
    for (int k = 0; k < 3; ++k) {
        stats.mu_source[k] /= stats.count;
        stats.mu_target[k] /= stats.count;
    }
    // Second pass: centered covariance, without centered-point arrays.
    for (int i = 0; i < n; ++i) {
        const int j = match_index_h[i];
        if (j == -1) {
            continue;
        }
        const Point x = transformed_h[i];
        const Point q = target_h[j];
        const double a[3] = {double(x.x) - stats.mu_source[0],
            double(x.y) - stats.mu_source[1], double(x.z) - stats.mu_source[2]};
        const double b[3] = {double(q.x) - stats.mu_target[0],
            double(q.y) - stats.mu_target[1], double(q.z) - stats.mu_target[2]};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                stats.H[3 * row + col] += a[row] * b[col];
            }
        }
    }
    *stats_h = stats;
    return ICP_OK;
}
