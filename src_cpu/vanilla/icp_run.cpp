#include "icp.h"
#include "icp_utils.h"
#include "icp_log.h"
#include <time.h>
#include <cmath>
#include <limits>

IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* icp_opt, IcpResult* result)
{
    *result = {};
    result->final_rmse = std::numeric_limits<double>::quiet_NaN();

    IcpBuffers buffers{};
    Pose current = icp_opt->initial_pose;

    IterationLog log{};
    IcpStatus s = openIterationLog("cpu", &log);
    if (s != ICP_OK) {
        result->status = s;
        return s;
    }
    std::puts("iteration_timing=host_steps_1_to_5");
    if (icp_opt->dump_iterations) {
        Pose identity;
        identityPose(&identity);
        s = dumpIterationPoints(&log, target, &identity, "target");
        if (s == ICP_OK) {
            s = dumpIterationPoints(&log, source, &current, "source_0000");
        }
    }
    if (s == ICP_OK) {
        s = allocateIcpBuffers(&buffers, source->count);
    }
    const float cutoff = float(icp_opt->max_distance * icp_opt->max_distance);
    bool final_converged = false;

    for (int iteration = 0; s == ICP_OK && iteration < icp_opt->max_iterations;
        ++iteration) {
        timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        IcpStatistics stats{};
        Pose delta{}, next{};
        bool converged = false;

        // Step 1: point transformation.
        s = transformPointsCPU(
            source->points_h, buffers.transformed_h, source->count, &current);
        if (s != ICP_OK) {
            break;
        }

        // Step 2: correspondence search.
        s = findCorrespondencesCPU(buffers.transformed_h, target->points_h,
            source->count, target->count, cutoff, buffers.match_index_h,
            buffers.match_dist2_h);
        if (s != ICP_OK) {
            break;
        }

        // Step 3: matched-pair statistics.
        s = computeStatisticsCPU(buffers.transformed_h, target->points_h,
            buffers.match_index_h, buffers.match_dist2_h, source->count,
            &stats);
        if (s != ICP_OK) {
            break;
        }

        // Step 4: incremental rigid transform.
        s = solveRigidUpdateCPU(&stats, &delta);
        if (s != ICP_OK) {
            break;
        }

        // Step 5: pose composition and convergence.
        s = updatePoseCPU(&delta, &current, icp_opt, &next, &converged);
        if (s != ICP_OK) {
            break;
        }

        current = next;
        final_converged = converged;
        ++result->iterations_completed;

        clock_gettime(CLOCK_MONOTONIC, &end);
        const double time_ms =
            (double(end.tv_sec) - double(start.tv_sec)) * 1000.0 +
            (double(end.tv_nsec) - double(start.tv_nsec)) / 1e6;
        s = recordIteration(&log, iteration + 1, time_ms, &current, converged);
        if (s == ICP_OK && icp_opt->dump_iterations) {
            char name[32];
            std::snprintf(name, sizeof(name), "source_%04d", iteration + 1);
            s = dumpIterationPoints(&log, source, &current, name);
        }
        if (s != ICP_OK) {
            break;
        }

        if (converged && !icp_opt->fixed_iterations)
            break;
    }

    // Reevaluate steps 1-3 at the final pose for the reported error.
    if (s == ICP_OK) {
        IcpStatistics stats{};
        s = transformPointsCPU(
            source->points_h, buffers.transformed_h, source->count, &current);
        if (s == ICP_OK)
            s = findCorrespondencesCPU(buffers.transformed_h, target->points_h,
                source->count, target->count, cutoff, buffers.match_index_h,
                buffers.match_dist2_h);
        if (s == ICP_OK)
            s = computeStatisticsCPU(buffers.transformed_h, target->points_h,
                buffers.match_index_h, buffers.match_dist2_h, source->count,
                &stats);
        if (s == ICP_OK && stats.count < 3)
            s = ICP_INSUFFICIENT_MATCHES;
        if (s == ICP_OK) {
            result->pose = current;
            result->pose_valid = true;
            result->converged = final_converged;
            result->final_inliers = stats.count;
            result->final_rmse = std::sqrt(stats.sse / stats.count);
        }
    }

    freeIcpBuffers(&buffers);
    const IcpStatus log_status = closeIterationLog(&log);
    if (s == ICP_OK) {
        s = log_status;
    }
    result->status = s;
    return s;
}
