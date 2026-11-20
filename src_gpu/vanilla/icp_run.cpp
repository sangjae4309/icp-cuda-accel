#include "icp.h"
#include "icp_utils.h"
#include "icp_log.h"
#include <cmath>
#include <time.h>
#include <limits>

IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* icp_opt, IcpResult* result)
{
    *result = {};
    result->final_rmse = std::numeric_limits<double>::quiet_NaN();

    CudaIcpBuffers buffers{};
    Pose current = icp_opt->initial_pose;
    const float cutoff = float(icp_opt->max_distance * icp_opt->max_distance);
    const char* label = "gpu-brute-force";
    switch (icp_opt->search) {
    case SEARCH_TILED:
        label = "gpu-tiled";
        break;
    case SEARCH_KD_TREE:
        label = "gpu-kd-tree";
        break;
    case SEARCH_VOXEL_GRID:
        label = "gpu-voxel-grid";
        break;
    default:
        break;
    }
    IterationLog log{};
    IcpStatus s = openIterationLog(label, &log);
    if (s != ICP_OK) {
        result->status = s;
        return s;
    }

    std::puts(icp_opt->benchmark ? "iteration_timing=disabled_for_benchmark"
        : "iteration_timing=cuda_events_upload_steps_1_to_3_download");

    GpuIterationTimer timer{};
    if (!icp_opt->benchmark) {
        s = createIterationTimer(&timer);
        if (s != ICP_OK) {
            closeIterationLog(&log);
            result->status = s;
            return s;
        }
    }
    s = allocateIcpBuffers(&buffers, source->count, target->count);
    if (s == ICP_OK) {
        s = uploadPointClouds(&buffers, source->points_h, target->points_h);
    }
    if (s == ICP_OK && icp_opt->search == SEARCH_VOXEL_GRID) {
        s = buildVoxelGridCUDA(&buffers, cutoff);
    }
    bool final_converged = false;

    // One wall-clock interval around the complete iteration loop.
    // Setup/transfers of original clouds are complete before this point.
    std::fflush(stdout);
    timespec loop_start, loop_end;
    clock_gettime(CLOCK_MONOTONIC, &loop_start);

    for (int iteration = 0;
        s == ICP_OK && iteration < icp_opt->max_iterations &&
        (!final_converged || icp_opt->fixed_iterations);
        ++iteration) {
        IcpStatistics stats{};
        Pose delta{}, next{};
        bool converged = false;
        float time_ms = 0;

        if (!icp_opt->benchmark) {
            s = startIterationTimer(&timer);
        }
        if (s == ICP_OK) {
            s = uploadPose(&buffers, &current);
        }
        if (s != ICP_OK) {
            break;
        }

        // Step 1: point transformation.
        s = transformPointsCUDA(&buffers);
        if (s != ICP_OK) {
            break;
        }

        // Step 2: correspondence search.
        s = findCorrespondencesCUDA(&buffers, icp_opt->search, cutoff);
        if (s != ICP_OK) {
            break;
        }

        // Step 3: matched-pair statistics.
        s = computeStatisticsCUDA(&buffers);
        if (s == ICP_OK) {
            s = downloadStatistics(&buffers, &stats);
        }
        if (s == ICP_OK && !icp_opt->benchmark) {
            // Same stream as the kernels; the download also completes work.
            // This event interval excludes host SVD and pose composition.
            s = stopIterationTimer(&timer, &time_ms);
        }
        if (s != ICP_OK) {
            break;
        }

        // Step 4: incremental rigid transform on the host.
        s = solveRigidUpdateCPU(&stats, &delta);
        if (s != ICP_OK) {
            break;
        }

        // Step 5: host pose composition and convergence.
        updatePoseCPU(&delta, &current, icp_opt, &next, &converged);

        current = next;
        final_converged = converged;
        ++result->iterations_completed;
        if (!icp_opt->benchmark) {
            s = recordIteration(
                &log, iteration + 1, double(time_ms), &current, converged);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &loop_end);
    result->loop_time_ms =
        (double(loop_end.tv_sec) - double(loop_start.tv_sec)) * 1000.0 +
        (double(loop_end.tv_nsec) - double(loop_start.tv_nsec)) / 1e6;

    // Final error evaluation is outside the measured iteration loop.
    // Reevaluate steps 1-3 at the final pose for the reported error.
    if (s == ICP_OK) {
        IcpStatistics stats{};
        s = uploadPose(&buffers, &current);
        if (s == ICP_OK) {
            s = transformPointsCUDA(&buffers);
        }
        if (s == ICP_OK) {
            s = findCorrespondencesCUDA(&buffers, icp_opt->search, cutoff);
        }
        if (s == ICP_OK) {
            s = computeStatisticsCUDA(&buffers);
        }
        if (s == ICP_OK) {
            s = downloadStatistics(&buffers, &stats);
        }
        if (s == ICP_OK && stats.count < 3) {
            s = ICP_INSUFFICIENT_MATCHES;
        }
        if (s == ICP_OK) {
            result->pose = current;
            result->pose_valid = true;
            result->converged = final_converged;
            result->final_inliers = stats.count;
            result->final_rmse = std::sqrt(stats.sse / stats.count);
        }
    }

    freeIcpBuffers(&buffers);
    if (!icp_opt->benchmark) {
        freeIterationTimer(&timer);
    }
    const IcpStatus log_status = closeIterationLog(&log);
    if (s == ICP_OK) {
        s = log_status;
    }
    result->status = s;
    return s;
}
