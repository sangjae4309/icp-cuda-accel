#include "icp.h"
#include <cmath>
#include <cfloat>
#include <limits>

IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* o, IcpResult* result) {
    if (!result) return ICP_INVALID_INPUT;
    *result = {}; result->status = ICP_INVALID_INPUT;
    result->final_rmse = std::numeric_limits<double>::quiet_NaN();
    if (!o || validateCloud(source) != ICP_OK || validateCloud(target) != ICP_OK)
        return result->status;
    if (o->backend < BACKEND_CPU || o->backend > BACKEND_CUDA_TILED ||
        o->max_iterations <= 0 || !std::isfinite(o->max_distance) || o->max_distance <= 0 ||
        o->max_distance > std::sqrt(double(FLT_MAX)) ||
        !std::isfinite(o->translation_tolerance) || o->translation_tolerance < 0 ||
        !std::isfinite(o->rotation_tolerance) || o->rotation_tolerance < 0) return result->status;
    for (double x : o->initial_pose.R) if (!std::isfinite(x)) return result->status;
    for (double x : o->initial_pose.t) if (!std::isfinite(x)) return result->status;
    // Validate a proper initial rotation; this is input validation, not ICP solving.
    const double* r = o->initial_pose.R;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        double dot=0;
        for (int k=0; k<3; ++k) dot += r[3*k+i]*r[3*k+j];
        if (std::abs(dot-(i==j ? 1.0 : 0.0)) > 1e-3) return result->status;
    }
    double det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    if (std::abs(det-1) > 1e-3) return result->status;
    HostWorkspace host{}; DeviceWorkspace device{};
    bool gpu = o->backend != BACKEND_CPU;
    Pose current = o->initial_pose;
    IcpStatus s = gpu ? preallocDeviceWorkspace(&device, source->count, target->count)
                      : preallocHostWorkspace(&host, source->count);
    if (s == ICP_OK && gpu) s = uploadPointClouds(&device, source->points_h, target->points_h);
    const float cutoff = float(o->max_distance * o->max_distance);
    for (int iteration = 0; s == ICP_OK && iteration < o->max_iterations; ++iteration) {
        IcpStatistics stats{};
        Pose delta{}, next{};
        bool converged = false;
        if (gpu) {
            s = uploadPose(&device, &current);
            if (s == ICP_OK) s = transformPointsCUDA(&device);
            if (s == ICP_OK) s = findCorrespondencesCUDA(&device, o->backend, cutoff);
            if (s == ICP_OK) s = computeStatisticsCUDA(&device);
            if (s == ICP_OK) s = downloadStatistics(&device, &stats);
        } else {
            s = transformPointsCPU(source->points_h, host.transformed_h, source->count, &current);
            if (s == ICP_OK) s = findCorrespondencesCPU(host.transformed_h, target->points_h,
                source->count, target->count, cutoff, host.match_index_h, host.match_dist2_h);
            if (s == ICP_OK) s = computeStatisticsCPU(host.transformed_h, target->points_h,
                host.match_index_h, host.match_dist2_h, source->count, target->count, &stats);
        }
        if (s != ICP_OK) break;
        if (stats.count < 3) { s = ICP_INSUFFICIENT_MATCHES; break; }
        s = solveRigidUpdateCPU(&stats, &delta);
        if (s == ICP_OK) s = composePoseCPU(&delta, &current, &next);
        if (s == ICP_OK) s = checkConvergenceCPU(&delta, o, &converged);
        if (s != ICP_OK) break;
        current = next;
        ++result->iterations_completed;
        if (converged && !o->fixed_iterations) break;
    }
    if (s == ICP_OK) {
        // TODO[VALIDATION]: Re-transform ORIGINAL source at current final pose,
        // redo correspondence/statistics, then publish pose/count/RMSE/convergence.
        // Never publish stale pre-update SSE or mark the initial pose as an estimate.
        s = ICP_NOT_IMPLEMENTED;
    }
    deallocHostWorkspace(&host);
    deallocDeviceWorkspace(&device);
    result->status = s;
    return s;
}
