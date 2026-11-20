#pragma once
#include "../common/point_cloud.h"

struct IcpStatistics
{
    int count;
    double mu_source[3], mu_target[3], H[9], sse;
};
struct CudaIcpBuffers;

// Step 1: transform original source points on the GPU.
IcpStatus transformPointsCUDA(CudaIcpBuffers* buffers);

// Step 2: dispatch the selected correspondence search.
IcpStatus findCorrespondencesCUDA(
    CudaIcpBuffers* buffers, CorrespondenceSearch search, float max_distance2);

// Alternative CUDA search implementations.
IcpStatus findCorrespondencesKdTreeCUDA(
    CudaIcpBuffers* buffers, float max_distance2);
IcpStatus findCorrespondencesVoxelGridCUDA(
    CudaIcpBuffers* buffers, float max_distance2);

// Step 3: two-pass matched-pair statistics on the GPU.
IcpStatus computeStatisticsCUDA(CudaIcpBuffers* buffers);

// Step 4: SVD rigid update on the host.
IcpStatus solveRigidUpdateCPU(const IcpStatistics* stats, Pose* delta);

// Step 5: host pose composition and convergence check.
IcpStatus updatePoseCPU(const Pose* delta, const Pose* current,
    const IcpOptions* icp_opt, Pose* next, bool* converged);

IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* icp_opt, IcpResult* result);
