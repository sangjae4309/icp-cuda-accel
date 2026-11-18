#pragma once
#include "icp.h"
#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

struct CentroidPartial
{
    int count;
    double sum_source[3], sum_target[3], sse;
};

struct CovariancePartial
{
    double H[9];
};

struct CudaIcpBuffers
{
    int num_source, num_target, num_blocks;
    Point *source_d, *target_d, *transformed_d;
    int* match_index_d;
    float* match_dist2_d;
    Pose* pose_d;
    IcpStatistics* stats_d;
    CentroidPartial* centroid_partial_d;
    CovariancePartial* covariance_partial_d;
};

IcpStatus cudaAvailability();
IcpStatus allocateIcpBuffers(CudaIcpBuffers* buffers, int n, int m);
void freeIcpBuffers(CudaIcpBuffers* buffers);
IcpStatus uploadPointClouds(
    CudaIcpBuffers* buffers, const Point* source, const Point* target);
IcpStatus uploadPose(CudaIcpBuffers* buffers, const Pose* pose);
IcpStatus downloadStatistics(
    const CudaIcpBuffers* buffers, IcpStatistics* stats);

struct GpuIterationTimer
{
    cudaEvent_t start, stop;
};

IcpStatus createIterationTimer(GpuIterationTimer* timer);
IcpStatus startIterationTimer(const GpuIterationTimer* timer);
IcpStatus stopIterationTimer(const GpuIterationTimer* timer, float* time_ms);
void freeIterationTimer(GpuIterationTimer* timer);
