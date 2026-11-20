#include "icp_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

static IcpStatus checked(cudaError_t e, const char* action)
{
    if (e == cudaSuccess)
        return ICP_OK;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(e));
    return ICP_CUDA_ERROR;
}

IcpStatus cudaAvailability()
{
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver ||
        (e == cudaSuccess && count == 0))
        return ICP_UNAVAILABLE;
    return checked(e, "cudaGetDeviceCount");
}

void freeIcpBuffers(CudaIcpBuffers* w)
{
    if (!w)
        return;
#define FREE(field)                                                            \
    if (w->field)                                                              \
    checked(cudaFree(w->field), "cudaFree " #field)
    FREE(source_d);
    FREE(target_d);
    FREE(transformed_d);
    FREE(match_index_d);
    FREE(match_dist2_d);
    FREE(pose_d);
    FREE(stats_d);
    FREE(centroid_partial_d);
    FREE(covariance_partial_d);
#undef FREE
    freeKdTreeCUDA(w);
    *w = {};
}

IcpStatus allocateIcpBuffers(CudaIcpBuffers* w, int n, int m)
{
    if (!w || n <= 0 || m <= 0 || w->source_d || w->target_d ||
        w->transformed_d || w->match_index_d || w->match_dist2_d || w->pose_d ||
        w->stats_d || w->centroid_partial_d || w->covariance_partial_d)
        return ICP_INVALID_INPUT;
    IcpStatus s = cudaAvailability();
    if (s != ICP_OK)
        return s;
    size_t blocks = (size_t(n) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (size_t(n) > SIZE_MAX / sizeof(Point) ||
        size_t(m) > SIZE_MAX / sizeof(Point) ||
        blocks > SIZE_MAX / sizeof(CentroidPartial) ||
        blocks > SIZE_MAX / sizeof(CovariancePartial))
        return ICP_ALLOCATION_ERROR;
    w->num_source = n;
    w->num_target = m;
    w->num_blocks = int(blocks);
#define ALLOC(field, count)                                                    \
    do {                                                                       \
        s = checked(cudaMalloc(reinterpret_cast<void**>(&w->field),            \
                        size_t(count) * sizeof(*w->field)),                    \
            "cudaMalloc " #field);                                             \
        if (s != ICP_OK) {                                                     \
            freeIcpBuffers(w);                                                 \
            return s;                                                          \
        }                                                                      \
    } while (0)
    ALLOC(source_d, n);
    ALLOC(target_d, m);
    ALLOC(transformed_d, n);
    ALLOC(match_index_d, n);
    ALLOC(match_dist2_d, n);
    ALLOC(pose_d, 1);
    ALLOC(stats_d, 1);
    ALLOC(centroid_partial_d, blocks);
    ALLOC(covariance_partial_d, blocks);
#undef ALLOC
    return ICP_OK;
}

IcpStatus uploadPointClouds(
    CudaIcpBuffers* w, const Point* source, const Point* target)
{
    if (!w || !source || !target || !w->source_d || !w->target_d ||
        w->num_source <= 0 || w->num_target <= 0)
        return ICP_INVALID_INPUT;
    IcpStatus s = checked(
        cudaMemcpy(w->source_d, source, size_t(w->num_source) * sizeof(Point),
            cudaMemcpyHostToDevice),
        "upload source");
    if (s != ICP_OK)
        return s;
    return checked(
        cudaMemcpy(w->target_d, target, size_t(w->num_target) * sizeof(Point),
            cudaMemcpyHostToDevice),
        "upload target");
}

IcpStatus uploadPose(CudaIcpBuffers* w, const Pose* p)
{
    if (!w || !p || !w->pose_d)
        return ICP_INVALID_INPUT;
    return checked(
        cudaMemcpy(w->pose_d, p, sizeof(Pose), cudaMemcpyHostToDevice),
        "upload pose");
}

IcpStatus downloadStatistics(const CudaIcpBuffers* w, IcpStatistics* s)
{
    if (!w || !s || !w->stats_d)
        return ICP_INVALID_INPUT;
    return checked(cudaMemcpy(s, w->stats_d, sizeof(IcpStatistics),
                       cudaMemcpyDeviceToHost),
        "download statistics");
}

IcpStatus createIterationTimer(GpuIterationTimer* timer)
{
    IcpStatus status =
        checked(cudaEventCreate(&timer->start), "create iteration start event");
    if (status != ICP_OK) {
        return status;
    }
    status =
        checked(cudaEventCreate(&timer->stop), "create iteration stop event");
    if (status != ICP_OK) {
        cudaEventDestroy(timer->start);
    }
    return status;
}

IcpStatus startIterationTimer(const GpuIterationTimer* timer)
{
    return checked(cudaEventRecord(timer->start), "record iteration start");
}

IcpStatus stopIterationTimer(const GpuIterationTimer* timer, float* time_ms)
{
    IcpStatus status =
        checked(cudaEventRecord(timer->stop), "record iteration stop");
    if (status != ICP_OK) {
        return status;
    }
    status =
        checked(cudaEventSynchronize(timer->stop), "wait for iteration stop");
    if (status != ICP_OK) {
        return status;
    }
    return checked(cudaEventElapsedTime(time_ms, timer->start, timer->stop),
        "read iteration time");
}

void freeIterationTimer(GpuIterationTimer* timer)
{
    checked(cudaEventDestroy(timer->start), "destroy iteration start event");
    checked(cudaEventDestroy(timer->stop), "destroy iteration stop event");
}
