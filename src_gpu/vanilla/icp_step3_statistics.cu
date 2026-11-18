#include "icp.h"
#include "icp_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

// All lanes enter these trees, including lanes with no source point or partial.
__device__ void reduceCentroids(CentroidPartial* values)
{
    const int lane = threadIdx.x;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            values[lane].count += values[lane + stride].count;
            values[lane].sse += values[lane + stride].sse;
            for (int d = 0; d < 3; ++d) {
                values[lane].sum_source[d] +=
                    values[lane + stride].sum_source[d];
                values[lane].sum_target[d] +=
                    values[lane + stride].sum_target[d];
            }
        }
        __syncthreads();
    }
}

__device__ void reduceCovariance(CovariancePartial* values)
{
    const int lane = threadIdx.x;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            for (int d = 0; d < 9; ++d) {
                values[lane].H[d] += values[lane + stride].H[d];
            }
        }
        __syncthreads();
    }
}

__global__ void accumulateCentroidPartialsKernel(const Point* x, const Point* q,
    const int* index, const float* dist2, int n, int m,
    CentroidPartial* partial)
{
    __shared__ CentroidPartial values[BLOCK_SIZE];
    const int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * blockDim.x + lane;
    CentroidPartial value{};
    if (i < size_t(n) && index[i] >= 0 && index[i] < m) {
        const Point source = x[i];
        const Point target = q[index[i]];
        value.count = 1;
        value.sse = double(dist2[i]);
        value.sum_source[0] = source.x;
        value.sum_source[1] = source.y;
        value.sum_source[2] = source.z;
        value.sum_target[0] = target.x;
        value.sum_target[1] = target.y;
        value.sum_target[2] = target.z;
    }
    values[lane] = value;
    reduceCentroids(values);
    if (lane == 0) {
        partial[blockIdx.x] = values[0];
    }
}

__global__ void finalizeCentroidsKernel(
    const CentroidPartial* partial, int blocks, IcpStatistics* stats)
{
    __shared__ CentroidPartial values[BLOCK_SIZE];
    const int lane = threadIdx.x;
    CentroidPartial value{};
    for (int b = lane; b < blocks; b += BLOCK_SIZE) {
        value.count += partial[b].count;
        value.sse += partial[b].sse;
        for (int d = 0; d < 3; ++d) {
            value.sum_source[d] += partial[b].sum_source[d];
            value.sum_target[d] += partial[b].sum_target[d];
        }
    }
    values[lane] = value;
    reduceCentroids(values);
    if (lane == 0) {
        IcpStatistics result{};
        result.count = values[0].count;
        result.sse = values[0].sse;
        if (result.count > 0) {
            for (int d = 0; d < 3; ++d) {
                result.mu_source[d] = values[0].sum_source[d] / result.count;
                result.mu_target[d] = values[0].sum_target[d] / result.count;
            }
        }
        *stats = result;
    }
}

__global__ void accumulateCovariancePartialsKernel(const Point* x,
    const Point* q, const int* index, int n, int m, const IcpStatistics* stats,
    CovariancePartial* partial)
{
    __shared__ CovariancePartial values[BLOCK_SIZE];
    const int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * blockDim.x + lane;
    CovariancePartial value{};
    if (i < size_t(n) && index[i] >= 0 && index[i] < m) {
        const Point source = x[i];
        const Point target = q[index[i]];
        const double a[3] = {double(source.x) - stats->mu_source[0],
            double(source.y) - stats->mu_source[1],
            double(source.z) - stats->mu_source[2]};
        const double b[3] = {double(target.x) - stats->mu_target[0],
            double(target.y) - stats->mu_target[1],
            double(target.z) - stats->mu_target[2]};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                value.H[3 * row + col] = a[row] * b[col];
            }
        }
    }
    values[lane] = value;
    reduceCovariance(values);
    if (lane == 0) {
        partial[blockIdx.x] = values[0];
    }
}

__global__ void finalizeCovarianceKernel(
    const CovariancePartial* partial, int blocks, IcpStatistics* stats)
{
    __shared__ CovariancePartial values[BLOCK_SIZE];
    const int lane = threadIdx.x;
    CovariancePartial value{};
    for (int b = lane; b < blocks; b += BLOCK_SIZE) {
        for (int d = 0; d < 9; ++d) {
            value.H[d] += partial[b].H[d];
        }
    }
    values[lane] = value;
    reduceCovariance(values);
    if (lane == 0) {
        for (int d = 0; d < 9; ++d) {
            stats->H[d] = values[0].H[d];
        }
    }
}

static IcpStatus checkLaunch(const char* kernel)
{
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        return ICP_OK;
    }
    std::fprintf(stderr, "%s: %s\n", kernel, cudaGetErrorString(error));
    return ICP_CUDA_ERROR;
}

IcpStatus computeStatisticsCUDA(CudaIcpBuffers* w)
{
    if (!w || !w->transformed_d || !w->target_d || !w->match_index_d ||
        !w->match_dist2_d || !w->stats_d || !w->centroid_partial_d ||
        !w->covariance_partial_d || w->num_source <= 0 || w->num_target <= 0) {
        return ICP_INVALID_INPUT;
    }
    if (size_t(w->num_blocks) !=
        (size_t(w->num_source) + BLOCK_SIZE - 1) / BLOCK_SIZE) {
        return ICP_INVALID_INPUT;
    }
    accumulateCentroidPartialsKernel<<<w->num_blocks, BLOCK_SIZE>>>(
        w->transformed_d, w->target_d, w->match_index_d, w->match_dist2_d,
        w->num_source, w->num_target, w->centroid_partial_d);
    IcpStatus status = checkLaunch("accumulateCentroidPartialsKernel");
    if (status != ICP_OK) {
        return status;
    }
    finalizeCentroidsKernel<<<1, BLOCK_SIZE>>>(
        w->centroid_partial_d, w->num_blocks, w->stats_d);
    status = checkLaunch("finalizeCentroidsKernel");
    if (status != ICP_OK) {
        return status;
    }
    accumulateCovariancePartialsKernel<<<w->num_blocks, BLOCK_SIZE>>>(
        w->transformed_d, w->target_d, w->match_index_d, w->num_source,
        w->num_target, w->stats_d, w->covariance_partial_d);
    status = checkLaunch("accumulateCovariancePartialsKernel");
    if (status != ICP_OK) {
        return status;
    }
    finalizeCovarianceKernel<<<1, BLOCK_SIZE>>>(
        w->covariance_partial_d, w->num_blocks, w->stats_d);
    // Execution errors are surfaced by the subsequent synchronous statistics
    // download.
    return checkLaunch("finalizeCovarianceKernel");
}
