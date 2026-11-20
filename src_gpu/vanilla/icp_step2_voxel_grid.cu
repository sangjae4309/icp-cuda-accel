#include "icp.h"
#include "icp_utils.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>

static IcpStatus checked(cudaError_t error, const char* action)
{
    if (error == cudaSuccess)
        return ICP_OK;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(error));
    return ICP_CUDA_ERROR;
}

__device__ __forceinline__ int voxelCoordinate(float value, float voxel_size)
{
    const double coordinate = floor(double(value) / double(voxel_size));
    if (coordinate < double(INT_MIN))
        return INT_MIN;
    if (coordinate > double(INT_MAX))
        return INT_MAX;
    return int(coordinate);
}

__device__ __forceinline__ uint32_t mixBits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__device__ __forceinline__ unsigned int voxelBucket(
    int x, int y, int z, int bucket_count)
{
    const uint32_t hash = mixBits(uint32_t(x) + 0x9e3779b9U) ^
        mixBits(uint32_t(y) + 0x85ebca6bU) ^
        mixBits(uint32_t(z) + 0xc2b2ae35U);
    return hash % uint32_t(bucket_count);
}

__global__ void buildVoxelGridKernel(const Point* target, int count,
    float voxel_size, int* heads, int* next)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const Point point = target[i];
    const int x = voxelCoordinate(point.x, voxel_size);
    const int y = voxelCoordinate(point.y, voxel_size);
    const int z = voxelCoordinate(point.z, voxel_size);
    const unsigned int bucket = voxelBucket(x, y, z, count);
    next[i] = atomicExch(&heads[bucket], i);
}

__device__ __forceinline__ int offsetVoxelCoordinate(int value, int offset)
{
    const long long result = static_cast<long long>(value) + offset;
    if (result < INT_MIN)
        return INT_MIN;
    if (result > INT_MAX)
        return INT_MAX;
    return int(result);
}

__device__ __forceinline__ float squaredDistance(Point a, Point b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__global__ void findCorrespondencesVoxelGridKernel(const Point* source,
    const Point* target, int source_count, int target_count, float cutoff,
    float voxel_size, const int* heads, const int* next, int* match_index,
    float* match_dist2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= source_count)
        return;

    const Point point = source[i];
    const int x = voxelCoordinate(point.x, voxel_size);
    const int y = voxelCoordinate(point.y, voxel_size);
    const int z = voxelCoordinate(point.z, voxel_size);
    float best = FLT_MAX;
    int best_index = -1;

    for (int dx = -1; dx <= 1; ++dx) {
        const int cell_x = offsetVoxelCoordinate(x, dx);
        for (int dy = -1; dy <= 1; ++dy) {
            const int cell_y = offsetVoxelCoordinate(y, dy);
            for (int dz = -1; dz <= 1; ++dz) {
                const int cell_z = offsetVoxelCoordinate(z, dz);
                const unsigned int bucket =
                    voxelBucket(cell_x, cell_y, cell_z, target_count);
                for (int j = heads[bucket]; j >= 0; j = next[j]) {
                    const Point candidate = target[j];
                    if (voxelCoordinate(candidate.x, voxel_size) != cell_x ||
                        voxelCoordinate(candidate.y, voxel_size) != cell_y ||
                        voxelCoordinate(candidate.z, voxel_size) != cell_z)
                        continue;
                    const float distance = squaredDistance(point, candidate);
                    if (distance < best ||
                        (best_index >= 0 && distance == best &&
                            j < best_index)) {
                        best = distance;
                        best_index = j;
                    }
                }
            }
        }
    }

    // The 27-cell neighborhood contains every point within the cutoff. If it
    // contains no accepted match, scan all targets to preserve the exact
    // nearest distance for rejected points too.
    if (best_index < 0 || best > cutoff) {
        best = FLT_MAX;
        best_index = -1;
        for (int j = 0; j < target_count; ++j) {
            const float distance = squaredDistance(point, target[j]);
            if (distance < best) {
                best = distance;
                best_index = j;
            }
        }
    }

    match_dist2[i] = best;
    match_index[i] = best <= cutoff ? best_index : -1;
}

IcpStatus initializeVoxelGridCUDA(CudaIcpBuffers* w)
{
    if (!w || !w->target_d || !w->voxel_heads_d || !w->voxel_next_d ||
        w->num_target <= 0 || !(w->voxel_size > 0))
        return ICP_INVALID_INPUT;
    IcpStatus status = checked(
        cudaMemset(w->voxel_heads_d, 0xff,
            size_t(w->num_target) * sizeof(*w->voxel_heads_d)),
        "clear voxel grid heads");
    if (status != ICP_OK)
        return status;
    const int blocks = (w->num_target - 1) / BLOCK_SIZE + 1;
    buildVoxelGridKernel<<<blocks, BLOCK_SIZE>>>(w->target_d, w->num_target,
        w->voxel_size, w->voxel_heads_d, w->voxel_next_d);
    return checked(cudaGetLastError(), "buildVoxelGridKernel launch");
}

IcpStatus findCorrespondencesVoxelGridCUDA(
    CudaIcpBuffers* w, float cutoff)
{
    if (!w || !w->transformed_d || !w->target_d || !w->match_index_d ||
        !w->match_dist2_d || !w->voxel_heads_d || !w->voxel_next_d ||
        !w->voxel_grid_ready || w->num_source <= 0 || w->num_target <= 0 ||
        w->num_blocks <= 0 || !(cutoff >= 0) || cutoff != w->voxel_cutoff)
        return ICP_INVALID_INPUT;
    if (w->num_blocks != (w->num_source + BLOCK_SIZE - 1) / BLOCK_SIZE)
        return ICP_INVALID_INPUT;
    findCorrespondencesVoxelGridKernel<<<w->num_blocks, BLOCK_SIZE>>>(
        w->transformed_d, w->target_d, w->num_source, w->num_target, cutoff,
        w->voxel_size, w->voxel_heads_d, w->voxel_next_d, w->match_index_d,
        w->match_dist2_d);
    return checked(
        cudaGetLastError(), "findCorrespondencesVoxelGridKernel launch");
}
