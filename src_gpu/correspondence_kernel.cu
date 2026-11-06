#include "icp.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

static IcpStatus checked(cudaError_t e, const char* action) {
    if (e == cudaSuccess) return ICP_OK;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(e));
    return ICP_CUDA_ERROR;
}

// Byte-identical to squaredDistance in icp_gold.cpp. dist2 accumulates in float, its own
// output type, so operand order and multiply-add contraction change the stored value and
// flip exact ties. Keep these two bodies textually in sync.
__device__ __forceinline__ float squaredDistance(Point a, Point b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__global__ void findCorrespondencesNaiveKernel(const Point* x, const Point* q, int n, int m,
    float cutoff, int* index, float* dist2) {
    // Device x[n], q[m], squared cutoff -> index[n], dist2[n].
    // One source per thread, scan all M targets; many-to-one, smallest index on exact
    // float distance tie. Reject nearest distance > cutoff with index=-1.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Returning early is safe in this kernel because it has no barriers. The tiled
    // kernel below must NOT do this; every lane there has to reach both __syncthreads.
    if (i >= n) return;

    const Point p = x[i]; // loaded once and reused across all m targets
    float best = FLT_MAX; // set to a value that any valid distance will be less than, so the first comparison will always succeed
    int best_j = -1; // initialize to -1 to indicate no valid match found yet
    
    // Same scan order and same strict < as findCorrespondencesCPU, so exact ties resolve
    // to the smallest target index on both sides.
    for (int j = 0; j < m; ++j) {
        const float d = squaredDistance(p, q[j]);
        if (d < best) { best = d; best_j = j; }
    }
    dist2[i] = best;
    index[i] = (best <= cutoff) ? best_j : -1;
}
__global__ void findCorrespondencesTiledKernel(const Point* x, const Point* q, int n, int m,
    float cutoff, int* index, float* dist2) {
    // Same device inputs/outputs and tie rule as naive/CPU.
    // Loads shared-memory target tiles; handles the partial final tile and source bounds.
    // EVERY lane, including i>=n, reaches BOTH barriers in every tile iteration.
    // Calculations and writes are guarded; there is no early return.
    __shared__ Point tile[BLOCK_SIZE];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;

    // Deliberately NOT `if (i >= n) return;`. A lane whose source index is out of range
    // still has to load its slot of every tile and still has to reach both barriers, so
    // out-of-range lanes are handled by guarding work rather than by leaving the kernel.
    const bool active = i < n;
    Point p = {0, 0, 0};
    if (active) p = x[i];

    float best = FLT_MAX;
    int best_j = -1;

    for (int base = 0; base < m; base += BLOCK_SIZE) {
        // Cooperative load indexed by LANE, never by i: the tile slot a lane fills has
        // nothing to do with whether that lane has a source point of its own.
        if (base + lane < m) tile[lane] = q[base + lane];
        __syncthreads(); // every slot of the tile is populated before anyone reads it

        // Partial final tile: only the first (m - base) slots were refilled this pass.
        const int count = (m - base < BLOCK_SIZE) ? (m - base) : BLOCK_SIZE;
        if (active) {
            // base+j walks the global target indices in the same ascending order as j does
            // in the naive kernel, so the strict < keeps the smallest GLOBAL index on an
            // exact tie, including a tie whose two targets fall in different tiles.
            for (int j = 0; j < count; ++j) {
                const float d = squaredDistance(p, tile[j]);
                if (d < best) { best = d; best_j = base + j; }
            }
        }
        __syncthreads(); // no lane may refill the tile while another warp still reads it
    }

    if (active) {
        dist2[i] = best;
        index[i] = (best <= cutoff) ? best_j : -1;
    }
}
IcpStatus findCorrespondencesCUDA(DeviceWorkspace* w, IcpBackend b, float cutoff) {
    // Device transformed/target -> match arrays. Selects the naive or tiled launch from
    // the backend, then checks launch errors. Nothing N*M is allocated.
    if (!w || !w->transformed_d || !w->target_d || !w->match_index_d || !w->match_dist2_d)
        return ICP_INVALID_INPUT;
    if (w->num_source <= 0 || w->num_target <= 0 || w->num_blocks <= 0) return ICP_INVALID_INPUT;
    if (w->num_blocks != (w->num_source + BLOCK_SIZE - 1) / BLOCK_SIZE) return ICP_INVALID_INPUT;
    if (!(cutoff >= 0)) return ICP_INVALID_INPUT; // also rejects NaN
    switch (b) {
    case BACKEND_CUDA_NAIVE:
        // Searches from transformed_d, the transform kernel's output, never from source_d.
        findCorrespondencesNaiveKernel<<<w->num_blocks, BLOCK_SIZE>>>(
            w->transformed_d, w->target_d, w->num_source, w->num_target,
            cutoff, w->match_index_d, w->match_dist2_d);
        return checked(cudaGetLastError(), "findCorrespondencesNaiveKernel launch");
    case BACKEND_CUDA_TILED:
        // Same grid as the naive launch. The kernel's tile load assumes the block is
        // exactly BLOCK_SIZE lanes wide, which is what the shared tile is sized for.
        findCorrespondencesTiledKernel<<<w->num_blocks, BLOCK_SIZE>>>(
            w->transformed_d, w->target_d, w->num_source, w->num_target,
            cutoff, w->match_index_d, w->match_dist2_d);
        return checked(cudaGetLastError(), "findCorrespondencesTiledKernel launch");
    default:
        return ICP_INVALID_INPUT; // includes BACKEND_CPU
    }
}
