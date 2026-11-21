#include "icp.h"
#include "icp_kdtree.h"
#include "icp_utils.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>

// GPU nearest-neighbour search over the host-built KdTree (icp_kdtree.h).
//
// The tree is built once on the host from the target cloud, copied to the
// device as three flat arrays (uploadKdTreeCUDA), and queried every ICP
// iteration by one thread per transformed source point. The traversal is a
// line-for-line port of KdTree::nearest(), so the result matches the CPU tree
// and both brute-force kernels bit for bit, ties included.

static IcpStatus checked(cudaError_t e, const char* action)
{
    if (e == cudaSuccess)
        return ICP_OK;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(e));
    return ICP_CUDA_ERROR;
}

// Per-thread traversal stack bound, shared with the host tree. A search pushes
// at most one deferred subtree per level, so depth <= kKdMaxStack is enough;
// uploadKdTreeCUDA rejects any deeper tree, and the kernel needs no overflow
// check.
constexpr int kKdMaxStack = KdTree::kMaxStack;

// Byte-identical to squaredDistance in icp_step2_correspondence.cu,
// icp_kdtree_build.cpp, and the CPU icp_step2_correspondence.cpp. Device
// functions cannot be shared across .cu files without -rdc, hence the copy.
// The query is always the first operand. Keep these bodies textually in sync.
static __device__ __forceinline__ float squaredDistance(Point a, Point b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static __device__ __forceinline__ float coord(Point p, int axis)
{
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

// A deferred subtree: node index range [lo, hi) plus a lower bound on dist2 to
// anything inside it (the squared split-plane distance when it was deferred).
struct KdRange
{
    int lo, hi;
    float bound2;
};

__global__ void findCorrespondencesKdTreeKernel(const Point* __restrict__ x,
    const Point* __restrict__ nodes, const int* __restrict__ node_index,
    const std::uint8_t* __restrict__ node_axis, int n, int m, float cutoff,
    int* __restrict__ index, float* __restrict__ dist2)
{
    // Device x[n] (transformed source), tree arrays of length m, squared
    // cutoff -> index[n], dist2[n]. One source point per thread.
    //
    // index[i] is the ORIGINAL target index (node_index[k]), not the tree
    // slot k, because step 3 reads target_d[index[i]] in original order.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Safe early return: this kernel has no __syncthreads.
    if (i >= n)
        return;

    const Point q = x[i];
    float best = FLT_MAX;
    int best_j = -1;

    // Lives in local memory (64 * 12 B per thread). Only the first `depth`
    // entries are ever touched.
    KdRange stack[kKdMaxStack];
    int sp = 0;
    stack[sp++] = {0, m, 0.0f};

    while (sp > 0) {
        const KdRange r = stack[--sp];
        if (r.bound2 > best)
            continue; // best shrank since this range was deferred
        int lo = r.lo, hi = r.hi;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            const Point p = nodes[mid];
            const int j = node_index[mid];

            // Same acceptance as the brute-force scan: a strict improvement,
            // or an exact tie with a smaller ORIGINAL index. best_j >= 0
            // reproduces brute force never accepting d == FLT_MAX.
            const float d = squaredDistance(q, p);
            if (d < best || (d == best && best_j >= 0 && j < best_j)) {
                best = d;
                best_j = j;
            }

            const int axis = node_axis[mid];
            const float diff = coord(q, axis) - coord(p, axis);
            const float diff2 = diff * diff;
            int near_lo, near_hi, far_lo, far_hi;
            if (diff < 0) { // query on the low side: left subtree is near
                near_lo = lo;
                near_hi = mid;
                far_lo = mid + 1;
                far_hi = hi;
            } else {
                near_lo = mid + 1;
                near_hi = hi;
                far_lo = lo;
                far_hi = mid;
            }
            // Exact pruning (see KdTree::nearest): diff2 <= squaredDistance to
            // any point beyond the plane, so a far side is dropped only when
            // everything in it is strictly farther than best. Points at
            // exactly best are kept for the smallest-index tie rule.
            if (far_lo < far_hi && diff2 <= best)
                stack[sp++] = {far_lo, far_hi, diff2};
            lo = near_lo;
            hi = near_hi;
        }
    }

    // Report the nearest distance even when rejected; a point exactly on the
    // cutoff is kept, matching findCorrespondencesCPU and the CUDA kernels.
    dist2[i] = best;
    index[i] = (best <= cutoff) ? best_j : -1;
}

void freeKdTreeCUDA(CudaIcpBuffers* w)
{
    if (!w)
        return;
    if (w->kd_points_d)
        checked(cudaFree(w->kd_points_d), "cudaFree kd_points_d");
    if (w->kd_indices_d)
        checked(cudaFree(w->kd_indices_d), "cudaFree kd_indices_d");
    if (w->kd_axes_d)
        checked(cudaFree(w->kd_axes_d), "cudaFree kd_axes_d");
    w->kd_points_d = nullptr;
    w->kd_indices_d = nullptr;
    w->kd_axes_d = nullptr;
    w->kd_count = 0;
}

IcpStatus uploadKdTreeCUDA(CudaIcpBuffers* w, const KdTree* tree)
{
    // Host tree -> three device arrays. Call once after allocateIcpBuffers;
    // the target never changes across ICP iterations, so neither does the
    // tree. The tree must be built over the same cloud uploaded as target_d.
    if (!w || !tree || tree->empty())
        return ICP_INVALID_INPUT;
    if (tree->count() != w->num_target)
        return ICP_INVALID_INPUT;
    if (tree->depth() > kKdMaxStack)
        return ICP_INVALID_INPUT; // the kernel's fixed stack could overflow
    if (w->kd_points_d || w->kd_indices_d || w->kd_axes_d)
        return ICP_INVALID_INPUT; // already uploaded; free first

    const size_t m = static_cast<size_t>(tree->count());
    IcpStatus s = checked(
        cudaMalloc(reinterpret_cast<void**>(&w->kd_points_d), m * sizeof(Point)),
        "cudaMalloc kd_points_d");
    if (s == ICP_OK)
        s = checked(cudaMalloc(reinterpret_cast<void**>(&w->kd_indices_d),
                        m * sizeof(int)),
            "cudaMalloc kd_indices_d");
    if (s == ICP_OK)
        s = checked(cudaMalloc(reinterpret_cast<void**>(&w->kd_axes_d),
                        m * sizeof(std::uint8_t)),
            "cudaMalloc kd_axes_d");
    if (s == ICP_OK)
        s = checked(cudaMemcpy(w->kd_points_d, tree->nodePoints(),
                        m * sizeof(Point), cudaMemcpyHostToDevice),
            "cudaMemcpy kd_points_d");
    if (s == ICP_OK)
        s = checked(cudaMemcpy(w->kd_indices_d, tree->nodeIndices(),
                        m * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy kd_indices_d");
    if (s == ICP_OK)
        s = checked(cudaMemcpy(w->kd_axes_d, tree->nodeAxes(),
                        m * sizeof(std::uint8_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kd_axes_d");

    if (s != ICP_OK) {
        freeKdTreeCUDA(w); // release whatever was allocated before the failure
        return s;
    }
    w->kd_count = tree->count();
    return ICP_OK;
}

IcpStatus findCorrespondencesKdTreeCUDA(CudaIcpBuffers* w, float cutoff)
{
    // Device transformed + tree -> match arrays. Same grid, outputs, and
    // validation as the brute-force launches in findCorrespondencesCUDA.
    if (!w || !w->transformed_d || !w->match_index_d || !w->match_dist2_d)
        return ICP_INVALID_INPUT;
    if (!w->kd_points_d || !w->kd_indices_d || !w->kd_axes_d) {
        std::fprintf(stderr,
            "findCorrespondencesKdTreeCUDA: KD-tree not uploaded "
            "(call uploadKdTreeCUDA first)\n");
        return ICP_INVALID_INPUT;
    }
    if (w->num_source <= 0 || w->num_target <= 0 || w->num_blocks <= 0)
        return ICP_INVALID_INPUT;
    if (w->kd_count != w->num_target)
        return ICP_INVALID_INPUT;
    if (w->num_blocks != (w->num_source + BLOCK_SIZE - 1) / BLOCK_SIZE)
        return ICP_INVALID_INPUT;
    if (!(cutoff >= 0))
        return ICP_INVALID_INPUT; // also rejects NaN

    // Searches from transformed_d (step 1 output), never source_d.
    findCorrespondencesKdTreeKernel<<<w->num_blocks, BLOCK_SIZE>>>(
        w->transformed_d, w->kd_points_d, w->kd_indices_d, w->kd_axes_d,
        w->num_source, w->kd_count, cutoff, w->match_index_d,
        w->match_dist2_d);
    return checked(
        cudaGetLastError(), "findCorrespondencesKdTreeKernel launch");
}