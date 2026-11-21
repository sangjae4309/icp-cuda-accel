#pragma once
#include "point_cloud.h"
#include <cstdint>
#include <vector>

// Static 3-D KD-tree over a target cloud (e.g. a KITTI frame from
// load_kitti()), built once on the host and queried for exact nearest
// neighbours.
//
// Layout: an implicit, pointer-free balanced tree stored in three parallel
// arrays of length count(). The node for index range [lo, hi) sits at
//     mid = lo + (hi - lo) / 2
// with its left subtree in [lo, mid) and its right subtree in [mid + 1, hi).
// No child pointers are stored, so the arrays can be copied to the GPU with
// three cudaMemcpy calls and traversed there with the same range arithmetic.
//
//   nodePoints()[k]   target point stored at node k (tree order)
//   nodeIndices()[k]  that point's index in the ORIGINAL target cloud
//   nodeAxes()[k]     split axis at node k: 0 = x, 1 = y, 2 = z
//
// Invariant along nodeAxes()[mid]: every point in [lo, mid) has a coordinate
// <= the node's, and every point in [mid + 1, hi) has a coordinate >= it.
//
// Search contract: identical to findCorrespondencesCPU and the brute-force
// CUDA kernels, bit for bit. The squared distance uses the same float formula
// and operand order, an exact distance tie resolves to the smallest ORIGINAL
// target index, and pruning is exact (see kdtree.cpp), so the tree returns the
// same index and dist2 as a full scan. This relies on the project's
// -ffp-contract=off / -fmad=false build flags, as brute-force parity already
// does.
class KdTree
{
  public:
    // Bound on the traversal stack. A balanced tree over at most INT_MAX
    // points is at most 31 levels deep, and a search keeps at most one
    // deferred subtree per level.
    static constexpr int kMaxStack = 64;

    KdTree() = default;

    // Copies the cloud's points; the cloud may be freed afterwards.
    // Rejects null/empty input and non-finite coordinates.
    IcpStatus build(const PointCloud* cloud);
    IcpStatus build(const Point* points, int count);

    // Exact nearest neighbour of `query`. Writes the original target index
    // (-1 only if every squared distance is >= FLT_MAX, which mirrors the
    // brute-force loop) and the squared distance (FLT_MAX in that case).
    void nearest(Point query, int* index, float* dist2) const;

    // Drop-in host equivalent of findCorrespondencesCPU over the tree's
    // target: index[i] = -1 when the nearest dist2 > max_distance2, and
    // dist2[i] always holds the nearest squared distance.
    IcpStatus findCorrespondences(const Point* queries, int n,
        float max_distance2, int* match_index, float* match_dist2) const;

    bool empty() const
    {
        return points_.empty();
    }

    int count() const
    {
        return static_cast<int>(points_.size());
    }

    int depth() const
    {
        return depth_;
    }

    const Point* nodePoints() const
    {
        return points_.data();
    }

    const int* nodeIndices() const
    {
        return indices_.data();
    }

    const std::uint8_t* nodeAxes() const
    {
        return axes_.data();
    }

    // Walks the whole tree and checks the split invariant and that
    // nodeIndices() is a permutation of [0, count()). For tests; O(n log n).
    bool checkInvariants() const;

  private:
    void buildRange(const Point* src, std::vector<int>& perm, int lo, int hi,
        int level);

    std::vector<Point> points_;
    std::vector<int> indices_;
    std::vector<std::uint8_t> axes_;
    int depth_ = 0;
};