#include "icp_kdtree.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <new>

// Byte-identical to squaredDistance in src_cpu/vanilla/icp_step2_correspondence
// .cpp and src_gpu/vanilla/icp_step2_correspondence.cu. The query is the first
// operand everywhere, so dx here is the same float the brute-force loop
// computes. Keep the bodies textually in sync.
static inline float squaredDistance(Point a, Point b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static inline float coord(const Point& p, int axis)
{
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

IcpStatus KdTree::build(const PointCloud* cloud)
{
    if (!cloud)
        return ICP_INVALID_INPUT;
    return build(cloud->points_h, cloud->count);
}

IcpStatus KdTree::build(const Point* src, int count)
{
    points_.clear();
    indices_.clear();
    axes_.clear();
    depth_ = 0;
    if (!src || count <= 0)
        return ICP_INVALID_INPUT;
    for (int i = 0; i < count; ++i)
        if (!std::isfinite(src[i].x) || !std::isfinite(src[i].y) ||
            !std::isfinite(src[i].z))
            return ICP_INVALID_INPUT; // NaN would break the median ordering

    try {
        points_.resize(count);
        indices_.resize(count);
        axes_.resize(count);
        std::vector<int> perm(count);
        for (int i = 0; i < count; ++i)
            perm[i] = i;
        buildRange(src, perm, 0, count, 1);
    } catch (const std::bad_alloc&) {
        points_.clear();
        indices_.clear();
        axes_.clear();
        depth_ = 0;
        return ICP_ALLOCATION_ERROR;
    }
    return ICP_OK;
}

// Places the median of perm[lo, hi) at mid along the axis of largest extent,
// then recurses into both halves. Each level is O(range) for the bounding box
// and nth_element, so the whole build is O(n log n). Recursion depth equals
// tree depth (about 17 for a 125k-point KITTI frame).
void KdTree::buildRange(
    const Point* src, std::vector<int>& perm, int lo, int hi, int level)
{
    if (lo >= hi)
        return;
    if (level > depth_)
        depth_ = level;

    // Split on the widest axis of this range's bounding box. For a KITTI scan
    // this picks x/y near the top (the scan is ~160 m wide, ~10 m tall) and
    // adapts lower down, unlike a fixed x->y->z cycle.
    Point mn = src[perm[lo]], mx = mn;
    for (int k = lo + 1; k < hi; ++k) {
        const Point& p = src[perm[k]];
        mn.x = std::min(mn.x, p.x);
        mx.x = std::max(mx.x, p.x);
        mn.y = std::min(mn.y, p.y);
        mx.y = std::max(mx.y, p.y);
        mn.z = std::min(mn.z, p.z);
        mx.z = std::max(mx.z, p.z);
    }
    const float ex = mx.x - mn.x, ey = mx.y - mn.y, ez = mx.z - mn.z;
    const int axis = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);

    // Total order (coordinate, original index): equal coordinates are split
    // deterministically, so the same input always yields the same tree on
    // every standard library. Equal coordinates may land on both sides of the
    // node; the search handles that because the invariant is <= / >=.
    const int mid = lo + (hi - lo) / 2;
    std::nth_element(perm.begin() + lo, perm.begin() + mid, perm.begin() + hi,
        [src, axis](int a, int b) {
            const float ca = coord(src[a], axis), cb = coord(src[b], axis);
            return ca < cb || (ca == cb && a < b);
        });

    points_[mid] = src[perm[mid]];
    indices_[mid] = perm[mid];
    axes_[mid] = static_cast<std::uint8_t>(axis);

    buildRange(src, perm, lo, mid, level + 1);
    buildRange(src, perm, mid + 1, hi, level + 1);
}

// Iterative depth-first search with an explicit stack, written so it ports to
// a CUDA kernel unchanged (no recursion, fixed-size local stack).
//
// Why pruning is exact in float, not just in real arithmetic: for a point r
// on the far side of a node p along axis a, |q.a - r.a| >= |q.a - p.a| in
// reals, and IEEE subtraction and squaring round monotonically, so the float
// diff2 below is <= fl((q.a - r.a)^2). squaredDistance then only adds
// non-negative terms, which also rounds monotonically, so diff2 <=
// squaredDistance(q, r). A subtree is skipped only when diff2 > best,
// which means every point in it is strictly farther than the current best.
// Points at exactly `best` are never pruned, so the smallest-index tie rule
// still sees them.
void KdTree::nearest(Point q, int* out_index, float* out_dist2) const
{
    float best = FLT_MAX;
    int best_j = -1;

    struct Range
    {
        int lo, hi;
        float bound2; // lower bound on dist2 to anything in this range
    };
    Range stack[kMaxStack];
    int sp = 0;
    if (!points_.empty())
        stack[sp++] = {0, count(), 0.0f};

    while (sp > 0) {
        Range r = stack[--sp];
        if (r.bound2 > best)
            continue; // best shrank since this range was deferred
        int lo = r.lo, hi = r.hi;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            const Point p = points_[mid];
            const int j = indices_[mid];

            // Same acceptance as the brute-force scan: a strict improvement,
            // or an exact tie with a smaller ORIGINAL index. best_j >= 0
            // reproduces brute force never accepting d == FLT_MAX.
            const float d = squaredDistance(q, p);
            if (d < best || (d == best && best_j >= 0 && j < best_j)) {
                best = d;
                best_j = j;
            }

            const int axis = axes_[mid];
            const float diff = coord(q, axis) - coord(p, axis);
            const float diff2 = diff * diff;
            int near_lo, near_hi, far_lo, far_hi;
            if (diff < 0) { // query is on the low side: left is near
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
            // Defer the far side only if it could still hold a point at or
            // below best. sp cannot overflow: at most one push per level.
            if (far_lo < far_hi && diff2 <= best)
                stack[sp++] = {far_lo, far_hi, diff2};
            lo = near_lo;
            hi = near_hi;
        }
    }
    *out_index = best_j;
    *out_dist2 = best;
}

IcpStatus KdTree::findCorrespondences(const Point* queries, int n,
    float max_distance2, int* match_index, float* match_dist2) const
{
    if (!queries || !match_index || !match_dist2 || n <= 0 || empty())
        return ICP_INVALID_INPUT;
    if (!(max_distance2 >= 0))
        return ICP_INVALID_INPUT; // also rejects NaN
    for (int i = 0; i < n; ++i) {
        int j;
        float d;
        nearest(queries[i], &j, &d);
        // Report the nearest distance even when rejected; a point exactly on
        // the cutoff is kept, as in findCorrespondencesCPU.
        match_dist2[i] = d;
        match_index[i] = (d <= max_distance2) ? j : -1;
    }
    return ICP_OK;
}

bool KdTree::checkInvariants() const
{
    const int n = count();
    if (n == 0)
        return indices_.empty() && axes_.empty() && depth_ == 0;
    if (static_cast<int>(indices_.size()) != n ||
        static_cast<int>(axes_.size()) != n || depth_ > kMaxStack)
        return false;

    std::vector<char> seen(n, 0);
    for (int k = 0; k < n; ++k) {
        const int j = indices_[k];
        if (j < 0 || j >= n || seen[j] || axes_[k] > 2)
            return false;
        seen[j] = 1;
    }

    // Each node must lie inside the box its ancestors' splits imply. Checking
    // one node against inherited bounds is equivalent to checking it against
    // every ancestor, so this is O(n) rather than O(n log n) comparisons.
    struct Frame
    {
        int lo, hi;
        float mn[3], mx[3];
    };
    std::vector<Frame> work;
    work.push_back({0, n, {-FLT_MAX, -FLT_MAX, -FLT_MAX},
        {FLT_MAX, FLT_MAX, FLT_MAX}});
    while (!work.empty()) {
        Frame f = work.back();
        work.pop_back();
        if (f.lo >= f.hi)
            continue;
        const int mid = f.lo + (f.hi - f.lo) / 2;
        const Point& p = points_[mid];
        for (int a = 0; a < 3; ++a) {
            const float c = coord(p, a);
            if (c < f.mn[a] || c > f.mx[a])
                return false;
        }
        const int axis = axes_[mid];
        Frame left = f, right = f;
        left.hi = mid;
        left.mx[axis] = coord(p, axis);
        right.lo = mid + 1;
        right.mn[axis] = coord(p, axis);
        work.push_back(left);
        work.push_back(right);
    }
    return true;
}