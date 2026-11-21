// Host-only KD-tree test: builds a KdTree on a KITTI target frame and checks
// it against the brute-force correspondence contract, bit for bit.
//
// Build (from the repo root, any Linux/macOS machine, no GPU needed):
//   g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off
//       -Isrc_gpu/common tests/kdtree_test.cpp src_gpu/common/kdtree.cpp
//       src_gpu/common/kitti_parser.cpp src_gpu/common/io.cpp
//       src_gpu/common/pose.cpp -o build-kdtree-test
// (all on one line; no trailing backslashes here, to keep -Wcomment quiet)
// Run:
//   ./build-kdtree-test [source.bin] [target.bin] [stride]
// stride N checks every Nth source point against brute force (default 10;
// 1 = all ~125k points, ~15.5e9 distance evaluations).
#include "icp_kdtree.h"
#include "kitti_parser.h"
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Verbatim body of findCorrespondencesCPU (src_cpu/vanilla), copied because
// the CPU and GPU trees carry separate point_cloud.h headers.
static inline float bfSquaredDistance(Point a, Point b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static void bruteForce(const Point* target, int m, Point p, float cutoff,
    int* index, float* dist2)
{
    float best = FLT_MAX;
    int best_j = -1;
    for (int j = 0; j < m; ++j) {
        const float d = bfSquaredDistance(p, target[j]);
        if (d < best) {
            best = d;
            best_j = j;
        }
    }
    *dist2 = best;
    *index = (best <= cutoff) ? best_j : -1;
}

static bool sameFloat(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0)
        .count();
}

// Compares tree vs brute force for queries[0, n) step `stride`.
static int compare(const KdTree& tree, const Point* target, int m,
    const Point* queries, int n, int stride, float cutoff, const char* label)
{
    std::vector<int> kd_index(n);
    std::vector<float> kd_dist2(n);
    auto t0 = std::chrono::steady_clock::now();
    if (tree.findCorrespondences(queries, n, cutoff, kd_index.data(),
            kd_dist2.data()) != ICP_OK) {
        std::printf("[FAIL] %s: findCorrespondences returned an error\n",
            label);
        return 1;
    }
    const double kd_ms = msSince(t0);

    int checked = 0, mismatches = 0, matched = 0;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i += stride) {
        int bf_index;
        float bf_dist2;
        bruteForce(target, m, queries[i], cutoff, &bf_index, &bf_dist2);
        ++checked;
        if (bf_index >= 0)
            ++matched;
        if (bf_index != kd_index[i] || !sameFloat(bf_dist2, kd_dist2[i])) {
            if (++mismatches <= 5)
                std::printf("  mismatch q=%d: bf (%d, %.9g) kd (%d, %.9g)\n",
                    i, bf_index, bf_dist2, kd_index[i], kd_dist2[i]);
        }
    }
    const double bf_ms = msSince(t0);
    std::printf("[%s] %s: %d/%d checked, %d within cutoff, %d mismatches | "
                "kd all %d queries %.1f ms, brute force %d queries %.1f ms\n",
        mismatches ? "FAIL" : "PASS", label, checked, n, matched, mismatches,
        n, kd_ms, checked, bf_ms);
    return mismatches ? 1 : 0;
}

// Small hand-made clouds that stress exact ties, duplicates and the cutoff.
static int syntheticTests()
{
    int failures = 0;

    // Integer grid with duplicated points: many exact distance ties, and
    // duplicate coordinates that straddle split planes.
    std::vector<Point> grid;
    for (int rep = 0; rep < 2; ++rep)
        for (int x = 0; x < 9; ++x)
            for (int y = 0; y < 7; ++y)
                for (int z = 0; z < 5; ++z)
                    grid.push_back({float(x), float(y), float(z)});
    std::vector<Point> q;
    for (int x = -2; x <= 20; ++x)
        for (int y = -2; y <= 14; ++y)
            for (int z = -2; z <= 10; ++z)
                q.push_back({x * 0.5f, y * 0.5f, z * 0.5f}); // half-steps tie
    KdTree t;
    if (t.build(grid.data(), int(grid.size())) != ICP_OK ||
        !t.checkInvariants()) {
        std::printf("[FAIL] grid build/invariants\n");
        return 1;
    }
    failures += compare(t, grid.data(), int(grid.size()), q.data(),
        int(q.size()), 1, 0.25f, "synthetic grid, cutoff on half-step ties");

    // All points identical: answer must be index 0.
    std::vector<Point> same(1000, Point{1.5f, -2.0f, 3.0f});
    Point probe[2] = {{1.5f, -2.0f, 3.0f}, {0, 0, 0}};
    t.build(same.data(), int(same.size()));
    failures += compare(t, same.data(), int(same.size()), probe, 2, 1, 1.0f,
        "1000 identical points");

    // Single point, and rejection of bad input.
    Point one = {3, 4, 0};
    t.build(&one, 1);
    failures += compare(t, &one, 1, probe, 2, 1, 25.0f, "single point");
    Point bad[2] = {{0, 0, 0}, {std::nanf(""), 0, 0}};
    if (t.build(bad, 2) != ICP_INVALID_INPUT || !t.empty() ||
        t.build(nullptr, 3) != ICP_INVALID_INPUT) {
        std::printf("[FAIL] invalid input not rejected\n");
        ++failures;
    } else {
        std::printf("[PASS] NaN / null input rejected\n");
    }
    return failures;
}

int main(int argc, char** argv)
{
    const char* source_path = argc > 1 ? argv[1] : "data/000001.bin";
    const char* target_path = argc > 2 ? argv[2] : "data/000000.bin";
    const int stride = argc > 3 ? std::atoi(argv[3]) : 10;
    if (stride <= 0) {
        std::fprintf(stderr, "stride must be positive\n");
        return 2;
    }

    int failures = syntheticTests();

    PointCloud source = {}, target = {};
    if (load_kitti(source_path, &source) != ICP_OK ||
        load_kitti(target_path, &target) != ICP_OK) {
        std::fprintf(stderr, "failed to load %s / %s\n", source_path,
            target_path);
        freePointCloud(&source);
        freePointCloud(&target);
        return 2;
    }
    std::printf("source %s: %d points, target %s: %d points\n", source_path,
        source.count, target_path, target.count);

    KdTree tree;
    auto t0 = std::chrono::steady_clock::now();
    const IcpStatus st = tree.build(&target);
    const double build_ms = msSince(t0);
    if (st != ICP_OK) {
        std::printf("[FAIL] build: %s\n", statusName(st));
        return 1;
    }
    bool stored_ok = true;
    for (int k = 0; k < tree.count(); ++k) {
        const Point a = tree.nodePoints()[k];
        const Point b = target.points_h[tree.nodeIndices()[k]];
        stored_ok = stored_ok && std::memcmp(&a, &b, sizeof(Point)) == 0;
    }
    const bool inv_ok = tree.checkInvariants();
    std::printf("[%s] built KD-tree: %d nodes, depth %d, %.1f ms; split "
                "invariant %s, node points match source %s\n",
        inv_ok && stored_ok ? "PASS" : "FAIL", tree.count(), tree.depth(),
        build_ms, inv_ok ? "ok" : "BROKEN", stored_ok ? "ok" : "BROKEN");
    failures += !(inv_ok && stored_ok);

    // Identity-pose correspondences: source frame vs target frame, using the
    // project's default max_distance = 1 m, i.e. squared cutoff 1.
    failures += compare(tree, target.points_h, target.count, source.points_h,
        source.count, stride, 1.0f, "KITTI source->target, cutoff 1 m");
    // Querying the target against itself: every point must find itself (or
    // the smallest index among exact duplicates) at distance 0.
    failures += compare(tree, target.points_h, target.count, target.points_h,
        target.count, stride, 0.0f, "KITTI target->target self-match");

    freePointCloud(&source);
    freePointCloud(&target);
    std::printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
    return failures ? 1 : 0;
}