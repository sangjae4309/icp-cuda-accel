#include "icp.h"
#include "icp_kdtree.h"
#include "icp_utils.h"
#include "kitti_parser.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool cudaOk(cudaError_t e, const char* what)
{
    if (e == cudaSuccess)
        return true;
    std::printf("  CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

static bool sameFloat(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// Runs one search over the queries in transformed_d and downloads the result.
// Outputs are poisoned first so a thread that never writes shows up as a
// mismatch. One warm-up run, then one run timed with CUDA events.
static bool runSearch(CudaIcpBuffers* w, CorrespondenceSearch search,
    float cutoff, std::vector<int>* index, std::vector<float>* dist2,
    float* kernel_ms)
{
    const size_t n = size_t(w->num_source);
    if (!cudaOk(cudaMemset(w->match_index_d, 0x7F, n * sizeof(int)),
            "poison match_index_d") ||
        !cudaOk(cudaMemset(w->match_dist2_d, 0x7F, n * sizeof(float)),
            "poison match_dist2_d"))
        return false;

    IcpStatus s = findCorrespondencesCUDA(w, search, cutoff); // warm-up
    if (s != ICP_OK) {
        std::printf("  findCorrespondencesCUDA: %s\n", statusName(s));
        return false;
    }
    // Catches faults inside the kernel (e.g. illegal address).
    if (!cudaOk(cudaDeviceSynchronize(), "kernel execution"))
        return false;

    cudaEvent_t start, stop;
    if (!cudaOk(cudaEventCreate(&start), "cudaEventCreate") ||
        !cudaOk(cudaEventCreate(&stop), "cudaEventCreate"))
        return false;
    cudaEventRecord(start);
    s = findCorrespondencesCUDA(w, search, cutoff);
    cudaEventRecord(stop);
    bool ok = s == ICP_OK && cudaOk(cudaEventSynchronize(stop), "timed run");
    if (ok)
        cudaEventElapsedTime(kernel_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!ok)
        return false;

    index->resize(n);
    dist2->resize(n);
    return cudaOk(cudaMemcpy(index->data(), w->match_index_d, n * sizeof(int),
                      cudaMemcpyDeviceToHost),
               "download match_index_d") &&
        cudaOk(cudaMemcpy(dist2->data(), w->match_dist2_d, n * sizeof(float),
                   cudaMemcpyDeviceToHost),
            "download match_dist2_d");
}

static int countMismatches(const char* name, const std::vector<int>& ref_i,
    const std::vector<float>& ref_d, const std::vector<int>& got_i,
    const std::vector<float>& got_d)
{
    int bad = 0;
    for (size_t i = 0; i < ref_i.size(); ++i) {
        if (ref_i[i] != got_i[i] || !sameFloat(ref_d[i], got_d[i])) {
            if (++bad <= 5)
                std::printf("  %s mismatch q=%zu: expected (%d, %.9g) got "
                            "(%d, %.9g)\n",
                    name, i, ref_i[i], ref_d[i], got_i[i], got_d[i]);
        }
    }
    return bad;
}

// Build the tree on the host, upload target + tree, run the GPU KD-tree and
// GPU naive search on the same queries, compare both ways.
static int runCase(const char* label, const Point* target, int m,
    const Point* queries, int n, float cutoff)
{
    KdTree tree;
    if (tree.build(target, m) != ICP_OK) {
        std::printf("[FAIL] %s: host tree build failed\n", label);
        return 1;
    }
    std::vector<int> host_i(n);
    std::vector<float> host_d(n);
    tree.findCorrespondences(queries, n, cutoff, host_i.data(), host_d.data());

    CudaIcpBuffers w{};
    IcpStatus s = allocateIcpBuffers(&w, n, m);
    if (s == ICP_OK)
        s = uploadPointClouds(&w, queries, target);
    if (s == ICP_OK &&
        !cudaOk(cudaMemcpy(w.transformed_d, w.source_d, size_t(n) * sizeof(Point),
                    cudaMemcpyDeviceToDevice),
            "copy queries to transformed_d"))
        s = ICP_CUDA_ERROR;
    if (s == ICP_OK)
        s = uploadKdTreeCUDA(&w, &tree);
    if (s != ICP_OK) {
        std::printf("[FAIL] %s: setup failed (%s)\n", label, statusName(s));
        freeKdTreeCUDA(&w);
        freeIcpBuffers(&w);
        return 1;
    }

    std::vector<int> kd_i, bf_i;
    std::vector<float> kd_d, bf_d;
    float kd_ms = 0, bf_ms = 0;
    const bool ran =
        runSearch(&w, SEARCH_KD_TREE, cutoff, &kd_i, &kd_d, &kd_ms) &&
        runSearch(&w, SEARCH_BRUTE_FORCE, cutoff, &bf_i, &bf_d, &bf_ms);
    freeKdTreeCUDA(&w);
    freeIcpBuffers(&w);
    if (!ran) {
        std::printf("[FAIL] %s: a GPU search did not run\n", label);
        return 1;
    }

    const int vs_bf = countMismatches("gpu-kd vs gpu-naive", bf_i, bf_d, kd_i, kd_d);
    const int vs_host = countMismatches("gpu-kd vs host-kd", host_i, host_d, kd_i, kd_d);
    int matched = 0;
    for (int i = 0; i < n; ++i)
        matched += kd_i[i] >= 0;
    const bool pass = vs_bf == 0 && vs_host == 0;
    std::printf("[%s] %s: n=%d m=%d depth=%d, %d within cutoff | mismatches "
                "vs naive %d, vs host tree %d | kd %.3f ms, naive %.3f ms "
                "(%.1fx)\n",
        pass ? "PASS" : "FAIL", label, n, m, tree.depth(), matched, vs_bf,
        vs_host, kd_ms, bf_ms, kd_ms > 0 ? bf_ms / kd_ms : 0.0);
    return pass ? 0 : 1;
}

// Small hand-made clouds that stress exact ties, duplicates and the cutoff.
static int syntheticCases()
{
    int failures = 0;

    std::vector<Point> grid; // integer grid, every point duplicated
    for (int rep = 0; rep < 2; ++rep)
        for (int x = 0; x < 9; ++x)
            for (int y = 0; y < 7; ++y)
                for (int z = 0; z < 5; ++z)
                    grid.push_back({float(x), float(y), float(z)});
    std::vector<Point> q; // half-steps: equidistant from several grid points
    for (int x = -2; x <= 20; ++x)
        for (int y = -2; y <= 14; ++y)
            for (int z = -2; z <= 10; ++z)
                q.push_back({x * 0.5f, y * 0.5f, z * 0.5f});
    failures += runCase("synthetic grid, half-step ties", grid.data(),
        int(grid.size()), q.data(), int(q.size()), 0.25f);

    std::vector<Point> same(1000, Point{1.5f, -2.0f, 3.0f});
    Point probe[3] = {{1.5f, -2.0f, 3.0f}, {0, 0, 0}, {100, 100, 100}};
    failures += runCase("1000 identical points", same.data(),
        int(same.size()), probe, 3, 1.0f);

    Point one = {3, 4, 0};
    failures += runCase("single point", &one, 1, probe, 3, 25.0f);

    // n not a multiple of BLOCK_SIZE: exercises the i >= n guard.
    failures += runCase("partial block (n=257)", grid.data(),
        int(grid.size()), q.data(), 257, 1.0f);
    return failures;
}

static int kittiCases(const char* source_path, const char* target_path)
{
    PointCloud source = {}, target = {};
    if (load_kitti(source_path, &source) != ICP_OK ||
        load_kitti(target_path, &target) != ICP_OK) {
        std::printf("[FAIL] could not load %s / %s\n", source_path,
            target_path);
        freePointCloud(&source);
        freePointCloud(&target);
        return 1;
    }
    std::printf("source %s: %d points, target %s: %d points\n", source_path,
        source.count, target_path, target.count);

    int failures = 0;
    failures += runCase("KITTI source->target, cutoff 1 m", target.points_h,
        target.count, source.points_h, source.count, 1.0f);

    // Source moved by 2 deg yaw + 0.5 m, as it would be mid-ICP.
    std::vector<Point> moved(source.count);
    const double c = std::cos(2.0 * M_PI / 180.0), sn = std::sin(2.0 * M_PI / 180.0);
    for (int i = 0; i < source.count; ++i) {
        const Point p = source.points_h[i];
        moved[i] = {float(c * p.x - sn * p.y + 0.5), float(sn * p.x + c * p.y),
            p.z};
    }
    failures += runCase("KITTI moved source->target, cutoff 1 m",
        target.points_h, target.count, moved.data(), source.count, 1.0f);

    failures += runCase("KITTI target->target self-match", target.points_h,
        target.count, target.points_h, target.count, 0.0f);

    freePointCloud(&source);
    freePointCloud(&target);
    return failures;
}

int main(int argc, char** argv)
{
    bool synthetic_only = false;
    const char* paths[2] = {"data/000001.bin", "data/000000.bin"};
    int npaths = 0;
    for (int a = 1; a < argc; ++a) {
        if (std::strcmp(argv[a], "--synthetic-only") == 0)
            synthetic_only = true;
        else if (npaths < 2)
            paths[npaths++] = argv[a];
    }

    if (cudaAvailability() != ICP_OK) {
        std::printf("No usable CUDA device.\n");
        return 2;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    int failures = syntheticCases();
    if (!synthetic_only)
        failures += kittiCases(paths[0], paths[1]);

    std::printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
    return failures ? 1 : 0;
}
