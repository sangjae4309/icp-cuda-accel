#include "vanilla/icp.h"
#include "common/kitti_parser.h"
#include "feature/icp.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <time.h>

// Minimal entry point: no result exporter, dry-run mode, or benchmark
// interface.
int main(int argc, char** argv)
{
    if (argc == 2 && !std::strcmp(argv[1], "--help")) {
        std::puts("Usage: icp_run source.bin target.bin [vanilla|feature]");
        return 0;
    }
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Use --help for usage.\n");
        return 1;
    }

    IcpOptions options{};
    defaultOptions(&options);
    const char* benchmark = std::getenv("ICP_BENCHMARK");
    options.benchmark = benchmark && std::strcmp(benchmark, "1") == 0;
    options.fixed_iterations = options.benchmark;
    const char* iterations = std::getenv("ICP_ITERATIONS");
    if (iterations) {
        options.max_iterations = std::atoi(iterations);
        if (options.max_iterations <= 0) {
            std::fprintf(stderr, "ICP_ITERATIONS must be positive.\n");
            return 1;
        }
    }
    const char* dump = std::getenv("ICP_DUMP");
    options.dump_iterations =
        !options.benchmark && dump && std::strcmp(dump, "1") == 0;

    PointCloud source{}, target{};
    IcpStatus status = load_kitti(argv[1], &source);
    if (status == ICP_OK)
        status = load_kitti(argv[2], &target);

    const char* method = argc == 4 ? argv[3] : "vanilla";
    if (std::strcmp(method, "vanilla") && std::strcmp(method, "feature"))
        status = ICP_INVALID_INPUT;

    if ((options.dump_iterations || options.benchmark) &&
        !std::strcmp(method, "feature")) {
        std::fprintf(stderr,
            "Benchmark and point dumps are supported for vanilla only.\n");
        status = ICP_INVALID_INPUT;
    }

    IcpResult result{};
    timespec start, end;
    if (status == ICP_OK && !std::strcmp(method, "feature")) {
        feature::Options feature_options;
        feature::Result feature_result;
        clock_gettime(CLOCK_MONOTONIC, &start);
        status =
            feature::run(&source, &target, feature_options, &feature_result);
        clock_gettime(CLOCK_MONOTONIC, &end);
        result = feature_result.icp;
        std::printf("source_planes=%d source_lines=%d target_planes=%d "
                    "target_lines=%d\n",
            feature_result.source_planes, feature_result.source_lines,
            feature_result.target_planes, feature_result.target_lines);
        if (status == ICP_OK) {
            std::printf("plane_matches=%d line_matches=%d feature_rmse=%.12g\n",
                feature_result.plane_matches, feature_result.line_matches,
                feature_result.feature_rmse);
        }
    } else if (status == ICP_OK) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        status = runICP(&source, &target, &options, &result);
        clock_gettime(CLOCK_MONOTONIC, &end);
    }

    std::printf("%s\n", statusName(status));
    if (status == ICP_OK) {
        const double execution_time_ms = options.benchmark
            ? result.loop_time_ms
            : (double(end.tv_sec) - double(start.tv_sec)) * 1000.0 +
                (double(end.tv_nsec) - double(start.tv_nsec)) / 1e6;
        std::printf("timing_scope=%s\n",
            options.benchmark ? "whole_loop_steps_1_to_5_no_logging"
                              : "complete_registration_call");
        std::printf("execution_time_ms=%.3f\n", execution_time_ms);
        std::printf("iterations=%d converged=%d inliers=%d rmse=%.12g\n",
            result.iterations_completed, int(result.converged),
            result.final_inliers, result.final_rmse);
        std::puts("T_target_source:");
        for (int row = 0; row < 3; ++row) {
            std::printf("%.12g %.12g %.12g %.12g\n", result.pose.R[3 * row],
                result.pose.R[3 * row + 1], result.pose.R[3 * row + 2],
                result.pose.t[row]);
        }
        std::puts("0 0 0 1");
    }

    freePointCloud(&source);
    freePointCloud(&target);
    return status == ICP_OK ? 0 : 1;
}
