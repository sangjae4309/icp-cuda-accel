#include "icp.hpp"
#include <iostream>
#include <exception>
#include <string>

static void usage() {
    std::cout << "Usage: icp_run --source P.xyz --target Q.xyz [options]\n"
              << "  --backend cuda-naive|cuda-tiled  (default cuda-tiled)\n"
              << "  --iterations 20 --max-distance 1.0 --fixed-iters\n"
              << "  --initial T_target_source.txt --output results/run.json\n"
              << "All lengths use the input cloud's units (KITTI: meters).\n";
}

int main(int argc, char** argv) {
    try {
        icp::Options options;
        std::string source, target, backend = "cuda-tiled", output = "results/run.json";
        for (int i = 1; i<argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            }
            if (arg == "--fixed-iters") {
                options.fixed_iterations = true;
                continue;
            }
            if (i + 1 >= argc) {
                std::cerr << "Missing value for "+arg << "\n";
                return 2;
            }
            std::string value = argv[++i];
            if (arg == "--source") {
                source = value;
            }
            else if (arg == "--target") {
                target = value;
            }
            else if (arg == "--backend") {
                backend = value;
            }
            else if (arg == "--output") {
                output = value;
            }
            else if (arg == "--iterations") {
                std::size_t parsed;
                options.iterations = std::stoi(value, &parsed);
                if (parsed != value.size()) {
                    std::cerr << "Invalid iterations" << "\n";
                    return 2;
                }
            }
            else if (arg == "--max-distance") {
                std::size_t parsed;
                options.max_distance = std::stod(value, &parsed);
                if (parsed != value.size()) {
                    std::cerr << "Invalid distance" << "\n";
                    return 2;
                }
            }
            else if (arg == "--initial") {
                options.initial = icp::read_pose(value);
            }
            else {
                std::cerr << "Unknown option "+arg << "\n";
                return 2;
            }
        }
        if (source.empty() || target.empty()) {
            usage();
            return 2;
        }
        auto source_cloud = icp::read_cloud(source);
        auto target_cloud = icp::read_cloud(target);
        const double pairs = double(source_cloud.size()) * double(target_cloud.size());
        if (pairs>1e8) {
            std::cerr << "WARNING: " << pairs << " pair distances per iteration. Downsample first.\n";
        }
        if (backend != "cuda-naive" && backend != "cuda-tiled") {
            std::cerr << "Unsupported backend for this executable: " << backend << "\n";
            return 2;
        }
        auto result = icp::run_cuda(source_cloud, target_cloud, options, backend == "cuda-tiled");
        icp::write_result(output, result, options);
        std::cout << "backend=" << backend << " points=" << source_cloud.size() << "x" << target_cloud.size() << " iterations=" << result.trace.size() << " final_inliers=" << result.final_inliers << " final_rmse=" << result.final_rmse << " loop_ms=" << result.loop_wall_ms << " registration_ms=" << result.registration_wall_ms << "\nSaved " << output << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
