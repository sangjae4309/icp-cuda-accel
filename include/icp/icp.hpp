#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace icp {
struct Point { float x, y, z; };
using Cloud = std::vector<Point>;
// Row-major. Maps ORIGINAL source coordinates into target coordinates.
struct Pose {
    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double t[3] = {0,0,0};
};
// Sufficient statistics for a rigid point-to-point update, in doubles:
// [0] count; [1:4] sum(x); [4:7] sum(q); [7:16] sum(x*q^T); [16] SSE.
// x = CURRENT transformed source, NOT the original source.
constexpr int kStats = 17;
struct Stats { double v[kStats] = {}; };
struct Times {
    double transform_ms=0, match_ms=0, reduce_ms=0, download_ms=0;
};
struct Step { Stats stats; Times times; };
struct Matches { std::vector<int> index; std::vector<float> squared_distance; };
struct Options {
    int iterations=20;
    double max_distance=1.0;
    double translation_tolerance=1e-6;
    double rotation_tolerance=1e-6; // radians
    bool fixed_iterations=false;
    Pose initial;
};
struct Iteration {
    int number=0, inliers=0;
    double rmse_before_update=0, delta_translation=0, delta_rotation=0, solve_ms=0;
    Times times;
};
struct Result {
    Pose pose;
    std::vector<Iteration> trace;
    std::string backend;
    std::size_t source_points=0, target_points=0;
    bool converged=false;
    int final_inliers=0;
    double final_rmse=0, setup_ms=0, loop_wall_ms=0, final_eval_ms=0;
    double registration_wall_ms=0;
};
class Backend {
public:
    virtual ~Backend()=default;
    virtual Step evaluate(const Pose& pose)=0;
    virtual Matches matches() const=0; // diagnostic, outside timed loop
};
std::unique_ptr<Backend> make_cpu(const Cloud&, const Cloud&, double max_distance);
std::unique_ptr<Backend> make_cuda(const Cloud&, const Cloud&, double max_distance, bool tiled);
bool cuda_available();
double now_ms(); // CLOCK_MONOTONIC; no std::chrono
Point transform_point(const Pose&, Point);
void transform_points(const Cloud& source, const Pose&, Cloud& current);
void find_correspondences(const Cloud& current, const Cloud& target,
                          float max_squared_distance, Matches&);
Pose compose(const Pose& left, const Pose& right);
Pose solve_increment(const Stats&);
double rotation_angle(const Pose&);
double translation_norm(const Pose&);
Stats serial_stats(const Cloud& current, const Cloud& target, const Matches&);
// Input coordinates must be finite; read_cloud validates file input.
Result run(const Cloud&, const Cloud&, const Options&);
Result run_cuda(const Cloud&, const Cloud&, const Options&, bool tiled);
Cloud read_cloud(const std::string& filename); // .xyz/.txt or KITTI .bin
Pose read_pose(const std::string& filename);   // 3x4 or 4x4 row-major text
void write_result(const std::string& filename, const Result&, const Options&);
} // namespace icp
