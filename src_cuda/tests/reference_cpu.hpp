#pragma once
#include "icp.hpp"

// Serial oracle used only by CUDA tests; not linked into the CUDA application.
namespace icp {
Point transform_point(const Pose&, Point);
void transform_points(const Cloud&, const Pose&, Cloud&);
void find_correspondences(const Cloud&, const Cloud&, float, Matches&);
Stats serial_stats(const Cloud&, const Cloud&, const Matches&);
std::unique_ptr<Backend> make_cpu(const Cloud&, const Cloud&, double);
Result run(const Cloud&, const Cloud&, const Options&);
}
