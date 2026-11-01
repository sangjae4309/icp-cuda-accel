#include "icp/icp.hpp"
#include <climits>
#include <cfloat>
#include <stdexcept>

namespace icp {

Result run_cuda(const Cloud& source, const Cloud& target, const Options& options, bool tiled) {
    if (source.empty() || target.empty()) {
        throw std::runtime_error("Empty source or target");
    }
    if (source.size() > INT_MAX || target.size() > INT_MAX) {
        throw std::runtime_error("Cloud exceeds 32-bit indexing");
    }
    if (options.iterations < 1 || !std::isfinite(options.max_distance) || options.max_distance <= 0 || options.max_distance > std::sqrt(FLT_MAX)) {
        throw std::runtime_error("Invalid iteration count or max distance");
    }
    // read_cloud validates that input coordinates are finite.
    Result result;
    result.backend = tiled ? "cuda-tiled" : "cuda-naive";
    result.source_points = source.size();
    result.target_points = target.size();
    result.pose = options.initial;
    result.trace.reserve(options.iterations);

    const double total_start = now_ms();
    const double setup_start = total_start;
    auto backend = make_cuda(source, target, options.max_distance, tiled);
    result.setup_ms = now_ms() - setup_start;

    const double loop_start = now_ms();
    for (int iteration_index = 0; iteration_index < options.iterations; ++iteration_index) {
        Step step = backend->evaluate(result.pose);
        const double solve_start = now_ms();
        Pose delta = solve_increment(step.stats);
        // Left-compose the increment fitted to the currently transformed source.
        result.pose = compose(delta, result.pose);

        Iteration iteration;
        iteration.number = iteration_index;
        iteration.inliers = int(step.stats.v[0]);
        iteration.rmse_before_update = std::sqrt(step.stats.v[16] / step.stats.v[0]);
        iteration.delta_translation = translation_norm(delta);
        iteration.delta_rotation = rotation_angle(delta);
        iteration.solve_ms = now_ms() - solve_start;
        iteration.times = step.times;
        result.trace.push_back(iteration);
        result.converged = (iteration.delta_translation < options.translation_tolerance &&
                            iteration.delta_rotation < options.rotation_tolerance);
        if (result.converged && !options.fixed_iterations) {
            break;
        }
    }
    result.loop_wall_ms = now_ms() - loop_start;

    const double eval_start = now_ms();
    // Recompute correspondences at the final pose before calculating RMSE.
    Step final_step = backend->evaluate(result.pose);
    result.final_inliers = int(final_step.stats.v[0]);
    if (result.final_inliers < 3) {
        throw std::runtime_error("Fewer than 3 final inliers");
    }
    result.final_rmse = std::sqrt(final_step.stats.v[16] / final_step.stats.v[0]);
    result.final_eval_ms = now_ms() - eval_start;
    result.registration_wall_ms = now_ms() - total_start;
    return result;
}
}
