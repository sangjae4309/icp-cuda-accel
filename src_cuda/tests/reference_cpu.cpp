#include "reference_cpu.hpp"

namespace icp {

Point transform_point(const Pose& pose, Point point) {
    // Use float arithmetic for point transforms, matching the GPU.
    float r[9];
    float t[3];
    for (int k = 0; k < 9; ++k) {
        r[k] = static_cast<float>(pose.R[k]);
    }
    for (int k = 0; k < 3; ++k) {
        t[k] = static_cast<float>(pose.t[k]);
    }

    return {
        r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0],
        r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1],
        r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]
    };
}

// Step 1: Apply the current pose to the original source.
void transform_points(const Cloud& source, const Pose& pose, Cloud& current) {
    current.resize(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        current[i] = transform_point(pose, source[i]);
    }
}

} // namespace icp

#include <cfloat>

namespace icp {

// Step 2: Find the nearest target point for each source point.
void find_correspondences(const Cloud& current, const Cloud& target,
                          float max_squared_distance, Matches& matches) {
    matches.index.resize(current.size());
    matches.squared_distance.resize(current.size());

    for (std::size_t i = 0; i < current.size(); ++i) {
        float best_distance = FLT_MAX;
        int best_index = -1;

        for (std::size_t j = 0; j < target.size(); ++j) {
            float dx = current[i].x - target[j].x;
            float dy = current[i].y - target[j].y;
            float dz = current[i].z - target[j].z;
            float distance = dx * dx + dy * dy + dz * dz;

            // Keep the smaller index when distances are equal.
            if (distance < best_distance) {
                best_distance = distance;
                best_index = int(j);
            }
        }

        if (best_distance > max_squared_distance) {
            best_index = -1;
        }
        matches.index[i] = best_index;
        matches.squared_distance[i] = best_distance;
    }
}

} // namespace icp


namespace icp {

// Step 3: Accumulate counts, coordinate sums, outer products, and squared distances in double precision.
Stats serial_stats(const Cloud& current, const Cloud& target, const Matches& matches) {
    Stats stats;

    for (std::size_t i = 0; i < current.size(); ++i) {
        int j = matches.index[i];
        if (j < 0) {
            continue;
        }

        double source_point[3] = {current[i].x, current[i].y, current[i].z};
        double target_point[3] = {target[j].x, target[j].y, target[j].z};

        stats.v[0] += 1;
        for (int row = 0; row < 3; ++row) {
            stats.v[1 + row] += source_point[row];
            stats.v[4 + row] += target_point[row];
            for (int column = 0; column < 3; ++column) {
                stats.v[7 + 3 * row + column] += source_point[row] * target_point[column];
            }
        }
        stats.v[16] += matches.squared_distance[i];
    }

    return stats;
}

} // namespace icp


namespace icp {

class CPUBackend final : public Backend {
    const Cloud& source_;
    const Cloud& target_;
    float max_squared_distance_;
    Cloud current_;
    Matches matches_;

public:
    CPUBackend(const Cloud& source, const Cloud& target, double max_distance)
        : source_(source), target_(target),
          max_squared_distance_(float(max_distance * max_distance)),
          current_(source.size()) {
        matches_.index.resize(source.size());
        matches_.squared_distance.resize(source.size());
    }

    Step evaluate(const Pose& pose) override {
        Step step;

        double start = now_ms();
        transform_points(source_, pose, current_);
        step.times.transform_ms = now_ms() - start;

        start = now_ms();
        find_correspondences(current_, target_, max_squared_distance_, matches_);
        step.times.match_ms = now_ms() - start;

        start = now_ms();
        step.stats = serial_stats(current_, target_, matches_);
        step.times.reduce_ms = now_ms() - start;

        return step;
    }

    Matches matches() const override {
        return matches_;
    }
};

std::unique_ptr<Backend> make_cpu(const Cloud& source, const Cloud& target,
                                double max_distance) {
    return std::make_unique<CPUBackend>(source, target, max_distance);
}

} // namespace icp

#include <climits>
#include <cfloat>
#include <stdexcept>

namespace icp {

Result run(const Cloud& source, const Cloud& target, const Options& options) {
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
    result.backend = "cpu";
    result.source_points = source.size();
    result.target_points = target.size();
    result.pose = options.initial;
    result.trace.reserve(options.iterations);

    const double total_start = now_ms();
    const double setup_start = total_start;
    auto backend = make_cpu(source, target, options.max_distance);
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
