#include "icp.hpp"

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
