#include "icp.h"
#include <algorithm>
#include <cmath>

namespace feature {

Pose applyIncrement(const Pose& current, const Vector6& increment)
{
    const Eigen::Vector3d velocity = increment.head<3>();
    const Eigen::Vector3d omega = increment.tail<3>();
    const double theta2 = omega.squaredNorm();
    double a, b, c;
    if (theta2 < 1e-8) {
        const double theta4 = theta2 * theta2;
        a = 1 - theta2 / 6 + theta4 / 120;
        b = 0.5 - theta2 / 24 + theta4 / 720;
        c = 1.0 / 6 - theta2 / 120 + theta4 / 5040;
    } else {
        const double theta = std::sqrt(theta2);
        a = std::sin(theta) / theta;
        b = (1 - std::cos(theta)) / theta2;
        c = (theta - std::sin(theta)) / (theta2 * theta);
    }
    const Eigen::Matrix3d W = makeSkew(omega);
    const Eigen::Matrix3d W2 = W * W;
    const Eigen::Matrix3d rotation =
        Eigen::Matrix3d::Identity() + a * W + b * W2;
    const Eigen::Vector3d translation =
        (Eigen::Matrix3d::Identity() + b * W + c * W2) * velocity;
    Pose next;
    for (int row = 0; row < 3; ++row) {
        next.t[row] = translation[row];
        for (int k = 0; k < 3; ++k) {
            next.t[row] += rotation(row, k) * current.t[k];
        }
        for (int col = 0; col < 3; ++col) {
            next.R[3 * row + col] = 0;
            for (int k = 0; k < 3; ++k) {
                next.R[3 * row + col] +=
                    rotation(row, k) * current.R[3 * k + col];
            }
        }
    }
    return next;
}

IcpStatus optimizePose(const std::vector<Match>& matches,
    const Options& options, Pose& pose, double& lambda, bool& converged)
{
    converged = false;
    if (matches.size() < 6) {
        return ICP_INSUFFICIENT_MATCHES;
    }
    if (!std::isfinite(lambda) || lambda <= 0 || options.lm_trials <= 0 ||
        !std::isfinite(options.icp.translation_tolerance) ||
        options.icp.translation_tolerance < 0 ||
        !std::isfinite(options.icp.rotation_tolerance) ||
        options.icp.rotation_tolerance < 0) {
        return ICP_INVALID_INPUT;
    }
    Matrix6 A;
    Vector6 g;
    double cost;
    accumulateSystem(matches, pose, A, g, cost);
    if (!A.allFinite() || !g.allFinite() || !std::isfinite(cost)) {
        return ICP_INVALID_INPUT;
    }
    const Vector6 diagonal = A.diagonal();
    if (diagonal.minCoeff() <= 0) {
        return ICP_DEGENERATE;
    }
    // Normalize translation/rotation column scales before testing
    // observability.
    const Vector6 scale = diagonal.array().sqrt().inverse().matrix();
    const Matrix6 normalized = scale.asDiagonal() * A * scale.asDiagonal();
    const Eigen::SelfAdjointEigenSolver<Matrix6> spectrum(
        normalized, Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success ||
        !spectrum.eigenvalues().allFinite() ||
        spectrum.eigenvalues()[0] <= 1e-10 * spectrum.eigenvalues()[5]) {
        return ICP_DEGENERATE;
    }
    const Vector6 scaled_gradient = scale.asDiagonal() * g;
    const Eigen::LDLT<Matrix6> undamped(normalized);
    const Vector6 gauss_newton =
        scale.asDiagonal() * undamped.solve(-scaled_gradient);
    if (undamped.info() != Eigen::Success || !gauss_newton.allFinite()) {
        return ICP_DEGENERATE;
    }
    // A small damped step alone can mean excessive damping, not convergence.
    if (gauss_newton.head<3>().norm() <= options.icp.translation_tolerance &&
        gauss_newton.tail<3>().norm() <= options.icp.rotation_tolerance) {
        converged = true;
        return ICP_OK;
    }
    for (int trial = 0; trial < options.lm_trials; ++trial) {
        Matrix6 damped = normalized;
        damped.diagonal().array() += lambda;
        const Eigen::LDLT<Matrix6> factor(damped);
        const Vector6 increment =
            scale.asDiagonal() * factor.solve(-scaled_gradient);
        if (factor.info() != Eigen::Success || !increment.allFinite()) {
            return ICP_DEGENERATE;
        }
        const double predicted =
            -g.dot(increment) - 0.5 * increment.dot(A * increment);
        const Pose candidate = applyIncrement(pose, increment);
        const double candidate_cost = evaluateCost(matches, candidate);
        const double actual = cost - candidate_cost;
        if (predicted > 0 && std::isfinite(candidate_cost) && actual > 0) {
            const double gain = actual / predicted;
            pose = candidate;
            if (gain > 0.75) {
                lambda = std::max(1e-12, lambda / 3);
            } else if (gain < 0.25) {
                lambda = std::min(1e12, lambda * 2);
            }
            return ICP_OK;
        }
        lambda = std::min(1e12, lambda * 10);
    }
    // No accepted step: leave the pose untouched and report a failed solve.
    return ICP_DEGENERATE;
}

} // namespace feature
