#include "icp.h"

namespace feature {

Eigen::Matrix3d makeSkew(const Eigen::Vector3d& value)
{
    Eigen::Matrix3d skew;
    skew << 0, -value.z(), value.y(), value.z(), 0, -value.x(), -value.y(),
        value.x(), 0;
    return skew;
}

Eigen::Vector3d transformPoint(const Pose& pose, const Eigen::Vector3d& point)
{
    Eigen::Vector3d transformed;
    for (int row = 0; row < 3; ++row) {
        transformed[row] = pose.R[3 * row] * point.x() +
            pose.R[3 * row + 1] * point.y() + pose.R[3 * row + 2] * point.z() +
            pose.t[row];
    }
    return transformed;
}

void accumulateSystem(const std::vector<Match>& matches, const Pose& pose,
    Matrix6& A, Vector6& g, double& cost)
{
    A.setZero();
    g.setZero();
    cost = 0;
    for (const Match& match : matches) {
        const Eigen::Vector3d x = transformPoint(pose, match.source);
        Eigen::Matrix<double, 3, 6> motion;
        motion.leftCols<3>().setIdentity();
        motion.rightCols<3>() = -makeSkew(x);
        if (match.kind == PLANE) {
            const double residual = match.direction.dot(x - match.center);
            const Eigen::Matrix<double, 1, 6> J =
                match.direction.transpose() * motion;
            A.noalias() += J.transpose() * J;
            g.noalias() += J.transpose() * residual;
            cost += 0.5 * residual * residual;
        } else if (match.kind == LINE) {
            const Eigen::Matrix3d projection = Eigen::Matrix3d::Identity() -
                match.direction * match.direction.transpose();
            const Eigen::Vector3d residual = projection * (x - match.center);
            const Eigen::Matrix<double, 3, 6> J = projection * motion;
            A.noalias() += J.transpose() * J;
            g.noalias() += J.transpose() * residual;
            cost += 0.5 * residual.squaredNorm();
        }
    }
}

double evaluateCost(const std::vector<Match>& matches, const Pose& pose)
{
    double cost = 0;
    for (const Match& match : matches) {
        const Eigen::Vector3d offset =
            transformPoint(pose, match.source) - match.center;
        if (match.kind == PLANE) {
            const double residual = match.direction.dot(offset);
            cost += 0.5 * residual * residual;
        } else if (match.kind == LINE) {
            const Eigen::Vector3d residual =
                offset - match.direction * match.direction.dot(offset);
            cost += 0.5 * residual.squaredNorm();
        }
    }
    return cost;
}

} // namespace feature
