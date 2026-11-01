#include "icp/icp.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace icp {

// 4단계: 중심점과 공분산을 구성하고 SVD로 증분 변환을 구합니다.
Pose solve_increment(const Stats& stats) {
    const double count = stats.v[0];
    if (count < 3) {
        throw std::runtime_error("Fewer than 3 inlier correspondences; check initialization/threshold/overlap");
    }
    Eigen::Vector3d source_centroid, target_centroid;
    for (int row = 0; row < 3; ++row) {
        source_centroid[row] = stats.v[1 + row] / count;
        target_centroid[row] = stats.v[4 + row] / count;
    }
    Eigen::Matrix3d H;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            H(row, column) = stats.v[7 + 3 * row + column] - count * source_centroid[row] * target_centroid[column];
        }
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU|Eigen::ComputeFullV);
    // Planar rank-2 data is allowed. Collinear/rank-1 data does not constrain a 3-D rotation.
    const auto sigma = svd.singularValues();
    if (sigma[0] <= 0 || sigma[1] <= 1e-10 * sigma[0]) {
        throw std::runtime_error("Degenerate correspondences: rank < 2");
    }
    const Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = ((V * U.transpose()).determinant() < 0) ? -1.0 : 1.0;
    Eigen::Matrix3d R = V * D * U.transpose();
    Eigen::Vector3d t = target_centroid - R * source_centroid;
    Pose delta;
    for (int row = 0; row < 3; ++row) {
        delta.t[row] = t[row];
        for (int column = 0; column < 3; ++column) {
            delta.R[3 * row + column] = R(row, column);
        }
    }
    return delta;
}
} // namespace icp
