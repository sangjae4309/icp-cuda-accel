#include "icp.h"
#include <Eigen/Dense>

IcpStatus solveRigidUpdateCPU(const IcpStatistics* stats_h, Pose* delta_h)
{
    if (stats_h->count < 3) {
        return ICP_INSUFFICIENT_MATCHES;
    }
    Eigen::Matrix3d H;
    Eigen::Vector3d source, target;
    for (int row = 0; row < 3; ++row) {
        source[row] = stats_h->mu_source[row];
        target[row] = stats_h->mu_target[row];
        for (int col = 0; col < 3; ++col) {
            H(row, col) = stats_h->H[3 * row + col];
        }
    }
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d singular = svd.singularValues();
    // Rank two permits planar non-collinear pairs; a line cannot determine a
    // unique rotation.
    if (singular[0] <= 0 || singular[1] <= 1e-12 * singular[0]) {
        return ICP_DEGENERATE;
    }
    Eigen::Matrix3d V = svd.matrixV();
    const Eigen::Matrix3d U = svd.matrixU();
    if ((V * U.transpose()).determinant() < 0) {
        V.col(2) *= -1;
    }
    const Eigen::Matrix3d rotation = V * U.transpose();
    const Eigen::Vector3d translation = target - rotation * source;
    for (int row = 0; row < 3; ++row) {
        delta_h->t[row] = translation[row];
        for (int col = 0; col < 3; ++col) {
            delta_h->R[3 * row + col] = rotation(row, col);
        }
    }
    return ICP_OK;
}
