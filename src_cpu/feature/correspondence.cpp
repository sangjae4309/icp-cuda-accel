#include "icp.h"
#include <cmath>

namespace feature {

void findMatches(const Tree& source,
    const std::vector<LocalFeature>& source_features, const Tree& target_planes,
    const Tree& target_lines, const Pose& pose, const Options& options,
    std::vector<Match>& matches)
{
    matches.clear();
    std::vector<int> neighbors;
    for (size_t i = 0; i < source.points.size(); ++i) {
        const Kind kind = source_features[i].kind;
        if (kind != PLANE && kind != LINE) {
            continue;
        }
        const Eigen::Vector3d transformed =
            transformPoint(pose, source.points[i]);
        const Tree& target = kind == PLANE ? target_planes : target_lines;
        findNeighbors(target, transformed, options.neighbors,
            options.icp.max_distance, neighbors);
        if (neighbors.size() < 3) {
            continue;
        }
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        for (int index : neighbors) {
            center += target.points[index];
        }
        center /= double(neighbors.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int index : neighbors) {
            const Eigen::Vector3d offset = target.points[index] - center;
            covariance.noalias() += offset * offset.transpose();
        }
        Eigen::Vector3d direction;
        if (kind == PLANE) {
            // Fit y_c = a*y_u + b*y_v about the centroid. Choose the coordinate
            // chart with largest projected area; planes through the origin work
            // too.
            int axis = 0;
            double largest_area = -1;
            for (int c = 0; c < 3; ++c) {
                const int u = (c + 1) % 3, v = (c + 2) % 3;
                const double area = covariance(u, u) * covariance(v, v) -
                    covariance(u, v) * covariance(u, v);
                if (area > largest_area) {
                    largest_area = area;
                    axis = c;
                }
            }
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            Eigen::MatrixXd design(neighbors.size(), 2);
            Eigen::VectorXd rhs(neighbors.size());
            for (size_t j = 0; j < neighbors.size(); ++j) {
                const Eigen::Vector3d offset =
                    target.points[neighbors[j]] - center;
                design(j, 0) = offset[u];
                design(j, 1) = offset[v];
                rhs[j] = offset[axis];
            }
            const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
            if (qr.rank() < 2) {
                continue;
            }
            // QR solve applies Q^T, triangular back substitution, then
            // permutation.
            const Eigen::Vector2d coefficients = qr.solve(rhs);
            direction[axis] = 1;
            direction[u] = -coefficients[0];
            direction[v] = -coefficients[1];
            direction.normalize();
        } else {
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
                covariance);
            if (eigen.info() != Eigen::Success || eigen.eigenvalues()[2] <= 0 ||
                (eigen.eigenvalues()[2] - eigen.eigenvalues()[1]) /
                        eigen.eigenvalues()[2] <
                    options.linearity) {
                continue;
            }
            direction = eigen.eigenvectors().col(2);
        }
        const Eigen::Vector3d offset = transformed - center;
        const double distance = kind == PLANE
            ? std::abs(direction.dot(offset))
            : (offset - direction * direction.dot(offset)).norm();
        if (direction.allFinite() && distance <= options.icp.max_distance) {
            matches.push_back({source.points[i], center, direction, kind});
        }
    }
}

} // namespace feature
