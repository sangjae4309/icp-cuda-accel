#include "icp.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>

namespace feature {

void buildTree(const PointCloud& cloud, Tree& tree)
{
    tree.points.clear();
    tree.nodes.clear();
    tree.root = -1;
    if (cloud.count <= 0) {
        return;
    }
    tree.points.reserve(cloud.count);
    tree.nodes.reserve(cloud.count);
    for (int i = 0; i < cloud.count; ++i) {
        const Point p = cloud.points_h[i];
        tree.points.emplace_back(double(p.x), double(p.y), double(p.z));
    }
    std::vector<int> order(cloud.count);
    std::iota(order.begin(), order.end(), 0);

    struct Work
    {
        int begin, end, depth, parent;
        bool right;
    };

    std::vector<Work> stack{{0, cloud.count, 0, -1, false}};
    while (!stack.empty()) {
        const Work work = stack.back();
        stack.pop_back();
        if (work.begin == work.end) {
            continue;
        }
        const int axis = work.depth % 3;
        const int middle = work.begin + (work.end - work.begin) / 2;
        std::nth_element(order.begin() + work.begin, order.begin() + middle,
            order.begin() + work.end, [&](int a, int b) {
                const double av = tree.points[a][axis],
                             bv = tree.points[b][axis];
                return av < bv || (av == bv && a < b);
            });
        const int node = int(tree.nodes.size());
        tree.nodes.push_back({order[middle], axis, -1, -1});
        if (work.parent < 0) {
            tree.root = node;
        } else if (work.right) {
            tree.nodes[work.parent].right = node;
        } else {
            tree.nodes[work.parent].left = node;
        }
        stack.push_back({middle + 1, work.end, work.depth + 1, node, true});
        stack.push_back({work.begin, middle, work.depth + 1, node, false});
    }
}

void findNeighbors(const Tree& tree, const Eigen::Vector3d& query, int k,
    double radius, std::vector<int>& indices)
{
    indices.clear();
    if (tree.root < 0 || k <= 0 || !(radius >= 0) || !query.allFinite()) {
        return;
    }
    const double radius2 = radius * radius;
    // The heap top is the worst retained distance, then the largest index on a
    // tie.
    std::priority_queue<std::pair<double, int>> nearest;

    struct Visit
    {
        int node;
        double lower_bound;
    };

    std::vector<Visit> stack{{tree.root, 0}};
    while (!stack.empty()) {
        const Visit visit = stack.back();
        stack.pop_back();
        const double limit = int(nearest.size()) < k
            ? radius2
            : std::min(radius2, nearest.top().first);
        if (visit.node < 0 || visit.lower_bound > limit) {
            continue;
        }
        const Node& node = tree.nodes[visit.node];
        const Eigen::Vector3d difference = query - tree.points[node.point];
        const double distance2 = difference.squaredNorm();
        const std::pair<double, int> candidate{distance2, node.point};
        if (distance2 <= radius2) {
            if (int(nearest.size()) < k) {
                nearest.push(candidate);
            } else if (candidate < nearest.top()) {
                nearest.pop();
                nearest.push(candidate);
            }
        }
        const double split = difference[node.axis];
        const int near = split <= 0 ? node.left : node.right;
        const int far = split <= 0 ? node.right : node.left;
        // A split-plane distance is a conservative subtree bound. Equality must
        // be visited because an equal-distance point may have a smaller
        // original index.
        stack.push_back({far, std::max(visit.lower_bound, split * split)});
        stack.push_back({near, visit.lower_bound});
    }
    indices.resize(nearest.size());
    for (size_t i = indices.size(); i > 0; --i) {
        indices[i - 1] = nearest.top().second;
        nearest.pop();
    }
}

void extractFeatures(const Tree& tree, const Options& options,
    std::vector<LocalFeature>& features)
{
    features.assign(tree.points.size(), LocalFeature{});
    std::vector<int> neighbors;
    const int k = std::min(options.neighbors, int(tree.points.size()));
    for (size_t i = 0; i < tree.points.size(); ++i) {
        findNeighbors(tree, tree.points[i], k + (k < int(tree.points.size())),
            options.neighborhood_radius, neighbors);
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), int(i)),
            neighbors.end());
        if (int(neighbors.size()) > k) {
            neighbors.resize(k);
        }
        if (neighbors.size() < 3) {
            continue;
        }
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        for (int index : neighbors) {
            center += tree.points[index];
        }
        center /= double(neighbors.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int index : neighbors) {
            const Eigen::Vector3d centered = tree.points[index] - center;
            covariance += centered * centered.transpose();
        }
        covariance /= double(neighbors.size());
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success ||
            !solver.eigenvalues().allFinite()) {
            continue;
        }
        // Eigen returns eigenvalues in ascending order.
        const Eigen::Vector3d values = solver.eigenvalues().cwiseMax(0.0);
        if (values[2] <= 0) {
            continue;
        }
        LocalFeature& feature = features[i];
        feature.center = center;
        if ((values[2] - values[1]) / values[2] >= options.linearity) {
            feature.kind = LINE;
            feature.direction = solver.eigenvectors().col(2);
        } else if ((values[1] - values[0]) / values[2] >= options.planarity) {
            feature.kind = PLANE;
            feature.direction = solver.eigenvectors().col(0);
        } else if (values[0] / values[2] >= options.isotropy) {
            feature.kind = CORNER;
        }
    }
}

} // namespace feature
