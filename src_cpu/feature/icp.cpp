#include "icp.h"
#include "../vanilla/icp.h"
#include <cfloat>
#include <cmath>
#include <limits>

namespace feature {
IcpStatus run(const PointCloud* source, const PointCloud* target,
    const Options& options, Result* result)
{
    if (!result) {
        return ICP_INVALID_INPUT;
    }
    *result = {};
    result->icp.status = ICP_INVALID_INPUT;
    result->icp.final_rmse = std::numeric_limits<double>::quiet_NaN();
    result->feature_rmse = std::numeric_limits<double>::quiet_NaN();
    const IcpOptions& o = options.icp;
    if (validateCloud(source) != ICP_OK || validateCloud(target) != ICP_OK ||
        o.max_iterations <= 0 || !std::isfinite(o.max_distance) ||
        o.max_distance <= 0 || o.max_distance > std::sqrt(double(FLT_MAX)) ||
        !std::isfinite(o.translation_tolerance) ||
        o.translation_tolerance < 0 || !std::isfinite(o.rotation_tolerance) ||
        o.rotation_tolerance < 0 || options.neighbors < 3 ||
        options.neighbors >= source->count ||
        options.neighbors >= target->count ||
        !std::isfinite(options.neighborhood_radius) ||
        options.neighborhood_radius <= 0 ||
        !(options.linearity > 0 && options.linearity <= 1) ||
        !(options.planarity > 0 && options.planarity <= 1) ||
        !(options.isotropy > 0 && options.isotropy <= 1) ||
        !std::isfinite(options.initial_lambda) || options.initial_lambda <= 0 ||
        options.lm_trials <= 0) {
        return result->icp.status;
    }
    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; ++row) {
        if (!std::isfinite(o.initial_pose.t[row])) {
            return result->icp.status;
        }
        for (int col = 0; col < 3; ++col) {
            rotation(row, col) = o.initial_pose.R[3 * row + col];
        }
    }
    if (!rotation.allFinite() ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() >
            1e-3 ||
        std::abs(rotation.determinant() - 1) > 1e-3) {
        return result->icp.status;
    }
    Tree source_tree, target_tree;
    buildTree(*source, source_tree);
    buildTree(*target, target_tree);
    std::vector<LocalFeature> source_features, target_features;
    extractFeatures(source_tree, options, source_features);
    extractFeatures(target_tree, options, target_features);
    for (const LocalFeature& f : source_features) {
        result->source_planes += f.kind == PLANE;
        result->source_lines += f.kind == LINE;
    }
    for (const LocalFeature& f : target_features) {
        result->target_planes += f.kind == PLANE;
        result->target_lines += f.kind == LINE;
    }
    // Page 12 has separate surface and edge search branches.
    std::vector<Point> plane_points, line_points;
    for (size_t i = 0; i < target_features.size(); ++i) {
        if (target_features[i].kind == PLANE) {
            plane_points.push_back(target->points_h[i]);
        } else if (target_features[i].kind == LINE) {
            line_points.push_back(target->points_h[i]);
        }
    }
    PointCloud plane_cloud{plane_points.data(), int(plane_points.size())};
    PointCloud line_cloud{line_points.data(), int(line_points.size())};
    Tree target_planes, target_lines;
    buildTree(plane_cloud, target_planes);
    buildTree(line_cloud, target_lines);
    Pose pose = o.initial_pose;
    double lambda = options.initial_lambda;
    std::vector<Match> matches;
    IcpStatus status = ICP_OK;
    bool converged = false;
    for (int iteration = 0; iteration < o.max_iterations; ++iteration) {
        findMatches(source_tree, source_features, target_planes, target_lines,
            pose, options, matches);
        if (matches.size() < 6) {
            status = ICP_INSUFFICIENT_MATCHES;
            break;
        }
        // A trial changes only pose: associations, local geometry and weights
        // remain fixed.
        status = optimizePose(matches, options, pose, lambda, converged);
        if (status != ICP_OK) {
            break;
        }
        ++result->icp.iterations_completed;
        if (converged && !o.fixed_iterations) {
            break;
        }
    }
    if (status == ICP_OK) {
        findMatches(source_tree, source_features, target_planes, target_lines,
            pose, options, matches);
        if (matches.size() < 6) {
            status = ICP_INSUFFICIENT_MATCHES;
        } else {
            const double cost = evaluateCost(matches, pose);
            result->feature_rmse = std::sqrt(2 * cost / matches.size());
            for (const Match& match : matches) {
                result->plane_matches += match.kind == PLANE;
                result->line_matches += match.kind == LINE;
            }
            // Comparable metric: all original source points, exhaustive NN,
            // same float cutoff as vanilla.
            std::vector<Point> transformed(source->count);
            std::vector<int> index(source->count);
            std::vector<float> distances(source->count);
            status = transformPointsCPU(
                source->points_h, transformed.data(), source->count, &pose);
            if (status == ICP_OK) {
                status = findCorrespondencesCPU(transformed.data(),
                    target->points_h, source->count, target->count,
                    float(o.max_distance * o.max_distance), index.data(),
                    distances.data());
            }
            double sse = 0;
            for (int i = 0; status == ICP_OK && i < source->count; ++i) {
                if (index[i] >= 0) {
                    ++result->icp.final_inliers;
                    sse += distances[i];
                }
            }
            if (status == ICP_OK && result->icp.final_inliers == 0) {
                status = ICP_INSUFFICIENT_MATCHES;
            }
            if (status == ICP_OK &&
                (!std::isfinite(cost) || !std::isfinite(sse))) {
                status = ICP_DEGENERATE;
            }
            if (status == ICP_OK) {
                result->icp.pose = pose;
                result->icp.pose_valid = true;
                result->icp.converged = converged;
                result->icp.final_rmse =
                    std::sqrt(sse / result->icp.final_inliers);
            }
        }
    }
    result->icp.status = status;
    return status;
}
} // namespace feature
