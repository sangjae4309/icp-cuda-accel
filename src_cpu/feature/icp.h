#pragma once
#include "../common/point_cloud.h"
#include <Eigen/Dense>
#include <vector>

namespace feature {
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

enum Kind
{
    NONE,
    PLANE,
    LINE,
    CORNER
};

struct Options
{
    IcpOptions icp;

    Options()
    {
        defaultOptions(&icp);
        icp.max_iterations = 100;
    }

    int neighbors = 20;
    double neighborhood_radius = 2.0;
    double linearity = 0.8;
    double planarity = 0.5;
    double isotropy = 0.5;
    double initial_lambda = 1e-3;
    int lm_trials = 10;
};

struct LocalFeature
{
    Kind kind = NONE;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
};

struct Node
{
    int point, axis, left, right;
};

struct Tree
{
    std::vector<Eigen::Vector3d> points;
    std::vector<Node> nodes;
    int root = -1;
};

struct Match
{
    Eigen::Vector3d source;
    Eigen::Vector3d center;
    Eigen::Vector3d direction;
    Kind kind;
};

struct Result
{
    IcpResult icp{};
    int source_planes = 0, source_lines = 0, target_planes = 0,
        target_lines = 0;
    int plane_matches = 0, line_matches = 0;
    double feature_rmse = 0;
};

void buildTree(const PointCloud& cloud, Tree& tree);
void findNeighbors(const Tree& tree, const Eigen::Vector3d& query, int k,
    double radius, std::vector<int>& indices);
void extractFeatures(const Tree& tree, const Options& options,
    std::vector<LocalFeature>& features);
void findMatches(const Tree& source,
    const std::vector<LocalFeature>& source_features, const Tree& target_planes,
    const Tree& target_lines, const Pose& pose, const Options& options,
    std::vector<Match>& matches);
Eigen::Matrix3d makeSkew(const Eigen::Vector3d& value);
Eigen::Vector3d transformPoint(const Pose& pose, const Eigen::Vector3d& point);
void accumulateSystem(const std::vector<Match>& matches, const Pose& pose,
    Matrix6& A, Vector6& g, double& cost);
double evaluateCost(const std::vector<Match>& matches, const Pose& pose);
Pose applyIncrement(const Pose& current, const Vector6& increment);
IcpStatus optimizePose(const std::vector<Match>& matches,
    const Options& options, Pose& pose, double& lambda, bool& converged);
IcpStatus run(const PointCloud* source, const PointCloud* target,
    const Options& options, Result* result);
} // namespace feature
