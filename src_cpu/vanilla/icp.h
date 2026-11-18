#ifndef ICP_CPU_VANILLA_H
#define ICP_CPU_VANILLA_H
#include "../common/point_cloud.h"

typedef struct
{
    int count;
    double mu_source[3], mu_target[3], H[9], sse;
} IcpStatistics;

// Step 1: transform original source with the current pose.
IcpStatus transformPointsCPU(const Point* source_h, Point* transformed_h,
    int num_source, const Pose* pose_h);

// Step 2: find the nearest target for each transformed source point.
IcpStatus findCorrespondencesCPU(const Point* transformed_h,
    const Point* target_h, int num_source, int num_target, float max_distance2,
    int* match_index_h, float* match_dist2_h);

// Step 3: accumulate matched-pair centroids, covariance and error.
IcpStatus computeStatisticsCPU(const Point* transformed_h,
    const Point* target_h, const int* match_index_h, const float* match_dist2_h,
    int num_source, IcpStatistics* stats_h);

// Step 4: solve the incremental rigid transform by SVD.
IcpStatus solveRigidUpdateCPU(const IcpStatistics* stats_h, Pose* delta_h);

// Step 5: left-compose the pose and check the incremental motion.
IcpStatus updatePoseCPU(const Pose* delta, const Pose* current,
    const IcpOptions* icp_opt, Pose* next, bool* converged);

IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* icp_opt, IcpResult* result);
#endif
