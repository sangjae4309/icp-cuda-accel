#pragma once

typedef enum
{
    ICP_OK = 0,
    ICP_INVALID_INPUT = 2,
    ICP_IO_ERROR = 3,
    ICP_INSUFFICIENT_MATCHES = 5,
    ICP_DEGENERATE = 6,
    ICP_ALLOCATION_ERROR = 8
} IcpStatus;

typedef struct
{
    float x, y, z;
} Point;

typedef struct
{
    Point* points_h;
    int count;
} PointCloud;

// Row-major matrices, column vectors. Maps original source into target.
typedef struct
{
    double R[9];
    double t[3];
} Pose;

typedef struct
{
    int max_iterations;
    double max_distance, translation_tolerance, rotation_tolerance;
    bool fixed_iterations;
    bool benchmark;
    bool dump_iterations;
    Pose initial_pose;
} IcpOptions;

typedef struct
{
    IcpStatus status;
    Pose pose;
    bool pose_valid, converged;
    int iterations_completed, final_inliers;
    double final_rmse;
    double loop_time_ms;
} IcpResult;

void identityPose(Pose* pose);
void defaultOptions(IcpOptions* options);
const char* statusName(IcpStatus status);
IcpStatus validateCloud(const PointCloud* cloud);
void freePointCloud(PointCloud* cloud);
IcpStatus composePoseCPU(
    const Pose* delta_h, const Pose* current_h, Pose* next_h);
IcpStatus checkConvergenceCPU(
    const Pose* delta_h, const IcpOptions* options, bool* converged);
