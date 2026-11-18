#pragma once

typedef enum
{
    ICP_OK = 0,
    ICP_NOT_IMPLEMENTED,
    ICP_INVALID_INPUT,
    ICP_IO_ERROR,
    ICP_CUDA_ERROR,
    ICP_INSUFFICIENT_MATCHES,
    ICP_DEGENERATE,
    ICP_UNAVAILABLE,
    ICP_ALLOCATION_ERROR
} IcpStatus;

typedef enum
{
    SEARCH_BRUTE_FORCE = 1,
    SEARCH_TILED = 2,
    SEARCH_KD_TREE = 3,
    SEARCH_VOXEL_GRID = 4
} CorrespondenceSearch;

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
    CorrespondenceSearch search;
    int max_iterations;
    double max_distance, translation_tolerance, rotation_tolerance;
    bool fixed_iterations;
    bool benchmark;
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
IcpStatus parseSearch(const char* name, CorrespondenceSearch* search);
IcpStatus validateCloud(const PointCloud* cloud);
void freePointCloud(PointCloud* cloud);
IcpStatus composePoseCPU(
    const Pose* delta_h, const Pose* current_h, Pose* next_h);
IcpStatus checkConvergenceCPU(
    const Pose* delta_h, const IcpOptions* options, bool* converged);
