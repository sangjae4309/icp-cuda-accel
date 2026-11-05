#ifndef ICP_DRAFT_H
#define ICP_DRAFT_H
#define BLOCK_SIZE 256

typedef enum { ICP_OK = 0, ICP_NOT_IMPLEMENTED, ICP_INVALID_INPUT,
    ICP_IO_ERROR, ICP_CUDA_ERROR, ICP_INSUFFICIENT_MATCHES,
    ICP_DEGENERATE, ICP_UNAVAILABLE, ICP_ALLOCATION_ERROR } IcpStatus;
typedef enum { BACKEND_CPU = 0, BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED } IcpBackend;
typedef struct { float x, y, z; } Point;
typedef struct { Point* points_h; int count; } PointCloud;
// Row-major matrices, column vectors. Maps original source into target.
typedef struct { double R[9]; double t[3]; } Pose;
typedef struct {
    int count;
    double mu_source[3], mu_target[3], H[9], sse;
} IcpStatistics;
typedef struct { int count; double sum_source[3], sum_target[3], sse; } CentroidPartial;
typedef struct { double H[9]; } CovariancePartial;
typedef struct { Point* transformed_h; int* match_index_h; float* match_dist2_h; } HostWorkspace;
typedef struct {
    int num_source, num_target, num_blocks;
    Point *source_d, *target_d, *transformed_d;
    int* match_index_d;
    float* match_dist2_d;
    Pose* pose_d;
    IcpStatistics* stats_d;
    CentroidPartial* centroid_partial_d;
    CovariancePartial* covariance_partial_d;
} DeviceWorkspace;
typedef struct {
    IcpBackend backend;
    int max_iterations;
    double max_distance, translation_tolerance, rotation_tolerance;
    bool fixed_iterations;
    Pose initial_pose;
} IcpOptions;
typedef struct {
    IcpStatus status;
    Pose pose;
    bool pose_valid, converged;
    int iterations_completed, final_inliers;
    double final_rmse;
} IcpResult;

void identityPose(Pose* pose);
void defaultOptions(IcpOptions* options);
const char* statusName(IcpStatus status);
IcpStatus parseBackend(const char* name, IcpBackend* backend);
IcpStatus validateCloud(const PointCloud* cloud);
IcpStatus loadXYZ(const char* path, PointCloud* cloud);
void freePointCloud(PointCloud* cloud);
// Owners must be zero-initialized and empty before allocation. Free is idempotent.
IcpStatus preallocHostWorkspace(HostWorkspace* workspace, int num_source);
void deallocHostWorkspace(HostWorkspace* workspace);
IcpStatus cudaAvailability();
IcpStatus preallocDeviceWorkspace(DeviceWorkspace* workspace, int num_source, int num_target);
void deallocDeviceWorkspace(DeviceWorkspace* workspace);
IcpStatus uploadPointClouds(DeviceWorkspace* workspace, const Point* source_h, const Point* target_h);
IcpStatus uploadPose(DeviceWorkspace* workspace, const Pose* pose_h);
IcpStatus downloadStatistics(const DeviceWorkspace* workspace, IcpStatistics* stats_h);

IcpStatus transformPointsCPU(const Point* source_h, Point* transformed_h, int num_source, const Pose* pose_h);
IcpStatus findCorrespondencesCPU(const Point* transformed_h, const Point* target_h,
    int num_source, int num_target, float max_distance2, int* match_index_h, float* match_dist2_h);
IcpStatus computeStatisticsCPU(const Point* transformed_h, const Point* target_h,
    const int* match_index_h, const float* match_dist2_h, int num_source, int num_target, IcpStatistics* stats_h);
IcpStatus solveRigidUpdateCPU(const IcpStatistics* stats_h, Pose* delta_h);
IcpStatus composePoseCPU(const Pose* delta_h, const Pose* current_h, Pose* next_h);
IcpStatus checkConvergenceCPU(const Pose* delta_h, const IcpOptions* options, bool* converged);
IcpStatus transformPointsCUDA(DeviceWorkspace* workspace);
IcpStatus findCorrespondencesCUDA(DeviceWorkspace* workspace, IcpBackend backend, float max_distance2);
IcpStatus computeStatisticsCUDA(DeviceWorkspace* workspace);
IcpStatus runICP(const PointCloud* source, const PointCloud* target,
    const IcpOptions* options, IcpResult* result);
#endif
