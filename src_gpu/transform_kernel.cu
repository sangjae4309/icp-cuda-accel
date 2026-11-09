#include "icp.h"
#include <cuda_runtime.h>
#include <cstdio>

static IcpStatus checked(cudaError_t e, const char* action) {
    if (e == cudaSuccess) return ICP_OK;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(e));
    return ICP_CUDA_ERROR;
}

__global__ void transformPointsKernel(const Point* source_d, Point* transformed_d, int n, const Pose* pose_d) {
    // Device original source[n] + accumulated pose -> transformed[n].
    // One thread per source, guard i<n; x=R*p+t. Does not mutate source_d.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return; // Guard against out-of-bounds access

    const double* R = pose_d->R; // Rotation matrix (3x3), row-major
    const double* t = pose_d->t; // Translation vector (3x1)

    // Read the original point from source_d into a register first
    // Pull in the coordinates as doubles to avoid precision loss during the transformation.
    const double x = double(source_d[i].x), y = double(source_d[i].y), z = double(source_d[i].z);

    // Use the transformation formula: transformed = R * p + t
    // transformed_d[i] is a point, so we can write directly to its fields.
    transformed_d[i].x = float(R[0] * x + R[1] * y + R[2] * z + t[0]);
    transformed_d[i].y = float(R[3] * x + R[4] * y + R[5] * z + t[1]);
    transformed_d[i].z = float(R[6] * x + R[7] * y + R[8] * z + t[2]);
}

IcpStatus transformPointsCUDA(DeviceWorkspace* w) {
    // Validate inputs and launch the kernel to transform points using the provided pose.
    if (!w || !w->source_d || !w->transformed_d || !w->pose_d ||
        w->num_source <= 0 || w->num_blocks <= 0) return ICP_INVALID_INPUT;
    if (w->num_blocks != (w->num_source + BLOCK_SIZE - 1) / BLOCK_SIZE) return ICP_INVALID_INPUT;
    
    // Launch the kernel with the specified number of blocks and threads per block.
    transformPointsKernel<<<w->num_blocks, BLOCK_SIZE>>>(
        w->source_d, w->transformed_d, w->num_source, w->pose_d);
    // Check for any errors during kernel launch or execution.
    return checked(cudaGetLastError(), "transformPointsKernel launch");
}
