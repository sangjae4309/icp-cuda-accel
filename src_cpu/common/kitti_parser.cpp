#include "kitti_parser.h"
#include <cstdio>
#include <cstdlib>
#include <climits>

IcpStatus load_kitti(const char* path, PointCloud* cloud)
{
    FILE* file = std::fopen(path, "rb");
    if (!file) {
        return ICP_IO_ERROR;
    }

    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::rewind(file);
    const long record_size = 4 * sizeof(float);
    if (bytes <= 0 || bytes % record_size != 0 ||
        bytes / record_size > INT_MAX) {
        std::fclose(file);
        return ICP_INVALID_INPUT;
    }

    cloud->count = int(bytes / record_size);
    cloud->points_h =
        static_cast<Point*>(std::malloc(size_t(cloud->count) * sizeof(Point)));
    if (!cloud->points_h) {
        cloud->count = 0;
        std::fclose(file);
        return ICP_ALLOCATION_ERROR;
    }

    IcpStatus status = ICP_OK;
    for (int i = 0; i < cloud->count; ++i) {
        float xyzi[4];
        if (std::fread(xyzi, sizeof(float), 4, file) != 4) {
            status = ICP_IO_ERROR;
            break;
        }
        cloud->points_h[i] = {xyzi[0], xyzi[1], xyzi[2]};
    }

    std::fclose(file);
    if (status == ICP_OK) {
        status = validateCloud(cloud);
    }
    if (status != ICP_OK) {
        freePointCloud(cloud);
    }
    return status;
}
