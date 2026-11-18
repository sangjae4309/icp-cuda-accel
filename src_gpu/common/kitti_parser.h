#pragma once
#include "point_cloud.h"

// Read original KITTI float32 XYZI records; retain XYZ, skip reflectance.
// The caller supplies an empty cloud and frees it with freePointCloud().
IcpStatus load_kitti(const char* path, PointCloud* cloud);
