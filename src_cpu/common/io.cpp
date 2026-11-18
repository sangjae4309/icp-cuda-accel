#include "point_cloud.h"
#include <cstdlib>
#include <cmath>

void defaultOptions(IcpOptions* o)
{
    if (!o)
        return;
    *o = {};
    o->max_iterations = 20;
    o->max_distance = 1;
    o->translation_tolerance = o->rotation_tolerance = 1e-6;
    identityPose(&o->initial_pose);
}

const char* statusName(IcpStatus s)
{
    switch (s) {
#define STATUS(x)                                                              \
    case x:                                                                    \
        return #x
        STATUS(ICP_OK);
        STATUS(ICP_INVALID_INPUT);
        STATUS(ICP_IO_ERROR);
        STATUS(ICP_INSUFFICIENT_MATCHES);
        STATUS(ICP_DEGENERATE);
        STATUS(ICP_ALLOCATION_ERROR);
#undef STATUS
    default:
        return "UNKNOWN_STATUS";
    }
}

IcpStatus validateCloud(const PointCloud* c)
{
    if (!c || !c->points_h || c->count <= 0)
        return ICP_INVALID_INPUT;
    for (int i = 0; i < c->count; ++i)
        if (!std::isfinite(c->points_h[i].x) ||
            !std::isfinite(c->points_h[i].y) ||
            !std::isfinite(c->points_h[i].z))
            return ICP_INVALID_INPUT;
    return ICP_OK;
}

void freePointCloud(PointCloud* c)
{
    if (c) {
        std::free(c->points_h);
        *c = {};
    }
}
