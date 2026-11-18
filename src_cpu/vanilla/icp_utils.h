#pragma once
#include "../common/point_cloud.h"

typedef struct
{
    Point* transformed_h;
    int* match_index_h;
    float* match_dist2_h;
} IcpBuffers;

// Owners must be zero-initialized and empty before allocation. Free is
// idempotent.
IcpStatus allocateIcpBuffers(IcpBuffers* buffers, int num_source);
void freeIcpBuffers(IcpBuffers* buffers);
