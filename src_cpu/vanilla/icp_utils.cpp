#include "icp_utils.h"
#include <cstdlib>

IcpStatus allocateIcpBuffers(IcpBuffers* buffers, int n)
{
    buffers->transformed_h =
        static_cast<Point*>(std::malloc(size_t(n) * sizeof(Point)));
    buffers->match_index_h =
        static_cast<int*>(std::malloc(size_t(n) * sizeof(int)));
    buffers->match_dist2_h =
        static_cast<float*>(std::malloc(size_t(n) * sizeof(float)));
    if (!buffers->transformed_h || !buffers->match_index_h ||
        !buffers->match_dist2_h) {
        freeIcpBuffers(buffers);
        return ICP_ALLOCATION_ERROR;
    }
    return ICP_OK;
}

void freeIcpBuffers(IcpBuffers* buffers)
{
    std::free(buffers->transformed_h);
    std::free(buffers->match_index_h);
    std::free(buffers->match_dist2_h);
    *buffers = {};
}
