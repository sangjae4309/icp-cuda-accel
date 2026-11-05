#include "icp.h"
__global__ void findCorrespondencesNaiveKernel(const Point* x, const Point* q, int n, int m,
    float cutoff, int* index, float* dist2) {
    // TODO[CORRESPONDENCE]: Device x[n], q[m], squared cutoff -> index[n], dist2[n].
    // One source per thread, scan all M targets; many-to-one, smallest index on exact
    // float distance tie. Reject nearest distance > cutoff with index=-1.
    (void)x; (void)q; (void)n; (void)m; (void)cutoff; (void)index; (void)dist2;
}
__global__ void findCorrespondencesTiledKernel(const Point* x, const Point* q, int n, int m,
    float cutoff, int* index, float* dist2) {
    // TODO[CORRESPONDENCE]: Same device inputs/outputs and tie rule as naive/CPU.
    // Load shared-memory target tiles; handle partial final tile and source bounds.
    // EVERY lane, including i>=n, reaches BOTH barriers in every tile iteration.
    // Guard calculations and writes, never return early before synchronization.
    (void)x; (void)q; (void)n; (void)m; (void)cutoff; (void)index; (void)dist2;
}
IcpStatus findCorrespondencesCUDA(DeviceWorkspace* w, IcpBackend b, float cutoff) {
    // TODO[CORRESPONDENCE]: Device transformed/target -> match arrays. Select naive
    // or tiled launch from backend, then check launch errors. No N*M allocation.
    (void)w; (void)b; (void)cutoff; return ICP_NOT_IMPLEMENTED;
}
