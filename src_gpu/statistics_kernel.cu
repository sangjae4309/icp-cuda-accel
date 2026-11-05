#include "icp.h"
__global__ void accumulateCentroidPartialsKernel(const Point* x, const Point* q,
    const int* index, const float* dist2, int n, int m, CentroidPartial* partial) {
    // TODO[STATISTICS]: Device valid pairs -> one double sums/count/SSE partial per block.
    // Ignore index=-1; count repeated target selections separately. Bounds n and m.
    (void)x; (void)q; (void)index; (void)dist2; (void)n; (void)m; (void)partial;
}
__global__ void finalizeCentroidsKernel(const CentroidPartial* partial, int blocks, IcpStatistics* stats) {
    // TODO[STATISTICS]: Device partial[blocks] -> stats count/SSE/centroids.
    // Reduce in double; handle zero count before division; reset fields each iteration.
    (void)partial; (void)blocks; (void)stats;
}
__global__ void accumulateCovariancePartialsKernel(const Point* x, const Point* q,
    const int* index, int n, int m, const IcpStatistics* stats, CovariancePartial* partial) {
    // TODO[STATISTICS]: Device valid pairs + centroids -> centered double H partial/block.
    // H=sum((x-mu_source)*(q-mu_target)^T), row-major; no centered-point arrays needed.
    (void)x; (void)q; (void)index; (void)n; (void)m; (void)stats; (void)partial;
}
__global__ void finalizeCovarianceKernel(const CovariancePartial* partial, int blocks, IcpStatistics* stats) {
    // TODO[STATISTICS]: Device covariance partial[blocks] -> stats.H[9], double reduction.
    (void)partial; (void)blocks; (void)stats;
}
IcpStatus computeStatisticsCUDA(DeviceWorkspace* w) {
    // TODO[STATISTICS]: Device transformed/target/matches/distances -> stats_d.
    // Launch pass A + centroid finalization, then pass B + covariance finalization
    // in the same stream. Check errors. Only stats is downloaded by orchestration.
    (void)w; return ICP_NOT_IMPLEMENTED;
}
