#include "icp.h"
#include <cstdio>
#include <cstring>
// Minimal entry point: no result exporter, dry-run mode, or benchmark interface.
int main(int argc, char** argv) {
    if(argc==2 && !std::strcmp(argv[1],"--help")) {
        std::puts("Usage: icp_run source.xyz target.xyz [cpu|cuda-naive|cuda-tiled]");
        return 0;
    }
    if(argc<3 || argc>4) { std::fprintf(stderr,"Use --help for usage.\n"); return 1; }
    IcpOptions options{}; defaultOptions(&options);
    options.backend = BACKEND_CUDA_NAIVE;
    IcpStatus status = argc==4 ? parseBackend(argv[3], &options.backend) : ICP_OK;
    PointCloud source{}, target{};
    if(status==ICP_OK) status=loadXYZ(argv[1],&source);
    if(status==ICP_OK) status=loadXYZ(argv[2],&target);
    IcpResult result{};
    if(status==ICP_OK) status=runICP(&source,&target,&options,&result);
    std::printf("%s: draft only; ICP algorithm is not implemented.\n",statusName(status));
    freePointCloud(&source); freePointCloud(&target);
    return status==ICP_OK ? 0 : 1;
}
