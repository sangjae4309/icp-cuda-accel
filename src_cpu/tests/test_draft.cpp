#include "icp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <initializer_list>
#include <unistd.h>
#ifdef ICP_HAS_CUDA
#include <cuda_runtime.h>
#endif
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main(int argc, char** argv) {
    const bool gpu=argc==2 && !std::strcmp(argv[1],"--gpu");
    Point source[257]{}, target[3]{{1,2,3},{4,5,6},{7,8,9}};
    for(int i=0;i<257;++i) source[i]={float(i),float(-i),float(i+1)};
    PointCloud a{source,257}, b{target,3};
    IcpOptions o{}; defaultOptions(&o);
    IcpResult result{};
    if(gpu) {
#ifdef ICP_HAS_CUDA
        IcpStatus available=cudaAvailability();
        if(available==ICP_UNAVAILABLE) { std::puts("SKIP: no CUDA device"); return 77; }
        CHECK(available==ICP_OK);
        DeviceWorkspace w{};
        CHECK(preallocDeviceWorkspace(&w,257,3)==ICP_OK);
        CHECK(w.num_blocks==2 && w.num_target==3);
        CHECK(uploadPointClouds(&w,source,target)==ICP_OK);
        CHECK(uploadPose(&w,&o.initial_pose)==ICP_OK);
        Point back[257]{}; Pose pose{};
        CHECK(cudaMemcpy(back,w.source_d,sizeof(source),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(back,source,sizeof(source)));
        CHECK(cudaMemcpy(back,w.target_d,sizeof(target),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(back,target,sizeof(target)));
        CHECK(cudaMemcpy(&pose,w.pose_d,sizeof(pose),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(&pose,&o.initial_pose,sizeof(pose)));
        IcpStatistics stats{}, copy{}; stats.count=3; stats.sse=12.5; stats.H[4]=7;
        CHECK(cudaMemcpy(w.stats_d,&stats,sizeof(stats),cudaMemcpyHostToDevice)==cudaSuccess);
        CHECK(downloadStatistics(&w,&copy)==ICP_OK);
        CHECK(copy.count==3 && copy.sse==12.5 && copy.H[4]==7);
        CHECK(transformPointsCUDA(&w)==ICP_NOT_IMPLEMENTED);
        CHECK(findCorrespondencesCUDA(&w,BACKEND_CUDA_TILED,1)==ICP_NOT_IMPLEMENTED);
        CHECK(computeStatisticsCUDA(&w)==ICP_NOT_IMPLEMENTED);
        deallocDeviceWorkspace(&w); deallocDeviceWorkspace(&w);
        CHECK(!w.source_d && !w.stats_d && w.num_blocks==0);
        for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
            o.backend=backend;
            CHECK(runICP(&a,&b,&o,&result)==ICP_NOT_IMPLEMENTED);
            CHECK(!result.pose_valid && result.iterations_completed==0);
        }
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        std::puts("PASS: CUDA memory movement and TODO propagation only"); return 0;
#else
        return 77;
#endif
    }
    HostWorkspace w{};
    CHECK(preallocHostWorkspace(&w,0)==ICP_INVALID_INPUT);
    CHECK(preallocHostWorkspace(&w,257)==ICP_OK);
    w.transformed_h[256]=source[256]; w.match_index_h[256]=-1; w.match_dist2_h[256]=2;
    CHECK(preallocHostWorkspace(&w,257)==ICP_INVALID_INPUT);
    deallocHostWorkspace(&w); deallocHostWorkspace(&w);
    CHECK(!w.transformed_h && !w.match_index_h && !w.match_dist2_h);
    CHECK(runICP(&a,&b,&o,&result)==ICP_NOT_IMPLEMENTED);
    CHECK(result.status==ICP_NOT_IMPLEMENTED && !result.pose_valid && !result.converged);
    CHECK(result.iterations_completed==0 && std::isnan(result.final_rmse));
    CHECK(parseBackend("bad",&o.backend)==ICP_INVALID_INPUT);
    o.backend=static_cast<IcpBackend>(99);
    CHECK(runICP(&a,&b,&o,&result)==ICP_INVALID_INPUT);
    defaultOptions(&o);
    PointCloud empty{};
    CHECK(runICP(&empty,&b,&o,&result)==ICP_INVALID_INPUT);
    source[0].x=std::numeric_limits<float>::infinity();
    CHECK(runICP(&a,&b,&o,&result)==ICP_INVALID_INPUT);
    source[0].x=0;
#ifndef ICP_HAS_CUDA
    o.backend=BACKEND_CUDA_NAIVE;
    CHECK(runICP(&a,&b,&o,&result)==ICP_UNAVAILABLE);
#endif
    const char* cases[]={"# comment\n\n1 2 3 4\n4 5 6\n", "1 nope 3\n", "", "nan 2 3\n", "1 2\n"};
    for(int i=0;i<5;++i) {
        char path[]="/tmp/icp-draft-XXXXXX";
        int fd=mkstemp(path); CHECK(fd>=0);
        FILE* f=fdopen(fd,"w"); CHECK(f);
        CHECK(std::fputs(cases[i],f)>=0); CHECK(std::fclose(f)==0);
        PointCloud c{}; IcpStatus s=loadXYZ(path,&c); unlink(path);
        if(i==0) { CHECK(s==ICP_OK && c.count==2 && c.points_h[1].z==6); }
        else { CHECK(s==ICP_INVALID_INPUT && !c.points_h && c.count==0); }
        freePointCloud(&c); freePointCloud(&c);
    }
    std::puts("PASS: host infrastructure and TODO propagation only");
    return 0;
}
