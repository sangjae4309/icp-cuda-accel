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
// Non-trivial rigid pose: 30 degrees about z plus a translation.
static Pose rotationZ30() {
    const double c=std::cos(M_PI/6), s=std::sin(M_PI/6);
    Pose p{};
    p.R[0]=c; p.R[1]=-s; p.R[2]=0;
    p.R[3]=s; p.R[4]= c; p.R[5]=0;
    p.R[6]=0; p.R[7]= 0; p.R[8]=1;
    p.t[0]=0.25; p.t[1]=-1.5; p.t[2]=3;
    return p;
}
// Rz(30)*Ry(20)*Rx(10): no exact zeros anywhere in R, so a host/device difference in
// multiply-add contraction shows up as a ULP mismatch. Keep this alongside rotationZ30.
static Pose generalPose() {
    const double cz=std::cos(M_PI/6), sz=std::sin(M_PI/6);
    const double cy=std::cos(M_PI/9), sy=std::sin(M_PI/9);
    const double cx=std::cos(M_PI/18), sx=std::sin(M_PI/18);
    Pose p{};
    p.R[0]=cz*cy; p.R[1]=cz*sy*sx-sz*cx; p.R[2]=cz*sy*cx+sz*sx;
    p.R[3]=sz*cy; p.R[4]=sz*sy*sx+cz*cx; p.R[5]=sz*sy*cx-cz*sx;
    p.R[6]=-sy;   p.R[7]=cy*sx;          p.R[8]=cy*cx;
    p.t[0]=-2.75; p.t[1]=0.5; p.t[2]=1.25;
    return p;
}
// Float results of the same double expression may differ by a ULP between host and
// device, so scale the tolerance with the magnitude instead of using a fixed epsilon.
static bool nearPoint(Point a, Point b) {
    const double tol[3]={1e-5*std::fmax(1.0,std::fabs(double(b.x))),
                         1e-5*std::fmax(1.0,std::fabs(double(b.y))),
                         1e-5*std::fmax(1.0,std::fabs(double(b.z)))};
    return std::fabs(double(a.x)-double(b.x))<=tol[0]
        && std::fabs(double(a.y)-double(b.y))<=tol[1]
        && std::fabs(double(a.z)-double(b.z))<=tol[2];
}
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
        // --- TR-1 transform: 257 sources means a partial final block (2 x 256). ---
        CHECK(transformPointsCUDA(nullptr)==ICP_INVALID_INPUT);
        DeviceWorkspace unallocated{};
        CHECK(transformPointsCUDA(&unallocated)==ICP_INVALID_INPUT);
        Point transformed[257]{}, gold[257]{};
        // Identity pose (uploaded above) must reproduce the input exactly.
        CHECK(transformPointsCUDA(&w)==ICP_OK);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        CHECK(cudaMemcpy(transformed,w.transformed_d,sizeof(transformed),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(transformed,source,sizeof(source)));
        // Known non-trivial pose: GPU must match the host gold.
        const Pose z30=rotationZ30();
        CHECK(uploadPose(&w,&z30)==ICP_OK);
        CHECK(transformPointsCUDA(&w)==ICP_OK);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        CHECK(cudaMemcpy(transformed,w.transformed_d,sizeof(transformed),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(transformPointsCPU(source,gold,257,&z30)==ICP_OK);
        for(int i=0;i<257;++i) CHECK(nearPoint(transformed[i],gold[i]));
        CHECK(!std::memcmp(transformed,gold,sizeof(gold))); // same double expr, same rounding
        // The kernel must never write through source_d.
        CHECK(cudaMemcpy(back,w.source_d,sizeof(source),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(back,source,sizeof(source)));
        // General pose with no exact zeros in R: the real test of host/device bit parity.
        const Pose gen=generalPose();
        CHECK(uploadPose(&w,&gen)==ICP_OK);
        CHECK(transformPointsCUDA(&w)==ICP_OK);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        CHECK(cudaMemcpy(transformed,w.transformed_d,sizeof(transformed),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(transformPointsCPU(source,gold,257,&gen)==ICP_OK);
        for(int i=0;i<257;++i) CHECK(nearPoint(transformed[i],gold[i]));
        CHECK(!std::memcmp(transformed,gold,sizeof(gold)));
        CHECK(cudaMemcpy(back,w.source_d,sizeof(source),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(back,source,sizeof(source)));
        // Idle-lane guard, checked without a sanitizer: 257 sources means block 1 runs 255
        // lanes with i>=n. Point transformed_d at a padded buffer so the bytes past element
        // n are observable, and require the sentinel tail to survive the launch.
        {
            Point* padded=nullptr;
            CHECK(cudaMalloc(reinterpret_cast<void**>(&padded),512*sizeof(Point))==cudaSuccess);
            Point sentinel[512]; for(int i=0;i<512;++i) sentinel[i]={-7.5f,-7.5f,-7.5f};
            CHECK(cudaMemcpy(padded,sentinel,sizeof(sentinel),cudaMemcpyHostToDevice)==cudaSuccess);
            DeviceWorkspace probe=w; // borrows source_d/pose_d; never deallocated as a workspace
            probe.transformed_d=padded;
            CHECK(probe.num_source==257 && probe.num_blocks==2);
            CHECK(transformPointsCUDA(&probe)==ICP_OK);
            CHECK(cudaDeviceSynchronize()==cudaSuccess);
            Point readback[512]{};
            CHECK(cudaMemcpy(readback,padded,sizeof(readback),cudaMemcpyDeviceToHost)==cudaSuccess);
            CHECK(!std::memcmp(readback,gold,sizeof(gold)));                 // 257 written
            CHECK(!std::memcmp(readback+257,sentinel+257,255*sizeof(Point))); // 255 untouched
            CHECK(cudaFree(padded)==cudaSuccess);
        }
        // A workspace whose num_blocks cannot cover num_source is rejected, not launched.
        {
            DeviceWorkspace bad=w; bad.num_blocks=1;
            CHECK(transformPointsCUDA(&bad)==ICP_INVALID_INPUT);
        }
        CHECK(uploadPose(&w,&o.initial_pose)==ICP_OK); // restore identity for later checks
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
        std::puts("PASS: CUDA memory movement, transform parity, TODO propagation"); return 0;
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
    // --- TR-1 transform, host gold ---
    {
        Point out[257]{};
        const Pose identity=o.initial_pose, z30=rotationZ30();
        CHECK(transformPointsCPU(nullptr,out,257,&identity)==ICP_INVALID_INPUT);
        CHECK(transformPointsCPU(source,nullptr,257,&identity)==ICP_INVALID_INPUT);
        CHECK(transformPointsCPU(source,out,257,nullptr)==ICP_INVALID_INPUT);
        CHECK(transformPointsCPU(source,out,0,&identity)==ICP_INVALID_INPUT);
        // Identity reproduces the input exactly and leaves the source alone.
        Point original[257]{}; std::memcpy(original,source,sizeof(source));
        CHECK(transformPointsCPU(source,out,257,&identity)==ICP_OK);
        CHECK(!std::memcmp(out,source,sizeof(source)));
        CHECK(!std::memcmp(original,source,sizeof(source)));
        // Known pose, verified against the closed form for source[i]={i,-i,i+1}.
        const double c=std::cos(M_PI/6), s=std::sin(M_PI/6);
        CHECK(transformPointsCPU(source,out,257,&z30)==ICP_OK);
        for(int i=0;i<257;++i) {
            Point want{float(i*(c+s)+0.25), float(i*(s-c)-1.5), float((i+1)+3.0)};
            CHECK(nearPoint(out[i],want));
        }
        CHECK(!std::memcmp(original,source,sizeof(source)));
        // In-place is safe: each point is copied before it is overwritten.
        Point inplace[257]{}; std::memcpy(inplace,source,sizeof(source));
        CHECK(transformPointsCPU(inplace,inplace,257,&z30)==ICP_OK);
        CHECK(!std::memcmp(inplace,out,sizeof(out)));
        // A general rotation + translation is a rigid motion: pairwise distances survive.
        // This checks the matrix application independently of the formula under test.
        const Pose gen=generalPose();
        CHECK(transformPointsCPU(source,out,257,&gen)==ICP_OK);
        for(int i=1;i<257;++i) {
            const double d0=std::hypot(std::hypot(double(source[i].x)-double(source[i-1].x),
                                                  double(source[i].y)-double(source[i-1].y)),
                                       double(source[i].z)-double(source[i-1].z));
            const double d1=std::hypot(std::hypot(double(out[i].x)-double(out[i-1].x),
                                                  double(out[i].y)-double(out[i-1].y)),
                                       double(out[i].z)-double(out[i-1].z));
            CHECK(std::fabs(d0-d1)<=1e-4*std::fmax(1.0,d0));
        }
        CHECK(!std::memcmp(original,source,sizeof(source)));
    }
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
    std::puts("PASS: host infrastructure, transform gold, TODO propagation");
    return 0;
}
