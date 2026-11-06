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
#ifdef ICP_HAS_CUDA
// Target cloud for the multi-tile cases: almost everything is far away, but two entries
// are an exact duplicate point placed at indices that fall in DIFFERENT shared-memory
// tiles (tiles are BLOCK_SIZE wide, so 5 is in tile 0 and 260 is in tile 1). The nearest
// match is then an exact tie spanning a tile boundary, and the smallest-index rule has to
// resolve it to 5 on every backend.
static const int kTieLowIndex = 5, kTieHighIndex = 260;
static void buildMultiTileTargets(Point* q, int m) {
    for(int j=0;j<m;++j) q[j]={float(1000+j),0,0};
    if(kTieLowIndex<m) q[kTieLowIndex]={0,0,0};
    if(kTieHighIndex<m) q[kTieHighIndex]={0,0,0};
}
// Runs the CPU gold, the naive kernel and the tiled kernel over one fixture and requires
// all three to agree bit-for-bit on both output arrays. The outputs are poisoned before
// each launch so a kernel that writes nothing cannot inherit the previous pass's answer.
static int compareBackends(const Point* xs, const Point* qs, int n, int m, float cutoff) {
    const int kCap=512;
    CHECK(n>0 && n<=kCap && m>0 && m<=kCap);
    Pose identity{}; identityPose(&identity);
    Point tgold[kCap]{}; int cidx[kCap]{}; float cd2[kCap]{};
    CHECK(transformPointsCPU(xs,tgold,n,&identity)==ICP_OK);
    CHECK(findCorrespondencesCPU(tgold,qs,n,m,cutoff,cidx,cd2)==ICP_OK);
    DeviceWorkspace w{};
    CHECK(preallocDeviceWorkspace(&w,n,m)==ICP_OK);
    CHECK(uploadPointClouds(&w,xs,qs)==ICP_OK);
    CHECK(uploadPose(&w,&identity)==ICP_OK);
    CHECK(transformPointsCUDA(&w)==ICP_OK); // identity pose: transformed_d == xs
    int gidx[kCap]{}; float gd2[kCap]{};
    for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
        CHECK(cudaMemset(w.match_index_d,0xAB,size_t(n)*sizeof(int))==cudaSuccess);
        CHECK(cudaMemset(w.match_dist2_d,0xAB,size_t(n)*sizeof(float))==cudaSuccess);
        CHECK(findCorrespondencesCUDA(&w,backend,cutoff)==ICP_OK);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        CHECK(cudaMemcpy(gidx,w.match_index_d,size_t(n)*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(cudaMemcpy(gd2,w.match_dist2_d,size_t(n)*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(!std::memcmp(gidx,cidx,size_t(n)*sizeof(int)));
        CHECK(!std::memcmp(gd2,cd2,size_t(n)*sizeof(float)));
    }
    deallocDeviceWorkspace(&w);
    return 0;
}
#endif
// --- NN-1 correspondence fixture: n=6, m=5. N != M and N is not a multiple of 256.
// Every coordinate and every resulting squared distance is exactly representable in
// float, so the expected tables below are exact values rather than approximations.
static const Point kCorrSource[6] = {
    {0,0,0},     // exact hit on target 0, dist2 == 0
    {1,0,0},     // exact TIE: target 0 and target 1 are both at dist2 == 1
    {0.5f,0,0},  // nearest is target 0 at 0.25 (many-to-one with the two above)
    {2.5f,0,0},  // nearest is target 1 at 0.25
    {0,10,0},    // exact hit on target 3, dist2 == 0
    {32,0,0},    // nearest is target 2 at 484, outside every cutoff tested
};
static const Point kCorrTarget[5] = {{0,0,0},{2,0,0},{10,0,0},{0,10,0},{64,0,0}};
// The nearest squared distance is reported regardless of the cutoff, so one table serves
// all three cases. Only the index column changes as the cutoff tightens.
static const float kCorrD2[6] = {0, 1, 0.25f, 0.25f, 0, 484};
static const int kCorrIdxCut1[6]   = { 0,  0,  0,  1, 3, -1}; // cutoff 1.00: tie point sits
                                                              // exactly ON the cutoff, kept
static const int kCorrIdxCut025[6] = { 0, -1,  0,  1, 3, -1}; // cutoff 0.25: 0.25 == cutoff
                                                              // kept, dist2 1 now rejected
static const int kCorrIdxCut0[6]   = { 0, -1, -1, -1, 3, -1}; // cutoff 0: only exact hits
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
        // --- NN-1 correspondence: own 6x5 workspace, driven through the real per-iteration
        // order (upload pose -> transform -> correspondence) rather than poking device memory.
        {
            DeviceWorkspace c{};
            CHECK(preallocDeviceWorkspace(&c,6,5)==ICP_OK);
            CHECK(c.num_blocks==1 && c.num_source==6 && c.num_target==5);
            CHECK(uploadPointClouds(&c,kCorrSource,kCorrTarget)==ICP_OK);
            CHECK(uploadPose(&c,&o.initial_pose)==ICP_OK); // identity: transformed == source
            CHECK(transformPointsCUDA(&c)==ICP_OK);
            // Backend dispatch: tiled is still NN-2, CPU and out-of-range are rejected.
            CHECK(findCorrespondencesCUDA(&c,BACKEND_CUDA_TILED,1)==ICP_OK); // NN-2 live
            CHECK(findCorrespondencesCUDA(&c,BACKEND_CPU,1)==ICP_INVALID_INPUT);
            CHECK(findCorrespondencesCUDA(&c,static_cast<IcpBackend>(99),1)==ICP_INVALID_INPUT);
            CHECK(findCorrespondencesCUDA(nullptr,BACKEND_CUDA_NAIVE,1)==ICP_INVALID_INPUT);
            DeviceWorkspace unalloc{};
            CHECK(findCorrespondencesCUDA(&unalloc,BACKEND_CUDA_NAIVE,1)==ICP_INVALID_INPUT);
            CHECK(findCorrespondencesCUDA(&c,BACKEND_CUDA_NAIVE,-1)==ICP_INVALID_INPUT);
            const float cutoff[3]={1.0f,0.25f,0.0f};
            const int* want[3]={kCorrIdxCut1,kCorrIdxCut025,kCorrIdxCut0};
            for(int k=0;k<3;++k) {
                CHECK(findCorrespondencesCUDA(&c,BACKEND_CUDA_NAIVE,cutoff[k])==ICP_OK);
                CHECK(cudaDeviceSynchronize()==cudaSuccess);
                int gidx[6]; float gd2[6];
                CHECK(cudaMemcpy(gidx,c.match_index_d,sizeof(gidx),cudaMemcpyDeviceToHost)==cudaSuccess);
                CHECK(cudaMemcpy(gd2,c.match_dist2_d,sizeof(gd2),cudaMemcpyDeviceToHost)==cudaSuccess);
                int cidx[6]; float cd2[6];
                CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,cutoff[k],cidx,cd2)==ICP_OK);
                CHECK(!std::memcmp(gidx,cidx,sizeof(gidx))); // bit-exact index parity
                CHECK(!std::memcmp(gd2,cd2,sizeof(gd2)));    // bit-exact dist2 parity
                for(int i=0;i<6;++i) { CHECK(gidx[i]==want[k][i]); CHECK(gd2[i]==kCorrD2[i]); }
            }
            // Idle-lane guard for the correspondence kernel, same technique as the transform
            // one: n=6 with BLOCK_SIZE lanes means 250 lanes have i>=n and must write nothing.
            {
                int* padIdx=nullptr; float* padD2=nullptr;
                CHECK(cudaMalloc(reinterpret_cast<void**>(&padIdx),BLOCK_SIZE*sizeof(int))==cudaSuccess);
                CHECK(cudaMalloc(reinterpret_cast<void**>(&padD2),BLOCK_SIZE*sizeof(float))==cudaSuccess);
                int sIdx[BLOCK_SIZE]; float sD2[BLOCK_SIZE];
                for(int i=0;i<BLOCK_SIZE;++i) { sIdx[i]=-9999; sD2[i]=-7.5f; }
                CHECK(cudaMemcpy(padIdx,sIdx,sizeof(sIdx),cudaMemcpyHostToDevice)==cudaSuccess);
                CHECK(cudaMemcpy(padD2,sD2,sizeof(sD2),cudaMemcpyHostToDevice)==cudaSuccess);
                DeviceWorkspace probe=c; // borrows transformed_d/target_d; not a real owner
                probe.match_index_d=padIdx; probe.match_dist2_d=padD2;
                // Both kernels must leave the idle-lane tail alone. The tiled kernel is the
                // interesting one: its out-of-range lanes stay alive to load tile slots.
                for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
                    CHECK(cudaMemcpy(padIdx,sIdx,sizeof(sIdx),cudaMemcpyHostToDevice)==cudaSuccess);
                    CHECK(cudaMemcpy(padD2,sD2,sizeof(sD2),cudaMemcpyHostToDevice)==cudaSuccess);
                    CHECK(findCorrespondencesCUDA(&probe,backend,1.0f)==ICP_OK);
                    CHECK(cudaDeviceSynchronize()==cudaSuccess);
                    int rIdx[BLOCK_SIZE]; float rD2[BLOCK_SIZE];
                    CHECK(cudaMemcpy(rIdx,padIdx,sizeof(rIdx),cudaMemcpyDeviceToHost)==cudaSuccess);
                    CHECK(cudaMemcpy(rD2,padD2,sizeof(rD2),cudaMemcpyDeviceToHost)==cudaSuccess);
                    for(int i=0;i<6;++i) { CHECK(rIdx[i]==kCorrIdxCut1[i]); CHECK(rD2[i]==kCorrD2[i]); }
                    for(int i=6;i<BLOCK_SIZE;++i) { CHECK(rIdx[i]==-9999); CHECK(rD2[i]==-7.5f); }
                }
                CHECK(cudaFree(padIdx)==cudaSuccess); CHECK(cudaFree(padD2)==cudaSuccess);
            }
            deallocDeviceWorkspace(&c);
        }
        // --- NN-1 parity across more than one block: n=257 (partial final block), m=3. ---
        {
            CHECK(transformPointsCUDA(&w)==ICP_OK); // identity pose restored just above
            CHECK(findCorrespondencesCUDA(&w,BACKEND_CUDA_NAIVE,1000.0f)==ICP_OK);
            CHECK(cudaDeviceSynchronize()==cudaSuccess);
            int gidx[257]; float gd2[257];
            CHECK(cudaMemcpy(gidx,w.match_index_d,sizeof(gidx),cudaMemcpyDeviceToHost)==cudaSuccess);
            CHECK(cudaMemcpy(gd2,w.match_dist2_d,sizeof(gd2),cudaMemcpyDeviceToHost)==cudaSuccess);
            Point tgold[257]{}; int cidx[257]; float cd2[257];
            CHECK(transformPointsCPU(source,tgold,257,&o.initial_pose)==ICP_OK);
            CHECK(findCorrespondencesCPU(tgold,target,257,3,1000.0f,cidx,cd2)==ICP_OK);
            CHECK(!std::memcmp(gidx,cidx,sizeof(gidx)));
            CHECK(!std::memcmp(gd2,cd2,sizeof(gd2)));
            // Guard against a degenerate case: the cutoff is chosen so that this fixture
            // keeps some pairs, rejects some, selects every target and reuses at least one.
            int kept=0, reused=0; bool seen[3]={false,false,false};
            for(int i=0;i<257;++i) if(cidx[i]>=0) {
                CHECK(cidx[i]<3);
                ++kept;
                if(seen[cidx[i]]) ++reused; else seen[cidx[i]]=true;
            }
            CHECK(kept>0 && kept<257);
            CHECK(seen[0] && seen[1] && seen[2]); // the scan is not stuck on one target
            CHECK(reused>0);                      // many-to-one genuinely occurs
        }
        // --- NN-2 tiled correspondence: identical results to the CPU gold and to the naive
        // kernel across every tiling shape that matters. ---
        {
            Pose identityLate{}; identityPose(&identityLate);
            // Sized for the largest m used below (two full tiles), not just for 300.
            static Point mtTarget[2*BLOCK_SIZE]; buildMultiTileTargets(mtTarget,2*BLOCK_SIZE);
            static Point mtSource[257];
            for(int i=0;i<257;++i) mtSource[i]={float(i)*0.25f,0,0};
            // n < BLOCK_SIZE while m spans two tiles. The 251 lanes with i>=n still have to
            // fill their tile slots, so an early return for them corrupts the second tile.
            CHECK(compareBackends(mtSource,mtTarget,5,300,10.0f)==0);
            // m exactly on a tile boundary (no partial tile at all), then the smallest
            // possible partial tile (one live slot), then a mid-sized partial tile.
            CHECK(compareBackends(mtSource,mtTarget,257,BLOCK_SIZE,10.0f)==0);
            CHECK(compareBackends(mtSource,mtTarget,257,BLOCK_SIZE+1,10.0f)==0);
            CHECK(compareBackends(mtSource,mtTarget,257,300,10.0f)==0);
            // A single source and a single target still has to clear both barriers.
            CHECK(compareBackends(mtSource,mtTarget,1,1,10.0f)==0);
            // Two FULL tiles, no partial tail. Also the widest window for a missing second
            // barrier, since the second tile load overwrites every slot of the first.
            CHECK(compareBackends(mtSource,mtTarget,257,2*BLOCK_SIZE,10.0f)==0);
            // The unique nearest target must be reported by its GLOBAL index even when it
            // lives in a tile after the first. A tiled kernel that stores the in-tile offset
            // instead of base+j agrees with everything else and fails only here.
            {
                static Point lateSource[3];
                const int lateWant[3]={BLOCK_SIZE, 270, 299}; // all beyond the first tile
                for(int k=0;k<3;++k) lateSource[k]=mtTarget[lateWant[k]];
                Point ltg[3]{}; int lidx[3]{}; float ld2[3]{};
                CHECK(transformPointsCPU(lateSource,ltg,3,&identityLate)==ICP_OK);
                CHECK(findCorrespondencesCPU(ltg,mtTarget,3,300,1.0f,lidx,ld2)==ICP_OK);
                for(int k=0;k<3;++k) { CHECK(lidx[k]==lateWant[k]); CHECK(ld2[k]==0); }
                CHECK(compareBackends(lateSource,mtTarget,3,300,1.0f)==0);
                DeviceWorkspace L{};
                CHECK(preallocDeviceWorkspace(&L,3,300)==ICP_OK);
                CHECK(uploadPointClouds(&L,lateSource,mtTarget)==ICP_OK);
                CHECK(uploadPose(&L,&identityLate)==ICP_OK);
                CHECK(transformPointsCUDA(&L)==ICP_OK);
                for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
                    int gidx[3]{};
                    CHECK(findCorrespondencesCUDA(&L,backend,1.0f)==ICP_OK);
                    CHECK(cudaDeviceSynchronize()==cudaSuccess);
                    CHECK(cudaMemcpy(gidx,L.match_index_d,sizeof(gidx),cudaMemcpyDeviceToHost)==cudaSuccess);
                    for(int k=0;k<3;++k) CHECK(gidx[k]==lateWant[k]);
                }
                deallocDeviceWorkspace(&L);
            }
            // The hand-checked small fixtures, now also through the tiled path.
            const float smallCut[3]={1.0f,0.25f,0.0f};
            for(int k=0;k<3;++k) CHECK(compareBackends(kCorrSource,kCorrTarget,6,5,smallCut[k])==0);
            CHECK(compareBackends(source,target,257,3,1000.0f)==0);
            // Exact tie spanning a tile boundary: targets 5 and 260 are the same point and
            // both are nearest. Every backend must report the lower global index, which is
            // what proves the tiled kernel's base+j bookkeeping and its strict < are right.
            Pose identity{}; identityPose(&identity);
            Point tg[5]{}; int cidx[5]{}; float cd2[5]{};
            CHECK(transformPointsCPU(mtSource,tg,5,&identity)==ICP_OK);
            CHECK(findCorrespondencesCPU(tg,mtTarget,5,300,10.0f,cidx,cd2)==ICP_OK);
            for(int i=0;i<5;++i) CHECK(cidx[i]==kTieLowIndex);
            DeviceWorkspace t{};
            CHECK(preallocDeviceWorkspace(&t,5,300)==ICP_OK);
            CHECK(uploadPointClouds(&t,mtSource,mtTarget)==ICP_OK);
            CHECK(uploadPose(&t,&identity)==ICP_OK);
            CHECK(transformPointsCUDA(&t)==ICP_OK);
            for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
                int gidx[5]{};
                CHECK(findCorrespondencesCUDA(&t,backend,10.0f)==ICP_OK);
                CHECK(cudaDeviceSynchronize()==cudaSuccess);
                CHECK(cudaMemcpy(gidx,t.match_index_d,sizeof(gidx),cudaMemcpyDeviceToHost)==cudaSuccess);
                for(int i=0;i<5;++i) CHECK(gidx[i]==kTieLowIndex);
            }
            deallocDeviceWorkspace(&t);
        }
        CHECK(computeStatisticsCUDA(&w)==ICP_NOT_IMPLEMENTED);
        deallocDeviceWorkspace(&w); deallocDeviceWorkspace(&w);
        CHECK(!w.source_d && !w.stats_d && w.num_blocks==0);
        for(IcpBackend backend : {BACKEND_CUDA_NAIVE, BACKEND_CUDA_TILED}) {
            o.backend=backend;
            CHECK(runICP(&a,&b,&o,&result)==ICP_NOT_IMPLEMENTED);
            CHECK(!result.pose_valid && result.iterations_completed==0);
        }
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        std::puts("PASS: CUDA memory, transform + correspondence parity, TODO propagation"); return 0;
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
    // --- NN-1 correspondence, host gold ---
    {
        int idx[6]; float d2[6];
        CHECK(findCorrespondencesCPU(nullptr,kCorrTarget,6,5,1,idx,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,nullptr,6,5,1,idx,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,1,nullptr,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,1,idx,nullptr)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,0,5,1,idx,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,0,1,idx,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,-1,idx,d2)==ICP_INVALID_INPUT);
        CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,
            std::numeric_limits<float>::quiet_NaN(),idx,d2)==ICP_INVALID_INPUT);
        // Tie-breaking, many-to-one, the exact cutoff boundary and rejection-with-distance.
        const float cutoff[3]={1.0f,0.25f,0.0f};
        const int* want[3]={kCorrIdxCut1,kCorrIdxCut025,kCorrIdxCut0};
        for(int k=0;k<3;++k) {
            CHECK(findCorrespondencesCPU(kCorrSource,kCorrTarget,6,5,cutoff[k],idx,d2)==ICP_OK);
            for(int i=0;i<6;++i) {
                CHECK(idx[i]==want[k][i]);
                CHECK(d2[i]==kCorrD2[i]); // exact: these distances are representable
            }
        }
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
    std::puts("PASS: host infrastructure, transform + correspondence gold, TODO propagation");
    return 0;
}
