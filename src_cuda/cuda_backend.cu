// Native CUDA reference + shared-memory tiled correspondence search.
// No Thrust/CUB/cuICP; no STL containers in device code.
#include "icp/icp.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>
namespace icp {
namespace {
constexpr int B=256; // Power of two; statistics shared memory = 17*256*8=34816 bytes.
void check(cudaError_t e,const char* call,const char* file,int line){
    if(e!=cudaSuccess)throw std::runtime_error(std::string(call)+" at "+file+":"+std::to_string(line)+": "+cudaGetErrorString(e));
}
#define CUDA_CHECK(call) check((call),#call,__FILE__,__LINE__)
template<class T> struct DeviceBuffer {
    T* p=nullptr;
    explicit DeviceBuffer(std::size_t n){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~DeviceBuffer(){if(p)cudaFree(p);}
    DeviceBuffer(const DeviceBuffer&)=delete;DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
struct PinnedStats {
    Stats* p=nullptr;
    PinnedStats(){CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p),sizeof(Stats)));}
    ~PinnedStats(){if(p)cudaFreeHost(p);}
};
struct Event {
    cudaEvent_t e=nullptr;
    Event(){CUDA_CHECK(cudaEventCreate(&e));}
    ~Event(){if(e)cudaEventDestroy(e);}
};
struct PoseF {float R[9],t[3];};
PoseF to_float(const Pose& p){PoseF f;for(int i=0;i<9;++i)f.R[i]=float(p.R[i]);for(int i=0;i<3;++i)f.t[i]=float(p.t[i]);return f;}
__global__ void transform_kernel(const Point* p,Point* x,int n,PoseF T){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Point a=p[i];
    x[i]={T.R[0]*a.x+T.R[1]*a.y+T.R[2]*a.z+T.t[0],
          T.R[3]*a.x+T.R[4]*a.y+T.R[5]*a.z+T.t[1],
          T.R[6]*a.x+T.R[7]*a.y+T.R[8]*a.z+T.t[2]};
}
__device__ float distance2(Point a,Point b){
    float x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return x*x+y*y+z*z;
}
__global__ void match_naive(const Point* x,const Point* q,int n,int m,float max_d2,int* indices,float* distances){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Point a=x[i];float best=FLT_MAX;int best_j=-1;
    for(int j=0;j<m;++j){float d=distance2(a,q[j]);if(d<best){best=d;best_j=j;}}
    indices[i]=(best<=max_d2)?best_j:-1;distances[i]=best;
}
__global__ void match_tiled(const Point* x,const Point* q,int n,int m,float max_d2,int* indices,float* distances){
    __shared__ Point tile[B];
    int lane=threadIdx.x,i=blockIdx.x*blockDim.x+lane;
    bool valid=i<n;Point a={0,0,0};if(valid)a=x[i];
    float best=FLT_MAX;int best_j=-1;
    // All lanes, including i>=n, must reach BOTH barriers in every tile.
    for(int base=0;base<m;base+=B){
        if(base+lane<m)tile[lane]=q[base+lane];
        __syncthreads();
        const int count=(m-base<B)?m-base:B;
        if(valid)for(int j=0;j<count;++j){float d=distance2(a,tile[j]);if(d<best){best=d;best_j=base+j;}}
        __syncthreads(); // Do not overwrite tile while another warp is still reading it.
    }
    if(valid){indices[i]=(best<=max_d2)?best_j:-1;distances[i]=best;}
}
__device__ void reduce_shared(double shared[kStats][B],Stats* out){
    int lane=threadIdx.x;
    __syncthreads();
    for(int stride=B/2;stride>0;stride>>=1){
        if(lane<stride)for(int k=0;k<kStats;++k)shared[k][lane]+=shared[k][lane+stride];
        __syncthreads();
    }
    if(lane==0)for(int k=0;k<kStats;++k)out->v[k]=shared[k][0];
}
__global__ void statistics_kernel(const Point* x,const Point* q,const int* indices,const float* distances,int n,Stats* partial){
    __shared__ double shared[kStats][B];
    const int lane=threadIdx.x,i=blockIdx.x*blockDim.x+lane;
    for(int k=0;k<kStats;++k)shared[k][lane]=0;
    if(i<n){int j=indices[i];if(j>=0){
        double a[3]={x[i].x,x[i].y,x[i].z};double b[3]={q[j].x,q[j].y,q[j].z};
        shared[0][lane]=1;
        for(int u=0;u<3;++u){shared[1+u][lane]=a[u];shared[4+u][lane]=b[u];
            for(int v=0;v<3;++v)shared[7+3*u+v][lane]=a[u]*b[v];}
        shared[16][lane]=distances[i];
    }}
    reduce_shared(shared,&partial[blockIdx.x]);
}
__global__ void final_reduce_kernel(const Stats* partial,int blocks,Stats* out){
    __shared__ double shared[kStats][B];int lane=threadIdx.x;
    for(int k=0;k<kStats;++k){double sum=0;for(int j=lane;j<blocks;j+=B)sum+=partial[j].v[k];shared[k][lane]=sum;}
    reduce_shared(shared,out);
}
float elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));return ms;}
class CUDABackend final : public Backend {
    int n_,m_,blocks_;float max_d2_;bool tiled_;
    DeviceBuffer<Point> p_,q_,x_;
    DeviceBuffer<int> indices_;
    DeviceBuffer<float> distances_;
    DeviceBuffer<Stats> partial_,out_;
    PinnedStats host_;
    Event start_,transformed_,matched_,reduced_,copied_;
public:
    CUDABackend(const Cloud& p,const Cloud& q,double d,bool tiled)
    :n_(int(p.size())),m_(int(q.size())),blocks_((n_+B-1)/B),max_d2_(float(d*d)),tiled_(tiled),
     p_(p.size()),q_(q.size()),x_(p.size()),indices_(p.size()),distances_(p.size()),partial_(blocks_),out_(1){
        CUDA_CHECK(cudaMemcpy(p_.p,p.data(),p.size()*sizeof(Point),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(q_.p,q.data(),q.size()*sizeof(Point),cudaMemcpyHostToDevice));
    }
    Step evaluate(const Pose& pose)override{
        Step out;
        CUDA_CHECK(cudaEventRecord(start_.e));
        transform_kernel<<<blocks_,B>>>(p_.p,x_.p,n_,to_float(pose));CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(transformed_.e));
        if(tiled_)match_tiled<<<blocks_,B>>>(x_.p,q_.p,n_,m_,max_d2_,indices_.p,distances_.p);
        else match_naive<<<blocks_,B>>>(x_.p,q_.p,n_,m_,max_d2_,indices_.p,distances_.p);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(matched_.e));
        statistics_kernel<<<blocks_,B>>>(x_.p,q_.p,indices_.p,distances_.p,n_,partial_.p);CUDA_CHECK(cudaGetLastError());
        final_reduce_kernel<<<1,B>>>(partial_.p,blocks_,out_.p);CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(reduced_.e));
        CUDA_CHECK(cudaMemcpyAsync(host_.p,out_.p,sizeof(Stats),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(copied_.e));CUDA_CHECK(cudaEventSynchronize(copied_.e));
        out.stats=*host_.p;
        out.times.transform_ms=elapsed(start_.e,transformed_.e);
        out.times.match_ms=elapsed(transformed_.e,matched_.e);
        out.times.reduce_ms=elapsed(matched_.e,reduced_.e);
        out.times.download_ms=elapsed(reduced_.e,copied_.e);
        return out;
    }
    Matches matches()const override{
        Matches m;m.index.resize(n_);m.squared_distance.resize(n_);
        CUDA_CHECK(cudaMemcpy(m.index.data(),indices_.p,n_*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m.squared_distance.data(),distances_.p,n_*sizeof(float),cudaMemcpyDeviceToHost));
        return m;
    }
};
} // anonymous namespace
bool cuda_available(){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}
std::unique_ptr<Backend> make_cuda(const Cloud& p,const Cloud& q,double d,bool tiled){
    return std::make_unique<CUDABackend>(p,q,d,tiled);
}
} // namespace icp
