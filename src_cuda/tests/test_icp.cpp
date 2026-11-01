#include "reference_cpu.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
using namespace icp;
namespace {
int passed=0;
void require(bool v,const std::string& msg){if(!v)throw std::runtime_error(msg);}
void test_ok(const char* msg){++passed;std::cout<<"PASS "<<msg<<"\n";}
double pose_diff(const Pose& a,const Pose& b){double d=0;for(int i=0;i<9;++i)d=std::max(d,std::abs(a.R[i]-b.R[i]));for(int i=0;i<3;++i)d=std::max(d,std::abs(a.t[i]-b.t[i]));return d;}
Cloud make_cloud(int n){std::mt19937 gen(123);std::uniform_real_distribution<float> u(-2,2);Cloud p;for(int i=0;i<n;++i)p.push_back({u(gen),0.8f*u(gen),0.6f*u(gen)});return p;}
Pose known_pose(){Pose p;double a=0.025;p.R[0]=std::cos(a);p.R[1]=-std::sin(a);p.R[3]=std::sin(a);p.R[4]=std::cos(a);p.t[0]=0.025;p.t[1]=-0.018;p.t[2]=0.012;return p;}
Cloud moved(const Cloud& p,const Pose& T){Cloud q;for(auto a:p)q.push_back(transform_point(T,a));return q;}
void cpu_tests(){
    Cloud p=make_cloud(257);Pose truth=known_pose();Cloud q=moved(p,truth);
    Options opt;opt.max_distance=0.7;opt.iterations=30;
    auto a=run(p,p,opt);require(pose_diff(a.pose,Pose{})<1e-6,"Identity pose");test_ok("identity ICP");
    auto b=run(p,q,opt);require(pose_diff(b.pose,truth)<2e-5,"Known transform not recovered");require(b.final_rmse<2e-5,"Final RMSE");test_ok("known rotation + translation recovered");
    require(pose_diff(compose(truth,Pose{}),truth)<1e-12,"Composition identity");test_ok("pose composition");
    Cloud tie_p={{0,0,0},{0.1f,0,0}},tie_q={{-1,0,0},{1,0,0},{5,0,0}};
    auto tie=make_cpu(tie_p,tie_q,2);tie->evaluate(Pose{});auto tm=tie->matches();
    require(tm.index[0]==0&&tm.index[1]==1,"Tie ordering");test_ok("nearest-neighbor ties choose lowest index");
    Cloud many={{0,0,0},{0.1f,0,0}};Cloud one={{0,0,0}};
    auto mb=make_cpu(many,one,1);mb->evaluate(Pose{});auto mm=mb->matches();
    require(mm.index[0]==0&&mm.index[1]==0,"Many-to-one association");test_ok("many source points may share one target");
    bool threw=false;try{Cloud far=moved(p,Pose{});for(auto& x:far)x.x+=100;run(p,far,opt);}catch(const std::exception&){threw=true;}
    require(threw,"Zero inliers must fail");test_ok("zero-inlier input rejected");
    threw=false;try{Cloud line={{0,0,0},{1,0,0},{2,0,0}};run(line,line,opt);}catch(const std::exception&){threw=true;}
    require(threw,"Collinear input must fail");test_ok("rank-one degeneracy rejected");
    Cloud plane={{0,0,0},{1,0,0},{0,1,0},{1,1,0}};auto pr=run(plane,plane,opt);
    require(pr.final_rmse<1e-6,"Planar data");test_ok("non-collinear planar data accepted");
    Cloud mirror=p;for(auto& x:mirror)x.x=-x.x;
    Matches direct;for(int i=0;i<int(p.size());++i){direct.index.push_back(i);direct.squared_distance.push_back(0);}
    auto reflected=solve_increment(serial_stats(p,mirror,direct));const double* r=reflected.R;
    double det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    require(std::abs(det-1)<1e-8,"Reflection correction");test_ok("SVD returns a proper rotation, not a reflection");
    opt.fixed_iterations=true;opt.iterations=7;auto fixed=run(p,q,opt);require(fixed.trace.size()==7,"Fixed iterations");test_ok("fixed-iteration benchmark mode");
    // Target permutation prevents accidental correspondence by input index.
    std::reverse(q.begin(),q.end());opt.fixed_iterations=false;opt.iterations=30;auto perm=run(p,q,opt);
    require(pose_diff(perm.pose,truth)<2e-5,"Target permutation");test_ok("permuted target cloud");
}
void gpu_tests(){
    for(bool tiled:{false,true}){
        for(int n:{1,31,256,257,513}){
            Cloud p=make_cloud(n),q=make_cloud(321);
            auto cpu=make_cpu(p,q,0.7),gpu=make_cuda(p,q,0.7,tiled);Pose pose=known_pose();
            auto cs=cpu->evaluate(pose),gs=gpu->evaluate(pose);auto cm=cpu->matches(),gm=gpu->matches();
            require(cm.index==gm.index,"GPU index mismatch (random test with no deliberate ties)");
            for(int k=0;k<kStats;++k)require(std::abs(cs.stats.v[k]-gs.stats.v[k])<2e-5*(1+std::abs(cs.stats.v[k])),"GPU statistics mismatch");
        }
        Cloud tp={{0,0,0},{0.1f,0,0}},tq={{-1,0,0},{1,0,0}};
        auto tg=make_cuda(tp,tq,2,tiled);tg->evaluate(Pose{});require(tg->matches().index[0]==0,"GPU tie order");
        Cloud mp={{0,0,0},{0.1f,0,0}},mq={{0,0,0}};
        auto mg=make_cuda(mp,mq,1,tiled);mg->evaluate(Pose{});
        require(mg->matches().index==std::vector<int>({0,0}),"GPU many-to-one association");
        Cloud far={{100,100,100}};
        auto empty=make_cuda(mp,far,.1,tiled);auto es=empty->evaluate(Pose{});
        require(es.stats.v[0]==0,"GPU zero-inlier statistics");
        Cloud p=make_cloud(257),q=moved(p,known_pose());Options o;o.max_distance=.7;o.iterations=30;
        auto c=run(p,q,o),g=run_cuda(p,q,o,tiled);
        require(pose_diff(c.pose,g.pose)<2e-4,"GPU full ICP mismatch");
        test_ok(tiled?"CUDA tiled: partial blocks, unequal N/M, ties, stats, pose":"CUDA naive: partial blocks, unequal N/M, ties, stats, pose");
    }
}
}
int main(int argc,char** argv){try{
    bool gpu=argc>1&&std::string(argv[1])=="--gpu";
    if(gpu){if(!cuda_available()){std::cout<<"SKIP no CUDA device\n";return 77;}gpu_tests();}
    else cpu_tests();
    std::cout<<"All "<<passed<<" groups passed.\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
