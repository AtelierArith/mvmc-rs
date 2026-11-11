// Optional complete-rebuild kernel experiment, independent of Cargo tests.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_panel.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_dense.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_complex.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_inverse_panel.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_batched.cu"
#include "../../gpu/mvmc-gpu-cuda/src/interval_paired.cu"

static void check(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
template<typename Ops,bool Panel>
__global__ void rebuild_probe(const typename Ops::T* input,typename Ops::T* a,typename Ops::T* w,
    typename Ops::T* out,typename Ops::T* pf,int* status,int n) {
  const size_t base=(size_t)blockIdx.x*n*n;
  for(int i=threadIdx.x;i<n*n;i+=blockDim.x) a[base+i]=input[base+i];
  __syncthreads();
  if(Panel) pfinv_panel_plane<Ops>(a+base,w+base,out+base,pf+blockIdx.x,status+blockIdx.x,n);
  else Ops::reference(a+base,w+base,out+base,pf+blockIdx.x,status+blockIdx.x,n);
}
template<typename T> T make(double re,double im);
template<> double make<double>(double re,double){return re;}
template<> rcx make<rcx>(double re,double im){return {re,im};}
static double magnitude(double x){return std::abs(x);}
static double magnitude(rcx x){return std::hypot(x.re,x.im);}
static double difference(double a,double b){return std::abs(a-b);}
static double difference(rcx a,rcx b){return std::hypot(a.re-b.re,a.im-b.im);}
template<typename Ops> void run(const char* label) {
  typedef typename Ops::T T;
  const int batch=16,reps=10;
  for(int n:{24,26,64,128,130,256,512}) {
    const int nt=sizeof(T)==8 && n>=128 ? 256 : (n<=32?32:(n<=64?64:(n<=128?128:256)));
    const size_t nn=(size_t)n*n, count=batch*nn;
    const size_t shared=std::max(std::max((size_t)(n*sizeof(T)+nt*12+n*8),
        sizeof(T)==8 && n>=64 ? (size_t)(40*n+384):0), (size_t)((5*n/2)*sizeof(T)+n*4));
    std::vector<T> input(count),reference(count),actual(count),pf_ref(batch),pf(batch);
    std::vector<int> st_ref(batch),st(batch);
    for(int b=0;b<batch;++b)for(int j=0;j<n;++j)for(int i=0;i<j;++i) {
      double re=i%2==0 && j==i+1 ? 8.0:((i*17+j*31+b*7)%23)/23.0-0.5;
      double im=((i*11+j*7+b*3)%19)/38.0;
      if(b==batch-1)re=im=0;
      input[b*nn+i+(size_t)j*n]=make<T>(re,im);
      input[b*nn+j+(size_t)i*n]=make<T>(-re,-im);
    }
    T *in,*a,*w,*out,*device_pf; int*device_st;
    check(cudaMalloc(&in,count*sizeof(T)));check(cudaMalloc(&a,count*sizeof(T)));
    check(cudaMalloc(&w,count*sizeof(T)));check(cudaMalloc(&out,count*sizeof(T)));
    check(cudaMalloc(&device_pf,batch*sizeof(T)));check(cudaMalloc(&device_st,batch*sizeof(int)));
    check(cudaMemcpy(in,input.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    rebuild_probe<Ops,false><<<batch,nt,shared>>>(in,a,w,out,device_pf,device_st,n);
    check(cudaMemcpy(reference.data(),out,count*sizeof(T),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(pf_ref.data(),device_pf,batch*sizeof(T),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(st_ref.data(),device_st,batch*sizeof(int),cudaMemcpyDeviceToHost));
    rebuild_probe<Ops,true><<<batch,nt,shared>>>(in,a,w,out,device_pf,device_st,n);
    check(cudaMemcpy(actual.data(),out,count*sizeof(T),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(pf.data(),device_pf,batch*sizeof(T),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(st.data(),device_st,batch*sizeof(int),cudaMemcpyDeviceToHost));
    double worst=0;
    for(int b=0;b<batch;++b) {
      if(st[b]!=st_ref[b]){fprintf(stderr,"%s n=%d status %d vs %d\n",label,n,st[b],st_ref[b]);std::exit(2);}
      const double pe=difference(pf[b],pf_ref[b])/std::fmax(1.0,magnitude(pf_ref[b]));
      if(!std::isfinite(pe)||pe>1e-10){fprintf(stderr,"%s n=%d pferr=%g\n",label,n,pe);std::exit(2);}
      double scale=1;
      for(size_t i=0;i<nn;++i)scale=std::fmax(scale,magnitude(reference[b*nn+i]));
      for(size_t i=0;i<nn;++i) {
        const double err=difference(actual[b*nn+i],reference[b*nn+i])/scale;
        if(!std::isfinite(err)||err>1e-10){fprintf(stderr,"%s n=%d inverse err=%g\n",label,n,err);std::exit(2);}
        worst=std::fmax(worst,err);
      }
    }
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    float old_ms,new_ms;
    check(cudaEventRecord(start));
    for(int r=0;r<reps;++r)rebuild_probe<Ops,false><<<batch,nt,shared>>>(in,a,w,out,device_pf,device_st,n);
    check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&old_ms,start,stop));
    check(cudaEventRecord(start));
    for(int r=0;r<reps;++r)rebuild_probe<Ops,true><<<batch,nt,shared>>>(in,a,w,out,device_pf,device_st,n);
    check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&new_ms,start,stop));
    printf("%s,%d,%d,%.6f,%.6f,%.4f,%.3e\n",label,n,batch,old_ms/reps,new_ms/reps,old_ms/new_ms,worst);
    if constexpr(sizeof(T)==8) {
      if(n>=128) {
        std::vector<double> table(count);
        for(size_t i=0;i<count;++i)table[i]=-input[i];
        check(cudaMemcpy(in,table.data(),count*sizeof(T),cudaMemcpyHostToDevice));
        int *flags,*error,*ele;
        check(cudaMalloc(&flags,sizeof(int)));check(cudaMalloc(&error,sizeof(int)));
        check(cudaMalloc(&ele,n*sizeof(int)));
        int one=1,zero=0;std::vector<int> cfg(n);
        for(int i=0;i<n;++i)cfg[i]=i%(n/2);
        check(cudaMemcpy(flags,&one,sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(error,&zero,sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(ele,cfg.data(),n*sizeof(int),cudaMemcpyHostToDevice));
        const int capacity=std::min(4,(48*1024-12*n)/(32*n));
        check(cudaEventRecord(start));
        for(int r=0;r<reps;++r)interval_paired_factor<<<batch,128,32*n*capacity+12*n>>>(flags,error,ele,
            in,a,w,out,device_pf,device_st,n,n/2,n/2,batch,capacity);
        check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));
        float paired_ms;check(cudaEventElapsedTime(&paired_ms,start,stop));
        printf("f64-current-resident,%d,%d,%.6f,%.6f,%.4f,0\n",n,batch,paired_ms/reps,new_ms/reps,paired_ms/new_ms);
        check(cudaFree(flags));check(cudaFree(error));check(cudaFree(ele));
      }
    }
    check(cudaFree(in));check(cudaFree(a));check(cudaFree(w));check(cudaFree(out));
    check(cudaFree(device_pf));check(cudaFree(device_st));check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
  }
}
int main(){printf("dtype,n,batch,reference_ms,panel_ms,speedup,max_inverse_difference\n");run<PfinvRealPanelOps>("f64");run<PfinvComplexPanelOps>("c64");}
