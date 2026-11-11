// Optional negative-control probe for original-input preservation. Override
// INVERSE_HEADER with a historical header to reproduce the pre-fix status.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_panel.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_dense.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_complex.cuh"
#ifndef INVERSE_HEADER
#define INVERSE_HEADER "../../gpu/mvmc-gpu-cuda/src/pfaffian_inverse_panel.cuh"
#endif
#include INVERSE_HEADER
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_batched.cu"

static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
template<typename Ops> __global__ void reference_probe(typename Ops::T* a,typename Ops::T* w,
    typename Ops::T* out,typename Ops::T* pf,int* st,int n){Ops::reference(a,w,out,pf,st,n);}
template<typename Ops> void run(int lanes){
  typedef typename Ops::T T;
  const int n=64,nt=64;
  std::vector<double> input(n*n*lanes,0.0),output(input.size()),pf(lanes);
  for(int i=0;i<n-1;++i){
    double v=(i==0?1e-308:(i==1?2.0:(i==2?1e308:1.0)));
    input[(i+(i+1)*n)*lanes]=v;input[(i+1+i*n)*lanes]=-v;
  }
  T *a,*w,*out,*device_pf;int*st;
  check(cudaMalloc(&a,input.size()*8));check(cudaMalloc(&w,input.size()*8));
  check(cudaMalloc(&out,input.size()*8));check(cudaMalloc(&device_pf,lanes*8));check(cudaMalloc(&st,4));
  const int shared=lanes==1?40*n+384:44*n;
  for(int ref=1;ref>=0;--ref){
    check(cudaMemcpy(a,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    if(ref)reference_probe<Ops><<<1,nt,shared>>>(a,w,out,device_pf,st,n);
    else if constexpr(std::is_same<T,double>::value)pfinv_f64<<<1,nt,shared>>>(a,w,out,device_pf,st,n);
    else pfinv_c64<<<1,nt,shared>>>(reinterpret_cast<Cx<double>*>(a),reinterpret_cast<Cx<double>*>(w),
        reinterpret_cast<Cx<double>*>(out),reinterpret_cast<Cx<double>*>(device_pf),st,n);
    int status;
    check(cudaMemcpy(&status,st,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(pf.data(),device_pf,lanes*8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(output.data(),out,input.size()*8,cudaMemcpyDeviceToHost));
    int nonfinite=0;for(double x:output)nonfinite+=!std::isfinite(x);
    printf("%d,%d,%d,%.17g,%.17g,%d\n",lanes,ref,status,pf[0],lanes==2?pf[1]:0.0,nonfinite);
  }
  check(cudaFree(a));check(cudaFree(w));check(cudaFree(out));check(cudaFree(device_pf));check(cudaFree(st));
}
int main(){printf("lanes,reference,status,pf_real,pf_imag,nonfinite_inverse_lanes\n");run<PfinvRealPanelOps>(1);run<PfinvComplexPanelOps>(2);}
