// Optional device-kernel probe; not compiled or run by Cargo tests.
// nvcc -O3 --fmad=false -arch=sm_86 gpu_probe.cu -o /tmp/mvmc-pf-panel-probe
// /tmp/mvmc-pf-panel-probe
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../gpu/mvmc-gpu-cuda/src/pfaffian_panel.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_dense.cuh"
#include "../../gpu/mvmc-gpu-cuda/src/resident_complex.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(1); }
}

template <typename Ops, bool Panel>
__global__ void probe(const typename Ops::T* input, typename Ops::T* work,
                     typename Ops::T* output, int n) {
  typedef typename Ops::T T;
  extern __shared__ char scratch[];
  T* tau = reinterpret_cast<T*>(scratch);
  T* u = tau + n;
  T* A = work + (size_t)blockIdx.x*n*n;
  const T* a = input + (size_t)blockIdx.x*n*n;
  for (int i = threadIdx.x; i < n*n; i += blockDim.x) A[i] = a[i];
  __syncthreads();
  T value;
  if (Panel) value = resident_panel_pfaffian<Ops>(A,n,tau,u);
  else if constexpr (sizeof(T) == sizeof(double)) value = resident_block_pfaffian_unblocked(A,n,tau,u);
  else value = rc_block_pfaffian_unblocked(A,n,tau,u);
  if (threadIdx.x == 0) output[blockIdx.x] = value;
}

template <typename T> T make(double re, double im);
template <> double make<double>(double re, double) { return re; }
template <> rcx make<rcx>(double re, double im) { return {re, im}; }
static double magnitude(double x) { return std::abs(x); }
static double magnitude(rcx x) { return std::hypot(x.re,x.im); }
static double difference(double x, double y) { return std::abs(x-y); }
static double difference(rcx x, rcx y) { return std::hypot(x.re-y.re,x.im-y.im); }

template <typename Ops> void run(const char* label) {
  typedef typename Ops::T T;
  const int batch = 32, reps = 40;
  for (int n : {2, 6, 24, 26, 32, 64, 128, 130, 256, 512}) {
    std::vector<T> input((size_t)batch*n*n);
    for (int b=0;b<batch;++b) for (int j=0;j<n;++j) for (int i=0;i<j;++i) {
      double re = i%2==0 && j==i+1 ? 8.0 : ((i*17+j*31+b*7)%23)/23.0-0.5;
      double im = ((i*11+j*7+b*3)%19)/38.0;
      // Last two planes have analytical Pfaffians: block diagonal and zero.
      if (b==batch-2) { re = i%2==0 && j==i+1 ? 2.0 : 0.0; im=0.0; }
      if (b==batch-1) re=im=0.0;
      input[(size_t)b*n*n+i+(size_t)j*n]=make<T>(re,im);
      input[(size_t)b*n*n+j+(size_t)i*n]=make<T>(-re,-im);
    }
    T *device_input,*work,*output;
    check(cudaMalloc(&device_input,input.size()*sizeof(T)));
    check(cudaMalloc(&work,input.size()*sizeof(T)));
    check(cudaMalloc(&output,batch*sizeof(T)));
    check(cudaMemcpy(device_input,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::vector<T> reference(batch), actual(batch);
    probe<Ops,false><<<batch,128,2*n*sizeof(T)>>>(device_input,work,output,n);
    check(cudaMemcpy(reference.data(),output,batch*sizeof(T),cudaMemcpyDeviceToHost));
    probe<Ops,true><<<batch,128,2*n*sizeof(T)>>>(device_input,work,output,n);
    check(cudaMemcpy(actual.data(),output,batch*sizeof(T),cudaMemcpyDeviceToHost));
    double worst=0.0;
    for (int b=0;b<batch;++b) {
      const T expected = b==batch-2 ? make<T>(std::pow(2.0,n/2),0.0) :
          (b==batch-1 ? make<T>(0.0,0.0) : reference[b]);
      const double error = difference(actual[b],expected)/std::fmax(1.0,magnitude(expected));
      if (!std::isfinite(error) || error>1e-11) {
        fprintf(stderr,"%s n=%d plane=%d relative=%g\n",label,n,b,error); std::exit(2);
      }
      worst=std::fmax(worst,error);
    }
    cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    float old_ms,new_ms;
    check(cudaEventRecord(start));
    for (int r=0;r<reps;++r) probe<Ops,false><<<batch,128,2*n*sizeof(T)>>>(device_input,work,output,n);
    check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&old_ms,start,stop));
    check(cudaEventRecord(start));
    for (int r=0;r<reps;++r) probe<Ops,true><<<batch,128,2*n*sizeof(T)>>>(device_input,work,output,n);
    check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&new_ms,start,stop));
    printf("%s,%d,%d,%.6f,%.6f,%.4f,%.3e\n",label,n,batch,old_ms/reps,new_ms/reps,old_ms/new_ms,worst);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(device_input)); check(cudaFree(work)); check(cudaFree(output));
  }
}

int main() {
  printf("dtype,n,batch,unblocked_ms,panel_ms,speedup,max_relative_error\n");
  run<ResidentRealPfOps>("f64"); run<ResidentComplexPfOps>("c64");
}
