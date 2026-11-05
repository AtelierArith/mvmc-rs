// Mean over walkers, then ordered accumulation over measurement windows.
// Hamiltonian term order and per-walker products of factored Green operators
// are retained. One block owns each output; there are no floating atomics.
__device__ double physcal_warp_sum(double x) {
  for (int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}

extern "C" __global__ void physcal_reduce(double* means, const double* values,
    const double* coefficients, int energy_terms, int greens, int ex,
    int terms, int walkers, int* error) {
  int row=blockIdx.x, tid=threadIdx.x;
  __shared__ int status;
  if (!tid) status=*error;
  __syncthreads();
  if (status) return;
  __shared__ double partial[8];
  double sum=0;
  for (int w=tid; w<walkers; w+=blockDim.x) {
    const double* v=values+(size_t)w*terms;
    double value=0;
    if (row==0) {
      for (int t=0;t<energy_terms;++t) value+=coefficients[t]*v[t];
    } else if (row<=greens) {
      value=v[energy_terms+row-1];
    } else {
      int k=energy_terms+greens+2*(row-greens-1);
      value=v[k]*v[k+1];
    }
    if (!isfinite(value)) atomicCAS(error,0,RESIDENT_ENERGY);
    sum+=value;
  }
  sum=physcal_warp_sum(sum);
  if (tid%32==0) partial[tid/32]=sum;
  __syncthreads();
  if (tid<32) {
    sum=physcal_warp_sum(tid<8?partial[tid]:0.0);
    if (!tid) means[row]+=sum/(double)walkers;
  }
}

// physcal_reduce for complex values (#678): the energy row sums the complex
// coefficients times the operator values (C calham.c), Green rows are the
// values, and a TwoBodyGEx row is C calgrn.c's LocalCisAjs[idx0] *
// conj(LocalCisAjs[idx1]). Values and `means` hold interleaved (re, im).
extern "C" __global__ void physcal_reduce_c(double* means, const double* values,
    const double* coefficients, int energy_terms, int greens, int ex,
    int terms, int walkers, int* error) {
  int row=blockIdx.x, tid=threadIdx.x;
  __shared__ int status;
  if (!tid) status=*error;
  __syncthreads();
  if (status) return;
  __shared__ double partial[16];
  double re=0, im=0;
  for (int w=tid; w<walkers; w+=blockDim.x) {
    const double* v=values+(size_t)2*w*terms;
    double vr=0, vi=0;
    if (row==0) {
      for (int t=0;t<energy_terms;++t) {
        const double cr=coefficients[2*t], ci=coefficients[2*t+1];
        vr=__dadd_rn(vr,__dsub_rn(__dmul_rn(cr,v[2*t]),__dmul_rn(ci,v[2*t+1])));
        vi=__dadd_rn(vi,__dadd_rn(__dmul_rn(cr,v[2*t+1]),__dmul_rn(ci,v[2*t])));
      }
    } else if (row<=greens) {
      const int k=energy_terms+row-1;
      vr=v[2*k]; vi=v[2*k+1];
    } else {
      const int k=energy_terms+greens+2*(row-greens-1);
      const double ar=v[2*k], ai=v[2*k+1], br=v[2*k+2], bi=-v[2*k+3];
      vr=__dsub_rn(__dmul_rn(ar,br),__dmul_rn(ai,bi));
      vi=__dadd_rn(__dmul_rn(ar,bi),__dmul_rn(ai,br));
    }
    if (!isfinite(vr) || !isfinite(vi)) atomicCAS(error,0,RESIDENT_ENERGY);
    re+=vr; im+=vi;
  }
  re=physcal_warp_sum(re);
  im=physcal_warp_sum(im);
  if (tid%32==0) { partial[tid/32]=re; partial[8+tid/32]=im; }
  __syncthreads();
  if (tid<32) {
    re=physcal_warp_sum(tid<8?partial[tid]:0.0);
    im=physcal_warp_sum(tid<8?partial[8+tid]:0.0);
    if (!tid) { means[2*row]+=re/(double)walkers; means[2*row+1]+=im/(double)walkers; }
  }
}
