// Mean over walkers, then ordered accumulation over measurement windows.
// Hamiltonian term order and per-walker products of factored Green operators
// are retained. One block owns each output; there are no floating atomics.
__device__ double physcal_warp_sum(double x) {
  for (int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}

extern "C" __global__ void physcal_reduce(double* means, const double* values,
    const double* coefficients, int energy_terms, int greens, int ex,
    int terms, int walkers, int* error, double* window_energy, double* window_greens) {
  int row=blockIdx.x, tid=threadIdx.x;
  __shared__ int status;
  if (!tid) status=*error;
  __syncthreads();
  if (status) return;
  __shared__ double partial[16];
  double sum=0, square=0;
  for (int w=tid; w<walkers; w+=blockDim.x) {
    const double* v=values+(size_t)w*terms;
    double value=0;
    if (row==0) {
      for (int t=0;t<energy_terms;++t) value+=coefficients[t]*v[t];
      square+=value*value;
    } else if (row<=greens) {
      value=v[energy_terms+row-1];
    } else {
      int k=energy_terms+greens+2*(row-greens-1);
      value=v[k]*v[k+1];
    }
    if (!isfinite(value) || (row==0 && !isfinite(square))) atomicCAS(error,0,RESIDENT_ENERGY);
    sum+=value;
  }
  sum=physcal_warp_sum(sum);
  if (row==0) square=physcal_warp_sum(square);
  if (tid%32==0) partial[tid/32]=sum;
  if (row==0 && tid%32==0) partial[8+tid/32]=square;
  __syncthreads();
  if (tid<32) {
    sum=physcal_warp_sum(tid<8?partial[tid]:0.0);
    if (row==0) square=physcal_warp_sum(tid<8?partial[8+tid]:0.0);
    if (!tid) {
      const double mean=sum/(double)walkers;
      means[row]+=mean;
      if (row==0) { window_energy[0]=mean; window_energy[1]=0; window_energy[2]=square/(double)walkers; }
      else if (window_greens) window_greens[row-1]=mean;
    }
  }
}

// physcal_reduce for complex values (#678): the energy row sums the complex
// coefficients times the operator values (C calham.c), Green rows are the
// values, and a TwoBodyGEx row is C calgrn.c's LocalCisAjs[idx0] *
// conj(LocalCisAjs[idx1]). Values and `means` hold interleaved (re, im).
extern "C" __global__ void physcal_reduce_c(double* means, const double* values,
    const double* coefficients, int energy_terms, int greens, int ex,
    int terms, int walkers, int* error, double* window_energy, double* window_greens) {
  int row=blockIdx.x, tid=threadIdx.x;
  __shared__ int status;
  if (!tid) status=*error;
  __syncthreads();
  if (status) return;
  __shared__ double partial[24];
  double re=0, im=0, square=0;
  for (int w=tid; w<walkers; w+=blockDim.x) {
    const double* v=values+(size_t)2*w*terms;
    double vr=0, vi=0;
    if (row==0) {
      for (int t=0;t<energy_terms;++t) {
        const double cr=coefficients[2*t], ci=coefficients[2*t+1];
        vr=__dadd_rn(vr,__dsub_rn(__dmul_rn(cr,v[2*t]),__dmul_rn(ci,v[2*t+1])));
        vi=__dadd_rn(vi,__dadd_rn(__dmul_rn(cr,v[2*t+1]),__dmul_rn(ci,v[2*t])));
      }
      // C vmccal.c: Etot2 += w * conj(e) * e.
      square+=__dadd_rn(__dmul_rn(vr,vr),__dmul_rn(vi,vi));
    } else if (row<=greens) {
      const int k=energy_terms+row-1;
      vr=v[2*k]; vi=v[2*k+1];
    } else {
      const int k=energy_terms+greens+2*(row-greens-1);
      const double ar=v[2*k], ai=v[2*k+1], br=v[2*k+2], bi=-v[2*k+3];
      vr=__dsub_rn(__dmul_rn(ar,br),__dmul_rn(ai,bi));
      vi=__dadd_rn(__dmul_rn(ar,bi),__dmul_rn(ai,br));
    }
    if (!isfinite(vr) || !isfinite(vi) || (row==0 && !isfinite(square))) atomicCAS(error,0,RESIDENT_ENERGY);
    re+=vr; im+=vi;
  }
  re=physcal_warp_sum(re);
  im=physcal_warp_sum(im);
  if (row==0) square=physcal_warp_sum(square);
  if (tid%32==0) { partial[tid/32]=re; partial[8+tid/32]=im; }
  if (row==0 && tid%32==0) partial[16+tid/32]=square;
  __syncthreads();
  if (tid<32) {
    re=physcal_warp_sum(tid<8?partial[tid]:0.0);
    im=physcal_warp_sum(tid<8?partial[8+tid]:0.0);
    if (row==0) square=physcal_warp_sum(tid<8?partial[16+tid]:0.0);
    if (!tid) {
      const double mr=re/(double)walkers, mi=im/(double)walkers;
      means[2*row]+=mr; means[2*row+1]+=mi;
      if (row==0) { window_energy[0]=mr; window_energy[1]=mi; window_energy[2]=square/(double)walkers; }
      else if (window_greens) { window_greens[2*(row-1)]=mr; window_greens[2*(row-1)+1]=mi; }
    }
  }
}
