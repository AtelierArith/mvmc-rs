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
    if (!isfinite(value)) atomicCAS(error,0,2);
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
