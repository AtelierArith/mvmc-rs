// Paired skew elimination: P A P^T = U D U^T, with unit upper U and
// 2x2 skew diagonal blocks D. Adapted from Julia-mVMC 47452f9,
// cuda_paired_factor.jl (PFAPACK DSKTF2 provenance). A panel keeps K
// rank-two updates in shared memory; inverse construction grows by two columns.
// Used only by the Julia-compatible resident sampler. The C-order factorizer
// remains the independent fallback, including amplified projection cancellation.

__device__ double paired_entry(const double* a, int i, int j, int n,
                              const double* raw, int pending) {
  double value = a[i + (size_t)j*n];
  for (int p=0; p<pending; ++p) {
    const double* v = raw + (size_t)p*4*n;
    if (v[j] != 0.0 || v[n+j] != 0.0)
      value = fma(-v[n+i], v[3*n+j], fma(v[i], v[2*n+j], value));
  }
  return value;
}

extern "C" __global__ void __launch_bounds__(128, 2) interval_paired_factor(
    const int* rebuild, const int* error, const int* ele, const double* slater,
    double* A, double* W, double* OUT, double* PF, int* ST,
    int n, int ne, int nsite, int nqp, int capacity) {
  int plane=blockIdx.x, w=plane/nqp, q=plane%nqp;
  if (*error || !rebuild[w]) return;
  int tid=threadIdx.x, nt=blockDim.x, lane=tid%32;
  size_t offset=(size_t)plane*n*n;
  double* a=A+offset; double* b=W+offset; double* out=OUT+offset;
  const int* e=ele+(size_t)w*n;
  extern __shared__ char paired_mem[];
  double* raw=reinterpret_cast<double*>(paired_mem);
  double* diagonal=raw+4*n*capacity;
  int* piv=reinterpret_cast<int*>(diagonal+n/2);
  int* perm=piv+n;
  __shared__ int pivot, bad;
  __shared__ double alpha, sign;
  for (int k=tid; k<n*n; k+=nt) {
    int i=k%n, j=k/n;
    a[k]=SLT(slater,q,e[i]+(i>=ne?nsite:0),e[j]+(j>=ne?nsite:0),2*nsite);
  }
  for (int i=tid; i<n; i+=nt) piv[i]=i;
  if (!tid) { bad=0; sign=1.; }
  __syncthreads();
  int pending=0;
  for (int k=n-1; k>=1; k-=2) {
    int kk=k-1;
    if (pending) {
      for (int i=tid; i<k; i+=nt) a[i+(size_t)k*n]=paired_entry(a,i,k,n,raw,pending);
      __syncthreads();
    }
    if (tid<32) {
      double vmax=fabs(a[(size_t)k*n]); int kp=0;
      for (int j=lane; j<k; j+=32) {
        double v=fabs(a[j+(size_t)k*n]);
        if (v>vmax || (v==vmax && j<kp)) {vmax=v;kp=j;}
      }
      for (int d=16; d; d>>=1) {
        double v=__shfl_down_sync(0xffffffff,vmax,d);
        int j=__shfl_down_sync(0xffffffff,kp,d);
        if (v>vmax || (v==vmax && j<kp)) {vmax=v;kp=j;}
      }
      if (!tid) { pivot=kp; if (vmax==0. || !isfinite(vmax)) bad=k; }
    }
    __syncthreads();
    if (bad) break;
    int kp=pivot;
    if (kp!=kk) {
      for (int j=tid; j<kp; j+=nt) {
        double v=a[j+(size_t)kk*n]; a[j+(size_t)kk*n]=a[j+(size_t)kp*n]; a[j+(size_t)kp*n]=v;
      }
      for (int j=kp+1+tid; j<kk; j+=nt) {
        double v=a[j+(size_t)kk*n]; a[j+(size_t)kk*n]=a[kp+(size_t)j*n]; a[kp+(size_t)j*n]=v;
      }
      for (int j=k+tid; j<n; j+=nt) {
        double v=a[kk+(size_t)j*n]; a[kk+(size_t)j*n]=a[kp+(size_t)j*n]; a[kp+(size_t)j*n]=v;
      }
      for (int c=tid; c<4*pending; c+=nt) {
        double* v=raw+(size_t)c*n; double t=v[kk];v[kk]=v[kp];v[kp]=t;
      }
      __syncthreads();
      for (int j=kp+tid; j<kk; j+=nt) a[j+(size_t)kk*n]=-a[j+(size_t)kk*n];
      for (int j=kp+1+tid; j<kk; j+=nt) a[kp+(size_t)j*n]=-a[kp+(size_t)j*n];
      if (!tid) sign=-sign;
      __syncthreads();
    }
    if (pending) for (int i=tid; i<kk; i+=nt)
      a[i+(size_t)kk*n]=paired_entry(a,i,kk,n,raw,pending);
    if (!tid) alpha=1./a[kk+(size_t)k*n];
    __syncthreads();
    double* v=raw+(size_t)pending*4*n;
    for (int j=tid; j<kk; j+=nt) {
      double u=a[j+(size_t)kk*n], x=a[j+(size_t)k*n];
      v[j]=x;v[n+j]=u;v[2*n+j]=u*alpha;v[3*n+j]=x*alpha;
      a[j+(size_t)j*n]=0.;a[j+(size_t)kk*n]=x*alpha;a[j+(size_t)k*n]=-u*alpha;
    }
    if (!tid) {piv[kk]=kp;diagonal[k/2]=a[kk+(size_t)k*n];a[kk+(size_t)k*n]=0.;}
    __syncthreads();
    if (++pending==capacity) {
      for (int j=tid/32; j<kk; j+=nt/32)
        for (int i=lane; i<j; i+=32) a[i+(size_t)j*n]=paired_entry(a,i,j,n,raw,pending);
      __syncthreads();pending=0;
    }
  }
  if (!tid) {
    double pf=1.;
    if (!bad) for (int i=0; i<n/2; ++i) pf*=diagonal[i];
    pf*=sign;
    if (!isfinite(pf) || pf==0.) bad=-1;
    PF[plane]=bad?0.:pf;
  }
  __syncthreads();
  if (bad) { if (!tid) ST[plane]=bad;return; }

  // Reuse the panel for U's new columns and their products with the current
  // leading inverse. Each row keeps the Julia ascending-j FMA order.
  for (int k=0; k<n; k+=2) {
    for (int i=tid; i<k; i+=nt) {raw[i]=a[i+(size_t)k*n];raw[n+i]=a[i+(size_t)(k+1)*n];}
    __syncthreads();
    for (int i=tid; i<k; i+=nt) {
      double x=0.,y=0.;
      for (int j=0; j<k; ++j) {double value=b[i+(size_t)j*n];x=fma(value,raw[j],x);y=fma(value,raw[n+j],y);}
      raw[2*n+i]=x;raw[3*n+i]=y;
    }
    __syncthreads();
    for (int i=tid; i<k; i+=nt) {
      double x=raw[2*n+i],y=raw[3*n+i];
      b[i+(size_t)k*n]=-x;b[k+(size_t)i*n]=x;
      b[i+(size_t)(k+1)*n]=-y;b[k+1+(size_t)i*n]=y;
    }
    if (!tid) {
      double value=-1./diagonal[k/2];
      for (int i=0; i<k; ++i) value=fma(raw[i],raw[3*n+i],value);
      b[k+(size_t)k*n]=0.;b[k+1+(size_t)(k+1)*n]=0.;
      b[k+(size_t)(k+1)*n]=value;b[k+1+(size_t)k*n]=-value;
    }
    __syncthreads();
  }
  if (!tid) {
    for (int i=0; i<n; ++i) perm[i]=i;
    for (int j=0; j<n; ++j) {int t=perm[j];perm[j]=perm[piv[j]];perm[piv[j]]=t;}
  }
  __syncthreads();
  bool invalid=false;
  for (int k=tid; k<n*n; k+=nt) {
    double value=b[perm[k%n]+(size_t)perm[k/n]*n];out[k]=value;invalid |= !isfinite(value);
  }
  if (__syncthreads_or(invalid)) {if (!tid) bad=-1;}
  if (!tid) ST[plane]=bad;
}

// Projection can amplify per-sector roundoff. Repeat these walkers using the
// original factorizer, entirely on device, before the commit checks status.
extern "C" __global__ void interval_paired_fallback(
    const int* rebuild, const int* status, const double* pf, const double* weights,
    int* fallback, int walkers, int nq) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if (w>=walkers) return;
  if (!rebuild[w]) {fallback[w]=0;return;}
  bool bad=false;double ip=0.,scale=0.;
  for (int q=0; q<nq; ++q) {
    double p=pf[w*nq+q],term=weights[q]*p;
    ip+=term;scale+=fabs(term);bad |= status[w*nq+q]!=0 || !isfinite(p) || p==0.;
  }
  fallback[w]=bad || !isfinite(ip) || fabs(ip)*1024.<scale;
}
