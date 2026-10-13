// Julia-mVMC 47452f9, cuda_delayed_sampling.jl: four-update queue, retaining
// the unit-vector correction separately:
// M += u v^T - v u^T + e_a v^T - v e_a^T. Every return flushes the queue.
// Each walker owns its vector packet and uses the factorization output as a
// second inverse buffer during sampling; the two lifetimes never overlap.
#ifndef SLT
#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])
#endif
enum { INTERVAL_DELAY = 4 };

__device__ double interval_warp_sum(double value) {
  for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff,value,d);
  return __shfl_sync(0xffffffff,value,0);
}

__device__ double interval_delayed_entry(const double* inv, const double* data,
    const int* sites, int count, int i, int j, int n) {
  if (i==j) return 0.;
  int row=min(i,j),col=max(i,j);
  double value=inv[(size_t)row*n+col];
  const double* u=data;const double* v=u+INTERVAL_DELAY*n;
  for (int k=0; k<count; ++k) {
    value=fma(u[k*n+row],v[k*n+col],fma(-u[k*n+col],v[k*n+row],value));
    if (col==sites[k]) value-=v[k*n+row];
    if (row==sites[k]) value+=v[k*n+col];
  }
  return i<j?value:-value;
}

__device__ void interval_delayed_ratio(const double* inv, double* data,
    const int* sites, int count, const double* row, const int* rs,
    int n, int a, int lane, double pf, double* candidate) {
  double ratio=0.;
  for (int j=lane; j<n; j+=32) ratio=fma(inv[(size_t)a*n+j],row[rs[j]],ratio);
  ratio=interval_warp_sum(ratio);
  const double* u=data;const double* v=u+INTERVAL_DELAY*n;
  double* dots=data+2*INTERVAL_DELAY*n;
  for (int k=0; k<count; ++k) {
    double udot=0.,vdot=0.;
    for (int j=lane; j<n; j+=32) {
      double x=row[rs[j]];
      udot=fma(u[k*n+j],x,udot);vdot=fma(v[k*n+j],x,vdot);
    }
    udot=interval_warp_sum(udot);vdot=interval_warp_sum(vdot);
    if (!lane) {
      udot+=row[rs[sites[k]]];dots[2*k]=udot;dots[2*k+1]=vdot;
      ratio=fma(u[k*n+a],vdot,fma(-v[k*n+a],udot,ratio));
      if (a==sites[k]) ratio+=vdot;
    }
  }
  if (!lane) {dots[2*INTERVAL_DELAY]=ratio;*candidate=-ratio*pf;}
}

__device__ void interval_delayed_update(const double* inv, double* data,
    const int* sites, int count, const double* tab, int qp, const int* rs,
    int n, int n2, int a, double* v1, double* v2, double* v3,
    int tid, int nt, bool active) {
  double* u=data;double* v=u+INTERVAL_DELAY*n;
  const double* dots=v+INTERVAL_DELAY*n;
  if (active) for (int j=tid; j<n; j+=nt) v3[j]=SLT(tab,qp,rs[a],rs[j],n2);
  __syncthreads();
  if (active) for (int i=tid; i<n; i+=nt) {
    double value=0.;
    if (i==a) value=dots[2*INTERVAL_DELAY];
    else {
      for (int j=0; j<n; ++j) value=fma(-inv[(size_t)j*n+i],v3[j],value);
      for (int k=0; k<count; ++k) {
        value=fma(u[k*n+i],dots[2*k+1],fma(-v[k*n+i],dots[2*k],value));
        if (i==sites[k]) value+=dots[2*k+1];
      }
    }
    v1[i]=value;
    v2[i]=interval_delayed_entry(inv,data,sites,count,a,i,n);
  }
  __syncthreads();
  if (active) {
    double scale=-1./v1[a];
    for (int i=tid; i<n; i+=nt) {u[count*n+i]=v1[i];v[count*n+i]=v2[i]*scale;}
  }
  __syncthreads();
}

__device__ void interval_delayed_flush(double*& inv, double*& alternate,
    const double* data, const int* sites, int count, int n, int nq, int tid, int nt) {
  if (!count) return;
  int packet=2*INTERVAL_DELAY*n+2*INTERVAL_DELAY+1;
  size_t nn=(size_t)n*n;
  for (size_t e=tid; e<nn*nq; e+=nt) {
    int q=e/nn,k=e%nn;
    alternate[e]=interval_delayed_entry(inv+q*nn,data+q*packet,sites,count,k/n,k%n,n);
  }
  __syncthreads();
  double* old=inv;inv=alternate;alternate=old;
}
