// GPU Opt state transitions. All matrices/configurations remain on one stream.

__device__ double opt_warp_sum(double x) {
  for (int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}

extern "C" __global__ void opt_fill(int* out, int value, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) out[i]=value;
}

__device__ int opt_bounded(const unsigned int* tape, int& cursor, int length, int bound, int& good) {
  unsigned int threshold=(0u-(unsigned int)bound)%(unsigned int)bound;
  while (cursor<length) {
    unsigned int word=tape[cursor++];
    if (word>=threshold) return word%(unsigned int)bound;
  }
  good=0; return 0;
}

extern "C" __global__ void opt_initial(int* cfg, int* scratch, const unsigned int* tape,
    const int* pending, int* candidate, int ns, int ne, int batch) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if (w>=batch || !pending[w]) return;
  int n=2*ne, words=n+2*ns+16, cursor=0, good=1;
  int* e=cfg+w*n; int* s=scratch+w*(n+ns);
  const unsigned int* t=tape+w*words;
  for (int a=0;a<n;++a) s[a]=a;
  for (int k=n;k>=2;--k) {
    int j=opt_bounded(t,cursor,words,k,good), v=s[k-1]; s[k-1]=s[j]; s[j]=v;
  }
  for (int r=0;r<ns;++r) s[n+r]=r;
  for (int spin=0;spin<2;++spin) {
    for (int k=ns;k>=2;--k) {
      int j=opt_bounded(t,cursor,words,k,good), v=s[n+k-1]; s[n+k-1]=s[n+j]; s[n+j]=v;
    }
    for (int a=0;a<ne;++a) e[spin*ne+a]=s[n+a];
  }
  if (!good) for (int a=0;a<n;++a) e[a]=a%ne;
  candidate[w]=good;
}

// An unsuccessful initialization is retried, rather than poisoning the run.
extern "C" __global__ void opt_initial_commit(int* pending, const int* candidate,
    const int* status, const double* out, const double* newpf, const double* weights,
    const double* logs, double* inv, double* pf, double* logip, int* aw,
    int* remaining, int n, int nq) {
  int w=blockIdx.x;
  if (!pending[w]) return;
  __shared__ int good; __shared__ double ip;
  if (!threadIdx.x) {
    good=candidate[w] && isfinite(logs[w]); ip=0;
    for (int q=0;q<nq;++q) {
      double p=newpf[w*nq+q];
      good &= status[w*nq+q]==0 && isfinite(p) && p!=0;
      ip=__dadd_rn(ip,__dmul_rn(weights[q],p));
    }
    good &= isfinite(ip) && ip!=0;
    if (!good) atomicAdd(remaining,1);
  }
  __syncthreads();
  if (!good) return;
  size_t base=(size_t)w*nq*n*n;
  for (size_t k=threadIdx.x;k<(size_t)nq*n*n;k+=blockDim.x) inv[base+k]=-out[base+k];
  for (int q=threadIdx.x;q<nq;q+=blockDim.x) pf[w*nq+q]=newpf[w*nq+q];
  if (!threadIdx.x) { logip[w]=log(fabs(ip) + 1e-100); aw[w]=0; pending[w]=0; }
}

extern "C" __global__ void opt_slater(double* slater, const double* params,
    const int* pa, const int* pb, const int* sa, const int* sb,
    const double* cs, const double* cc, const double* ss, int ns, int nq) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if (k>=ns*ns*nq) return;
  int q=k/(ns*ns), r=(k/ns)%ns, c=k%ns, n2=2*ns;
  double a=__dmul_rn(params[pa[k]],(double)sa[k]), b=__dmul_rn(params[pb[k]],(double)sb[k]);
  double d=__dsub_rn(a,b);
  double* s=slater+(size_t)q*n2*n2;
  s[r*n2+c]=-__dmul_rn(d,cs[q]);
  s[r*n2+c+ns]=__dadd_rn(__dmul_rn(a,cc[q]),__dmul_rn(b,ss[q]));
  s[(r+ns)*n2+c]=__dsub_rn(-__dmul_rn(a,ss[q]),__dmul_rn(b,cc[q]));
  s[(r+ns)*n2+c+ns]=__dmul_rn(d,cs[q]);
}

// Integer counts and electron-label lookup are rebuilt without downloading cfg.
extern "C" __global__ void opt_counts(const int* cfg, int* occ, int* lookup,
    int* counts, const int* gutz, const int* jast, const double* params,
    const double* pf, const double* weights, double* overlaps, double* logs,
    int ns, int ne, int nq, int ng, int nj) {
  int w=blockIdx.x, tid=threadIdx.x, n=2*ne, nproj=ng+nj;
  int* o=occ+w*2*ns; int* map=lookup+w*2*ns; int* cnt=counts+w*nproj;
  for (int r=tid;r<2*ns;r+=blockDim.x) { o[r]=0; map[r]=-1; }
  for (int p=tid;p<nproj;p+=blockDim.x) cnt[p]=0;
  __syncthreads();
  for (int a=tid;a<n;a+=blockDim.x) { int r=cfg[w*n+a]+(a>=ne?ns:0); o[r]=1; map[r]=a; }
  __syncthreads();
  for (int r=tid;r<ns;r+=blockDim.x) {
    if (gutz[r]>=0) atomicAdd(cnt+gutz[r],o[r]*o[r+ns]);
    if (nj) for (int s=r+1;s<ns;++s) {
      int p=jast[r*ns+s];
      if (p>=0) atomicAdd(cnt+ng+p,(o[r]+o[r+ns]-1)*(o[s]+o[s+ns]-1));
    }
  }
  __syncthreads();
  if (!tid) {
    double logc=0, ip=0;
    for (int p=0;p<nproj;++p) logc=__dadd_rn(logc,__dmul_rn(params[p],(double)cnt[p]));
    for (int q=0;q<nq;++q) ip=__dadd_rn(ip,__dmul_rn(weights[q],pf[w*nq+q]));
    logs[w]=logc; overlaps[w]=ip;
  }
}

extern "C" __global__ void opt_derivatives(double* O, const int* counts,
    const int* lookup, const int* columns, const int* xyz, const double* coeff,
    const double* inv, const double* pf, const double* weights, const double* overlaps,
    int np, int nproj, int ns, int ne, int nq, int batch, const int* error) {
  int lane=threadIdx.x%32;
  unsigned int index=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if (index>=np*batch || *error) return;
  int w=index%batch, p=index/batch, n=2*ne, output=w*np+p;
  if (p<nproj) { if (!lane) O[output]=counts[w*nproj+p]; return; }
  double sum=0, reciprocal=1.0/overlaps[w];
  for (int k=columns[p]+lane;k<columns[p+1];k+=32) {
    // Julia CSC rows decode to (C column, C row) in its column-major view.
    size_t coordinate=(size_t)3*k;
    int i=lookup[w*2*ns+xyz[coordinate+1]], j=lookup[w*2*ns+xyz[coordinate]], q=xyz[coordinate+2];
    if (i>=0 && i<j) {
      double cofactor=pf[w*nq+q]*inv[((size_t)w*nq+q)*n*n+j*n+i];
      double sensitivity=isfinite(reciprocal) && fabs(reciprocal)>=2.2250738585072014e-308
          ? -weights[q]*cofactor*reciprocal : -weights[q]*cofactor/overlaps[w];
      sum += coeff[k]*sensitivity;
    }
  }
  sum=opt_warp_sum(sum);
  if (!lane) O[output]=sum;
}

__device__ int opt_site(const int* e,int i,int a,int k,int da,int dk) {
  return i==a?da:(i==k?dk:e[i]);
}

// Correlation difference for at most two moves. Merge coincident sites first;
// include each touched/untouched and touched/touched Jastrow pair exactly once.
__device__ double opt_delta(const int* occ,const int* gutz,const int* jast,
    const double* params,const int* op,int ns,int ng,int nj,int lane) {
  int sites[4]={op[0],op[1],op[3],op[4]}, spins[4]={op[2],op[2],op[5],op[5]};
  int change[4]={1,-1,op[6]==2?1:0,op[6]==2?-1:0};
  int s[4],du[4],dd[4],count=0;
  for(int t=0;t<4;++t) {
    int i=0; for(;i<count && s[i]!=sites[t];++i) {}
    if (i==count) {s[count]=sites[t];du[count]=0;dd[count]=0;++count;}
    if(spins[t]) dd[i]+=change[t]; else du[i]+=change[t];
  }
  double result=0;
  if(!lane) for(int i=0;i<count;++i) {
    int r=s[i],u=occ[r],d=occ[r+ns],p=gutz[r];
    if(p>=0) result+=params[p]*((u+du[i])*(d+dd[i])-u*d);
  }
  if(nj) for(int r=lane;r<ns;r+=32) {
    int t=0; for(;t<count && s[t]!=r;++t) {}
    if(t<count) continue;
    int charge=occ[r]+occ[r+ns]-1;
    for(int i=0;i<count;++i) {
      int lo=r<s[i]?r:s[i],hi=r<s[i]?s[i]:r,p=jast[lo*ns+hi];
      if(p>=0) result+=params[ng+p]*(du[i]+dd[i])*charge;
    }
  }
  if(nj && !lane) for(int i=0;i<count;++i) for(int j=i+1;j<count;++j) {
    int lo=s[i]<s[j]?s[i]:s[j],hi=s[i]<s[j]?s[j]:s[i],p=jast[lo*ns+hi];
    int a=occ[s[i]]+occ[s[i]+ns]-1,b=occ[s[j]]+occ[s[j]+ns]-1;
    if(p>=0) result+=params[ng+p]*((a+du[i]+dd[i])*(b+du[j]+dd[j])-a*b);
  }
  return opt_warp_sum(result);
}

// A warp owns each Hamiltonian term. Electron labels encode operator signs;
// applying the right pair then the left avoids a mutable per-term configuration.
extern "C" __global__ void opt_energy_terms(double* values,const int* ops,
    const int* cfg,const int* occ,const int* lookup,const double* overlaps,
    const int* gutz,const int* jast,const double* params,const double* slater,
    const double* weights,const double* inv,const double* pf,
    int ns,int ne,int nq,int ng,int nj,int terms,int batch,const int* error) {
  int lane=threadIdx.x%32;
  unsigned int index=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(index>=terms*batch || *error) return;
  int t=index%terms,w=index/terms,n=2*ne,n2=2*ns;
  const int* op=ops+(size_t)7*t; const int* map=lookup+w*n2; const int* e=cfg+w*n;
  int k=-1,a=-1;
  if(op[6]==2) {
    k=map[op[4]+op[5]*ns];
    if(k<0 || (op[3]!=op[4] && map[op[3]+op[5]*ns]>=0)) {if(!lane) values[index]=0;return;}
  }
  // The right operation may have changed the source/destination of the left.
  for(int i=op[2]*ne;i<(op[2]+1)*ne;++i) {
    int r=i==k?op[3]:e[i];
    if(r==op[1]) a=i;
    if(op[0]!=op[1] && r==op[0]) {if(!lane) values[index]=0;return;}
  }
  if(a<0) {if(!lane) values[index]=0;return;}
  int da=op[0],dk=op[3];
  // Ignore unchanged electron slots; the left pair wins if labels coincide.
  if(k==a) k=-1;
  if(k>=0 && dk==e[k]) k=-1;
  if(da==e[a]) a=-1;
  if(a<0 && k<0) {if(!lane) values[index]=1;return;}
  if(a<0) {a=k;da=dk;k=-1;}
  double delta=opt_delta(occ+w*n2,gutz,jast,params,op,ns,ng,nj,lane),newip=0;
  for(int q=0;q<nq;++q) {
    const double* S=slater+(size_t)q*n2*n2;
    const double* M=inv+((size_t)w*nq+q)*n*n;
    int ra=da+(a>=ne?ns:0),rk=dk+(k>=ne?ns:0);
    double ratio=0;
    if(k<0) {
      double sum=0;
      for(int j=lane;j<n;j+=32) sum+=M[a*n+j]*S[ra*n2+e[j]+(j>=ne?ns:0)];
      ratio=-opt_warp_sum(sum);
    } else {
      double pa=0,pb=0,qa=0,qb=0,bma=0;
      for(int i=lane;i<n;i+=32) {
        int ri=opt_site(e,i,a,k,da,dk)+(i>=ne?ns:0);
        double av=S[ra*n2+ri],bv=S[rk*n2+ri],tmp=0;
        for(int j=0;j<n;++j) {
          int rj=opt_site(e,j,a,k,da,dk)+(j>=ne?ns:0);
          tmp+=M[i*n+j]*S[ra*n2+rj];
        }
        bma+=bv*tmp; pa+=M[i*n+a]*av; pb+=M[i*n+k]*av;
        qa+=M[i*n+a]*bv; qb+=M[i*n+k]*bv;
      }
      pa=opt_warp_sum(pa);pb=opt_warp_sum(pb);qa=opt_warp_sum(qa);qb=opt_warp_sum(qb);bma=opt_warp_sum(bma);
      ratio=M[a*n+k]*S[rk*n2+ra]+M[a*n+k]*bma+pa*qb-pb*qa;
    }
    newip+=weights[q]*(ratio*pf[w*nq+q]);
  }
  if(!lane) values[index]=exp(delta)*newip/overlaps[w];
}

extern "C" __global__ void opt_energy(double* energies,const double* values,
    const double* coefficients,int terms,int batch,int* error) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if(w>=batch || *error) return;
  double e=0;
  for(int t=0;t<terms;++t) e+=coefficients[t]*values[w*terms+t];
  energies[w]=e;
  if(!isfinite(e)) atomicCAS(error,0,2);
}

extern "C" __global__ void opt_means(double* mean,const double* O,const double* energy,
    int np,int batch) {
  int p=blockIdx.x,tid=threadIdx.x;
  __shared__ double partial[8];
  double sum=0;
  for(int w=tid;w<batch;w+=blockDim.x) sum+=p==np?energy[w]:O[w*np+p];
  sum=opt_warp_sum(sum);
  if(tid%32==0) partial[tid/32]=sum;
  __syncthreads();
  if(tid<32) {
    sum=opt_warp_sum(tid<8?partial[tid]:0.0);
    if(!tid) mean[p]=sum/(double)batch;
  }
}

// Julia materializes the mean outer product before covariance subtraction.
// An FMA here invents nonzero variance for one sample: the stored Gram product
// has rounded, while the fused mean product has not. Share the same rounding
// between pruning and the solver matrix, including exactly zero covariance.
__device__ double opt_covariance(double gram,double mi,double mj,int batch) {
  return __dsub_rn(gram/(double)batch,__dmul_rn(mi,mj));
}

extern "C" __global__ void opt_variance(double* variance,const double* gram,
    const double* mean,int np,int batch) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<np) variance[p]=opt_covariance(gram[p*np+p],mean[p],mean[p],batch);
}

extern "C" __global__ void opt_cutoff(double* cutoff,const double* variance,
    const double* mean,double* history,int step,int np,double threshold) {
  if(threadIdx.x || blockIdx.x) return;
  double value=0;
  for(int p=0;p<np;++p) value=fmax(value,fabs(variance[p]));
  *cutoff=value*threshold; history[step]=mean[np];
}

extern "C" __global__ void opt_system(double* gram,double* rhs,const double* ho,
    const double* mean,const double* variance,const double* cutoff,const int* flags,
    int np,int batch,double delta,double dt) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=np) return;
  bool active=flags[i] && variance[i]>*cutoff;
  rhs[i]=active?-2*dt*__dsub_rn(ho[i]/(double)batch,__dmul_rn(mean[np],mean[i])):0;
  for(int j=0;j<np;++j) {
    bool other=flags[j] && variance[j]>*cutoff;
    double s=opt_covariance(gram[j*np+i],mean[i],mean[j],batch);
    gram[j*np+i]=active&&other?(i==j?s*(1+delta):s):(i==j?1.0:0.0);
  }
}

extern "C" __global__ void opt_update(double* params,const double* update,
    const int* info,int* error,int np) {
  if(info[0] || info[1]) {if(!threadIdx.x && !blockIdx.x) atomicCAS(error,0,3);return;}
  if(*error) return;
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<np) {
    double value=params[p]+update[p];
    if(!isfinite(value)) atomicCAS(error,0,4); else params[p]=value;
  }
}

extern "C" __global__ void opt_gauge(double* params,double* projection,
    int np,int nproj,int shift,const int* error) {
  if(threadIdx.x || blockIdx.x || *error) return;
  double average=0,largest=0;
  if(shift) {for(int p=0;p<nproj;++p) average=__dadd_rn(average,params[p]);average/=nproj;}
  for(int p=nproj;p<np;++p) largest=fmax(largest,fabs(params[p]));
  double scale=largest>0?4.0/largest:1.0;
  for(int p=0;p<nproj;++p) {params[p]=__dsub_rn(params[p],average);projection[p]=params[p];}
  for(int p=nproj;p<np;++p) params[p]=__dmul_rn(params[p],scale);
}
