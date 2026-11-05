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

// Julia initial_configuration_kernel! (cuda_initial_sampling.jl:13-80). A
// uniform permutation of electron labels assigns the local-spin sites in site
// order; each spin then shuffles the nonlocal sites (cumulatively) and fills
// its unassigned labels in order. `loc` is null when there are no local spins,
// which consumes the same words in the same order.
extern "C" __global__ void opt_initial(int* cfg, int* scratch, const unsigned int* tape,
    const int* pending, int* candidate, const int* loc, int ns, int ne, int batch) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if (w>=batch || !pending[w]) return;
  int n=2*ne, words=n+2*ns+16, cursor=0, good=1;
  int* e=cfg+w*n; int* s=scratch+w*(n+ns);
  const unsigned int* t=tape+w*words;
  for (int a=0;a<n;++a) { e[a]=-1; s[a]=a; }
  for (int k=n;k>=2;--k) {
    int j=opt_bounded(t,cursor,words,k,good), v=s[k-1]; s[k-1]=s[j]; s[j]=v;
  }
  int nlocal=0, nonlocal=0;
  for (int r=0;r<ns;++r) {
    if (loc!=nullptr && loc[r]==1) {
      // The host checks nlocal <= 2Ne; keep every address valid regardless.
      if (nlocal<n) e[s[nlocal]]=r; else good=0;
      ++nlocal;
    } else {
      s[n+nonlocal++]=r;
    }
  }
  for (int spin=0;spin<2;++spin) {
    int unused=0;
    for (int a=0;a<ne;++a) unused+=e[spin*ne+a]==-1;
    if (unused>nonlocal) good=0;
    for (int k=nonlocal;k>=2;--k) {
      int j=opt_bounded(t,cursor,words,k,good), v=s[n+k-1]; s[n+k-1]=s[n+j]; s[n+j]=v;
    }
    int next=0;
    for (int a=0;a<ne;++a) {
      if (e[spin*ne+a]==-1) {
        if (next<nonlocal) e[spin*ne+a]=s[n+next];
        ++next;
      }
    }
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
    int ns, int ne, int nq, int ng, int nj, const int* dh, int n2, int n4) {
  int w=blockIdx.x, tid=threadIdx.x, n=2*ne, nproj=ng+nj+6*n2+10*n4;
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
  // C MakeProjCnt DH2/DH4 counters (integers, so atomic order is irrelevant).
  if (n2+n4) {
    const ResidentOcc occupation{o,ns};
    for (int k=tid;k<(n2+n4)*ns;k+=blockDim.x) {
      int p=resident_dh_index(occupation,dh,n2,n4,ns,ng+nj,k);
      if (p>=0) atomicAdd(cnt+p,1);
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
    int np, int nproj, int ns, int ne, int nq, int batch, int* error, const int* valid, int nrbm,
    const double* fix, int nfix, int nopt) {
  int lane=threadIdx.x%32;
  unsigned int index=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if (index>=np*batch || *error) return;
  int w=index%batch, p=index/batch, n=2*ne, output=w*np+p;
  if (p<nproj) { if (!lane) O[output]=counts[w*nproj+p]; return; }
  if (p<nproj+nrbm) return;  // opt_rbm_diff writes the RBM block
  if (p>=np-nopt) {
    // C calculateOptTransDiff (vmccal.c:652-665): the sector's fixed-weight
    // overlap over the total, O_i = sum_j QPFixWeight[j] PfM[i*NQPFix+j] / ip.
    // An invalid plane publishes Pf = 0, its exact value.
    if (!lane) {
      const double* pf_w=pf+(size_t)w*nq+(size_t)(p-(np-nopt))*nfix;
      double ip=0;
      for (int j=0;j<nfix;++j) ip=__dadd_rn(ip,__dmul_rn(fix[j],pf_w[j]));
      const double value=ip/overlaps[w];
      O[output]=value;
      if (!isfinite(value)) atomicCAS(error,0,RESIDENT_DERIVATIVES);
    }
    return;
  }
  double sum=0, reciprocal=1.0/overlaps[w];
  for (int k=columns[p]+lane;k<columns[p+1];k+=32) {
    // Julia CSC rows decode to (C column, C row) in its column-major view.
    size_t coordinate=(size_t)3*k;
    int i=lookup[w*2*ns+xyz[coordinate+1]], j=lookup[w*2*ns+xyz[coordinate]], q=xyz[coordinate+2];
    if (i>=0 && i<j) {
      // An invalid plane stores its exact minor cofactor here (opt_cofactors).
      double stored=inv[((size_t)w*nq+q)*n*n+j*n+i];
      double cofactor=valid[w*nq+q] ? pf[w*nq+q]*stored : stored;
      double sensitivity=isfinite(reciprocal) && fabs(reciprocal)>=2.2250738585072014e-308
          ? -weights[q]*cofactor*reciprocal : -weights[q]*cofactor/overlaps[w];
      sum += coeff[k]*sensitivity;
    }
  }
  sum=opt_warp_sum(sum);
  if (!lane) {
    O[output]=sum;
    // A nonfinite derivative would otherwise reach SR as a generic solve failure.
    if (!isfinite(sum)) atomicCAS(error,0,RESIDENT_DERIVATIVES);
  }
}

__device__ int opt_site(const int* e,int i,int a,int k,int da,int dk) {
  return i==a?da:(i==k?dk:e[i]);
}

// Correlation difference for at most two moves. Merge coincident sites first;
// include each touched/untouched and touched/touched Jastrow pair exactly once.
__device__ double opt_delta(const int* occ,const int* gutz,const int* jast,
    const double* params,const int* op,int ns,int ng,int nj,int lane,
    const int* dh,int n2,int n4,const int* rbm,const double* rbmw,const double* theta_w,int nh) {
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
  // C UpdateProjCnt recounts DH2/DH4 for the moved configuration.
  if(n2+n4) {
    const ResidentOcc before{occ,ns};
    const ResidentOccMoved after{occ,ns,count,s,du,dd};
    result+=resident_dh_delta(before,after,dh,n2,n4,ns,ng+nj,params,lane,32);
  }
  // C calham.c/calgrn.c: UpdateRBMCnt for the right pair, then the left pair,
  // and the RBM ratio of the moved configuration (LogRBMRatio).
  if(rbm!=nullptr) {
    const int two=op[6]==2;
    const int ri[2]={two?op[4]:op[1],op[1]},rj[2]={two?op[3]:op[0],op[0]};
    const int sp[2]={two?op[5]:op[2],op[2]};
    result+=resident_rbm_hidden_delta(rbm,rbmw,theta_w,ns,nh,ri,rj,sp,two+1,lane,32);
    if(!lane) result+=resident_rbm_phys_delta(rbm,rbmw,ns,ri,rj,sp,two+1);
  }
  return opt_warp_sum(result);
}

// Resolve one operator against a walker's labels: -1 gives value 0, 1 gives
// value 1, 0 leaves moved slots a (to da) and optionally k (to dk). Electron
// labels encode operator signs; applying the right pair then the left avoids a
// mutable per-term configuration.
__device__ int opt_resolve(const int* op,const int* map,const int* e,int ne,int ns,
    int& a,int& k,int& da,int& dk) {
  k=-1;a=-1;da=op[0];dk=op[3];
  if(op[6]==2) {
    k=map[op[4]+op[5]*ns];
    if(k<0 || (op[3]!=op[4] && map[op[3]+op[5]*ns]>=0)) return -1;
  }
  // The right operation may have changed the source/destination of the left.
  for(int i=op[2]*ne;i<(op[2]+1)*ne;++i) {
    int r=i==k?op[3]:e[i];
    if(r==op[1]) a=i;
    if(op[0]!=op[1] && r==op[0]) return -1;
  }
  if(a<0) return -1;
  // Ignore unchanged electron slots; the left pair wins if labels coincide.
  if(k==a) k=-1;
  if(k>=0 && dk==e[k]) k=-1;
  if(da==e[a]) a=-1;
  if(a<0 && k<0) return 1;
  if(a<0) {a=k;da=dk;k=-1;}
  return 0;
}

__device__ int opt_all_valid(const int* valid,int w,int nq) {
  int all=1;
  for(int q=0;q<nq;++q) all&=valid[w*nq+q]!=0;
  return all;
}

// A warp owns each Hamiltonian term. Walkers with an invalid QP plane are
// evaluated by opt_energy_fallback instead.
extern "C" __global__ void opt_energy_terms(double* values,const int* ops,
    const int* cfg,const int* occ,const int* lookup,const double* overlaps,
    const int* gutz,const int* jast,const double* params,const double* slater,
    const double* weights,const double* inv,const double* pf,
    int ns,int ne,int nq,int ng,int nj,int terms,int batch,const int* error,
    const int* valid,const int* dh,int n2dh,int n4dh,const int* rbm,const double* rbmw,
    const double* theta,int nh) {
  int lane=threadIdx.x%32;
  unsigned int index=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  if(index>=terms*batch || *error) return;
  int t=index%terms,w=index/terms,n=2*ne,n2=2*ns;
  if(!opt_all_valid(valid,w,nq)) return;
  const int* op=ops+(size_t)7*t; const int* map=lookup+w*n2; const int* e=cfg+w*n;
  int a,k,da,dk;
  int resolved=opt_resolve(op,map,e,ne,ns,a,k,da,dk);
  if(resolved) {if(!lane) values[index]=resolved>0?1:0;return;}
  double delta=opt_delta(occ+w*n2,gutz,jast,params,op,ns,ng,nj,lane,dh,n2dh,n4dh,
      rbm,rbmw,rbm?theta+(size_t)w*nh:nullptr,nh),newip=0;
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

// Julia observable_kernel!(fallback_only=true) with updated_overlap!: one
// block per walker that has an invalid plane evaluates every term with full
// Pfaffians of the moved configuration (cuda_observables.jl:109-122,283-295).
// All-valid walkers return immediately. `fac` is per-walker nq*n*n scratch.
// Dynamic shared memory: n ints of moved sites (rounded to even), 2n doubles.
extern "C" __global__ void opt_energy_fallback(double* values,const int* ops,
    const int* cfg,const int* occ,const int* lookup,const double* overlaps,
    const int* gutz,const int* jast,const double* params,const double* slater,
    const double* weights,const int* valid,double* fac,
    int ns,int ne,int nq,int ng,int nj,int terms,int batch,const int* error,
    const int* dh,int n2dh,int n4dh,const int* rbm,const double* rbmw,const double* theta,int nh) {
  int w=blockIdx.x,tid=threadIdx.x,n=2*ne,n2=2*ns;
  if(w>=batch || *error || opt_all_valid(valid,w,nq)) return;
  extern __shared__ char fallback_mem[];
  int* rs=reinterpret_cast<int*>(fallback_mem);
  double* vec=reinterpret_cast<double*>(rs+((n+1)&~1));
  __shared__ int s_resolved,s_a,s_k,s_da,s_dk;
  __shared__ double s_delta,s_ip;
  double* A=fac+(size_t)w*nq*n*n;
  const int* map=lookup+w*n2; const int* e=cfg+w*n;
  for(int t=0;t<terms;++t) {
    const int* op=ops+(size_t)7*t;
    if(!tid) {
      int a,k,da,dk;
      s_resolved=opt_resolve(op,map,e,ne,ns,a,k,da,dk);
      s_a=a;s_k=k;s_da=da;s_dk=dk;s_ip=0;
    }
    __syncthreads();
    const int resolved=s_resolved;
    if(resolved) {
      if(!tid) values[(size_t)w*terms+t]=resolved>0?1:0;
      __syncthreads();
      continue;
    }
    if(tid<32) {
      double d=opt_delta(occ+w*n2,gutz,jast,params,op,ns,ng,nj,tid,dh,n2dh,n4dh,
          rbm,rbmw,rbm?theta+(size_t)w*nh:nullptr,nh);
      if(!tid) s_delta=d;
    }
    for(int i=tid;i<n;i+=blockDim.x)
      rs[i]=opt_site(e,i,s_a,s_k,s_da,s_dk)+(i>=ne?ns:0);
    __syncthreads();
    for(int q=0;q<nq;++q) {
      resident_gather(A,rs,n,-1,-1,slater,q,n2);
      const double p=resident_block_pfaffian(A,n,vec,vec+n);
      if(!tid) s_ip+=weights[q]*p;
    }
    if(!tid) values[(size_t)w*terms+t]=exp(s_delta)*s_ip/overlaps[w];
    __syncthreads();
  }
}

// Exact cofactors of an invalid plane (Julia cuda_sr.jl:154-187):
// dPf(A)/dA(i,j) = (-1)^(i+j+1) Pf(A without rows/columns i,j) for 0-based
// i<j. They are stored where opt_derivatives otherwise reads the inverse
// entry, which no all-valid path reads for an invalid plane. One block per
// (walker, QP); valid planes return. `fac` is per-plane n*n scratch.
// Dynamic shared memory: n ints of sites (rounded to even), then 2n doubles.
extern "C" __global__ void opt_cofactors(double* inv,const int* valid,const int* cfg,
    const double* slater,double* fac,int ns,int ne,int nq,int batch,const int* error) {
  int plane=blockIdx.x,w=plane/nq,q=plane%nq,n=2*ne,n2=2*ns;
  if(w>=batch || *error || valid[plane]) return;
  extern __shared__ char cofactor_mem[];
  int* rs=reinterpret_cast<int*>(cofactor_mem);
  double* vec=reinterpret_cast<double*>(rs+((n+1)&~1));
  for(int i=threadIdx.x;i<n;i+=blockDim.x) rs[i]=cfg[w*n+i]+(i>=ne?ns:0);
  __syncthreads();
  double* A=fac+(size_t)plane*n*n;
  double* C=inv+(size_t)plane*n*n;
  for(int j=1;j<n;++j) for(int i=0;i<j;++i) {
    const int m=resident_gather(A,rs,n,i,j,slater,q,n2);
    const double p=m?resident_block_pfaffian(A,m,vec,vec+m):1.0;
    if(!threadIdx.x) C[j*n+i]=((i+j+1)&1)?-p:p;
    __syncthreads();
  }
}

extern "C" __global__ void opt_energy(double* energies,const double* values,
    const double* coefficients,int terms,int batch,int* error) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if(w>=batch || *error) return;
  double e=0;
  for(int t=0;t<terms;++t) e+=coefficients[t]*values[w*terms+t];
  energies[w]=e;
  if(!isfinite(e)) atomicCAS(error,0,RESIDENT_ENERGY);
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
  if(info[0] || info[1]) {if(!threadIdx.x && !blockIdx.x) atomicCAS(error,0,RESIDENT_SR);return;}
  if(*error) return;
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<np) {
    double value=params[p]+update[p];
    if(!isfinite(value)) atomicCAS(error,0,RESIDENT_PARAMETERS); else params[p]=value;
  }
}

// C SyncModifiedParameter (parameter.c:134-176): shiftDH2/shiftDH4 move each
// d/h generator's mean into the Gutzwiller factors, then shiftGJ. The flags
// follow C SetFlagShift; the Rust C-order sync_modified_parameter matches.
// Slater is rescaled, then OptTrans is normalized to maximum amplitude one,
// and C UpdateQPWeight (qp.c:130-148) rebuilds the full QP weights for the
// next rebuild and sampling interval.
extern "C" __global__ void opt_gauge(double* params,double* projection,
    int np,int ng,int nj,int n2,int n4,int shift,int shift2,int shift4,const int* error,
    double* rbmw,int nrbm,int nopt,double* weights,const double* fix,int nfix) {
  if(threadIdx.x || blockIdx.x || *error) return;
  const int gj=ng+nj,nproj=gj+6*n2+10*n4,slater=nproj+nrbm,trans=np-nopt;
  double g=0;
  if(shift2) for(int x=0;x<2*n2;++x) {
    double* d=params+gj+x;
    double mean=__dadd_rn(__dadd_rn(d[0],d[2*n2]),d[4*n2])/3.0;
    for(int b=0;b<3;++b) d[2*b*n2]=__dsub_rn(d[2*b*n2],mean);
    g=__dadd_rn(g,mean);
  }
  if(shift4) {
    double g4=0;
    for(int x=0;x<2*n4;++x) {
      double* d=params+gj+6*n2+x;
      double mean=d[0];
      for(int b=1;b<5;++b) mean=__dadd_rn(mean,d[2*b*n4]);
      mean/=5.0;
      for(int b=0;b<5;++b) d[2*b*n4]=__dsub_rn(d[2*b*n4],mean);
      g4=__dadd_rn(g4,mean);
    }
    g=__dadd_rn(g,g4);
  }
  if(g!=0) for(int p=0;p<ng;++p) params[p]=__dadd_rn(params[p],g);
  double average=0,largest=0;
  if(shift) {for(int p=0;p<gj;++p) average=__dadd_rn(average,params[p]);average/=gj;}
  // C rescales only Slater (Para + NProj + NRBM); RBM is left unchanged.
  for(int p=slater;p<trans;++p) largest=fmax(largest,fabs(params[p]));
  double scale=largest>0?4.0/largest:1.0;
  for(int p=0;p<gj;++p) params[p]=__dsub_rn(params[p],average);
  for(int p=0;p<nproj;++p) projection[p]=params[p];
  if(rbmw) for(int p=0;p<nrbm;++p) rbmw[p]=params[nproj+p];
  for(int p=slater;p<trans;++p) params[p]=__dmul_rn(params[p],scale);
  if(nopt) {
    // OptTrans *= 1/max|OptTrans|; a zero maximum is left unscaled, as the
    // Rust C-order sync does (C would divide by zero).
    double xmax=0;
    for(int i=0;i<nopt;++i) xmax=fmax(xmax,fabs(params[trans+i]));
    if(xmax>0) {
      const double ratio=__ddiv_rn(1.0,xmax);
      for(int i=0;i<nopt;++i) params[trans+i]=__dmul_rn(params[trans+i],ratio);
    }
    for(int i=0;i<nopt;++i) for(int j=0;j<nfix;++j)
      weights[i*nfix+j]=__dmul_rn(params[trans+i],fix[j]);
  }
}

// C MakeRBMCnt hidden fields of every walker from its occupation table: one
// thread per (walker, neuron), in C operation order (resident_rbm_make).
extern "C" __global__ void opt_rbm_theta(double* theta,const int* occ,const int* rbm,
    const double* rbmw,int ns,int nh,int batch,const int* error) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=batch*nh || *error) return;
  int w=k/nh,h=k%nh;
  theta[k]=resident_rbm_make(rbm,rbmw,occ+(size_t)w*2*ns,ns,nh,h);
}

// Real part of C RBMDiff (rbm.c:325-385) into the RBM columns of O, one
// thread per walker in C order: physical counters (C MakeRBMCnt sums), then
// per neuron tanh(theta) into its hidden coefficient and x_r tanh(theta)
// into its couplings. Tied indices accumulate in that order.
extern "C" __global__ void opt_rbm_diff(double* O,const int* occ,const double* theta,
    const int* rbm,int np,int nproj,int nrbm,int ns,int nh,int batch,int* error) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if(w>=batch || *error) return;
  double* o=O+(size_t)w*np+nproj;
  const int* x=occ+(size_t)w*2*ns;
  for(int p=0;p<nrbm;++p) o[p]=0.0;
  for(int r=0;r<ns;++r) {
    if(rbm[r]>=0) o[rbm[r]]+=(double)(x[r]+x[r+ns]-1);
    if(rbm[ns+r]>=0) o[rbm[ns+r]]+=(double)(x[r]-x[r+ns]);
  }
  for(int rs=0;rs<2*ns;++rs) if(rbm[2*ns+rs]>=0) o[rbm[2*ns+rs]]+=(double)(2*x[rs]-1);
  for(int h=0;h<nh;++h) {
    const int family=resident_rbm_family(rbm,ns,h),b=rbm[4*ns+nh+h];
    const int* c=resident_rbm_coupling(rbm,ns,nh,h);
    const double t=tanh(theta[(size_t)w*nh+h]);
    if(b>=0) o[b]=__dadd_rn(o[b],t);
    if(c[0]<0) continue;
    const int width=family==2?2*ns:ns;
    for(int rs=0;rs<width;++rs) {
      const int xi=family==0?x[rs]+x[rs+ns]-1:family==1?x[rs]-x[rs+ns]:2*x[rs]-1;
      o[c[rs]]=__dadd_rn(o[c[rs]],__dmul_rn((double)xi,t));
    }
  }
  for(int p=0;p<nrbm;++p) if(!isfinite(o[p])) atomicCAS(error,0,RESIDENT_DERIVATIVES);
}
