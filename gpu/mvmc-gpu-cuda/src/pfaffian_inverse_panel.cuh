// Shared paired factorization for the complete Pfaffian/inverse rebuild.
// Exactly the existing A/W/OUT matrices are used; no extra device allocation.
// Real n=64..126 and complex n>=64 select this path. The existing Julia
// paired real n>=128 path is faster and keeps its independent C fallback.
#ifndef MVMC_PFAFFIAN_INVERSE_PANEL
#define MVMC_PFAFFIAN_INVERSE_PANEL
struct PfinvRealPanelOps : ResidentRealPfOps {
  __device__ static bool finite(T x) { return isfinite(x); }
  __device__ static void reference(T*,T*,T*,T*,int*,int);
};
struct PfinvComplexPanelOps : ResidentComplexPfOps {
  __device__ static bool finite(T x) { return rc_finite(x); }
  __device__ static void reference(T*,T*,T*,T*,int*,int);
};

template <typename Ops>
__device__ void pfinv_panel_plane(typename Ops::T* A, typename Ops::T* W,
    typename Ops::T* OUT, typename Ops::T* PF, int* ST, int n) {
  typedef typename Ops::T T;
  extern __shared__ char panel_mem[];
  T* tau=reinterpret_cast<T*>(panel_mem);
  T* u=tau+n;
  T* diagonal=u+n;
  int* permutation=reinterpret_cast<int*>(diagonal+n/2);
  __shared__ int status;
  const int tid=threadIdx.x, nt=blockDim.x;
  const size_t nn=(size_t)n*n;
  // Preserve input for the exact reference failure/status path.
  for (size_t i=tid;i<nn;i+=nt) W[i]=A[i];
  __syncthreads();
  const T pf=resident_panel_pfaffian<Ops,true>(A,n,tau,u,OUT,diagonal,permutation,&status);
  if (status!=0) {
    for (size_t i=tid;i<nn;i+=nt) A[i]=W[i];
    __syncthreads();
    Ops::reference(A,W,OUT,PF,ST,n);
    return;
  }
  if (!Ops::finite(pf)) {
    for (size_t i=tid;i<nn;i+=nt) A[i]=W[i];
    __syncthreads();
    Ops::reference(A,W,OUT,PF,ST,n);
    return;
  }
  int bad_reciprocal=0;
  for (int i=tid;i<n/2;i+=nt) {
    diagonal[i]=Ops::div(Ops::one(),diagonal[i]);
    bad_reciprocal |= !Ops::finite(diagonal[i]);
  }
  // Removing products with structural zeros is valid for finite L. Preserve
  // the reference fallback when a factor is nonfinite: the old full dot
  // would encounter nonfinite*0 on its diagonal even if pf stayed finite.
  for (size_t i=tid;i<nn;i+=nt) bad_reciprocal |= !Ops::finite(OUT[i]);
  bad_reciprocal=__syncthreads_or(bad_reciprocal);
  if (bad_reciprocal) {
    for (size_t i=tid;i<nn;i+=nt) A[i]=W[i];
    __syncthreads();
    Ops::reference(A,W,OUT,PF,ST,n);
    return;
  }
  // Reuse A for B=L^-1, row-major for coalesced accesses across columns.
  // Keep W's exact original input through inversion, not just factorization:
  // a finite inverse can have nonrepresentable intermediate lower factors.
  for (size_t i=tid;i<nn;i+=nt) A[i]=Ops::zero();
  __syncthreads();
  for (int row=0;row<n;++row) {
    for (int col=tid;col<=row;col+=nt) {
      T value=row==col ? Ops::one() : Ops::zero();
      // Skip complete warp-width runs of structural zeros, retaining a
      // common k across lanes for coalesced B loads and broadcast L loads.
      // Remaining terms stay in their original order, including publication.
      for (int k=col & ~31;k<row;++k)
        value=Ops::sub(value,Ops::mul(OUT[row+(size_t)k*n],A[col+(size_t)k*n]));
      A[col+(size_t)row*n]=value;
    }
    __syncthreads();
  }
  // B^T D^-1 B: only one triangle is evaluated. Pair products avoid
  // materializing D^-1 B and reuse the diagonal in shared memory.
  int bad=0;
  for (int j=tid;j<n;j+=nt) OUT[permutation[j]+(size_t)permutation[j]*n]=Ops::zero();
  for (size_t index=tid;index<nn;index+=nt) {
    const int row=index%n, col=index/n;
    if (row<=col) continue;
    T value=Ops::zero();
    for (int k=row & ~1;k<n;k+=2) {
      const T delta=Ops::sub(Ops::mul(A[row+(size_t)k*n],A[col+(size_t)(k+1)*n]),
                            Ops::mul(A[row+(size_t)(k+1)*n],A[col+(size_t)k*n]));
      value=Ops::add(value,Ops::mul(delta,diagonal[k/2]));
    }
    bad |= !Ops::finite(value);
    OUT[permutation[row]+(size_t)permutation[col]*n]=value;
    OUT[permutation[col]+(size_t)permutation[row]*n]=Ops::neg(value);
  }
  bad=__syncthreads_or(bad);
  if (bad) {
    for (size_t i=tid;i<nn;i+=nt) A[i]=W[i];
    __syncthreads();
    Ops::reference(A,W,OUT,PF,ST,n);
    return;
  }
  if (tid==0) { *PF=pf; *ST=0; }
}
#endif
