// Device-resident SR-CG (#670, #572 milestone 4): C StochasticOptCG_real
// (stcopt_cg_impl.c) without host round trips. The host enqueues an upper
// bound of iterations; every iteration kernel returns at once after the
// device-side stop flag is set, so the C stopping rule and iteration count are
// decided on the device.
//
// The active parameters are a mask rather than C's compacted smatToParaIdx
// arrays: every masked operation adds exact zeros, so it equals the compacted
// arithmetic up to reduction order. Elementwise updates use explicit _rn
// intrinsics (no FMA), as the C/Rust CPU loops round each product separately.
//
// st:  [0] delta, [1] threshold, [2] alpha, [3] coef (<O>.x), [4] 1/Wc
// ist: [0] stop flag, [1] completed iterations, [2] max iterations, [3] nSmat

#define CG_BLOCK 1024

__device__ double cg_block_sum(double x, double* scratch) {
  x = opt_warp_sum(x);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (!lane) scratch[warp] = x;
  __syncthreads();
  if (warp == 0) {
    x = threadIdx.x < (blockDim.x >> 5) ? scratch[threadIdx.x] : 0.0;
    x = opt_warp_sum(x);
  }
  __syncthreads();
  if (!threadIdx.x) scratch[0] = x;
  __syncthreads();
  return scratch[0];
}

// Masked fixed-order dot product of one block.
__device__ double cg_block_dot(const double* a, const double* b, const int* active, int n,
    double* scratch) {
  double s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    if (active[i]) s = __dadd_rn(s, __dmul_rn(a[i], b[i]));
  return cg_block_sum(s, scratch);
}

// SROptO/OOdiag/HO of the real path: one thread per parameter accumulates the
// samples sequentially, then divides by Wc (C weightAverageSROpt, w=1). Thread
// np computes <H>. mean has np+1 entries ([np] = <H>), as in opt_means.
extern "C" __global__ void cg_moments(double* mean, double* variance, double* ho,
    const double* O, const double* energy, int np, int batch) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p > np) return;
  const double wc = (double)batch;
  if (p == np) {
    double e = 0;
    for (int w = 0; w < batch; ++w) e = __dadd_rn(e, energy[w]);
    mean[np] = e / wc;
    return;
  }
  double o = 0, oo = 0, eo = 0;
  for (int w = 0; w < batch; ++w) {
    const double x = O[(size_t)w * np + p];
    o = __dadd_rn(o, x);
    oo = __dadd_rn(oo, __dmul_rn(x, x));
    eo = __dadd_rn(eo, __dmul_rn(energy[w], x));
  }
  const double m = o / wc;
  mean[p] = m;
  // stcopt_cg_impl.c:113 S[pi][pi] = OOdiag[pi] - O[pi]*O[pi].
  variance[p] = __dsub_rn(oo / wc, __dmul_rn(m, m));
  ho[p] = eo / wc;
}

// stcopt_cg_impl.c:111-160 (diagonal cut) and StochasticOptCG_Init (:463-492):
// x=0, d=r=g, and the initial stop test of StochasticOptCG_Main.
extern "C" __global__ void cg_prepare(double* x, double* vec, int* active, double* st, int* ist,
    double* trace, double* history, const double* mean, const double* variance,
    const double* ho, const int* flags, int np, int batch, int step, double red_cut,
    double step_dt, double tolerance, int max_iter, const int* error) {
  __shared__ double scratch[32];
  __shared__ int count[32];
  double* g = vec;
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  double* sdiag = vec + 4 * (size_t)np;
  double* mo = vec + 5 * (size_t)np;
  // C's maximum: sDiagMax starts at sDiagElm[0] and grows on `>`, so NaN
  // entries are skipped unless the first one is NaN.
  double v = -__longlong_as_double(0x7ff0000000000000LL);
  for (int p = threadIdx.x; p < np; p += blockDim.x) v = fmax(v, variance[p]);
  for (int o = 16; o; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffff, v, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (!lane) scratch[warp] = v;
  __syncthreads();
  if (!threadIdx.x) {
    double m = scratch[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) m = fmax(m, scratch[i]);
    if (isnan(variance[0])) m = variance[0];
    scratch[0] = m;
  }
  __syncthreads();
  const double threshold = scratch[0] * red_cut;
  const double dt = 2.0 * step_dt;
  const double energy = mean[np];
  int n = 0;
  for (int p = threadIdx.x; p < np; p += blockDim.x) {
    // C: fixed by OptFlag, else cut when sDiag < diagCutThreshold.
    const bool a = flags[p] && !(variance[p] < threshold);
    active[p] = a;
    n += a;
    x[p] = 0;
    // StochasticOptCG_Init: g = -dt*(HO - H*O), dt = 2*DSROptStepDt.
    const double gp = a ? __dmul_rn(-dt, __dsub_rn(ho[p], __dmul_rn(energy, mean[p]))) : 0.0;
    g[p] = d[p] = r[p] = gp;
    sdiag[p] = a ? variance[p] : 0.0;
    mo[p] = a ? mean[p] : 0.0;
  }
  for (int o = 16; o; o >>= 1) n += __shfl_down_sync(0xffffffff, n, o);
  __syncthreads();
  if (!lane) count[warp] = n;
  __syncthreads();
  if (!threadIdx.x) {
    int total = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) total += count[i];
    count[0] = total;
  }
  __syncthreads();
  const int nsmat = count[0];
  const double delta = cg_block_dot(r, r, active, np, scratch);
  if (!threadIdx.x) {
    // stcopt_cg_impl.c:264-265, evaluated left to right.
    const double thresh = tolerance * tolerance * (double)nsmat * (double)nsmat;
    const int limit = max_iter > 0 ? max_iter : nsmat;
    st[0] = delta;
    st[1] = thresh;
    st[4] = 1.0 / (double)batch;
    ist[1] = 0;
    ist[2] = limit;
    ist[3] = nsmat;
    ist[0] = *error || limit <= 0 || delta < thresh;
    trace[0] = delta;
    history[step] = energy;
  }
}

// y[w] = sum_si v[si] O[si][w] (C DGEMV 'T'); block `batch` computes <O>.v.
extern "C" __global__ void cg_ty(double* y, double* st, const int* ist, const double* O,
    const double* v, const int* active, const double* mo, int np, int batch) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  const int w = blockIdx.x;
  double s = 0;
  if (w < batch) {
    const double* row = O + (size_t)w * np;
    for (int p = threadIdx.x; p < np; p += blockDim.x)
      if (active[p]) s += row[p] * v[p];
    s = cg_block_sum(s, scratch);
    if (!threadIdx.x) y[w] = s;
  } else {
    s = cg_block_dot(mo, v, active, np, scratch);
    if (!threadIdx.x) st[3] = s;
  }
}

// operate_by_S (stcopt_cg_impl.c:376-432): z = invW*(O y) - coef*<O>, then
// z += sdiag*DSROptStaDel*v.
extern "C" __global__ void cg_oy(double* z, const double* st, const int* ist, const double* O,
    const double* y, const double* v, const int* active, const double* mo,
    const double* sdiag, int np, int batch, double sta_del) {
  if (ist[0]) return;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= np) return;
  if (!active[p]) {z[p] = 0; return;}
  double s = 0;
  for (int w = 0; w < batch; ++w) s += O[(size_t)w * np + p] * y[w];
  const double value = __dsub_rn(__dmul_rn(st[4], s), __dmul_rn(st[3], mo[p]));
  z[p] = __dadd_rn(value, __dmul_rn(__dmul_rn(sdiag[p], sta_del), v[p]));
}

// alpha = delta/(d.q), x += alpha*d (stcopt_cg_impl.c:307-316).
__device__ double cg_alpha_update(double* x, const double* d, const double* q,
    const int* active, double* st, int np, double* scratch) {
  const double alpha = st[0] / cg_block_dot(d, q, active, np, scratch);
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) x[p] = __dadd_rn(x[p], __dmul_rn(alpha, d[p]));
  return alpha;
}

// beta = (r.r)/delta, delta = beta*delta, d = r + beta*d and the stop test of
// the next C loop iteration (stcopt_cg_impl.c:333-343, :270).
__device__ void cg_direction(double* d, const double* r, const int* active, double* st,
    int* ist, double* trace, int np, double* scratch) {
  const double delta = st[0];
  const double beta = cg_block_dot(r, r, active, np, scratch) / delta;
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) d[p] = __dadd_rn(r[p], __dmul_rn(beta, d[p]));
  if (!threadIdx.x) {
    const double next = beta * delta;
    const int iter = ist[1] + 1;
    st[0] = next;
    ist[1] = iter;
    trace[iter] = next;
    ist[0] = iter >= ist[2] || next < st[1];
  }
}

// One ordinary iteration after q = S d: r -= alpha*q.
extern "C" __global__ void cg_iterate(double* x, double* vec, const int* active, double* st,
    int* ist, double* trace, int np) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  const double* q = vec + 3 * (size_t)np;
  const double alpha = cg_alpha_update(x, d, q, active, st, np, scratch);
  if (!threadIdx.x) st[2] = alpha;
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) r[p] = __dsub_rn(r[p], __dmul_rn(alpha, q[p]));
  cg_direction(d, r, active, st, ist, trace, np, scratch);
}

// Every 20th iteration: alpha and x first, then r = S x is formed by
// cg_ty/cg_oy, and cg_refresh replaces it by g - S x.
extern "C" __global__ void cg_alpha(double* x, double* vec, const int* active, double* st,
    const int* ist, int np) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  const double alpha = cg_alpha_update(x, vec + np, vec + 3 * (size_t)np, active, st, np,
      scratch);
  if (!threadIdx.x) st[2] = alpha;
}

extern "C" __global__ void cg_refresh(double* vec, const int* active, double* st, int* ist,
    double* trace, int np) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  const double* g = vec;
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) r[p] = __dsub_rn(g[p], r[p]);
  cg_direction(d, r, active, st, ist, trace, np, scratch);
}

// stcopt_cg_impl.c:221-233: a nonfinite increment returns info=1 and C stops
// (vmcmain.c:501-504). Nothing is published; opt_update observes the status.
extern "C" __global__ void cg_finish(const double* x, const int* active, int* info,
    const int* ist, int* iterations, int step, int np, int* error) {
  __shared__ int bad[32];
  int b = 0;
  for (int p = threadIdx.x; p < np; p += blockDim.x) b |= active[p] && !isfinite(x[p]);
  b = __any_sync(0xffffffff, b);
  if (!(threadIdx.x & 31)) bad[threadIdx.x >> 5] = b;
  __syncthreads();
  if (threadIdx.x) return;
  for (int i = 1; i < (int)(blockDim.x >> 5); ++i) b |= bad[i];
  info[0] = 0;
  info[1] = 0;
  iterations[step] = ist[1];
  if (b) atomicCAS(error, 0, RESIDENT_CG);
}
