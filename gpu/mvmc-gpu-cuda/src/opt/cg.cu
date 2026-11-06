// Device-resident SR-CG (#670, #694, #572 milestone 4): C 1.4.0 StochasticOptCG_real
// (stcopt_cg_impl.c) without host round trips: the plain StochasticOptCG_Main
// and the DiagScale StochasticOptPreconCG_DiagScale_Main, the NSRCGFallback
// rerun and Rescale4SRCG_real (RescaleSmat). The host enqueues an upper bound
// of iterations per solve; every iteration kernel returns at once after the
// device-side stop flag is set, so the C stopping rule, iteration count and
// fallback decision are made on the device.
//
// The active parameters are a mask rather than C's compacted smatToParaIdx
// arrays: every masked operation adds exact zeros, so it equals the compacted
// arithmetic up to reduction order. Elementwise updates use explicit _rn
// intrinsics (no FMA), as the C/Rust CPU loops round each product separately.
//
// st:  [0] delta, [1] threshold, [2] alpha, [3] coef (<O>.x), [4] 1/Wc,
//      [5] rescaleOO_SlaterRatio, [6] rescaleO_SlaterRatio (1 when not rescaled)
// ist: [0] stop flag, [1] completed iterations, [2] max iterations, [3] nSmat,
//      [4] C 1.4.0 Main status of the running solve: iterations, or -(iter+1)
//          on breakdown or non-convergence (stcopt_cg_impl.c:405-469, 526-583),
//      [5] status of the first solve when the fallback ran, [6] fallback ran,
//      [7] RescaleSmat applied (Rescale4SRCG did not return early)

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

// Block maximum by fmax, which skips NaN operands.
__device__ double cg_block_fmax(double v, double* scratch) {
  for (int o = 16; o; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffff, v, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (!lane) scratch[warp] = v;
  __syncthreads();
  if (!threadIdx.x) {
    double m = scratch[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) m = fmax(m, scratch[i]);
    scratch[0] = m;
  }
  __syncthreads();
  const double m = scratch[0];
  __syncthreads();
  return m;
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
    double* oodiag, const double* O, const double* energy, int np, int batch) {
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
  oodiag[p] = oo / wc;
  ho[p] = eo / wc;
  // stcopt_cg_impl.c:129 S[pi][pi] = OOdiag[pi] - O[pi]*O[pi].
  variance[p] = __dsub_rn(oo / wc, __dmul_rn(m, m));
}

// C get_absmax (stcopt_cg.c:35-46) over p[start..end): 0 for an empty range;
// the maximum starts at |p[start]| and grows on `<`, so a NaN is skipped unless
// it is the first entry.
__device__ double cg_block_absmax(const double* p, int start, int end, double* scratch) {
  if (end <= start) return 0.0;
  double v = 0;
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) v = fmax(v, fabs(p[i]));
  v = cg_block_fmax(v, scratch);
  return isnan(p[start]) ? p[start] : v;
}

// Rescale4SRCG_real (stcopt_cg_impl.c:772-875) on the averaged operands, in
// place, before the diagonal cut. r = OOdiag - O^2 is `variance`, of all
// parameters (before OptFlag and the cut). C returns without rescaling when
// max|r| over the projections is below 1e-10; the static early returns (no
// projection, no Slater block, NProj >= NPara) are decided by the host. The
// Slater block [slt_start, slt_end) is Para + NProj + NRBM (NProjBF = 0): OO
// and the variance use rescaleOO, O and HO rescaleO, and the Slater parameters
// are divided by rescaleO. cg_rescale_samples scales the stored derivatives.
extern "C" __global__ void cg_rescale(double* mean, double* variance, double* ho,
    double* oodiag, double* params, double* st, int* ist, int nproj, int slt_start,
    int slt_end, const int* error) {
  __shared__ double scratch[32];
  __shared__ double ratio[2];
  __shared__ int apply;
  const double proj_max = cg_block_absmax(variance, 0, nproj, scratch);
  double slater_max = cg_block_absmax(variance, slt_start, slt_end, scratch);
  if (!threadIdx.x) {
    const int a = !*error && !(proj_max < 1e-10);
    if (slater_max < 1e-10) slater_max = 1e-10;
    const double oo = proj_max / slater_max;
    const double o = sqrt(oo);
    ratio[0] = oo;
    ratio[1] = o;
    apply = a;
    ist[7] = a;
    st[5] = a ? oo : 1.0;
    st[6] = a ? o : 1.0;
  }
  __syncthreads();
  if (!apply) return;
  const double oo = ratio[0], o = ratio[1];
  for (int p = slt_start + threadIdx.x; p < slt_end; p += blockDim.x) {
    const double od = __dmul_rn(oodiag[p], oo);
    const double m = __dmul_rn(mean[p], o);
    oodiag[p] = od;
    mean[p] = m;
    ho[p] = __dmul_rn(ho[p], o);
    // The diagonal of the rescaled S (stcopt_cg_impl.c:129, 747).
    variance[p] = __dsub_rn(od, __dmul_rn(m, m));
    params[p] = __ddiv_rn(params[p], o);
  }
}

// SROptO_Store *= rescaleO_SlaterRatio on the Slater block of every sample
// (stcopt_cg_impl.c:856-862); one block per walker.
extern "C" __global__ void cg_rescale_samples(double* O, const double* st, const int* ist,
    int np, int slt_start, int slt_end) {
  if (!ist[7]) return;
  const double o = st[6];
  double* row = O + (size_t)blockIdx.x * np;
  for (int p = slt_start + threadIdx.x; p < slt_end; p += blockDim.x)
    row[p] = __dmul_rn(row[p], o);
}


// x = 0 and the start of a solve (StochasticOptCG_Init, then the head of the
// Main). Plain CG (stcopt_cg_impl.c:519-526): d = r = g, delta = r.r and the
// initial stop test. DiagScale (:376-392) instead forms r = g - S x through the
// operator, so it leaves the stop flag at `stop` for that product and
// cg_diag_init completes the start. Inactive entries of g are exact zeros.
__device__ void cg_begin(double* x, double* vec, const int* active, double* st, int* ist,
    double* trace, int np, int diag, int stop, double* scratch) {
  const double* g = vec;
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  for (int p = threadIdx.x; p < np; p += blockDim.x) {
    x[p] = 0;
    d[p] = r[p] = g[p];
  }
  const double delta = diag ? 0.0 : cg_block_dot(r, r, active, np, scratch);
  if (!threadIdx.x) {
    ist[1] = 0;
    ist[4] = 0;
    if (!diag) {
      st[0] = delta;
      trace[0] = delta;
      // C returns -(0+1) for a nonfinite delta and stops at fabs(delta) < thresh.
      if (!stop && !isfinite(delta)) {
        stop = 1;
        ist[4] = -1;
      } else if (!stop && fabs(delta) < st[1]) {
        stop = 1;
      }
    }
    ist[0] = stop;
  }
}

// stcopt_cg_impl.c:127-176 (diagonal cut) and StochasticOptCG_Init (:681-751),
// then cg_begin for the selected variant (`diag`: useDiagScale).
extern "C" __global__ void cg_prepare(double* x, double* vec, int* active, double* st, int* ist,
    double* trace, double* history, const double* mean, const double* variance,
    const double* ho, const int* flags, int np, int batch, int step, double red_cut,
    double step_dt, double tolerance, int max_iter, const int* error, int diag) {
  __shared__ double scratch[32];
  __shared__ int count[32];
  double* g = vec;
  double* sdiag = vec + 4 * (size_t)np;
  double* mo = vec + 5 * (size_t)np;
  // C's maximum: sDiagMax starts at sDiagElm[0] and grows on `>`, so NaN
  // entries are skipped unless the first one is NaN.
  double v = -__longlong_as_double(0x7ff0000000000000LL);
  for (int p = threadIdx.x; p < np; p += blockDim.x) v = fmax(v, variance[p]);
  v = cg_block_fmax(v, scratch);
  if (isnan(variance[0])) v = variance[0];
  const double threshold = v * red_cut;
  const double dt = 2.0 * step_dt;
  const double energy = mean[np];
  int n = 0;
  for (int p = threadIdx.x; p < np; p += blockDim.x) {
    // C: fixed by OptFlag, else cut when sDiag < diagCutThreshold.
    const bool a = flags[p] && !(variance[p] < threshold);
    active[p] = a;
    n += a;
    // StochasticOptCG_Init: g = -dt*(HO - H*O), dt = 2*DSROptStepDt.
    g[p] = a ? __dmul_rn(-dt, __dsub_rn(ho[p], __dmul_rn(energy, mean[p]))) : 0.0;
    sdiag[p] = a ? variance[p] : 0.0;
    mo[p] = a ? mean[p] : 0.0;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
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
  if (!threadIdx.x) {
    // stcopt_cg_impl.c:368, 492, evaluated left to right.
    st[1] = tolerance * tolerance * (double)nsmat * (double)nsmat;
    st[4] = 1.0 / (double)batch;
    ist[2] = max_iter > 0 ? max_iter : nsmat;
    ist[3] = nsmat;
    ist[5] = 0;
    ist[6] = 0;
    history[step] = energy;
  }
  __syncthreads();
  // An empty active set is an exact no-op, as on the CPU (#682).
  cg_begin(x, vec, active, st, ist, trace, np, diag, *error || nsmat == 0, scratch);
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

// operate_by_S (stcopt_cg_impl.c:618-674): z = invW*(O y) - coef*<O>, then
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


// Record the C 1.4.0 breakdown status -(iter+1) of the running iteration and stop.
__device__ void cg_breakdown(int* ist) {
  if (!threadIdx.x) {
    ist[4] = -(ist[1] + 1);
    ist[0] = 1;
  }
}

// DiagScale preconditioner M_ii = (1 + DSROptStaDel) S_ii, clamped to
// +-precon_diag_eps = 1e-12 when nonfinite or smaller (stcopt_cg_impl.c:383-389,
// :450-456); `factor` is the host's 1.0 + DSROptStaDel (:362).
__device__ double cg_precon(double sdiag, double factor) {
  double dq = __dmul_rn(factor, sdiag);
  if (!isfinite(dq) || fabs(dq) < 1.0e-12) dq = dq < 0.0 ? -1.0e-12 : 1.0e-12;
  return dq;
}

// Completes the DiagScale start after r = S x (stcopt_cg_impl.c:376-395):
// r = g - r, d = M^-1 r, delta = r.d, then the tests at the head of the loop.
extern "C" __global__ void cg_diag_init(double* vec, const int* active, double* st, int* ist,
    double* trace, int np, double factor) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  const double* g = vec;
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  const double* sdiag = vec + 4 * (size_t)np;
  for (int p = threadIdx.x; p < np; p += blockDim.x) {
    if (active[p]) {
      r[p] = __dsub_rn(g[p], r[p]);
      d[p] = __ddiv_rn(r[p], cg_precon(sdiag[p], factor));
    } else {
      d[p] = 0;
    }
  }
  const double delta = cg_block_dot(r, d, active, np, scratch);
  if (!threadIdx.x) {
    st[0] = delta;
    trace[0] = delta;
    if (!isfinite(delta)) {
      ist[0] = 1;
      ist[4] = -1;
    } else if (fabs(delta) < st[1]) {
      ist[0] = 1;
    }
  }
}

// dq = d.q, alpha = delta/dq, x += alpha*d (stcopt_cg_impl.c:539-549, :408-418).
// C 1.4.0 stops before the update when dq is nonfinite or below 1e-20, or alpha
// is nonfinite. Returns false after such a breakdown; `alpha` is this thread's copy.
__device__ bool cg_alpha_update(double* x, const double* d, const double* q,
    const int* active, double* st, int* ist, int np, double* scratch, double* alpha_out) {
  const double dq = cg_block_dot(d, q, active, np, scratch);
  const double alpha = st[0] / dq;
  if (!isfinite(dq) || fabs(dq) < 1.0e-20 || !isfinite(alpha)) {
    cg_breakdown(ist);
    return false;
  }
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) x[p] = __dadd_rn(x[p], __dmul_rn(alpha, d[p]));
  if (!threadIdx.x) st[2] = alpha;
  *alpha_out = alpha;
  return true;
}

// End of a C loop iteration and the tests at the start of the next one
// (stcopt_cg_impl.c:565-583, :528-535). Plain: delta_new = r.r, beta =
// delta_new/delta, delta = beta*delta, d = r + beta*d. DiagScale (:449-469):
// q = M^-1 r, delta_new = r.q, delta = delta_new, d = q + beta*d.
__device__ void cg_direction(double* vec, const int* active, double* st, int* ist,
    double* trace, int np, int diag, double factor, double* scratch) {
  double* d = vec + np;
  const double* r = vec + 2 * (size_t)np;
  double* q = vec + 3 * (size_t)np;
  const double* sdiag = vec + 4 * (size_t)np;
  const double delta = st[0];
  if (diag)
    for (int p = threadIdx.x; p < np; p += blockDim.x)
      if (active[p]) q[p] = __ddiv_rn(r[p], cg_precon(sdiag[p], factor));
  const double* z = diag ? q : r;
  const double delta_new = cg_block_dot(r, z, active, np, scratch);
  const double beta = delta_new / delta;
  if (!isfinite(delta_new) || fabs(delta) < 1.0e-20 || !isfinite(beta)) {
    cg_breakdown(ist);
    return;
  }
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) d[p] = __dadd_rn(z[p], __dmul_rn(beta, d[p]));
  if (!threadIdx.x) {
    const double next = diag ? delta_new : beta * delta;
    const int iter = ist[1] + 1;
    st[0] = next;
    ist[1] = iter;
    trace[iter] = next;
    if (iter >= ist[2]) {
      // :583 and :469: the limit is reached; fabs(NaN) >= thresh is false, as in C.
      ist[0] = 1;
      ist[4] = fabs(next) >= st[1] ? -(ist[2] + 1) : iter;
    } else if (!isfinite(next)) {
      ist[0] = 1;
      ist[4] = -(iter + 1);
    } else if (fabs(next) < st[1]) {
      ist[0] = 1;
      ist[4] = iter;
    }
  }
}

// One ordinary iteration after q = S d: r -= alpha*q.
extern "C" __global__ void cg_iterate(double* x, double* vec, const int* active, double* st,
    int* ist, double* trace, int np, int diag, double factor) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  double* d = vec + np;
  double* r = vec + 2 * (size_t)np;
  const double* q = vec + 3 * (size_t)np;
  double alpha;
  if (!cg_alpha_update(x, d, q, active, st, ist, np, scratch, &alpha)) return;
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) r[p] = __dsub_rn(r[p], __dmul_rn(alpha, q[p]));
  cg_direction(vec, active, st, ist, trace, np, diag, factor, scratch);
}

// Every 20th iteration: alpha and x first, then r = S x is formed by
// cg_ty/cg_oy, and cg_refresh replaces it by g - S x.
extern "C" __global__ void cg_alpha(double* x, double* vec, const int* active, double* st,
    int* ist, int np) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  double alpha;
  cg_alpha_update(x, vec + np, vec + 3 * (size_t)np, active, st, ist, np, scratch, &alpha);
}

extern "C" __global__ void cg_refresh(double* vec, const int* active, double* st, int* ist,
    double* trace, int np, int diag, double factor) {
  if (ist[0]) return;
  __shared__ double scratch[32];
  const double* g = vec;
  double* r = vec + 2 * (size_t)np;
  for (int p = threadIdx.x; p < np; p += blockDim.x)
    if (active[p]) r[p] = __dsub_rn(g[p], r[p]);
  cg_direction(vec, active, st, ist, trace, np, diag, factor, scratch);
}

// NSRCGFallback (stcopt_cg_impl.c:225-251): after a failed first solve (status
// < 0), C reinitializes VecCG (x = 0) and runs the other variant (`diag`). The
// first status is kept for zqp_SRinfo.dat. Otherwise the second solve's
// enqueued kernels return at once.
extern "C" __global__ void cg_fallback(double* x, double* vec, const int* active, double* st,
    int* ist, double* trace, int np, int diag, const int* error) {
  __shared__ double scratch[32];
  __shared__ int run;
  if (!threadIdx.x) {
    run = !*error && ist[4] < 0;
    if (run) {
      ist[5] = ist[4];
      ist[6] = 1;
    } else {
      ist[0] = 1;
    }
  }
  __syncthreads();
  if (run) cg_begin(x, vec, active, st, ist, trace, np, diag, 0, scratch);
}

// stcopt_cg_impl.c:203-310: a failed solve aborts with NSRCGAbortOnFail (C 1.4.0
// MPI_Abort): without the fallback when the solve failed, with it when the
// fallback solve failed too. Otherwise C continues with the approximate
// solution; a nonfinite increment then returns info=1 and C stops
// (vmcmain.c:503-506). Nothing is published; opt_update observes the status.
// `iterations[step]` is the zqp_SRinfo.dat info: the first solve's status
// (C's -(fallback_iter+1) after a fallback). `fallback[step]` is the fallback
// solve's status, or INT_MIN when it did not run.
extern "C" __global__ void cg_finish(const double* x, const int* active, int* info,
    const int* ist, int* iterations, int* fallback, int step, int np, int* error,
    int abort_on_fail) {
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
  const int ran = ist[6];
  iterations[step] = ran ? ist[5] : ist[4];
  fallback[step] = ran ? ist[4] : (-2147483647 - 1);
  if (ist[4] < 0 && abort_on_fail) {
    atomicCAS(error, 0, RESIDENT_CG_FAILED);
  } else if (b) {
    atomicCAS(error, 0, RESIDENT_CG);
  }
}
