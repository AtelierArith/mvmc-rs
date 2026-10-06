// Kernels of the device-resident SR pipeline (issue #447). Compile with --fmad=false: the
// elementwise formulas keep the rounding of crates/mvmc-core/src/sr_backend.rs
// (`assemble_s_g`) and sr_cg.rs (`apply_with_backend`).

// G holds the upper triangle of the symmetric Gram matrix O O^T (cuBLAS dsyrk, 'U'), column
// major with leading dimension ld; element (a, b) is read at min(a,b) + max(a,b)*ld.
__device__ inline double sym(const double* g, long ld, long a, long b) {
  return a <= b ? g[a + b * ld] : g[b + a * ld];
}

// S[si + sj*nmap] = OO(pj+off, pi+off) - OO(pi+off, 0) * OO(pj+off, 0), diagonal scaled by
// ratio_diag = 1 + DSROptStaDel (the C stcopt.c:69 formula; `oo.at(flat)` of the CPU path).
extern "C" __global__ void k_assemble_s(const double* g, long ld, const long* map, long nmap,
                                        long off, double ratio_diag, double* s) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nmap * nmap) return;
  const long si = idx % nmap, sj = idx / nmap;
  const long pi = map[si], pj = map[sj];
  const double tmp = sym(g, ld, pi + off, 0);
  double v = sym(g, ld, pj + off, pi + off) - tmp * sym(g, ld, pj + off, 0);
  if (si == sj) v *= ratio_diag;
  s[idx] = v;
}

// g[si] = -2 dt (HO(pi+off) - HO(0) OO(pi+off, 0))
extern "C" __global__ void k_assemble_g(const double* ho, const double* g, long ld,
                                        const long* map, long nmap, long off, double step_dt,
                                        double* out) {
  const long si = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (si >= nmap) return;
  const long pi = map[si];
  const double v = ho[pi + off] - ho[0] * sym(g, ld, pi + off, 0);
  out[si] = -2.0 * step_dt * v;
}

// z = inv_weight * y - coef * mean + shift * diag * x      (SampledSrOperator::apply)
extern "C" __global__ void k_cg_combine(const double* y, const double* x, const double* mean,
                                        const double* diag, double inv_weight, double coef,
                                        double shift, long n, double* z) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  z[i] = inv_weight * y[i] - coef * mean[i] + shift * diag[i] * x[i];
}

// 1 if any element is non-finite
extern "C" __global__ void k_nonfinite(const double* x, long n, int* flag) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !(x[i] - x[i] == 0.0)) atomicExch(flag, 1);
}
