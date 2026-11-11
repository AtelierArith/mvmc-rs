// Complex FP64 helpers of the resident complex normal-orbital path (#678).
//
// `rcx` is an interleaved (re, im) pair, the layout of `Cx<double>` in
// pfaffian_batched.cu and of num_complex::Complex64 on the host. Every
// product and sum is separately rounded (no FMA contraction), so a result
// depends only on the operation order written here. Division follows Smith's
// algorithm, as `Cx` does.
//
// The block helpers mirror resident_dense.cuh (#573) with complex arithmetic:
// gather, PFAPACK pfaffian_LTL with partial pivoting (pivot magnitude
// |re|+|im|, the izamax norm), Gauss-Jordan inverse, skew publication and the
// per-walker refactor. Matrices are column-major, A(r,c) = A[r + c*n].
#ifndef MVMC_RESIDENT_COMPLEX
#define MVMC_RESIDENT_COMPLEX
#ifndef SLT
#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])
#endif

struct rcx {
  double re, im;
};

__device__ __forceinline__ rcx rc(double re, double im) {
  rcx z;
  z.re = re;
  z.im = im;
  return z;
}
__device__ __forceinline__ rcx rc_add(rcx a, rcx b) {
  return rc(__dadd_rn(a.re, b.re), __dadd_rn(a.im, b.im));
}
__device__ __forceinline__ rcx rc_sub(rcx a, rcx b) {
  return rc(__dsub_rn(a.re, b.re), __dsub_rn(a.im, b.im));
}
__device__ __forceinline__ rcx rc_neg(rcx a) { return rc(-a.re, -a.im); }
__device__ __forceinline__ rcx rc_conj(rcx a) { return rc(a.re, -a.im); }
__device__ __forceinline__ rcx rc_mul(rcx a, rcx b) {
  return rc(__dsub_rn(__dmul_rn(a.re, b.re), __dmul_rn(a.im, b.im)),
            __dadd_rn(__dmul_rn(a.re, b.im), __dmul_rn(a.im, b.re)));
}
__device__ __forceinline__ rcx rc_scale(rcx a, double s) {
  return rc(__dmul_rn(a.re, s), __dmul_rn(a.im, s));
}
__device__ inline rcx rc_div(rcx a, rcx b) {
  if (fabs(b.re) >= fabs(b.im)) {
    const double r = __ddiv_rn(b.im, b.re), d = __dadd_rn(b.re, __dmul_rn(b.im, r));
    return rc(__ddiv_rn(__dadd_rn(a.re, __dmul_rn(a.im, r)), d),
              __ddiv_rn(__dsub_rn(a.im, __dmul_rn(a.re, r)), d));
  }
  const double r = __ddiv_rn(b.re, b.im), d = __dadd_rn(__dmul_rn(b.re, r), b.im);
  return rc(__ddiv_rn(__dadd_rn(__dmul_rn(a.re, r), a.im), d),
            __ddiv_rn(__dsub_rn(__dmul_rn(a.im, r), a.re), d));
}
__device__ __forceinline__ double rc_abs1(rcx a) { return fabs(a.re) + fabs(a.im); }
__device__ __forceinline__ double rc_abs(rcx a) { return hypot(a.re, a.im); }
__device__ __forceinline__ bool rc_finite(rcx a) { return isfinite(a.re) && isfinite(a.im); }
__device__ __forceinline__ bool rc_zero(rcx a) { return a.re == 0.0 && a.im == 0.0; }

// Fixed-order warp reduction of both components; every lane receives the sum.
__device__ inline rcx rc_warp_sum(rcx x) {
  for (int d = 16; d; d >>= 1) {
    x.re += __shfl_down_sync(0xffffffff, x.re, d);
    x.im += __shfl_down_sync(0xffffffff, x.im, d);
  }
  return rc(__shfl_sync(0xffffffff, x.re, 0), __shfl_sync(0xffffffff, x.im, 0));
}

// Gather the skew plane of global sites rs[0..n), omitting electron slots
// `skip_a` and `skip_b` (pass -1 to keep all). Returns the gathered side.
__device__ inline int rc_gather(rcx* A, const int* rs, int n, int skip_a, int skip_b,
                                const rcx* slater, int q, int n2) {
  const int m = n - (skip_a >= 0) - (skip_b >= 0);
  for (int index = threadIdx.x; index < m * m; index += blockDim.x) {
    int r = index % m, c = index / m;
    r += (skip_a >= 0 && r >= skip_a);
    r += (skip_b >= 0 && r >= skip_b);
    c += (skip_a >= 0 && c >= skip_a);
    c += (skip_b >= 0 && c >= skip_b);
    A[index] = rc_neg(SLT(slater, q, rs[c], rs[r], n2));
  }
  __syncthreads();
  return m;
}

// First index of the largest |re|+|im| over begin <= i < end (ties keep the
// smallest index). Returns -1 when every entry is zero and -2 when any entry
// is nonfinite.
__device__ int rc_block_argmax(const rcx* x, int begin, int end) {
  __shared__ double s_value[32];
  __shared__ int s_index[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  double best = -1.0;
  int index = end, nonfinite = 0;
  for (int i = begin + tid; i < end; i += blockDim.x) {
    const double v = rc_abs1(x[i]);
    nonfinite |= !isfinite(v);
    if (v > best) {
      best = v;
      index = i;
    }
  }
  for (int d = 16; d; d >>= 1) {
    const double other = __shfl_down_sync(0xffffffff, best, d);
    const int at = __shfl_down_sync(0xffffffff, index, d);
    if (other > best || (other == best && at < index)) {
      best = other;
      index = at;
    }
  }
  if (lane == 0) {
    s_value[warp] = best;
    s_index[warp] = index;
  }
  nonfinite = __syncthreads_or(nonfinite);
  if (warp == 0) {
    best = lane < warps ? s_value[lane] : -1.0;
    index = lane < warps ? s_index[lane] : end;
    for (int d = 16; d; d >>= 1) {
      const double other = __shfl_down_sync(0xffffffff, best, d);
      const int at = __shfl_down_sync(0xffffffff, index, d);
      if (other > best || (other == best && at < index)) {
        best = other;
        index = at;
      }
    }
    if (lane == 0) s_index[0] = nonfinite ? -2 : (best > 0.0 ? index : -1);
  }
  __syncthreads();
  const int result = s_index[0];
  __syncthreads();
  return result;
}

// Complex Pfaffian by skew Parlett-Reid elimination with partial pivoting,
// resident_block_pfaffian with complex arithmetic. An exactly zero pivot
// column gives Pf = 0; a nonfinite column gives NaN. Destroys A; tau and u
// hold n values each. Every thread of the block receives the same value.
__device__ rcx rc_block_pfaffian_unblocked(rcx* A, int n, rcx* tau, rcx* u) {
  __shared__ rcx s_pf;
  const int tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) s_pf = rc(1.0, 0.0);
  __syncthreads();
  for (int k = 0; k + 1 < n; k += 2) {
    const int kp = rc_block_argmax(A + (size_t)k * n, k + 1, n);
    if (kp < 0) {
      if (tid == 0) {
        const double nan = __longlong_as_double(0x7ff8000000000000LL);
        s_pf = kp == -1 ? rc(0.0, 0.0) : rc(nan, nan);
      }
      __syncthreads();
      break;
    }
    if (kp != k + 1) {
      for (int c = tid; c < n; c += nt) {
        const rcx x = A[(k + 1) + (size_t)c * n];
        A[(k + 1) + (size_t)c * n] = A[kp + (size_t)c * n];
        A[kp + (size_t)c * n] = x;
      }
      __syncthreads();
      for (int r = tid; r < n; r += nt) {
        const rcx x = A[r + (size_t)(k + 1) * n];
        A[r + (size_t)(k + 1) * n] = A[r + (size_t)kp * n];
        A[r + (size_t)kp * n] = x;
      }
      if (tid == 0) s_pf = rc_neg(s_pf);
      __syncthreads();
    }
    const rcx pivot = A[k + (size_t)(k + 1) * n];
    if (tid == 0) s_pf = rc_mul(s_pf, pivot);
    for (int c = k + 2 + tid; c < n; c += nt) {
      tau[c] = rc_div(A[k + (size_t)c * n], pivot);
      u[c] = A[c + (size_t)(k + 1) * n];
    }
    __syncthreads();
    const int width = n - k - 2;
    for (int index = tid; index < width * width; index += nt) {
      const int r = k + 2 + index % width, c = k + 2 + index / width;
      const rcx delta = rc_sub(rc_mul(tau[r], u[c]), rc_mul(u[r], tau[c]));
      A[r + (size_t)c * n] = rc_add(A[r + (size_t)c * n], delta);
    }
    __syncthreads();
  }
  const rcx result = s_pf;
  __syncthreads();
  return result;
}

struct ResidentComplexPfOps {
  typedef rcx T;
  __device__ static T one() { return rc(1.0, 0.0); }
  __device__ static T zero() { return rc(0.0, 0.0); }
  __device__ static T nan() { const double x = __longlong_as_double(0x7ff8000000000000LL); return rc(x,x); }
  __device__ static T neg(T x) { return rc_neg(x); }
  __device__ static T add(T x, T y) { return rc_add(x,y); }
  __device__ static T sub(T x, T y) { return rc_sub(x,y); }
  __device__ static T mul(T x, T y) { return rc_mul(x,y); }
  __device__ static T div(T x, T y) { return rc_div(x,y); }
  __device__ static int argmax(const T* x, int begin, int end) {
    return rc_block_argmax(x, begin, end);
  }
};

__device__ rcx rc_block_pfaffian(rcx* A, int n, rcx* tau, rcx* u) {
  if (n < 64) return rc_block_pfaffian_unblocked(A, n, tau, u);
  return resident_panel_pfaffian<ResidentComplexPfOps>(A, n, tau, u);
}

// Gauss-Jordan inverse with partial pivoting: Y = A^-1 (column-major).
// Destroys A; col holds n values. Returns 0 on a zero or nonfinite pivot column.
__device__ int rc_block_inverse(rcx* A, rcx* Y, int n, rcx* col) {
  __shared__ rcx s_scale;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int index = tid; index < n * n; index += nt)
    Y[index] = rc(index % n == index / n ? 1.0 : 0.0, 0.0);
  __syncthreads();
  int ok = 1;
  for (int k = 0; k < n; ++k) {
    const int p = rc_block_argmax(A + (size_t)k * n, k, n);
    if (p < 0) {
      ok = 0;
      break;
    }
    if (p != k) {
      for (int c = tid; c < n; c += nt) {
        rcx x = A[k + (size_t)c * n];
        A[k + (size_t)c * n] = A[p + (size_t)c * n];
        A[p + (size_t)c * n] = x;
        x = Y[k + (size_t)c * n];
        Y[k + (size_t)c * n] = Y[p + (size_t)c * n];
        Y[p + (size_t)c * n] = x;
      }
      __syncthreads();
    }
    if (tid == 0) s_scale = rc_div(rc(1.0, 0.0), A[k + (size_t)k * n]);
    __syncthreads();
    const rcx scale = s_scale;
    for (int c = tid; c < n; c += nt) {
      A[k + (size_t)c * n] = rc_mul(A[k + (size_t)c * n], scale);
      Y[k + (size_t)c * n] = rc_mul(Y[k + (size_t)c * n], scale);
    }
    __syncthreads();
    for (int r = tid; r < n; r += nt) col[r] = A[r + (size_t)k * n];
    __syncthreads();
    for (int index = tid; index < n * n; index += nt) {
      const int r = index % n, c = index / n;
      if (r == k) continue;
      const rcx x = col[r];
      if (c >= k) A[index] = rc_sub(A[index], rc_mul(x, A[k + (size_t)c * n]));
      Y[index] = rc_sub(Y[index], rc_mul(x, Y[k + (size_t)c * n]));
    }
    __syncthreads();
  }
  return ok;
}

// Publish Y = A^-1 as the mVMC inverse invM = -A^-1, mirroring one rounded
// skew pair (resident_publish_inverse). Returns 0 for a nonfinite entry.
__device__ int rc_publish_inverse(rcx* inv, const rcx* Y, int n) {
  __shared__ int s_bad;
  if (threadIdx.x == 0) s_bad = 0;
  __syncthreads();
  for (int index = threadIdx.x; index < n * n; index += blockDim.x) {
    const int r = index % n, c = index / n;
    const rcx x = r == c ? rc(0.0, 0.0)
                         : rc_scale(rc_sub(Y[r + (size_t)c * n], Y[c + (size_t)r * n]), 0.5);
    if (!rc_finite(x)) s_bad = 1;
    inv[index] = rc_neg(x);
  }
  __syncthreads();
  const int ok = !s_bad;
  __syncthreads();
  return ok;
}

// Refactor every plane of one walker on its current sites
// (resident_refactor_walker). Returns whether every plane is valid.
__device__ int rc_refactor_walker(rcx* inv, rcx* pf, int* valid, const int* rs, const rcx* slater,
                                  int n, int nqp, int n2, rcx* fac, rcx* work, rcx* vec) {
  __shared__ int s_all;
  if (threadIdx.x == 0) s_all = 1;
  for (int q = 0; q < nqp; ++q) {
    rc_gather(fac, rs, n, -1, -1, slater, q, n2);
    const rcx p = rc_block_pfaffian(fac, n, vec, vec + n);
    int ok = rc_finite(p) && !rc_zero(p);
    if (ok) {
      rc_gather(fac, rs, n, -1, -1, slater, q, n2);
      ok = rc_block_inverse(fac, work, n, vec);
    }
    if (ok) ok = rc_publish_inverse(inv + (size_t)q * n * n, work, n);
    if (threadIdx.x == 0) {
      pf[q] = p;
      valid[q] = ok;
      if (!ok) s_all = 0;
    }
    __syncthreads();
  }
  const int all = s_all;
  __syncthreads();
  return all;
}
#endif
