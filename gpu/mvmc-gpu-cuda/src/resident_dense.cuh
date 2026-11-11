// Block-cooperative dense helpers for singular QP planes (#573).
//
// Julia GPU recovers per plane (Julia-mVMC 7314015): walkers with an invalid
// plane use full Pfaffians for proposals and observables
// (cuda_cooperative_sampling.jl:3-22, cuda_observables.jl:283-295), rebuild
// in the hop kernel (cuda_observables.jl:355-450), and use exact (n-2)-minor
// Pfaffians for derivatives (cuda_sr.jl:154-187). These helpers run on one
// block, keep matrices in per-walker global scratch and need two n-vectors of
// shared or global scratch. They serve only walkers with an invalid plane;
// the all-valid path keeps its inverse-update arithmetic.
//
// Matrices are column-major, A(r,c) = A[r + c*n], as gathered by
// interval_resident_factor: A[i*n + j] = -S[q][rs_i][rs_j].
#ifndef MVMC_RESIDENT_DENSE
#define MVMC_RESIDENT_DENSE
#ifndef SLT
#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])
#endif

// Gather the skew plane of global sites rs[0..n), omitting electron slots
// `skip_a` and `skip_b` (pass -1 to keep all). Returns the gathered side.
__device__ inline int resident_gather(double* A, const int* rs, int n, int skip_a, int skip_b,
                                      const double* slater, int q, int n2) {
  const int m = n - (skip_a >= 0) - (skip_b >= 0);
  for (int index = threadIdx.x; index < m * m; index += blockDim.x) {
    int r = index % m, c = index / m;
    // Map the minor index back to the electron slot, skipping removed slots.
    r += (skip_a >= 0 && r >= skip_a);
    r += (skip_b >= 0 && r >= skip_b);
    c += (skip_a >= 0 && c >= skip_a);
    c += (skip_b >= 0 && c >= skip_b);
    A[index] = -SLT(slater, q, rs[c], rs[r], n2);
  }
  __syncthreads();
  return m;
}

// First index of the largest |x[i]| over begin <= i < end, reduced by the
// whole block (blockDim a multiple of 32). Ties keep the smallest index, as a
// serial scan does. Returns -1 when every entry is zero and -2 when any entry
// is nonfinite, so a NaN can never be mistaken for a singular pivot.
__device__ int resident_block_argmax(const double* x, int begin, int end) {
  __shared__ double s_value[32];
  __shared__ int s_index[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  double best = -1.0;
  int index = end, nonfinite = 0;
  for (int i = begin + tid; i < end; i += blockDim.x) {
    const double v = fabs(x[i]);
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

// Pfaffian by skew Parlett-Reid elimination with row/column partial pivoting
// (PFAPACK pfaffian_LTL). An exactly zero pivot column gives Pf = 0; a
// nonfinite column gives NaN. The rank-two update uses separately rounded
// products, so A remains exactly skew-symmetric. Destroys A; tau and u hold n
// doubles each. Every thread of the block must call it and receives the same
// value.
__device__ double resident_block_pfaffian_unblocked(double* A, int n, double* tau, double* u) {
  __shared__ double s_pf;
  const int tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) s_pf = 1.0;
  __syncthreads();
  for (int k = 0; k + 1 < n; k += 2) {
    const int kp = resident_block_argmax(A + (size_t)k * n, k + 1, n);
    if (kp < 0) {
      if (tid == 0) s_pf = kp == -1 ? 0.0 : __longlong_as_double(0x7ff8000000000000LL);
      __syncthreads();
      break;
    }
    if (kp != k + 1) {
      for (int c = tid; c < n; c += nt) {
        const double x = A[(k + 1) + (size_t)c * n];
        A[(k + 1) + (size_t)c * n] = A[kp + (size_t)c * n];
        A[kp + (size_t)c * n] = x;
      }
      __syncthreads();
      for (int r = tid; r < n; r += nt) {
        const double x = A[r + (size_t)(k + 1) * n];
        A[r + (size_t)(k + 1) * n] = A[r + (size_t)kp * n];
        A[r + (size_t)kp * n] = x;
      }
      if (tid == 0) s_pf = -s_pf;
      __syncthreads();
    }
    const double pivot = A[k + (size_t)(k + 1) * n];
    if (tid == 0) s_pf = __dmul_rn(s_pf, pivot);
    for (int c = k + 2 + tid; c < n; c += nt) {
      tau[c] = __ddiv_rn(A[k + (size_t)c * n], pivot);
      u[c] = A[c + (size_t)(k + 1) * n];
    }
    __syncthreads();
    const int width = n - k - 2;
    for (int index = tid; index < width * width; index += nt) {
      const int r = k + 2 + index % width, c = k + 2 + index / width;
      const double delta = __dsub_rn(__dmul_rn(tau[r], u[c]), __dmul_rn(u[r], tau[c]));
      A[r + (size_t)c * n] = __dadd_rn(A[r + (size_t)c * n], delta);
    }
    __syncthreads();
  }
  const double result = s_pf;
  __syncthreads();
  return result;
}

struct ResidentRealPfOps {
  typedef double T;
  __device__ static T one() { return 1.0; }
  __device__ static T zero() { return 0.0; }
  __device__ static T nan() { return __longlong_as_double(0x7ff8000000000000LL); }
  __device__ static T neg(T x) { return -x; }
  __device__ static T add(T x, T y) { return __dadd_rn(x, y); }
  __device__ static T sub(T x, T y) { return __dsub_rn(x, y); }
  __device__ static T mul(T x, T y) { return __dmul_rn(x, y); }
  __device__ static T div(T x, T y) { return __ddiv_rn(x, y); }
  __device__ static int argmax(const T* x, int begin, int end) {
    return resident_block_argmax(x, begin, end);
  }
};

__device__ double resident_block_pfaffian(double* A, int n, double* tau, double* u) {
  // Structured proposal/observable matrices favor this loop: using the panel
  // here regressed L128 Opt by 7%, despite its dense standalone benchmark gain.
  // Shared-factor full rebuilds select panels separately in pfaffian_batched.cu.
  return resident_block_pfaffian_unblocked(A, n, tau, u);
}

// Gauss-Jordan inverse with partial pivoting: Y = A^-1 (column-major).
// Destroys A; col holds n doubles. Returns 0 on a zero or nonfinite pivot column.
__device__ int resident_block_inverse(double* A, double* Y, int n, double* col) {
  __shared__ double s_scale;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int index = tid; index < n * n; index += nt) Y[index] = index % n == index / n;
  __syncthreads();
  int ok = 1;
  for (int k = 0; k < n; ++k) {
    const int p = resident_block_argmax(A + (size_t)k * n, k, n);
    if (p < 0) {
      ok = 0;
      break;
    }
    if (p != k) {
      for (int c = tid; c < n; c += nt) {
        double x = A[k + (size_t)c * n];
        A[k + (size_t)c * n] = A[p + (size_t)c * n];
        A[p + (size_t)c * n] = x;
        x = Y[k + (size_t)c * n];
        Y[k + (size_t)c * n] = Y[p + (size_t)c * n];
        Y[p + (size_t)c * n] = x;
      }
      __syncthreads();
    }
    if (tid == 0) s_scale = 1.0 / A[k + (size_t)k * n];
    __syncthreads();
    const double scale = s_scale;
    for (int c = tid; c < n; c += nt) {
      A[k + (size_t)c * n] *= scale;
      Y[k + (size_t)c * n] *= scale;
    }
    __syncthreads();
    for (int r = tid; r < n; r += nt) col[r] = A[r + (size_t)k * n];
    __syncthreads();
    for (int index = tid; index < n * n; index += nt) {
      const int r = index % n, c = index / n;
      if (r == k) continue;
      const double x = col[r];
      if (c >= k) A[index] -= x * A[k + (size_t)c * n];
      Y[index] -= x * Y[k + (size_t)c * n];
    }
    __syncthreads();
  }
  return ok;
}

// Publish Y = A^-1 as the mVMC inverse invM = -A^-1 (the interval_resident_commit
// sign), mirroring one rounded skew pair as Julia does after its Gauss-Jordan
// rebuild (cuda_sampling_update.jl:38-60). Returns 0 for a nonfinite entry.
__device__ int resident_publish_inverse(double* inv, const double* Y, int n) {
  __shared__ int s_bad;
  if (threadIdx.x == 0) s_bad = 0;
  __syncthreads();
  for (int index = threadIdx.x; index < n * n; index += blockDim.x) {
    const int r = index % n, c = index / n;
    const double x = r == c ? 0.0 : 0.5 * (Y[r + (size_t)c * n] - Y[c + (size_t)r * n]);
    if (!isfinite(x)) s_bad = 1;
    inv[index] = -x;
  }
  __syncthreads();
  const int ok = !s_bad;
  __syncthreads();
  return ok;
}

// Refactor every plane of one walker on its current sites. Valid planes get
// a published inverse; an invalid plane (zero/nonfinite Pfaffian or a zero
// Gauss-Jordan pivot) keeps its Pfaffian and stale inverse storage, which no
// all-valid path reads. Returns whether every plane is valid.
__device__ int resident_refactor_walker(double* inv, double* pf, int* valid, const int* rs,
                                        const double* slater, int n, int nqp, int n2,
                                        double* fac, double* work, double* vec) {
  __shared__ int s_all;
  if (threadIdx.x == 0) s_all = 1;
  for (int q = 0; q < nqp; ++q) {
    resident_gather(fac, rs, n, -1, -1, slater, q, n2);
    const double p = resident_block_pfaffian(fac, n, vec, vec + n);
    int ok = isfinite(p) && p != 0.0;
    if (ok) {
      resident_gather(fac, rs, n, -1, -1, slater, q, n2);
      ok = resident_block_inverse(fac, work, n, vec);
    }
    if (ok) ok = resident_publish_inverse(inv + (size_t)q * n * n, work, n);
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
