// Batched Pfaffian + inverse of skew-symmetric matrices, one thread block per plane.
//
// Same math as crates/pfapack: dsktf2/zsktf2 (Parlett-Reid style skew LTL^T with partial
// pivoting, upper storage), utu2pfa, utu2inv. Planes are column-major n x n, stored back to
// back (plane p at offset p*n*n). The kernel works in global memory (the planes of n = 2*Ne up
// to 128 do not fit shared memory for complex data) with small per-plane vectors in dynamic
// shared memory. Compile with --fmad=false so the rank-2 update keeps pfapack's operation
// order, (c + x t1) - y t2.
//
// Status codes (int per plane): 0 ok, k > 0 zero pivot (pfapack INFO, 1-based row),
// -1 non-finite Pfaffian. For status != 0 the output plane is zero; a zero-pivot plane
// reports pf = 0.

template <typename R> struct Cx { R re, im; };

template <typename R> __device__ inline Cx<R> operator+(Cx<R> a, Cx<R> b) { return {a.re + b.re, a.im + b.im}; }
template <typename R> __device__ inline Cx<R> operator-(Cx<R> a, Cx<R> b) { return {a.re - b.re, a.im - b.im}; }
template <typename R> __device__ inline Cx<R> operator-(Cx<R> a) { return {-a.re, -a.im}; }
template <typename R> __device__ inline Cx<R> operator*(Cx<R> a, Cx<R> b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
// Smith's algorithm.
template <typename R> __device__ inline Cx<R> operator/(Cx<R> a, Cx<R> b) {
  if (fabs(b.re) >= fabs(b.im)) {
    R r = b.im / b.re, d = b.re + b.im * r;
    return {(a.re + a.im * r) / d, (a.im - a.re * r) / d};
  } else {
    R r = b.re / b.im, d = b.re * r + b.im;
    return {(a.re * r + a.im) / d, (a.im * r - a.re) / d};
  }
}

template <typename T> struct Tr {  // real scalar
  typedef T R;
  __device__ static T make(R re, R) { return re; }
  __device__ static R mag(T x) { return fabs(x); }
  __device__ static bool finite(T x) { return x - x == R(0); }
};
template <typename Rr> struct Tr<Cx<Rr>> {  // complex scalar
  typedef Rr R;
  __device__ static Cx<Rr> make(R re, R im) { return {re, im}; }
  __device__ static R mag(Cx<Rr> x) { return fabs(x.re) + fabs(x.im); }  // izamax norm
  __device__ static bool finite(Cx<Rr> x) { return (x.re - x.re == Rr(0)) && (x.im - x.im == Rr(0)); }
};

template <typename T>
__device__ void pfinv_plane(T* A, T* W, T* OUT, T* PF, int* ST, int n) {
  typedef Tr<T> TR;
  typedef typename TR::R R;
  extern __shared__ char smem_raw[];
  T* vt = reinterpret_cast<T*>(smem_raw);                 // n
  R* red_v = reinterpret_cast<R*>(vt + n);                // blockDim
  int* red_i = reinterpret_cast<int*>(red_v + blockDim.x);  // blockDim
  int* idx = red_i + blockDim.x;                          // n
  int* piv = idx + n;                                     // n
  __shared__ int s_kp, s_info, s_zero, s_status;
  __shared__ R s_colmax;

  const int tid = threadIdx.x, nt = blockDim.x;
  const T one = TR::make(R(1), R(0));
  const T zero = TR::make(R(0), R(0));
  const size_t nn = (size_t)n * n;
  for (int i = tid; i < n; i += nt) piv[i] = i;
  if (tid == 0) s_info = 0;
  __syncthreads();

  // ---- Phase 1: LTL^T factorization, k0 = n-1 .. 1 (0-based), kk = k0-1 -------------------
  for (int k0 = n - 1; k0 >= 1; --k0) {
    const int kk = k0 - 1;
    // First-index argmax of mag(A[0..kk, k0]); NaN loses except at index 0 (as the scalar
    // `v > colmax` scan does).
    R bv = R(-1);
    int bi = 0x7fffffff;
    for (int j = tid; j <= kk; j += nt) {
      R v = TR::mag(A[j + (size_t)k0 * n]);
      if (v != v) v = (j == 0) ? R(1e300) * R(1e300) : R(-1);
      if (v > bv || (v == bv && j < bi)) { bv = v; bi = j; }
    }
    red_v[tid] = bv;
    red_i[tid] = bi;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        R ov = red_v[tid + s];
        int oi = red_i[tid + s];
        if (ov > red_v[tid] || (ov == red_v[tid] && oi < red_i[tid])) { red_v[tid] = ov; red_i[tid] = oi; }
      }
      __syncthreads();
    }
    if (tid == 0) {
      int kp = red_i[0];
      if (kp > kk) kp = kk;
      s_kp = kp;
      s_colmax = TR::mag(A[kp + (size_t)k0 * n]);
      s_zero = (s_colmax == R(0)) ? 1 : 0;
      if (s_zero) {
        if (s_info == 0) s_info = kk + 1;
      } else {
        piv[kk] = kp;
      }
    }
    __syncthreads();
    if (s_zero) continue;
    const int kp = s_kp;

    if (kp != kk) {
      for (int j = tid; j < kp; j += nt) {
        T t = A[j + (size_t)kk * n]; A[j + (size_t)kk * n] = A[j + (size_t)kp * n]; A[j + (size_t)kp * n] = t;
      }
      for (int j = kp + 1 + tid; j < kk; j += nt) {
        T t = A[j + (size_t)kk * n]; A[j + (size_t)kk * n] = A[kp + (size_t)j * n]; A[kp + (size_t)j * n] = t;
      }
      for (int j = k0 + tid; j < n; j += nt) {
        T t = A[kk + (size_t)j * n]; A[kk + (size_t)j * n] = A[kp + (size_t)j * n]; A[kp + (size_t)j * n] = t;
      }
      __syncthreads();
      for (int j = kp + tid; j < kk; j += nt) A[j + (size_t)kk * n] = -A[j + (size_t)kk * n];
      for (int j = kp + 1 + tid; j < kk; j += nt) A[kp + (size_t)j * n] = -A[kp + (size_t)j * n];
      __syncthreads();
    }

    if (kk >= 1) {
      const T alpha = one / A[kk + (size_t)k0 * n];
      // Upper rank-2 update of A[0..kk, 0..kk]: A_ij = (A_ij + x_i t1_j) - y_i t2_j, i < j,
      // x = A[:, k0], y = A[:, kk], t1 = alpha y_j, t2 = alpha x_j; the diagonal is set to 0.
      for (int e = tid; e < kk * kk; e += nt) {
        const int i = e % kk, j = e / kk;
        if (i > j) continue;
        if (i == j) { A[j + (size_t)j * n] = zero; continue; }
        const T x = A[i + (size_t)k0 * n], y = A[i + (size_t)kk * n];
        const T t1 = alpha * A[j + (size_t)kk * n];
        const T t2 = alpha * A[j + (size_t)k0 * n];
        A[i + (size_t)j * n] = (A[i + (size_t)j * n] + x * t1) - y * t2;
      }
      __syncthreads();
      for (int i = tid; i < kk; i += nt) A[i + (size_t)k0 * n] = A[i + (size_t)k0 * n] * alpha;
      __syncthreads();
    }
  }
  __syncthreads();

  // ---- Phase 2: Pfaffian and status -------------------------------------------------------
  if (tid == 0) {
    int status = 0;
    T pf = zero;
    if (s_info > 0) {
      status = s_info;
    } else {
      pf = one;
      for (int i = 0; i + 1 < n; i += 2) pf = pf * A[i + (size_t)(i + 1) * n];
      int flips = 0;
      for (int k = 0; k < n; ++k) flips += (piv[k] != k);
      if (flips & 1) pf = zero - pf;
      if (!TR::finite(pf)) status = -1;
    }
    s_status = status;
    PF[0] = pf;
    ST[0] = status;
  }
  __syncthreads();
  if (s_status != 0) {
    for (size_t e = tid; e < nn; e += nt) OUT[e] = zero;
    return;
  }

  // ---- Phase 3: inverse (utu2inv) ----------------------------------------------------------
  // M = I-like workspace: columns 0..n-2 hold X = S^-1 (S[i,k] = A[i,k+1], unit upper), the last
  // column is e_{n-1}. One thread per column (back substitution).
  for (size_t e = tid; e < nn; e += nt) W[e] = zero;
  __syncthreads();
  for (int c = tid; c < n - 1; c += nt) {
    T* xc = W + (size_t)c * n;
    xc[c] = one;
    for (int i = c - 1; i >= 0; --i) {
      T acc = zero;
      for (int k = i + 1; k <= c; ++k) acc = acc + A[i + (size_t)(k + 1) * n] * xc[k];
      xc[i] = zero - acc;
    }
  }
  if (tid == 0) W[(n - 1) + (size_t)(n - 1) * n] = one;
  for (int i = tid; i < n - 1; i += nt) vt[i] = -A[i + (size_t)(i + 1) * n];
  __syncthreads();
  // Skew-tridiagonal solve C = T^-1 M per column (C overwrites A).
  for (int j = tid; j < n; j += nt) {
    const T* b = W + (size_t)j * n;
    T* c = A + (size_t)j * n;
    c[1] = b[0] / (-vt[0]);
    for (int i = 2; i < n; i += 2) c[i + 1] = (b[i] - c[i - 1] * vt[i - 1]) / (-vt[i]);
    c[n - 2] = b[n - 1] / vt[n - 2];
    for (int i = n - 3; i >= 1; i -= 2) c[i - 1] = (b[i] + c[i + 1] * vt[i]) / vt[i - 1];
  }
  // Composed pivot permutation (swap sequence j = 0..n-1).
  if (tid == 0) {
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int j = 0; j < n; ++j) { int t = piv[j], a = idx[j]; idx[j] = idx[t]; idx[t] = a; }
  }
  __syncthreads();
  // Out[r,c] = sum_{k <= idx[r]} M[k, idx[r]] * C[k, idx[c]].
  for (int e = tid; e < n * n; e += nt) {
    const int r = e % n, c = e / n;
    const int ir = idx[r], ic = idx[c];
    T acc = zero;
    for (int k = 0; k <= ir; ++k) acc = acc + W[k + (size_t)ir * n] * A[k + (size_t)ic * n];
    OUT[r + (size_t)c * n] = acc;
  }
}


// ---------------------------------------------------------------------------------------------
// Fast path for real double matrices, n >= 64 and even (issue #517).
//
// The generic `pfinv_plane<T>` above still serves the f32/c64 variants and the small real
// batches unchanged; this path is used only by `pfinv_f64`, i.e. the production kernel of the
// device-sampler recompute and the measurement side. Every per-element formula is unchanged,
// so the two paths agree bit for bit through the whole factorization and the Pfaffian and
// differ by rounding only in the mirrored half of the inverse (see phase 3(d)). Phase 3(b) is
// left in the exact C operation order on purpose: splitting each dot product into partial sums
// was measured and gave no gain (the phase is memory-bound), so it is not kept.
//
//  * phase 1 stages x = A[:, k0], y = A[:, kk], t1 = alpha*y and t2 = alpha*x in shared memory
//    once per step (4n doubles), scales column k0 in the same pass, and then applies the
//    strict-upper rank-2 update A_ij = (A_ij + x_i t1_j) - y_i t2_j with one double2 load and
//    one double2 store per two elements, with warp-cyclic columns. The generic path re-reads
//    five global values per element and computes every index with % and /; the pivot search
//    here reduces with __shfl_down_sync plus one shared-memory round instead of log2(nt)
//    __syncthreads. Sweeping whole columns per warp amortizes the t1s/t2s reads over 64 rows.
//  * phase 3(d) computes M^T C as a register-tiled GEMM restricted to the upper triangle of
//    the pre-permutation index space and writes the transposed entry as its exact negative:
//    for n a multiple of 64 that is n^3/6 multiplies instead of n^3, and each 64x64 macro tile
//    fetches a column pair with two double2 loads per 4-wide k block. The k loop stops at the
//    end of the macro row block; the extra terms are the exact zeros of the upper-triangular M
//    and every entry keeps the original k = 0,1,2,... accumulation order. Tiles are swept in
//    strides of nt so any block size is handled.
// ---------------------------------------------------------------------------------------------
__device__ inline void pfinv_plane_f64(double* A, double* W, double* OUT, double* PF, int* ST, int n) {
  extern __shared__ char smem_raw[];
  double* xs = reinterpret_cast<double*>(smem_raw);  // n
  double* ys = xs + n;                               // n
  double* t1s = ys + n;                              // n
  double* t2s = t1s + n;                             // n
  double* vt = xs;                                   // n (phase 3 reuses xs)
  int* red_i = reinterpret_cast<int*>(t2s + n);      // 32 (also used as the inverse permutation)
  double* red_v = reinterpret_cast<double*>(red_i + 32);  // 32
  int* piv = reinterpret_cast<int*>(red_v + 32);     // n
  int* idx = piv + n;                                // n
  __shared__ int s_kp, s_info, s_zero, s_status;
  __shared__ double s_colmax;

  const int tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid & 31, wid = tid >> 5, nw = nt >> 5;
  const size_t nn = (size_t)n * n;
  for (int i = tid; i < n; i += nt) piv[i] = i;
  if (tid == 0) s_info = 0;
  __syncthreads();

  // ---- Phase 1: LTL^T factorization, k0 = n-1 .. 1 (0-based), kk = k0-1 ----------------------
  for (int k0 = n - 1; k0 >= 1; --k0) {
    const int kk = k0 - 1;
    // First-index argmax of mag(A[0..kk, k0]) (same semantics as the scalar scan), reduced
    // warp-wise with shuffles and then across the warps.
    double bv = -1.0;
    int bi = 0x7fffffff;
    for (int j = tid; j <= kk; j += nt) {
      double v = fabs(A[j + (size_t)k0 * n]);
      if (v != v) v = (j == 0) ? 1e300 * 1e300 : -1.0;
      if (v > bv || (v == bv && j < bi)) { bv = v; bi = j; }
    }
    for (int s = 16; s > 0; s >>= 1) {
      double ov = __shfl_down_sync(0xffffffffu, bv, s);
      int oi = __shfl_down_sync(0xffffffffu, bi, s);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
      red_v[wid] = bv;
      red_i[wid] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      double v = red_v[0];
      int i0 = red_i[0];
      for (int w = 1; w < nw; ++w) {
        double ov = red_v[w];
        int oi = red_i[w];
        if (ov > v || (ov == v && oi < i0)) { v = ov; i0 = oi; }
      }
      int kp = i0;
      if (kp > kk) kp = kk;
      s_kp = kp;
      s_colmax = fabs(A[kp + (size_t)k0 * n]);
      s_zero = (s_colmax == 0.0) ? 1 : 0;
      if (s_zero) {
        if (s_info == 0) s_info = kk + 1;
      } else {
        piv[kk] = kp;
      }
    }
    __syncthreads();
    if (s_zero) continue;
    const int kp = s_kp;

    if (kp != kk) {
      for (int j = tid; j < kp; j += nt) {
        double t = A[j + (size_t)kk * n]; A[j + (size_t)kk * n] = A[j + (size_t)kp * n]; A[j + (size_t)kp * n] = t;
      }
      for (int j = kp + 1 + tid; j < kk; j += nt) {
        double t = A[j + (size_t)kk * n]; A[j + (size_t)kk * n] = A[kp + (size_t)j * n]; A[kp + (size_t)j * n] = t;
      }
      for (int j = k0 + tid; j < n; j += nt) {
        double t = A[kk + (size_t)j * n]; A[kk + (size_t)j * n] = A[kp + (size_t)j * n]; A[kp + (size_t)j * n] = t;
      }
      __syncthreads();
      for (int j = kp + tid; j < kk; j += nt) A[j + (size_t)kk * n] = -A[j + (size_t)kk * n];
      for (int j = kp + 1 + tid; j < kk; j += nt) A[kp + (size_t)j * n] = -A[kp + (size_t)j * n];
      __syncthreads();
    }

    if (kk >= 1) {
      const double alpha = 1.0 / A[kk + (size_t)k0 * n];
      // Stage the pivot columns and scale column k0 in one pass.
      for (int i = tid; i < kk; i += nt) {
        const double x = A[i + (size_t)k0 * n];
        const double y = A[i + (size_t)kk * n];
        xs[i] = x;
        ys[i] = y;
        t1s[i] = alpha * y;
        t2s[i] = alpha * x;
        A[i + (size_t)k0 * n] = alpha * x;
      }
      __syncthreads();
      // A_ij = (A_ij + x_i t1_j) - y_i t2_j over the strict upper triangle (i < j); the
      // diagonal is zeroed in the same sweep. Warp w handles columns j = w, w+nw, ... and all
      // of their rows in double2 chunks of 64 rows.
      {
        const int lane2 = lane << 1;
        for (int j = wid; j < kk; j += nw) {
          const double t1 = t1s[j], t2 = t2s[j];
          for (int base = 0; base < j; base += 64) {
            const int i = base + lane2;
            if (i + 1 < j) {
              const double2 x = *reinterpret_cast<const double2*>(xs + i);
              const double2 y = *reinterpret_cast<const double2*>(ys + i);
              double2 v = *reinterpret_cast<const double2*>(A + i + (size_t)j * n);
              v.x = (v.x + x.x * t1) - y.x * t2;
              v.y = (v.y + x.y * t1) - y.y * t2;
              *reinterpret_cast<double2*>(A + i + (size_t)j * n) = v;
            } else if (i < j) {
              const double v = (A[i + (size_t)j * n] + xs[i] * t1) - ys[i] * t2;
              A[i + (size_t)j * n] = v;
            }
          }
        }
        // Diagonal of the trailing block (the update only touches the strict upper triangle).
        for (int j = tid; j < kk; j += nt) A[j + (size_t)j * n] = 0.0;
      }
      __syncthreads();
    }
  }

  __syncthreads();

  // ---- Phase 2: Pfaffian and status ---------------------------------------------------------
  if (tid == 0) {
    int status = 0;
    double pf = 0.0;
    if (s_info > 0) {
      status = s_info;
    } else {
      pf = 1.0;
      for (int i = 0; i + 1 < n; i += 2) pf = pf * A[i + (size_t)(i + 1) * n];
      int flips = 0;
      for (int k = 0; k < n; ++k) flips += (piv[k] != k);
      if (flips & 1) pf = 0.0 - pf;
      if (!(pf - pf == 0.0)) status = -1;
    }
    s_status = status;
    PF[0] = pf;
    ST[0] = status;
  }
  __syncthreads();
  if (s_status != 0) {
    for (size_t e = tid; e < nn; e += nt) OUT[e] = 0.0;
    return;
  }

  // ---- Phase 3: inverse (utu2inv) -----------------------------------------------------------
  for (size_t e = tid; e < nn; e += nt) W[e] = 0.0;
  __syncthreads();
  for (int c = tid; c < n - 1; c += nt) {
    double* xc = W + (size_t)c * n;
    xc[c] = 1.0;
    for (int i = c - 1; i >= 0; --i) {
      double acc = 0.0;
      for (int k = i + 1; k <= c; ++k) acc = acc + A[i + (size_t)(k + 1) * n] * xc[k];
      xc[i] = 0.0 - acc;
    }
  }

  if (tid == 0) W[(n - 1) + (size_t)(n - 1) * n] = 1.0;
  for (int i = tid; i < n - 1; i += nt) vt[i] = -A[i + (size_t)(i + 1) * n];
  __syncthreads();
  for (int j = tid; j < n; j += nt) {
    const double* b = W + (size_t)j * n;
    double* c = A + (size_t)j * n;
    c[1] = b[0] / (-vt[0]);
    for (int i = 2; i < n; i += 2) c[i + 1] = (b[i] - c[i - 1] * vt[i - 1]) / (-vt[i]);
    c[n - 2] = b[n - 1] / vt[n - 2];
    for (int i = n - 3; i >= 1; i -= 2) c[i - 1] = (b[i] + c[i + 1] * vt[i]) / vt[i - 1];
  }
  if (tid == 0) {
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int j = 0; j < n; ++j) { int t = piv[j], a = idx[j]; idx[j] = idx[t]; idx[t] = a; }
    if (n % 64 == 0) {
      // inverse permutation in place of `piv` (no longer needed)
      for (int r = 0; r < n; ++r) piv[idx[r]] = r;
    }
  }
  __syncthreads();
  // Out[r,c] = sum_{k <= idx[r]} M[k, idx[r]] * C[k, idx[c]].
  //
  // Fast path (issue #517): n a multiple of 64. The product G = M^T C is skew, so only the
  // upper triangle G[i,j] (i <= j in pre-permutation indices) is computed and the transposed
  // entry is written as its exact negative. The 64x64 macro tiles use 4x4 register tiles per
  // thread with the k loop running to the end of the macro row block: the extra terms are the
  // exact zeros of the upper-triangular M, so each computed entry keeps the original
  // k = 0,1,2,... accumulation order. The two permuted output positions are written scattered;
  // that is n^2 single stores against n^3/6 multiplies. The macro tile holds 16x16 thread
  // tiles, swept in strides of nt so any block size is correct.
  if (n % 64 == 0) {
    const int* inv = piv;
    const int nm = n >> 6;
    for (int ma = 0; ma < nm; ++ma) {
      const int kmax = (ma + 1) << 6;
      for (int mb = ma; mb < nm; ++mb) {
        for (int tt = tid; tt < 256; tt += nt) {
          const int tx = tt >> 4, ty = tt & 15;
          if (mb == ma && ty < tx) continue;
          const int i0 = (ma << 6) + (tx << 2);
          const int j0 = (mb << 6) + (ty << 2);
          const double* wi = W + (size_t)i0 * n;
          const double* cj = A + (size_t)j0 * n;
          double g[4][4] = {{0.0, 0.0, 0.0, 0.0},
                             {0.0, 0.0, 0.0, 0.0},
                             {0.0, 0.0, 0.0, 0.0},
                             {0.0, 0.0, 0.0, 0.0}};
          // k blocks of 4: every column pair is fetched with two double2 loads, so each load
          // moves 32 contiguous bytes (16 column loads + 16 FMA per 64 multiplies) and the
          // accumulation order per output stays k = 0,1,2,3.
          for (int kb = 0; kb < kmax; kb += 4) {
            double2 w2[4][2], c2[4][2];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              w2[p][0] = *reinterpret_cast<const double2*>(wi + (size_t)p * n + kb);
              w2[p][1] = *reinterpret_cast<const double2*>(wi + (size_t)p * n + kb + 2);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              c2[q][0] = *reinterpret_cast<const double2*>(cj + (size_t)q * n + kb);
              c2[q][1] = *reinterpret_cast<const double2*>(cj + (size_t)q * n + kb + 2);
            }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              const double wv[4] = {kk < 2 ? (kk == 0 ? w2[0][0].x : w2[0][0].y)
                                           : (kk == 2 ? w2[0][1].x : w2[0][1].y),
                                    kk < 2 ? (kk == 0 ? w2[1][0].x : w2[1][0].y)
                                           : (kk == 2 ? w2[1][1].x : w2[1][1].y),
                                    kk < 2 ? (kk == 0 ? w2[2][0].x : w2[2][0].y)
                                           : (kk == 2 ? w2[2][1].x : w2[2][1].y),
                                    kk < 2 ? (kk == 0 ? w2[3][0].x : w2[3][0].y)
                                           : (kk == 2 ? w2[3][1].x : w2[3][1].y)};
              const double cv[4] = {kk < 2 ? (kk == 0 ? c2[0][0].x : c2[0][0].y)
                                           : (kk == 2 ? c2[0][1].x : c2[0][1].y),
                                    kk < 2 ? (kk == 0 ? c2[1][0].x : c2[1][0].y)
                                           : (kk == 2 ? c2[1][1].x : c2[1][1].y),
                                    kk < 2 ? (kk == 0 ? c2[2][0].x : c2[2][0].y)
                                           : (kk == 2 ? c2[2][1].x : c2[2][1].y),
                                    kk < 2 ? (kk == 0 ? c2[3][0].x : c2[3][0].y)
                                           : (kk == 2 ? c2[3][1].x : c2[3][1].y)};
#pragma unroll
              for (int p = 0; p < 4; ++p)
#pragma unroll
                for (int q = 0; q < 4; ++q) g[p][q] = g[p][q] + wv[p] * cv[q];
            }
          }
          const int ris[4] = {inv[i0], inv[i0 + 1], inv[i0 + 2], inv[i0 + 3]};
          const int cis[4] = {inv[j0], inv[j0 + 1], inv[j0 + 2], inv[j0 + 3]};
          for (int p = 0; p < 4; ++p) {
            for (int q = 0; q < 4; ++q) {
              const int i = i0 + p, j = j0 + q;
              if (i > j) continue;
              OUT[ris[p] + (size_t)cis[q] * n] = g[p][q];
              if (i < j) OUT[cis[q] + (size_t)ris[p] * n] = 0.0 - g[p][q];
            }
          }
        }
      }
    }
    __syncthreads();
    return;
  }
  for (int e = tid; e < n * n; e += nt) {
    const int r = e % n, c = e / n;
    const int ir = idx[r], ic = idx[c];
    double acc = 0.0;
    for (int k = 0; k <= ir; ++k) acc = acc + W[k + (size_t)ir * n] * A[k + (size_t)ic * n];
    OUT[r + (size_t)c * n] = acc;
  }

}

template <typename T>
__device__ inline void entry(T* A, T* W, T* OUT, T* PF, int* ST, int n) {
  const size_t p = blockIdx.x;
  pfinv_plane<T>(A + p * (size_t)n * n, W + p * (size_t)n * n, OUT + p * (size_t)n * n, PF + p, ST + p, n);
}

// Real-double body of the batched entry points: n >= 64 keeps the small-n batches on the generic
// path (bit-identical to the pre-#517 kernel there); odd n has no 16-byte-aligned double2 staging
// and stays generic as well.
__device__ inline void body_f64(double* A, double* W, double* OUT, double* PF, int* ST, int n) {
  if (n >= 64 && (n & 1) == 0) {
    pfinv_plane_f64(A + blockIdx.x * (size_t)n * n, W + blockIdx.x * (size_t)n * n,
                    OUT + blockIdx.x * (size_t)n * n, PF + blockIdx.x, ST + blockIdx.x, n);
  } else {
    entry<double>(A, W, OUT, PF, ST, n);
  }
}

extern "C" __global__ void __launch_bounds__(256, 2) pfinv_f64(double* A, double* W, double* OUT, double* PF, int* ST, int n) {
  body_f64(A, W, OUT, PF, ST, n);
}

// Wide-block variant of `pfinv_f64` (issue #517 sampler hops): identical math, one 1024-thread
// block per plane instead of 256 threads with two blocks per SM. Phase 1's per-step rank-2 sweep
// is latency-bound, so a batch that under-uses the device (the sampler's recompute is 8 planes =
// 8 SMs out of 108) runs 1.2-1.6x faster; the saturated large batches prefer `pfinv_f64`
// (measured in `bench_recompute_roofline`; see docs/design/gpu-readiness.md 16.10).
extern "C" __global__ void __launch_bounds__(1024, 1) pfinv_f64_wide(double* A, double* W, double* OUT, double* PF, int* ST, int n) {
  body_f64(A, W, OUT, PF, ST, n);
}
extern "C" __global__ void pfinv_c64(Cx<double>* A, Cx<double>* W, Cx<double>* OUT, Cx<double>* PF, int* ST, int n) {
  entry<Cx<double>>(A, W, OUT, PF, ST, n);
}
extern "C" __global__ void pfinv_f32(float* A, float* W, float* OUT, float* PF, int* ST, int n) {
  entry<float>(A, W, OUT, PF, ST, n);
}
extern "C" __global__ void pfinv_c32(Cx<float>* A, Cx<float>* W, Cx<float>* OUT, Cx<float>* PF, int* ST, int n) {
  entry<Cx<float>>(A, W, OUT, PF, ST, n);
}
