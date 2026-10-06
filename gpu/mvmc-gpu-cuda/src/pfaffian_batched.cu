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

template <typename T>
__device__ inline void entry(T* A, T* W, T* OUT, T* PF, int* ST, int n) {
  const size_t p = blockIdx.x;
  pfinv_plane<T>(A + p * (size_t)n * n, W + p * (size_t)n * n, OUT + p * (size_t)n * n, PF + p, ST + p, n);
}

extern "C" __global__ void pfinv_f64(double* A, double* W, double* OUT, double* PF, int* ST, int n) {
  entry<double>(A, W, OUT, PF, ST, n);
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
