// Panelized Parlett-Reid Pfaffian, following the delayed rank-two update
// organization of tensor4all/faer (Wimmer Algorithm 923). A CUDA block owns
// one matrix. Eliminated columns hold panel factors, so caller scratch and
// host/device transfers do not grow. This is a device implementation, not a
// call to faer's CPU-only API.
#ifndef MVMC_PFAFFIAN_PANEL
#define MVMC_PFAFFIAN_PANEL

template <typename Ops, bool Factor = false>
__device__ typename Ops::T resident_panel_pfaffian(
    typename Ops::T* A, int n, typename Ops::T* tau, typename Ops::T* u,
    typename Ops::T* lower = nullptr, typename Ops::T* diagonal = nullptr,
    int* permutation = nullptr, int* factor_status = nullptr) {
  typedef typename Ops::T T;
  __shared__ T product;
  const int tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) product = Ops::one();
  if (Factor) {
    if (tid == 0) *factor_status = 0;
    for (int i=tid;i<n;i+=nt) permutation[i]=i;
    for (int i=tid;i<n*n;i+=nt) lower[i] = i%n==i/n ? Ops::one() : Ops::zero();
  }
  __syncthreads();
  if (n & 1) return Ops::zero();
  const int panel = 8;
  for (int start = 0; start < n; start += panel) {
    const int end = min(start + panel, n);
    for (int k = start; k + 1 < end; k += 2) {
      // Materialize only this pivot column, with pending panel corrections.
      for (int r = k + 1 + tid; r < n; r += nt) {
        T value = A[r + (size_t)k * n];
        for (int p = start; p < k; p += 2) {
          value = Ops::add(value, Ops::sub(
              Ops::mul(A[r + (size_t)p*n], A[k + (size_t)(p+1)*n]),
              Ops::mul(A[r + (size_t)(p+1)*n], A[k + (size_t)p*n])));
        }
        tau[r] = value;
      }
      __syncthreads();
      const int kp = Ops::argmax(tau, k + 1, n);
      if (kp < 0) {
        if (tid == 0) product = kp == -1 ? Ops::zero() : Ops::nan();
        if (Factor && tid == 0) *factor_status = kp == -1 ? 1 : -1;
        __syncthreads();
        const T value = product;
        __syncthreads();
        return value;
      }
      // Store the current pivot column before its congruence permutation.
      for (int r = k + 1 + tid; r < n; r += nt) A[r + (size_t)k*n] = tau[r];
      __syncthreads();
      if (kp != k + 1) {
        for (int c = tid; c < n; c += nt) {
          const T x = A[k + 1 + (size_t)c*n];
          A[k + 1 + (size_t)c*n] = A[kp + (size_t)c*n];
          A[kp + (size_t)c*n] = x;
        }
        if (Factor) {
          for (int c=tid;c<k;c+=nt) {
            const T x=lower[k+1+(size_t)c*n];
            lower[k+1+(size_t)c*n]=lower[kp+(size_t)c*n];
            lower[kp+(size_t)c*n]=x;
          }
          if (tid==0) { const int p=permutation[k+1]; permutation[k+1]=permutation[kp]; permutation[kp]=p; }
        }
        __syncthreads();
        for (int r = tid; r < n; r += nt) {
          const T x = A[r + (size_t)(k+1)*n];
          A[r + (size_t)(k+1)*n] = A[r + (size_t)kp*n];
          A[r + (size_t)kp*n] = x;
        }
        if (tid == 0) product = Ops::neg(product);
        __syncthreads();
      }
      const T d = A[k + 1 + (size_t)k*n];
      if (tid == 0) product = Ops::mul(product, Ops::neg(d));
      if (Factor && tid==0) diagonal[k/2]=d;
      for (int r = k + 2 + tid; r < n; r += nt) {
        T value = A[r + (size_t)(k+1)*n];
        for (int p = start; p < k; p += 2) {
          value = Ops::add(value, Ops::sub(
              Ops::mul(A[r + (size_t)p*n], A[k+1 + (size_t)(p+1)*n]),
              Ops::mul(A[r + (size_t)(p+1)*n], A[k+1 + (size_t)p*n])));
        }
        u[r] = value;
        tau[r] = Ops::div(A[r + (size_t)k*n], d);
        if (Factor) {
          lower[r+(size_t)k*n]=Ops::neg(Ops::div(value,d));
          lower[r+(size_t)(k+1)*n]=tau[r];
        }
      }
      __syncthreads();
      for (int r = k + 2 + tid; r < n; r += nt) {
        A[r + (size_t)k*n] = tau[r];
        A[r + (size_t)(k+1)*n] = u[r];
      }
      __syncthreads();
    }
    // Fuse every pending pair into one read/write of each trailing entry.
    // Only one triangle is evaluated; the other is its exact negative.
    const int width = n - end;
    for (int index = tid; index < width * width; index += nt) {
      const int r = end + index % width, c = end + index / width;
      if (r <= c) continue;
      T value = A[r + (size_t)c*n];
      for (int p = start; p < end; p += 2) {
        value = Ops::add(value, Ops::sub(
            Ops::mul(A[r + (size_t)p*n], A[c + (size_t)(p+1)*n]),
            Ops::mul(A[r + (size_t)(p+1)*n], A[c + (size_t)p*n])));
      }
      A[r + (size_t)c*n] = value;
      A[c + (size_t)r*n] = Ops::neg(value);
    }
    __syncthreads();
  }
  const T value = product;
  __syncthreads();
  return value;
}
#endif
