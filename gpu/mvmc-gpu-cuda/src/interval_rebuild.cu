// Device-side rebuild helpers for the interval sampler (issue #547).
//
// Compiled together with `pfaffian_batched.cu` (`pfinv_f64`) and `interval_kernels.cu`
// (`SLT`). When a walker's accepted-move window passes `nsite`, the host batches every walker
// that needs a rebuild into one assemble + `pfinv_f64` + commit launch on tenferro's stream.
// This keeps the recomputation on the device instead of a per-walker host
// `calc_m_all_real`, which is the rebuild bottleneck at L=128.

// Assemble the skew plane `A[item][qp] = -S[qp][rs[msi]][rs[j]]` (the mVMC `calculateMAll`
// plane of the candidate configuration) for the walkers in `rec_walker`. One block per
// (item, qp); the element map matches `k_assemble` in sampler_kernels.cu.
extern "C" __global__ void interval_assemble(const int* rec_walker, const int* ele,
                                             const double* slater, double* A, int n, int ne,
                                             int nsite, int nqp) {
  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int w = rec_walker[item];
  const int* e = ele + (size_t)w * n;
  const int n2 = 2 * nsite;
  double* plane = A + ((size_t)item * nqp + qp) * n * n;
  for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) {
    const int msi = idx / n, j = idx % n;
    const int rsi = e[msi] + (msi >= ne ? nsite : 0);
    const int rsj = e[j] + (j >= ne ? nsite : 0);
    plane[idx] = -SLT(slater, qp, rsi, rsj, n2);
  }
}

// Commit `pfinv_f64`'s output for the walkers in `rec_walker`: `invM = -OUT` (the
// `calc_m_all_real` sign flip), `pf = PF`, reset the accepted-move window and refresh the
// cached `log|inner product|`.
extern "C" __global__ void interval_commit(const int* rec_walker, const double* OUT,
                                           const double* PF, const double* weights, double* inv,
                                           double* pf, double* logip, int* aw, int n, int nqp) {
  const int item = blockIdx.x;
  const int w = rec_walker[item];
  const size_t plane = (size_t)n * n;
  for (size_t e = threadIdx.x; e < (size_t)nqp * plane; e += blockDim.x)
    inv[(size_t)w * nqp * plane + e] = -OUT[(size_t)item * nqp * plane + e];
  for (int q = threadIdx.x; q < nqp; q += blockDim.x)
    pf[(size_t)w * nqp + q] = PF[(size_t)item * nqp + q];
  __syncthreads();
  if (threadIdx.x == 0) {
    double ip = 0.0;
    for (int q = 0; q < nqp; ++q) ip += weights[q] * pf[(size_t)w * nqp + q];
    logip[w] = log(ip < 0.0 ? -ip : ip);
    aw[w] = 0;
  }
}
