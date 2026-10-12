// Resident interval orchestration (#560). Rebuild flags never leave the device.
// A launch can accept at most one move per hop. After a rebuild the next one needs
// nsite+1 accepts, so the host can enqueue a bounded schedule without reading flags.

extern "C" __global__ void interval_resident_hop(
    int* ele, int* occ, double* inv, double* pf, int* aw, double* logip,
    const double* slater, const double* weights, const int* gutz, const int* jast,
    const double* params, const unsigned int* tape, int total_hops, int* hop_off,
    int n, int ne, int nsite, int nqp, int qp_groups, int ngutz, int nproj,
    int* hops_done, int* rebuild, int* accepts, int* total_accepts, const int* error) {
  if (*error) return;  // block-uniform; previous launch owns the error flag
  if (n == 128) {
    interval_hop_impl(ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast,
      params, tape, total_hops, hop_off, 128, ne, nsite, nqp, qp_groups, ngutz,
      nproj, 0.0, hops_done, rebuild, accepts);
  } else {
    interval_hop_impl(ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast,
      params, tape, total_hops, hop_off, n, ne, nsite, nqp, qp_groups, ngutz,
      nproj, 0.0, hops_done, rebuild, accepts);
  }
  // interval_hop_impl publishes accepts from thread 0 of the same block.
  if (threadIdx.x == 0) total_accepts[blockIdx.x] += accepts[blockIdx.x];
}

// Assemble and factor only a requested walker. No host-generated compact index
// list and no copying resident inverses/Pfaffians back for a rebuild.
extern "C" __global__ void interval_resident_factor(
    const int* rebuild, const int* error, const int* ele, const double* slater,
    double* A, double* W, double* OUT, double* PF, int* ST,
    int n, int ne, int nsite, int nqp) {
  const int plane = blockIdx.x, w = plane / nqp, q = plane % nqp;
  if (*error || !rebuild[w]) return;
  const size_t offset = (size_t)plane * n * n;
  double* a = A + offset;
  const int* e = ele + (size_t)w * n;
  for (int index = threadIdx.x; index < n*n; index += blockDim.x) {
    const int i = index / n, j = index % n;
    const int ri = e[i] + (i >= ne ? nsite : 0);
    const int rj = e[j] + (j >= ne ? nsite : 0);
    a[index] = -SLT(slater, q, ri, rj, 2*nsite);
  }
  __syncthreads();
  body_f64(A, W, OUT, PF, ST, n);
}

// Check the entire walker's factorization before publishing anything. The error
// flag is sticky across steps; subsequent sampling stops until the owner reports
// the failure. All planes of a failed walker retain their previous tables.
extern "C" __global__ void interval_resident_commit(
    const int* rebuild, int* error, const int* ST, const double* OUT,
    const double* PF, const double* weights, double* inv, double* pf,
    double* logip, int* aw, int n, int nqp) {
  const int w = blockIdx.x;
  if (!rebuild[w]) return;
  __shared__ int good;
  __shared__ double log_overlap;
  if (threadIdx.x == 0) {
    good = 1;
    double ip = 0.0;
    for (int q = 0; q < nqp; ++q) {
      if (ST[w*nqp+q] != 0) good = 0;
      ip += weights[q] * PF[w*nqp+q];
    }
    if (!isfinite(ip) || ip == 0.0) good = 0;
    log_overlap = log(fabs(ip));
    if (!good) atomicCAS(error, 0, 1);
  }
  __syncthreads();
  if (!good) return;
  const size_t base = (size_t)w*nqp*n*n;
  for (size_t i = threadIdx.x; i < (size_t)nqp*n*n; i += blockDim.x)
    inv[base+i] = -OUT[base+i];
  for (int q = threadIdx.x; q < nqp; q += blockDim.x)
    pf[w*nqp+q] = PF[w*nqp+q];
  if (threadIdx.x == 0) {
    logip[w] = log_overlap;
    aw[w] = 0;
  }
}
