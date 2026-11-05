// Resident interval orchestration (#560). Rebuild flags never leave the device.
// A launch can accept at most one move per hop. After a rebuild the next one needs
// nsite+1 accepts, so the host can enqueue a bounded schedule without reading flags.

extern "C" __global__ void interval_resident_hop(
    int* ele, int* occ, double* inv, double* pf, int* aw, double* logip,
    const double* slater, const double* weights, const int* gutz, const int* jast,
    const double* params, const unsigned int* tape, int total_hops, int* hop_off,
    int n, int ne, int nsite, int nqp, int qp_groups, int ngutz, int nproj,
    int* hops_done, int* rebuild, int* accepts, int* total_accepts, int* error, double* delayed, double* flush, int delayed_updates,
    int* valid, double* fac, double* work, const int* loc, double exchange_fraction,
    const int* dh, int n2dh, int n4dh, const int* rbm, const double* rbmw, double* theta,
    int nneuron, int fsz, int spin_free) {
  if (*error) return;  // block-uniform; previous launch owns the error flag
  if (n == 128) {
    interval_hop_impl<true>(ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast,
      params, tape, total_hops, hop_off, 128, ne, nsite, nqp, qp_groups, ngutz,
      nproj, exchange_fraction, hops_done, rebuild, accepts,
      n >= 128 && delayed_updates ? delayed : nullptr, flush, valid, fac, work, error, loc, dh, n2dh, n4dh, rbm, rbmw, theta, nneuron,
      fsz, spin_free);
  } else {
    interval_hop_impl<true>(ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast,
      params, tape, total_hops, hop_off, n, ne, nsite, nqp, qp_groups, ngutz,
      nproj, exchange_fraction, hops_done, rebuild, accepts,
      n >= 128 && delayed_updates ? delayed : nullptr, flush, valid, fac, work, error, loc, dh, n2dh, n4dh, rbm, rbmw, theta, nneuron,
      fsz, spin_free);
  }
  // interval_hop_impl publishes accepts from thread 0 of the same block.
  if (threadIdx.x == 0) total_accepts[blockIdx.x] += accepts[blockIdx.x];
}

// Assemble and factor only a requested walker. No host-generated compact index
// list and no copying resident inverses/Pfaffians back for a rebuild. `ne` is
// the first slot with the down-spin offset: Ne for normal walkers, 2Ne for FSZ
// walkers, whose slots already hold rs (#714).
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

// Publish a walker whose weighted overlap is finite and nonzero, recording
// per-plane validity as Julia does (cuda_batched_inverse.jl:68): no zero pivot
// and a finite nonzero Pfaffian. An invalid plane publishes Pf = 0 from the
// zero-pivot factor; no all-valid path reads its inverse storage. A zero or
// nonfinite total latches the sticky error, and every table of that walker
// retains its previous contents. Later sampling stops until the owner reports.
extern "C" __global__ void interval_resident_commit(
    const int* rebuild, int* error, const int* ST, const double* OUT,
    const double* PF, const double* weights, double* inv, double* pf,
    double* logip, int* aw, int* valid, int n, int nqp) {
  const int w = blockIdx.x;
  if (!rebuild[w]) return;
  __shared__ int good;
  __shared__ double log_overlap;
  if (threadIdx.x == 0) {
    double ip = 0.0;
    for (int q = 0; q < nqp; ++q)
      ip = __dadd_rn(ip, __dmul_rn(weights[q], PF[w*nqp+q]));
    good = isfinite(ip) && ip != 0.0;
    log_overlap = log(fabs(ip) + 1e-100);
    if (!good) atomicCAS(error, 0, RESIDENT_REBUILD);
  }
  __syncthreads();
  if (!good) return;
  const size_t base = (size_t)w*nqp*n*n;
  for (size_t i = threadIdx.x; i < (size_t)nqp*n*n; i += blockDim.x)
    inv[base+i] = -OUT[base+i];
  for (int q = threadIdx.x; q < nqp; q += blockDim.x) {
    const double p = PF[w*nqp+q];
    pf[w*nqp+q] = p;
    valid[w*nqp+q] = ST[w*nqp+q] == 0 && isfinite(p) && p != 0.0;
  }
  if (threadIdx.x == 0) {
    logip[w] = log_overlap;
    aw[w] = 0;
  }
}

// The host enqueues a bounded number of hop/rebuild rounds without reading
// flags. In-kernel refactors keep that bound; this guards the invariant so a
// walker with unfinished hops never publishes a short interval.
extern "C" __global__ void interval_resident_complete(
    const int* hop_off, int hops, int* error, int walkers) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walkers && hop_off[w] < hops) atomicCAS(error, 0, RESIDENT_SCHEDULE);
}
