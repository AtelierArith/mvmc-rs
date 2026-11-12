// Resident complex normal-orbital sampler and rebuild (#678, #572 milestone 6).
//
// Julia GPU has no complex path ("CUDA runner requires real parameters",
// ext/cuda_state.jl:39), so the physics follows C's complex normal path:
// CalculateMAll_fcmp (PFAPACK zsktf2/utu2pfa/utu2inv, here
// pfinv_plane<Cx<double>>), CalculateNewPfM2 and UpdateMAll_child_fcmp
// (pfupdate.c), CalculateLogIP_fcmp (qp.c, log|ip|) and the vmcmake.c
// acceptance exp(2 (Re(logIpNew - logIpOld) + LogProjRatio)). LogProjRatio
// uses the real projection parameters (projection.c creal(Proj)).
//
// The proposal and decision contract is the Julia GPU tape of
// interval_kernels.cu (five words per proposal, empty-destination hops,
// exchange threshold word 3, log-space acceptance with the 1e-100 overlap
// floor). Updates are immediate C rank-two updates (no delayed queue, no
// paired factor). The #573 per-plane validity is kept: a walker with an
// invalid plane or an exchange proposal scores the candidate with full
// complex Pfaffians, and such an accept, or an accept whose candidate plane
// Pfaffian is zero or nonfinite, refactors the walker in this kernel.
// Local spins, DH and RBM are rejected on the host in complex mode.
//
// Complex FSZ walkers (#714 (c), C vmcmake_fsz.c, pfupdate_fsz.c) hold rs
// per slot and use the FSZ proposals of resident_propose (interval_kernels.cu);
// a slot then needs no spin offset (split = n).

// Complex C UpdateMAll_child_fcmp for one QP plane, by its block or one warp:
// vec1 = -invM^T sltE_a, PfM *= -vec1[a], vec2 = invM_a / -vec1[a],
// invM += vec1 vec2^T - vec2 vec1^T with the unit-vector corrections.
template <bool Warp>
__device__ inline void interval_complex_update(rcx* invq, rcx* pfq, const rcx* tab, int qp,
                                               const int* rs, int n, int n2, int msa, rcx* v1,
                                               rcx* v2, rcx* v3) {
  const int tid = Warp ? (threadIdx.x & 31) : threadIdx.x;
  const int nt = Warp ? 32 : blockDim.x;
  const int rsa = rs[msa];
  for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
  if (Warp) __syncwarp(); else __syncthreads();
  for (int i = tid; i < n; i += nt) {
    rcx acc = rc(0.0, 0.0);
    for (int j = 0; j < n; ++j) acc = rc_sub(acc, rc_mul(invq[(size_t)j * n + i], v3[j]));
    v1[i] = acc;
  }
  if (Warp) __syncwarp(); else __syncthreads();
  // v3 is dead after the dot products. Share the identical reciprocal within
  // this warp instead of evaluating Smith division in every lane.
  rcx inv_vec1_a;
  if (Warp) {
    if (tid == 0) v3[0] = rc_div(rc(-1.0, 0.0), v1[msa]);
    __syncwarp();
    inv_vec1_a = v3[0];
  } else {
    inv_vec1_a = rc_div(rc(-1.0, 0.0), v1[msa]);
  }
  for (int i = tid; i < n; i += nt) v2[i] = rc_mul(invq[(size_t)msa * n + i], inv_vec1_a);
  if (tid == 0) *pfq = rc_mul(*pfq, rc_neg(v1[msa]));
  if (Warp) __syncwarp(); else __syncthreads();
  const int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
  for (int i = warp; i < n; i += nwarp) {
    const rcx vi1 = v1[i], vi2 = v2[i];
    for (int j = lane; j < n; j += 32) {
      const size_t e = (size_t)i * n + j;
      rcx x = rc_add(invq[e], rc_sub(rc_mul(vi1, v2[j]), rc_mul(v1[j], vi2)));
      if (j == msa) x = rc_sub(x, vi2);
      if (i == msa) x = rc_add(x, v2[j]);
      invq[e] = x;
    }
  }
  if (Warp) __syncwarp(); else __syncthreads();
}

// One block per walker. State per walker: ele [n], occ [2 nsite], inv
// [nqp][n*n] (invM = -X^-1), pf [nqp], aw (accepts since rebuild), logip
// (log|ip| with the 1e-100 floor). Dynamic shared memory: aligned n ints,
// nqp complex candidate Pfaffians, 3n complex vectors per update group, 8 ints.
#define MVMC_COMPLEX_HOP_PARAMS \
    int* ele, \
    int* occ, \
    rcx* inv, \
    rcx* pf, \
    int* aw, \
    double* logip, \
    const rcx* slater, \
    const rcx* weights, \
    const int* gutz, \
    const int* jast, \
    const double* params, \
    const unsigned int* tape, \
    int total_hops, \
    int* hop_off, \
    int n, \
    int ne, \
    int nsite, \
    int nqp, \
    int ngutz, \
    int nproj, \
    int* hops_done, \
    int* rebuild, \
    int* accepts, \
    int* total_accepts, \
    int* error, \
    int* valid, \
    rcx* fac, \
    rcx* work, \
    double exchange_fraction, \
    int fsz, \
    int spin_free, \
    double flip_fraction
#define MVMC_COMPLEX_HOP_FORWARD \
    ele, occ, inv, pf, aw, logip, \
    slater, weights, gutz, jast, params, tape, \
    total_hops, hop_off, n, ne, nsite, nqp, \
    ngutz, nproj, hops_done, rebuild, accepts, total_accepts, \
    error, valid, fac, work, exchange_fraction, fsz, \
    spin_free, flip_fraction

template <bool WarpUpdates>
__device__ __forceinline__ void interval_complex_hop_body(MVMC_COMPLEX_HOP_PARAMS) {
  if (*error) return;  // block-uniform; previous launch owns the error flag
  const int nt = blockDim.x;
  // The host selects the specialized entry and its matching scratch layout.
  const bool warp_updates = WarpUpdates;
  const int update_groups = WarpUpdates ? (nt >> 5) : 1;
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  rcx* pfnew = reinterpret_cast<rcx*>(rs + ((n + 1) & ~1));  // [nqp]
  rcx* scratch = pfnew + nqp;                                 // [3n]
  int* mvw = reinterpret_cast<int*>(scratch + 3 * (size_t)n * update_groups);  // [8]
  rcx* v1 = scratch;
  rcx* v2 = v1 + n;
  rcx* v3 = v2 + n;
  const int w = blockIdx.x, tid = threadIdx.x;
  const int n2 = 2 * nsite;
  int* ele_w = ele + (size_t)w * n;
  int* occ_w = occ + (size_t)w * 2 * nsite;
  rcx* inv_w = inv + (size_t)w * nqp * n * n;
  rcx* pf_w = pf + (size_t)w * nqp;
  int* valid_w = valid + (size_t)w * nqp;
  rcx* fac_w = fac + (size_t)w * nqp * n * n;
  rcx* work_w = work + (size_t)w * nqp * n * n;
  const unsigned int* tape_w = tape + (size_t)w * 5;
  const size_t tape_stride = 5 * (size_t)gridDim.x;
  const int split = fsz ? n : ne;
  __shared__ int s_rank, s_exchange, s_partner, s_from, s_to, s_pfrom, s_pto;
  __shared__ double s_log;
  if (tid == 0) {
    int all = 1;
    for (int q = 0; q < nqp; ++q) all &= valid_w[q] != 0;
    s_rank = all;
  }
  __syncthreads();
  double log_old = logip[w];
  int window = aw[w], advanced = 0, accepted = 0, need_rebuild = 0;
  if (tid == 0) mvw[6] = 0;
  const int start = hop_off[w];
  for (int hop = start; hop < total_hops; ++hop) {
    const unsigned int* t = tape_w + (size_t)hop * tape_stride;
    if (tid == 0) {
      // No local spins in complex mode (rejected on the host), so a local
      // spin flip proposal is always invalid.
      const ResidentMove m = resident_propose(t, ele_w, occ_w, nullptr, n, ne, nsite,
                                              exchange_fraction, flip_fraction, fsz, spin_free);
      s_exchange = m.exchange;
      s_partner = m.partner;
      s_from = m.from;
      s_to = m.to;
      s_pfrom = m.pfrom;
      s_pto = m.pto;
      mvw[0] = m.valid;
      mvw[1] = m.msa;
    }
    __syncthreads();
    const int msa = mvw[1];
    const int exchange = s_exchange, partner = s_partner;
    const int from = s_from, to = s_to, pfrom = s_pfrom, pto = s_pto;
    const int ri = from % nsite, rj = to % nsite;
    if (!mvw[0]) {
      __syncthreads();
      ++advanced;
      continue;
    }
    if (tid == 0) {
      ele_w[msa] = fsz ? to : rj;
      occ_w[from] = 0;
      occ_w[to] = 1;
      if (exchange) {
        ele_w[partner] = fsz ? pto : ri;
        occ_w[pfrom] = 0;
        occ_w[pto] = 1;
      }
    }
    __syncthreads();
    for (int j = tid; j < n; j += nt) rs[j] = ele_w[j] + (j >= split ? nsite : 0);
    __syncthreads();
    if (!s_rank || exchange) {
      // Full candidate Pfaffians of every plane (Julia cooperative_overlap!).
      for (int q = 0; q < nqp; ++q) {
        rc_gather(fac_w, rs, n, -1, -1, slater, q, n2);
        const rcx p = rc_block_pfaffian(fac_w, n, scratch, scratch + n);
        if (tid == 0) pfnew[q] = p;
      }
      __syncthreads();
    } else {
      // C CalculateNewPfM2: pfMNew = -pfM sum_j invM[a][j] sltE[a][j].
      const int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
      for (int q = warp; q < nqp; q += nwarp) {
        const rcx* inv_row = inv_w + (size_t)q * n * n + (size_t)msa * n;
        const rcx* row = slater + ((size_t)q * n2 + rs[msa]) * n2;
        rcx ratio = rc(0.0, 0.0);
        for (int j = lane; j < n; j += 32) ratio = rc_add(ratio, rc_mul(inv_row[j], row[rs[j]]));
        ratio = rc_warp_sum(ratio);
        if (lane == 0) pfnew[q] = rc_neg(rc_mul(ratio, pf_w[q]));
      }
      __syncthreads();
    }
    if (tid == 0) {
      rcx ip_new = rc(0.0, 0.0);
      for (int q = 0; q < nqp; ++q) ip_new = rc_add(ip_new, rc_mul(weights[q], pfnew[q]));
      const double log_new = log(rc_abs(ip_new) + 1e-100);
      // An exchange or an on-site FSZ spin flip keeps every counter.
      const double proj =
          exchange || ri == rj
              ? 0.0
              : interval_proj_delta(occ_w, nsite, ngutz, nproj, gutz, jast, params, ri, rj);
      const double logratio = 2.0 * (proj + log_new - log_old);
      const double u = (double)t[4] / 4294967296.0;
      const int accept = rc_finite(ip_new) && !rc_zero(ip_new) && log(u) < logratio;
      mvw[5] = accept;
      int singular = 0;
      if (accept && s_rank && !exchange) {
        for (int q = 0; q < nqp; ++q) singular |= !rc_finite(pfnew[q]) || rc_zero(pfnew[q]);
      }
      mvw[7] = accept && (singular || !s_rank || exchange);
      if (accept) {
        s_log = log_new;
      } else {
        ele_w[msa] = fsz ? from : ri;
        occ_w[from] = 1;
        occ_w[to] = 0;
        if (exchange) {
          ele_w[partner] = fsz ? pfrom : rj;
          occ_w[pfrom] = 1;
          occ_w[pto] = 0;
        }
      }
    }
    __syncthreads();
    if (mvw[5] && mvw[7]) {
      const int all =
          rc_refactor_walker(inv_w, pf_w, valid_w, rs, slater, n, nqp, n2, fac_w, work_w, scratch);
      if (tid == 0) {
        rcx ip = rc(0.0, 0.0);
        for (int q = 0; q < nqp; ++q) ip = rc_add(ip, rc_mul(weights[q], pf_w[q]));
        ++accepted;
        if (!rc_finite(ip) || rc_zero(ip)) {
          atomicCAS(error, 0, RESIDENT_REBUILD);
          mvw[6] = 2;
        } else {
          log_old = log(rc_abs(ip) + 1e-100);
          window = 0;
        }
        s_rank = all;
      }
      __syncthreads();
    } else if (mvw[5]) {
      if (warp_updates) {
        const int warp = tid >> 5;
        rcx* vectors = scratch + (size_t)warp * 3 * n;
        for (int q = warp; q < nqp; q += nt >> 5)
          interval_complex_update<true>(inv_w + (size_t)q * n * n, pf_w + q, slater, q, rs,
                                        n, n2, msa, vectors, vectors + n, vectors + 2 * n);
      } else {
        for (int q = 0; q < nqp; ++q)
          interval_complex_update<false>(inv_w + (size_t)q * n * n, pf_w + q, slater, q, rs,
                                         n, n2, msa, v1, v2, v3);
      }
      if (tid == 0) {
        log_old = s_log;
        ++accepted;
        if (++window > nsite) mvw[6] = 1;
      }
      __syncthreads();
    }
    ++advanced;
    if (mvw[6]) {
      need_rebuild = mvw[6] == 1;  // 2: in-kernel refactor failed and latched
      break;
    }
  }
  if (tid == 0) {
    hop_off[w] = start + advanced;
    hops_done[w] = advanced;
    rebuild[w] = need_rebuild;
    accepts[w] = accepted;
    total_accepts[w] += accepted;
    aw[w] = window;
    logip[w] = log_old;
  }
}

extern "C" __global__ void interval_complex_hop(MVMC_COMPLEX_HOP_PARAMS) {
  interval_complex_hop_body<false>(MVMC_COMPLEX_HOP_FORWARD);
}

extern "C" __global__ void interval_complex_hop_warp(MVMC_COMPLEX_HOP_PARAMS) {
  interval_complex_hop_body<true>(MVMC_COMPLEX_HOP_FORWARD);
}
#undef MVMC_COMPLEX_HOP_PARAMS
#undef MVMC_COMPLEX_HOP_FORWARD

// Assemble the complex plane A = -S(rs, rs) of a requested walker and factor
// it with a paired panel factorization (n>=64) or the C-order reference.
// Failure paths retain the original PFAPACK status through a device-only fallback.
extern "C" __global__ void interval_complex_factor(
    const int* rebuild, const int* error, const int* ele, const Cx<double>* slater,
    Cx<double>* A, Cx<double>* W, Cx<double>* OUT, Cx<double>* PF, int* ST,
    int n, int ne, int nsite, int nqp) {
  const int plane = blockIdx.x, w = plane / nqp, q = plane % nqp;
  if (*error || !rebuild[w]) return;
  Cx<double>* a = A + (size_t)plane * n * n;
  const int* e = ele + (size_t)w * n;
  for (int index = threadIdx.x; index < n * n; index += blockDim.x) {
    const int i = index / n, j = index % n;
    const int ri = e[i] + (i >= ne ? nsite : 0);
    const int rj = e[j] + (j >= ne ? nsite : 0);
    a[index] = -SLT(slater, q, ri, rj, 2 * nsite);
  }
  __syncthreads();
  body_c64(A, W, OUT, PF, ST, n);
}

// interval_resident_commit with complex tables: publish a walker whose
// weighted overlap is finite and nonzero, with per-plane validity; a zero or
// nonfinite total latches the sticky error and keeps every previous table.
extern "C" __global__ void interval_complex_commit(
    const int* rebuild, int* error, const int* ST, const rcx* OUT, const rcx* PF,
    const rcx* weights, rcx* inv, rcx* pf, double* logip, int* aw, int* valid, int n, int nqp) {
  const int w = blockIdx.x;
  if (!rebuild[w]) return;
  __shared__ int good;
  __shared__ double log_overlap;
  if (threadIdx.x == 0) {
    rcx ip = rc(0.0, 0.0);
    for (int q = 0; q < nqp; ++q) ip = rc_add(ip, rc_mul(weights[q], PF[w * nqp + q]));
    good = rc_finite(ip) && !rc_zero(ip);
    log_overlap = log(rc_abs(ip) + 1e-100);
    if (!good) atomicCAS(error, 0, RESIDENT_REBUILD);
  }
  __syncthreads();
  if (!good) return;
  const size_t base = (size_t)w * nqp * n * n;
  for (size_t i = threadIdx.x; i < (size_t)nqp * n * n; i += blockDim.x)
    inv[base + i] = rc_neg(OUT[base + i]);
  for (int q = threadIdx.x; q < nqp; q += blockDim.x) {
    const rcx p = PF[w * nqp + q];
    pf[w * nqp + q] = p;
    valid[w * nqp + q] = ST[w * nqp + q] == 0 && rc_finite(p) && !rc_zero(p);
  }
  if (threadIdx.x == 0) {
    logip[w] = log_overlap;
    aw[w] = 0;
  }
}
