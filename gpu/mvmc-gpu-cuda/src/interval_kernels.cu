#ifndef SLT
#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])
#endif

// Device-resident interval sampler (issue #547).
//
// One block advances one walker's configuration for a whole interval of ordinary
// hopping proposals on the device, with the candidate selection, the Gutzwiller/Jastrow
// projection delta and the Metropolis decision all on the device. This removes the
// per-hop host<->device round trip that dominates the production lock-step sampler
// (`k_propose`/`k_accept` driven once per hop from the host).
//
// The legacy entry retains `k_propose` / `k_accept`'s separate products and
// ordered sums. At n >= 128 the resident entry follows Julia GPU's FMA
// arithmetic and four queued rank-two updates with warp-reduced ratios. The
// supplied tape, projection delta and log-space decision contract are shared.
//
// Proposal contract: the tape is the Julia GPU sampler's device tape
// (MVMCOptimizers.jl/ext/cuda_sampling_update.jl:cached_sampling_kernel!), so a tape can be
// shared with `gpu_sample!(s; tape=...)` for parity. A device u32 tape of 5 words per step,
// laid out [word][walker][step] (word-fastest, then walker, then step), linear index
// 5*(w + W*step) + word:
//   w0 -> source electron slot index: mi = w0 % ne
//   w1 -> spin sector (top bit): spin = w1 >> 31
//   w2 -> hop: destination rank among the empty sites of that spin without a local spin
//         (the Julia `empty_destinations` proposal, the default for nsite >= 64);
//         exchange: opposite-spin partner slot, w2 % ne
//   w3 -> exchange threshold: exchange = w3/2^32 < exchange_fraction (Julia GPU
//         0 / 0.5 / 1 for NExUpdatePath 0 / 1 / 2, cuda_runners.jl:52-58)
//   w4 -> acceptance uniform: u = w4 / 2^32, accept when log(u) < logratio
//   Every step consumes exactly 5 words, valid or not, so the tape length is fixed and
//   decision independent.
//
// Local spins and exchanges follow cuda_sampling_update.jl:215-341 and exist only on
// the resident entry; the legacy entry is hopping-only without local spins. A hop
// from a local-spin site is invalid. An exchange swaps an electron with an
// opposite-spin partner when both sites are singly occupied. Its overlap is
// recomputed from full Pfaffians and an accept refactors the walker, as Julia does.
// The projection counts (doublons, charges) are unchanged by an exchange.
// Only the real, non-FSZ normal mode is handled; the caller rejects anything else.

// ---------------------------------------------------------------------------------------------
// One-electron rank-one ratio for one QP plane: r = -pf_qp * sum_j inv[msa*n+j] * S[rsa][rs[j]].
// `rs[j]` is the candidate site index (spin offset included). Parallel products, serial sum.
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// One-electron rank-two inverse update (the k_accept `!two` branch) for one QP plane.
// ---------------------------------------------------------------------------------------------
// All groups reach every block barrier, including an inactive final QP tile.
// Immediate updates retain the original separate products and ascending-j order.
__device__ inline void interval_hop_update(double* invq, const double* tab, int qp, const int* rs,
                                           int n, int n2, int msa, double* v1, double* v2,
                                           double* v3, int tid, int nt, bool active) {
  const int rsa = rs[msa];
  if (active) {
    for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
  }
  __syncthreads();
  if (active) {
    for (int i = tid; i < n; i += nt) {
      double acc = 0.0;
#pragma unroll 8
      for (int j = 0; j < n; ++j) acc += -invq[(size_t)j * n + i] * v3[j];
      v1[i] = acc;
    }
  }
  __syncthreads();
  if (active) {
    const double inv_vec1_a = -1.0 / v1[msa];
    for (int i = tid; i < n; i += nt) v2[i] = invq[(size_t)msa * n + i] * inv_vec1_a;
  }
  __syncthreads();
  if (active) {
    const int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    for (int i = warp; i < n; i += nwarp) {
      const double vi1 = v1[i], vi2 = v2[i];
      const int rowoff = i * n;
      for (int j = lane; j < n; j += 32) {
        const int e = rowoff + j;
        double x = invq[e];
        x += vi1 * v2[j] - v1[j] * vi2;
        if (j == msa) x -= vi2;
        if (i == msa) x += v2[j];
        invq[e] = x;
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Gutzwiller/Jastrow projection delta of a hop ri -> rj, using the post-hop occupancy
// `occ` ([site] up then [nsite+site] down). Collapses update_proj_cnt + log_proj_ratio:
// only the touched counters change, so the delta is
//   sum_i params[i] * (new_count[i] - old_count[i]).
// ---------------------------------------------------------------------------------------------
__device__ inline double interval_proj_delta(const int* occ, int nsite, int ngutz, int nproj,
                                             const int* gutz, const int* jast,
                                             const double* params, int ri, int rj) {
  double delta = 0.0;
  const int n0ri = occ[ri], n1ri = occ[nsite + ri];
  const int n0rj = occ[rj], n1rj = occ[nsite + rj];
  if (ngutz > 0) {
    if (gutz != nullptr) {
      const int ii = gutz[ri];
      if (ii >= 0 && ii < nproj) delta -= params[ii] * (double)(n0ri + n1ri);
      const int ij = gutz[rj];
      if (ij >= 0 && ij < nproj) delta += params[ij] * (double)(n0rj * n1rj);
    } else if (nproj > 0) {
      delta += params[0] * (double)(n0rj * n1rj - (n0ri + n1ri));
    }
  }
  if (jast != nullptr) {
    const int offset = ngutz;
    const int xi_ri = n0ri + n1ri, xi_rj = n0rj + n1rj;
    const int a = ri < rj ? ri : rj, b = ri < rj ? rj : ri;
    const int idx = jast[a * nsite + b];
    if (idx >= 0 && offset + idx < nproj) {
      delta += params[offset + idx] * (double)(xi_ri - xi_rj + 1);
    }
    for (int rk = 0; rk < nsite; ++rk) {
      if (rk == ri || rk == rj) continue;
      const double x = (double)(occ[rk] + occ[nsite + rk] - 1);
      if (x == 0.0) continue;
      const int a1 = ri < rk ? ri : rk, b1 = ri < rk ? rk : ri;
      const int ia = jast[a1 * nsite + b1];
      if (ia >= 0 && offset + ia < nproj) delta -= params[offset + ia] * x;
      const int a2 = rj < rk ? rj : rk, b2 = rj < rk ? rk : rj;
      const int ib = jast[a2 * nsite + b2];
      if (ib >= 0 && offset + ib < nproj) delta += params[offset + ib] * x;
    }
  }
  return delta;
}

// ---------------------------------------------------------------------------------------------
// The fused interval kernel. One block per walker; `threads` should be >= n (clamped).
//
// State (per walker, resident across calls):
//   ele  [W][n]          configuration (site index per electron slot)
//   occ  [W][2*nsite]    occupancy (ele_num)
//   inv  [W][nqp][n*n]   mVMC inverse (invM = -X^-1)
//   pf   [W][nqp]        Pfaffians
//   aw   [W]             accepted moves since the last rebuild (in/out)
//   logip[W]             cached log|inner product| (in/out)
//
// Returns (per walker, written by thread 0):
//   hops_done[W]   hops actually advanced (< max_hops when a rebuild was requested)
//   rebuild[W]     1 when `aw` passed `nsite`, or an accepted move has a zero/nonfinite
//                  candidate plane Pfaffian (host must recompute before continuing)
//
// Shared memory: aligned n ints, nqp+1 doubles, reusable ratio/update scratch, 8 ints.
// ---------------------------------------------------------------------------------------------
template <bool resident = false>
__device__ __forceinline__ void interval_hop_impl(
    int* ele, int* occ, double* inv, double* pf, int* aw, double* logip,
    const double* slater, const double* weights, const int* gutz, const int* jast,
    const double* params, const unsigned int* tape, int total_hops, int* hop_off,
    int n, int ne, int nsite, int nqp, int qp_groups, int ngutz, int nproj, double exchange_fraction,
    int* hops_done, int* rebuild, int* accepts, double* delayed = nullptr, double* flush = nullptr,
    int* valid = nullptr, double* fac = nullptr, double* work = nullptr, int* error = nullptr,
    const int* loc = nullptr, const int* dh = nullptr, int n2dh = 0, int n4dh = 0,
    const int* rbm = nullptr, const double* rbmw = nullptr, double* theta = nullptr,
    int nneuron = 0) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  double* sh = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  double* pfnew = sh;        // [nqp]
  double* out = sh + nqp;    // [1]
  double* scratch = out + 1;
  const int w = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nwarp = nt >> 5;
  const int ratio_vectors = nwarp < nqp ? nwarp : nqp;
  const int scratch_vectors = ratio_vectors > 3 * qp_groups ? ratio_vectors : 3 * qp_groups;
  int* mvw = reinterpret_cast<int*>(scratch + (size_t)scratch_vectors * n);  // [8]
  const int group_threads = nt / qp_groups;
  const int group = tid / group_threads, group_tid = tid % group_threads;
  double* v1 = scratch + (size_t)group * 3 * n;
  double* v2 = v1 + n;
  double* v3 = v2 + n;

  const int n2 = 2 * nsite;
  const double* tab = slater;
  int* ele_w = ele + (size_t)w * n;
  int* occ_w = occ + (size_t)w * 2 * nsite;
  double* inv_w = inv + (size_t)w * nqp * n * n;
  double* pf_w = pf + (size_t)w * nqp;
  const bool queue = resident && delayed != nullptr;
  const int packet = 2*INTERVAL_DELAY*n+2*INTERVAL_DELAY+1;
  double* delay_w = queue ? delayed+(size_t)w*nqp*packet : nullptr;
  double* original_inv = inv_w;
  double* alternate = queue ? flush+(size_t)w*nqp*n*n : nullptr;
  int* sites = mvw+8;
  int pending = 0;
  // Julia tape layout [word][walker][step]: 5*(walker + walkers*step) + word.
  const unsigned int* tape_w = tape + (size_t)w * 5;
  const size_t tape_stride = 5 * (size_t)gridDim.x;

  // Per-plane recovery (#573) on the resident path: a walker with an invalid
  // plane proposes with full Pfaffians and refactors in this kernel on every
  // accept, as Julia's cached sampler does (cuda_sampling_update.jl:291-371).
  const bool recover = resident && valid != nullptr;
  int* valid_w = recover ? valid + (size_t)w * nqp : nullptr;
  double* fac_w = recover ? fac + (size_t)w * nqp * n * n : nullptr;
  double* work_w = recover ? work + (size_t)w * nqp * n * n : nullptr;
  // Exchanges and local spins need the resident full-Pfaffian/refactor path.
  const double exchange_p = recover ? exchange_fraction : 0.0;
  if (!recover) loc = nullptr;
  __shared__ int s_rank, s_exchange, s_partner;
  if (tid == 0) {
    int all = 1;
    if (recover) for (int q = 0; q < nqp; ++q) all &= valid_w[q] != 0;
    s_rank = all;
  }
  __syncthreads();

  double log_old = logip[w];
  int window = aw[w];
  int advanced = 0;
  int accepted = 0;
  int need_rebuild = 0;
  mvw[6] = 0;
  const int start = hop_off[w];
  for (int hop = start; hop < total_hops; ++hop) {
    const unsigned int* t = tape_w + (size_t)hop * tape_stride;
    if (tid == 0) {
      const int mi = (int)(t[0] % (unsigned)ne);
      const int spin = (int)(t[1] >> 31);
      const int ri = ele_w[mi + spin * ne];
      const int exchange = (double)t[3] / 4294967296.0 < exchange_p;
      int rj = 0, valid, partner = -1;
      if (exchange) {
        // Julia: partner of the other spin; neither site may hold the moved spin's
        // counterpart (cuda_sampling_update.jl:269-279).
        partner = (int)(t[2] % (unsigned)ne) + (1 - spin) * ne;
        rj = ele_w[partner];
        valid = ri != rj && occ_w[rj + spin * nsite] == 0 && occ_w[ri + (1 - spin) * nsite] == 0;
      } else {
        // empty_hopping_destination: empty sites of this spin without a local spin.
        int count = 0;
        for (int r = 0; r < nsite; ++r) {
          if (occ_w[r + spin * nsite] == 0 && (loc == nullptr || loc[r] == 0)) ++count;
        }
        valid = count > 0 && (loc == nullptr || loc[ri] == 0);
        if (count > 0) {
          int rank = (int)(t[2] % (unsigned)count);
          for (int r = 0; r < nsite; ++r) {
            if (occ_w[r + spin * nsite] == 0 && (loc == nullptr || loc[r] == 0)) {
              if (rank == 0) {
                rj = r;
                break;
              }
              --rank;
            }
          }
        }
      }
      s_exchange = exchange;
      s_partner = partner;
      mvw[0] = valid;
      mvw[1] = mi;
      mvw[2] = spin;
      mvw[3] = ri;
      mvw[4] = rj;
    }
    __syncthreads();
    const int mi = mvw[1], spin = mvw[2], ri = mvw[3], rj = mvw[4];
    const int exchange = s_exchange, partner = s_partner;
    if (!mvw[0]) {
      // Invalid proposals are frequent with local spins and exchanges: every
      // thread must read this step's move before thread 0 writes the next one.
      __syncthreads();
      ++advanced;
      continue;
    }
    if (tid == 0) {
      ele_w[mi + spin * ne] = rj;
      occ_w[ri + spin * nsite] = 0;
      occ_w[rj + spin * nsite] = 1;
      if (exchange) {
        ele_w[partner] = ri;
        occ_w[rj + (1 - spin) * nsite] = 0;
        occ_w[ri + (1 - spin) * nsite] = 1;
      }
    }
    __syncthreads();
    const int msa = mi + spin * ne;
    for (int j = tid; j < n; j += nt) rs[j] = ele_w[j] + (j >= ne ? nsite : 0);
    __syncthreads();

    if (resident && (!s_rank || exchange)) {
      // Julia cooperative_overlap!: every plane from the candidate sites.
      for (int q = 0; q < nqp; ++q) {
        resident_gather(fac_w, rs, n, -1, -1, tab, q, n2);
        const double p = resident_block_pfaffian(fac_w, n, scratch, scratch + n);
        if (tid == 0) pfnew[q] = p;
      }
      __syncthreads();
    } else {
      const int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;
      for (int q = warp; q < nqp; q += nwarp) {
        const double* invq = inv_w + (size_t)q * n * n;
        double* sc = scratch + (size_t)warp * n;
        const int rsa = rs[msa];
        const double* row = tab + (((size_t)q * n2) + rsa) * n2;
        const double* inv_row = invq + (size_t)msa * n;
        if (queue) {
          interval_delayed_ratio(invq,delay_w+q*packet,sites,pending,row,rs,n,msa,lane,pf_w[q],pfnew+q);
        } else {
          for (int j = lane; j < n; j += 32) sc[j] = inv_row[j] * row[rs[j]];
          __syncwarp();
          if (lane == 0) {
            double ratio = 0.0;
#pragma unroll 8
            for (int j = 0; j < n; ++j) ratio += sc[j];
            pfnew[q] = -ratio * pf_w[q];
          }
        }
        __syncwarp();  // lane 0 must finish reading before the next QP reuses scratch
      }
      __syncthreads();
    }

    // Doublon-holon change of a hop (resident only; an exchange keeps every
    // site's charge, so no DH counter changes). Block-parallel over the DH
    // items, then a fixed-order warp and warp-partial reduction. The ratio
    // scratch is free here: the update reuses it only after the decision.
    double dh_delta = 0.0;
    if (resident && dh != nullptr && !exchange) {
      const ResidentOcc after{occ_w, nsite};
      const ResidentOccBeforeHop before{occ_w, nsite, ri, rj, spin};
      double part = resident_dh_delta(before, after, dh, n2dh, n4dh, nsite,
                                      nproj - 6 * n2dh - 10 * n4dh, params, tid, nt);
      for (int d = 16; d; d >>= 1) part += __shfl_down_sync(0xffffffff, part, d);
      if ((tid & 31) == 0) scratch[tid >> 5] = part;
      __syncthreads();
      if (tid == 0) for (int i = 0; i < nwarp; ++i) dh_delta += scratch[i];
    }

    // RBM log-ratio of the move (resident only): C UpdateRBMCnt applied to
    // the cached hidden fields, then LogRBMRatio. An exchange is C's two hops
    // (ri -> rj with spin s, then rj -> ri with 1-s, vmcmake.c:250-260).
    const bool use_rbm = resident && rbm != nullptr;
    double* theta_w = use_rbm ? theta + (size_t)w * nneuron : nullptr;
    const int hops = exchange ? 2 : 1;
    const int hop_i[2] = {ri, rj}, hop_j[2] = {rj, ri}, hop_s[2] = {spin, 1 - spin};
    double rbm_delta = 0.0;
    if (use_rbm) {
      __syncthreads();  // the DH reduction above may still be reading scratch
      double part = resident_rbm_hidden_delta(rbm, rbmw, theta_w, nsite, nneuron, hop_i, hop_j,
                                              hop_s, hops, tid, nt);
      for (int d = 16; d; d >>= 1) part += __shfl_down_sync(0xffffffff, part, d);
      if ((tid & 31) == 0) scratch[tid >> 5] = part;
      __syncthreads();
      if (tid == 0) {
        rbm_delta = resident_rbm_phys_delta(rbm, rbmw, nsite, hop_i, hop_j, hop_s, hops);
        for (int i = 0; i < nwarp; ++i) rbm_delta += scratch[i];
      }
    }

    if (tid == 0) {
      double ip_new = 0.0;
      for (int q = 0; q < nqp; ++q) ip_new = __dadd_rn(ip_new, __dmul_rn(weights[q], pfnew[q]));
      const double log_new = log(fabs(ip_new) + 1e-100);
      // An exchange keeps every site's charge and doublon count.
      const double proj = (exchange ? 0.0
          : interval_proj_delta(occ_w, nsite, ngutz, nproj, gutz, jast, params, ri, rj) + dh_delta)
          + rbm_delta;
      const double logratio = 2.0 * (proj + log_new - log_old);
      const double u = (double)t[4] / 4294967296.0;
      // Keep log-space decisions for finite overlaps (including u == 0 and
      // extreme ratios), but never publish an overflowed weighted sum (#564).
      const int accept = isfinite(ip_new) && ip_new != 0.0 && log(u) < logratio;
      mvw[5] = accept;
      // A finite nonzero total can hide a candidate plane whose Pfaffian is 0 or
      // nonfinite. Its rank-two update divides by that plane's ratio, so the
      // accept instead requests an immediate walker rebuild (Julia
      // cuda_sampling_update.jl:343-371). The rebuild latches a singular plane.
      // The resident path refactors in this kernel; the legacy host-driven
      // path returns a rebuild request instead.
      int singular = 0;
      if (accept && s_rank && !exchange) {
        for (int q = 0; q < nqp; ++q) singular |= !isfinite(pfnew[q]) || pfnew[q] == 0.0;
      }
      // Julia refactors after every accepted exchange (cuda_sampling_update.jl:341).
      mvw[7] = accept && (singular || !s_rank || exchange);
      if (accept) {
        out[0] = log_new;
      } else {
        ele_w[mi + spin * ne] = ri;
        occ_w[ri + spin * nsite] = 1;
        occ_w[rj + spin * nsite] = 0;
        if (exchange) {
          ele_w[partner] = rj;
          occ_w[rj + (1 - spin) * nsite] = 1;
          occ_w[ri + (1 - spin) * nsite] = 0;
        }
      }
    }
    __syncthreads();
    // An accepted move advances every hidden field (C copies rbmCntNew).
    // Each thread rewrites the neurons it read above.
    if (use_rbm && mvw[5]) {
      for (int h = tid; h < nneuron; h += nt) {
        double t = theta_w[h];
        for (int k = 0; k < hops; ++k)
          t = resident_rbm_hop(rbm, rbmw, nsite, nneuron, h, t, hop_i[k], hop_j[k], hop_s[k]);
        theta_w[h] = t;
      }
    }
    if (mvw[5] && mvw[7] && recover) {
      // Queued updates belong to the replaced inverse; discard, do not flush.
      pending = 0;
      const int all = resident_refactor_walker(inv_w, pf_w, valid_w, rs, tab, n, nqp, n2,
                                               fac_w, work_w, scratch);
      if (tid == 0) {
        double ip = 0.0;
        for (int q = 0; q < nqp; ++q) ip = __dadd_rn(ip, __dmul_rn(weights[q], pf_w[q]));
        ++accepted;
        if (!isfinite(ip) || ip == 0.0) {
          atomicCAS(error, 0, RESIDENT_REBUILD);
          mvw[6] = 2;
        } else {
          log_old = log(fabs(ip) + 1e-100);
          window = 0;
        }
        s_rank = all;
      }
      __syncthreads();
    } else if (mvw[5] && mvw[7]) {
      // No inverse/Pfaffian update; the rebuild refactors the accepted configuration.
      if (tid == 0) {
        log_old = out[0];
        ++accepted;
        ++window;
        mvw[6] = 1;
      }
      __syncthreads();
    } else if (mvw[5]) {
      for (int base = 0; base < nqp; base += qp_groups) {
        const int q = base + group;
        const bool active = q < nqp;
        double* invq = active ? inv_w + (size_t)q * n * n : inv_w;
        if (queue)
          interval_delayed_update(invq,delay_w+(active?q:0)*packet,sites,pending,tab,q,rs,n,n2,msa,
                                  v1,v2,v3,group_tid,group_threads,active);
        else
          interval_hop_update(invq, tab, q, rs, n, n2, msa, v1, v2, v3,
                                    group_tid, group_threads, active);
        // Publish the Pfaffian with the pivot that updated this inverse. The
        // immediate path follows C UpdateMAll (PfM *= -vec1[msa]); the queue
        // pivots with the proposal ratio itself (Julia). Mixing the column
        // contraction pivot with the row-dot proposal ratio leaves an
        // unrecoverable error after a near-singular plane (#656).
        if (active && group_tid == 0) pf_w[q] = queue ? pfnew[q] : pf_w[q] * -v1[msa];
      }
      __syncthreads();
      if (queue) {
        if (tid == 0) sites[pending] = msa;
        __syncthreads();
        if (++pending == INTERVAL_DELAY) {
          interval_delayed_flush(inv_w,alternate,delay_w,sites,pending,n,nqp,tid,nt);
          pending = 0;
        }
      }
      if (tid == 0) {
        log_old = out[0];
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

  if (queue) {
    interval_delayed_flush(inv_w,alternate,delay_w,sites,pending,n,nqp,tid,nt);
    if (inv_w != original_inv) {
      for (size_t e=tid; e<(size_t)nqp*n*n; e+=nt) original_inv[e]=inv_w[e];
      __syncthreads();
    }
  }
  if (tid == 0) {
    hop_off[w] = start + advanced;
    hops_done[w] = advanced;
    rebuild[w] = need_rebuild;
    accepts[w] = accepted;
    aw[w] = window;
    logip[w] = log_old;
  }
}

// Compile the common matrix side into the ordered loops and address arithmetic.
// Other geometries retain the same general kernel. No arithmetic is reassociated.
extern "C" __global__ void k_interval_hop(
    int* ele, int* occ, double* inv, double* pf, int* aw, double* logip,
    const double* slater, const double* weights, const int* gutz, const int* jast,
    const double* params, const unsigned int* tape, int total_hops, int* hop_off,
    int n, int ne, int nsite, int nqp, int qp_groups, int ngutz, int nproj, double exchange_fraction,
    int* hops_done, int* rebuild, int* accepts) {
  if (n == 128) {
    interval_hop_impl(
      ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast, params, tape,
      total_hops, hop_off, 128, ne, nsite, nqp, qp_groups, ngutz, nproj,
      exchange_fraction, hops_done, rebuild, accepts);
  } else {
    interval_hop_impl(
      ele, occ, inv, pf, aw, logip, slater, weights, gutz, jast, params, tape,
      total_hops, hop_off, n, ne, nsite, nqp, qp_groups, ngutz, nproj,
      exchange_fraction, hops_done, rebuild, accepts);
  }
}
