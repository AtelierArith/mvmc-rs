// Device-resident interval sampler (issue #547).
//
// One block advances one walker's configuration for a whole interval of ordinary
// hopping proposals on the device, with the candidate selection, the Gutzwiller/Jastrow
// projection delta and the Metropolis decision all on the device. This removes the
// per-hop host<->device round trip that dominates the production lock-step sampler
// (`k_propose`/`k_accept` driven once per hop from the host).
//
// The arithmetic of the rank-one ratio and the rank-two inverse update is the same as
// `k_propose` / `k_accept` (sampler_kernels.cu): products are formed in parallel and
// summed in index order by one thread, so each serial sum keeps the host's C-order
// rounding while the gather latency is spread over the block.
//
// Proposal contract (a separate GPU RNG contract, like the Julia GPU path):
//   A fixed tape of u32 words, 4 per hop, laid out [word][walker][hop]:
//     w0 -> source electron slot index: mi = w0 % ne
//     w1 -> spin sector (top bit): spin = w1 >> 31
//     w2 -> destination rank among the empty sites of that spin: rj = rank-th empty site
//     w3 -> acceptance uniform: u = w3 / 2^32
//   Every hop consumes exactly 4 words and is always a valid hop (the destination is
//   chosen from the empty set), so the tape length is fixed and decision independent.
//
// Only the real, non-FSZ, ordinary-hopping normal mode without local spins is handled;
// the caller rejects anything else before launching.

// ---------------------------------------------------------------------------------------------
// One-electron rank-one ratio for one QP plane: r = -pf_qp * sum_j inv[msa*n+j] * S[rsa][rs[j]].
// `rs[j]` is the candidate site index (spin offset included). Parallel products, serial sum.
// ---------------------------------------------------------------------------------------------
__device__ inline void interval_hop_ratio(const double* tab, const double* invq, double pfq,
                                          int qp, const int* rs, int n, int n2, int msa,
                                          double* scratch, int tid, int nt, double* out) {
  const int rsa = rs[msa];
  const double* row = tab + (((size_t)qp * n2) + rsa) * n2;
  const double* inv_row = invq + (size_t)msa * n;
  for (int j = tid; j < n; j += nt) scratch[j] = inv_row[j] * row[rs[j]];
  __syncthreads();
  if (tid == 0) {
    double ratio = 0.0;
#pragma unroll 8
    for (int j = 0; j < n; ++j) ratio += scratch[j];
    *out = -ratio * pfq;
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// One-electron rank-two inverse update (the k_accept `!two` branch) for one QP plane.
// ---------------------------------------------------------------------------------------------
__device__ inline void interval_hop_update(double* invq, const double* tab, int qp, const int* rs,
                                           int n, int n2, int msa, double* v1, double* v2,
                                           double* v3, int tid, int nt) {
  const int rsa = rs[msa];
  for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
  __syncthreads();
  // vec1[msi] = sum_msj (-inv[msj*n+msi]) * slt[msj]  (column-major sweep, msj ascending)
  for (int i = tid; i < n; i += nt) {
    double acc = 0.0;
#pragma unroll 8
    for (int j = 0; j < n; ++j) acc += -invq[(size_t)j * n + i] * v3[j];
    v1[i] = acc;
  }
  __syncthreads();
  const double inv_vec1_a = -1.0 / v1[msa];
  for (int i = tid; i < n; i += nt) v2[i] = invq[(size_t)msa * n + i] * inv_vec1_a;
  __syncthreads();
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
//   rebuild[W]     1 when `aw` passed `nsite` (host must recompute before continuing)
//
// Shared memory: n ints (rs), then (nqp + 1 + 3n) doubles, then 8 ints (mv).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void k_interval_hop(
    int* ele, int* occ, double* inv, double* pf, int* aw, double* logip, int* mv,
    const double* const* slater_tab, const double* weights, const int* gutz, const int* jast,
    const double* params, const unsigned int* tape, int total_hops, int hop_offset, int max_hops,
    int n, int ne, int nsite, int nqp, int ngutz, int nproj, int* hops_done, int* rebuild) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  double* sh = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  double* pfnew = sh;        // [nqp]
  double* out = sh + nqp;    // [1]
  double* v1 = out + 1;
  double* v2 = v1 + n;
  double* v3 = v2 + n;
  int* mvw = reinterpret_cast<int*>(sh + nqp + 1 + 3 * n);  // [8]

  const int w = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int n2 = 2 * nsite;
  const double* tab = slater_tab[w];
  int* ele_w = ele + (size_t)w * n;
  int* occ_w = occ + (size_t)w * 2 * nsite;
  double* inv_w = inv + (size_t)w * nqp * n * n;
  double* pf_w = pf + (size_t)w * nqp;
  const unsigned int* tape_w = tape + (size_t)w * total_hops * 4;

  double log_old = logip[w];
  int window = aw[w];
  int advanced = 0;
  int need_rebuild = 0;
  mvw[6] = 0;
  for (int hop = 0; hop < max_hops; ++hop) {
    const unsigned int* t = tape_w + (size_t)(hop_offset + hop) * 4;
    if (tid == 0) {
      const int mi = (int)(t[0] % (unsigned)ne);
      const int spin = (int)(t[1] >> 31);
      const int ri = ele_w[mi + spin * ne];
      int count = 0;
      for (int r = 0; r < nsite; ++r) {
        if (occ_w[r + spin * nsite] == 0) ++count;
      }
      int rj = 0;
      const int valid = count > 0;
      if (valid) {
        int rank = (int)(t[2] % (unsigned)count);
        for (int r = 0; r < nsite; ++r) {
          if (occ_w[r + spin * nsite] == 0) {
            if (rank == 0) {
              rj = r;
              break;
            }
            --rank;
          }
        }
      }
      mvw[0] = valid;
      mvw[1] = mi;
      mvw[2] = spin;
      mvw[3] = ri;
      mvw[4] = rj;
    }
    __syncthreads();
    if (!mvw[0]) {
      ++advanced;
      continue;
    }
    const int mi = mvw[1], spin = mvw[2], ri = mvw[3], rj = mvw[4];
    if (tid == 0) {
      ele_w[mi + spin * ne] = rj;
      occ_w[ri + spin * nsite] = 0;
      occ_w[rj + spin * nsite] = 1;
    }
    __syncthreads();
    const int msa = mi + spin * ne;
    for (int j = tid; j < n; j += nt) rs[j] = ele_w[j] + (j >= ne ? nsite : 0);
    __syncthreads();

    for (int q = 0; q < nqp; ++q) {
      interval_hop_ratio(tab, inv_w + (size_t)q * n * n, pf_w[q], q, rs, n, n2, msa, v3, tid, nt,
                         out);
      if (tid == 0) pfnew[q] = out[0];
      __syncthreads();
    }

    if (tid == 0) {
      double ip_new = 0.0;
      for (int q = 0; q < nqp; ++q) ip_new += weights[q] * pfnew[q];
      const double log_new = log(ip_new < 0.0 ? -ip_new : ip_new);
      const double proj =
          interval_proj_delta(occ_w, nsite, ngutz, nproj, gutz, jast, params, ri, rj);
      double weight = exp(2.0 * (proj + log_new - log_old));
      if (!isfinite(weight)) weight = -1.0;
      const double u = (double)t[3] / 4294967296.0;
      const int accept = weight > u;
      mvw[5] = accept;
      if (accept) {
        out[0] = log_new;
      } else {
        ele_w[mi + spin * ne] = ri;
        occ_w[ri + spin * nsite] = 1;
        occ_w[rj + spin * nsite] = 0;
      }
    }
    __syncthreads();
    if (mvw[5]) {
      for (int q = 0; q < nqp; ++q) {
        interval_hop_update(inv_w + (size_t)q * n * n, tab, q, rs, n, n2, msa, v1, v2, v3, tid, nt);
        if (tid == 0) pf_w[q] = pfnew[q];
      }
      __syncthreads();
      if (tid == 0) {
        log_old = out[0];
        if (++window > nsite) mvw[6] = 1;
      }
      __syncthreads();
    }
    ++advanced;
    if (mvw[6]) {
      need_rebuild = 1;
      break;
    }
  }

  if (tid == 0) {
    hops_done[w] = advanced;
    rebuild[w] = need_rebuild;
    aw[w] = window;
    logip[w] = log_old;
  }
}
