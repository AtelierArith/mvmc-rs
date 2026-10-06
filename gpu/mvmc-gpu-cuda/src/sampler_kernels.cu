// Device-resident sampler kernels (issue #434), real (f64) normal mode.
//
// Ports of crates/mvmc-core/src/sampling/updates.rs (rank-one / rank-two ratios and inverse
// updates) and of the plane assembly of crates/mvmc-core/src/pfaffian.rs, in the CPU operation
// order. Compile with --fmad=false (explicit fma() calls below reproduce Rust's mul_add).
//
// Layout (all per walker w, QP plane q, n = 2*Ne, n2 = 2*Nsite):
//   inv   [w][q][n*n]   mVMC invM = -X^-1, element (row r, col c) at r*n + c as the CPU flat table
//   pf    [w][q]        current Pfaffians
//   pfnew [w][q]        Pfaffians of the pending proposal
//   ele   [w][n]        accepted configuration (site index per electron slot)
//   pend  [w][8]        pending proposal: slotA, siteA, spinA, slotB(-1: none), siteB, spinB
//   slater_tab[w]       pointer to the walker's Slater table [q][n2][n2]
//
// Spin of slot j: j >= Ne -> spin 1 (site + Nsite), exactly as the CPU kernels.

#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])

__device__ inline int cand_site(const int* ele_w, const int* mv, int j) {
  // mv = {slotA, siteA, spinA, slotB, siteB, spinB}
  if (j == mv[0]) return mv[1];
  if (mv[3] >= 0 && j == mv[3]) return mv[4];
  return ele_w[j];
}

// ---------------------------------------------------------------------------------------------
// proposals: one block per (item, qp). items: 8 ints each (walker, slotA, siteA, spinA, slotB,
// siteB, spinB, pad). Writes out[out_off + item*nqp + qp] and pfnew[w][qp]; records the pending
// move. The products of every serial sum are formed in parallel (a product's rounding does not
// depend on its position) and then summed in index order by one thread, so each sum has exactly
// the CPU rounding while the gather latency is spread over the block.
// Dynamic shared memory: n ints + (8*n + 8) doubles.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void k_propose(const int* items, int count, const double* const* slater_tab,
                                     const int* ele, const double* inv, const double* pf,
                                     double* pfnew, int* pend, double* out, int out_off,
                                     int n, int ne, int nsite, int nqp) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  double* sh = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  double* t0 = sh;           // products / vec_a
  double* t1 = sh + n;       // products / vec_b
  double* t2 = sh + 2 * n;
  double* t3 = sh + 3 * n;
  double* t4 = sh + 4 * n;   // dots
  double* sc = sh + 5 * n;   // scalars

  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int* it = items + (size_t)item * 8;
  const int w = it[0];
  const int mv[6] = {it[1], it[2], it[3], it[4], it[5], it[6]};
  const int* ele_w = ele + (size_t)w * n;
  const double* tab = slater_tab[w];
  const int n2 = 2 * nsite;
  const double* invq = inv + ((size_t)w * nqp + qp) * (size_t)n * n;
  const double pfq = pf[(size_t)w * nqp + qp];
  const int tid = threadIdx.x, nt = blockDim.x;
  if (qp == 0 && tid < 6) pend[(size_t)w * 8 + tid] = mv[tid];
  for (int j = tid; j < n; j += nt) rs[j] = cand_site(ele_w, mv, j) + (j >= ne ? nsite : 0);
  __syncthreads();
  const bool two = (mv[3] >= 0 && mv[3] != mv[0]);
  double result = 0.0;
  if (!two) {
    // one-electron move; the degenerate same-slot exchange is a hop of its second move
    const int slot = (mv[3] >= 0) ? mv[3] : mv[0];
    const int site = (mv[3] >= 0) ? mv[4] : mv[1];
    const int rsa = site + (slot >= ne ? nsite : 0);
    const double* row = tab + (((size_t)qp * n2) + rsa) * n2;
    const double* inv_row = invq + (size_t)slot * n;
    for (int j = tid; j < n; j += nt) t0[j] = inv_row[j] * row[rs[j]];
    __syncthreads();
    if (tid == 0) {
      double ratio = 0.0;
#pragma unroll 8
      for (int j = 0; j < n; ++j) ratio += t0[j];
      sc[0] = -ratio * pfq;
    }
    __syncthreads();
    result = sc[0];
  } else {
    // two-electron exchange: two_ratio_real::<false> (Julia @turbo reduction tree, explicit FMA)
    const int msa = mv[0], msb = mv[3];
    const int rsa = mv[1] + (mv[2] ? nsite : 0), rsb = mv[4] + (mv[5] ? nsite : 0);
    const double* rowa = tab + (((size_t)qp * n2) + rsa) * n2;
    const double* rowb = tab + (((size_t)qp * n2) + rsb) * n2;
    const double* inv_a = invq + (size_t)msa * n;
    const double* inv_b = invq + (size_t)msb * n;
    // vec_a -> t0, vec_b -> t1; products for p_a, p_b, q_a, q_b -> t2.. needs 4 arrays: reuse
    // t4 region after the dots: first the sums.
    for (int j = tid; j < n; j += nt) {
      t0[j] = rowa[rs[j]];
      t1[j] = rowb[rs[j]];
    }
    __syncthreads();
    // four serial sums, one per thread 0..3: p_a, p_b, q_a, q_b
    if (tid < 4) {
      const double* invx = (tid & 1) ? inv_b : inv_a;
      const double* vec = (tid < 2) ? t0 : t1;
      double acc = 0.0;
#pragma unroll 8
      for (int i = 0; i < n; ++i) acc += invx[i] * vec[i];
      sc[tid] = acc;
    }
    // bilinear form  sum_i b_i sum_j inv[i*n+j] a_j  with the LoopVectorization tree
    for (int i = tid; i < n; i += nt) {
      double inner[4] = {0, 0, 0, 0};
      const double* invi = invq + (size_t)i * n;
#pragma unroll 4
      for (int j = 0; j < n; ++j) inner[j & 3] = fma(invi[j], t0[j], inner[j & 3]);
      t4[i] = (inner[0] + inner[2]) + (inner[1] + inner[3]);
    }
    __syncthreads();
    if (tid == 0) {
      double outer[6] = {0, 0, 0, 0, 0, 0};
      for (int i = 0; i < n; ++i) outer[i % 6] = fma(t1[i], t4[i], outer[i % 6]);
      const double bma = (outer[4] + (outer[0] + outer[2])) + (outer[5] + (outer[3] + outer[1]));
      const double p_a = sc[0], p_b = sc[1], q_a = sc[2], q_b = sc[3];
      const double inv_ab = inv_a[msb];
      const double vec_ba = t1[msa];
      const double ratio = inv_ab * vec_ba + inv_ab * bma + p_a * q_b - p_b * q_a;
      sc[4] = ratio * pfq;
    }
    __syncthreads();
    result = sc[4];
  }
  if (tid == 0) {
    pfnew[(size_t)w * nqp + qp] = result;
    out[(size_t)out_off + (size_t)item * nqp + qp] = result;
  }
}

// ---------------------------------------------------------------------------------------------
// accepted moves: one block per (accepted walker, qp). Dynamic shared memory:
//   n ints (candidate sites) + 6*n doubles (a, b, c, d, e, f vectors) + 16 doubles (scalars).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void k_accept(const int* acc_w, const double* const* slater_tab,
                                    const int* ele, double* inv, double* pf, const double* pfnew,
                                    const int* pend, int n, int ne, int nsite, int nqp) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);                       // [n] candidate rs index
  double* vec = reinterpret_cast<double*>(rs + ((n + 1) & ~1));      // 6 vectors of n + scalars
  double* v1 = vec;            // hop: vec1 / two: p
  double* v2 = vec + n;        // hop: vec2 / two: q
  double* v3 = vec + 2 * n;    // hop: slt  / two: s
  double* v4 = vec + 3 * n;    // two: t
  double* sc = vec + 4 * n;    // scalars

  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int w = acc_w[item];
  const int* pd = pend + (size_t)w * 8;
  const int mv[6] = {pd[0], pd[1], pd[2], pd[3], pd[4], pd[5]};
  const int* ele_w = ele + (size_t)w * n;
  const double* tab = slater_tab[w];
  const int n2 = 2 * nsite;
  double* invq = inv + ((size_t)w * nqp + qp) * (size_t)n * n;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int j = tid; j < n; j += nt) rs[j] = cand_site(ele_w, mv, j) + (j >= ne ? nsite : 0);
  __syncthreads();
  const bool two = (mv[3] >= 0 && mv[3] != mv[0]);

  if (!two) {
    const int msa = (mv[3] >= 0) ? mv[3] : mv[0];
    const int site = (mv[3] >= 0) ? mv[4] : mv[1];
    const int rsa = site + (msa >= ne ? nsite : 0);
    for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
    __syncthreads();
    // vec1[msi] = sum_msj (-inv[msj*n+msi]) * slt[msj]   (column-major sweep, msj ascending)
    for (int i = tid; i < n; i += nt) {
      double acc = 0.0;
#pragma unroll 8
      for (int j = 0; j < n; ++j) acc += -invq[(size_t)j * n + i] * v3[j];
      v1[i] = acc;
    }
    __syncthreads();
    const double tmp = v1[msa];
    const double inv_vec1_a = -1.0 / tmp;
    for (int i = tid; i < n; i += nt) v2[i] = invq[(size_t)msa * n + i] * inv_vec1_a;
    __syncthreads();
    for (int e = tid; e < n * n; e += nt) {
      const int i = e / n, j = e % n;
      double x = invq[e];
      x += v1[i] * v2[j] - v1[j] * v2[i];
      if (j == msa) x -= v2[i];
      if (i == msa) x += v2[j];
      invq[e] = x;
    }
    if (tid == 0) pf[(size_t)w * nqp + qp] = pfnew[(size_t)w * nqp + qp];
    return;
  }

  // two-electron exchange: update_two_real
  const int msa = mv[0], msb = mv[3];
  const int rsa = mv[1] + (mv[2] ? nsite : 0), rsb = mv[4] + (mv[5] ? nsite : 0);
  const int ra_old = ele_w[msa];
  const int rsa_old = ra_old + (mv[2] ? nsite : 0), rsb_old = ra_old + (mv[5] ? nsite : 0);
  double* vp = v1;  double* vq = v2;  double* vs = v3;  double* vt = v4;
  for (int i = tid; i < n; i += nt) {
    vs[i] = SLT(tab, qp, rsa, rs[i], n2);
    vt[i] = SLT(tab, qp, rsb, rs[i], n2);
  }
  __syncthreads();
  if (tid == 0) vs[msb] = SLT(tab, qp, rsa_old, rsb_old, n2);
  __syncthreads();
  for (int i = tid; i < n; i += nt) {
    double p = 0.0, q = 0.0;
#pragma unroll 4
    for (int j = 0; j < n; ++j) {
      const double x = invq[(size_t)i * n + j];
      p += x * vs[j];
      q += x * vt[j];
    }
    vp[i] = p;
    vq[i] = q;
  }
  __syncthreads();
  if (tid == 0) {
    double bma = 0.0;
    for (int i = 0; i < n; ++i) bma += vt[i] * vp[i];
    const double inv_ab = invq[(size_t)msa * n + msb];
    const double ratio = inv_ab * vt[msa] + inv_ab * bma + vp[msa] * vq[msb] - vp[msb] * vq[msa];
    pf[(size_t)w * nqp + qp] *= ratio;
    const double a = -vp[msa], b = vp[msb], c = vq[msa], d = -vq[msb];
    const double e = -bma - vt[msa], f = inv_ab;
    const double det = a * d - b * c - e * f;
    sc[0] = a; sc[1] = b; sc[2] = c; sc[3] = d; sc[4] = e; sc[5] = f;
    sc[6] = det; sc[7] = 1.0 / det;
  }
  __syncthreads();
  const double a = sc[0], b = sc[1], c = sc[2], d = sc[3], e = sc[4], f = sc[5];
  const double det = sc[6], inv_det = sc[7];
  // s/t are overwritten by inv_det * inv columns (read before any inverse element changes)
  for (int i = tid; i < n; i += nt) {
    vs[i] = inv_det * invq[(size_t)msa * n + i];
    vt[i] = inv_det * invq[(size_t)msb * n + i];
  }
  __syncthreads();
  for (int idx = tid; idx < n * n; idx += nt) {
    const int i = idx / n, j = idx % n;
    const double p_i = vp[i], q_i = vq[i], s_i = vs[i], t_i = vt[i];
    const double p_j = vp[j], q_j = vq[j], s_j = vs[j], t_j = vt[j];
    double x = invq[idx];
    x += a * (q_i * t_j - q_j * t_i) + b * (q_i * s_j - q_j * s_i) +
         c * (p_i * t_j - p_j * t_i) + d * (p_i * s_j - p_j * s_i) +
         e * det * (s_i * t_j - s_j * t_i) + f * inv_det * (p_i * q_j - q_i * p_j);
    if (j == msa) x += -c * t_i - d * s_i - f * inv_det * q_i;
    if (j == msb) x += -a * t_i - b * s_i + f * inv_det * p_i;
    if (i == msa) x += c * t_j + d * s_j + f * inv_det * q_j;
    if (i == msb) x += a * t_j + b * s_j - f * inv_det * p_j;
    if (i == msa && j == msb) x += f * inv_det;
    if (i == msb && j == msa) x -= f * inv_det;
    invq[idx] = x;
  }
}

// apply the accepted moves to the resident configuration (after all k_accept blocks)
extern "C" __global__ void k_commit_moves(const int* acc_w, int count, int* ele, const int* pend,
                                          int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int w = acc_w[i];
  const int* pd = pend + (size_t)w * 8;
  ele[(size_t)w * n + pd[0]] = pd[1];
  if (pd[3] >= 0) ele[(size_t)w * n + pd[3]] = pd[4];
}

// ---------------------------------------------------------------------------------------------
// recomputation: planes of the staged configurations. items: R entries; rec_ele [R][n] ints.
// A[item][q][msi*n + j] = -S[q][rs[msi]][rs[j]]   (the plane assembled by assemble_inv_m_real)
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void k_assemble(const int* rec_walker, const int* rec_ele,
                                      const double* const* slater_tab, double* planes, int n,
                                      int ne, int nsite, int nqp) {
  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int w = rec_walker[item];
  const int* e = rec_ele + (size_t)item * n;
  const double* tab = slater_tab[w];
  const int n2 = 2 * nsite;
  double* plane = planes + ((size_t)item * nqp + qp) * (size_t)n * n;
  for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) {
    const int msi = idx / n, j = idx % n;
    const int rsi = e[msi] + (msi >= ne ? nsite : 0);
    const int rsj = e[j] + (j >= ne ? nsite : 0);
    plane[idx] = -SLT(tab, qp, rsi, rsj, n2);
  }
}

// commit a recomputation: invM = -inv (the calc_m_all_real sign flip), pf, configuration.
// pf_set != nullptr (begin): the host's Pfaffians replace the computed ones.
// out: out[off_pf + item*nqp + q] pf, out[off_failed + item] failed (double 0/1).
extern "C" __global__ void k_commit_recompute(const int* rec_walker, const int* rec_ele,
                                              const double* inv_out, const double* pf_comp,
                                              const int* status, const double* pf_set, double* inv,
                                              double* pf, int* ele, double* out,
                                              int off_pf, int off_failed, int n, int nqp) {
  __shared__ int ok;
  const int item = blockIdx.x;
  const int w = rec_walker[item];
  if (threadIdx.x == 0) {
    int good = 1;
    for (int q = 0; q < nqp; ++q) good &= (status[(size_t)item * nqp + q] == 0);
    ok = good;
    out[off_failed + item] = good ? 0.0 : 1.0;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < n; j += blockDim.x) ele[(size_t)w * n + j] = rec_ele[(size_t)item * n + j];
  if (!ok) return;
  const size_t plane = (size_t)n * n;
  for (size_t e = threadIdx.x; e < (size_t)nqp * plane; e += blockDim.x)
    inv[(size_t)w * nqp * plane + e] = -inv_out[(size_t)item * nqp * plane + e];
  for (int q = threadIdx.x; q < nqp; q += blockDim.x) {
    const double v = pf_set ? pf_set[(size_t)item * nqp + q] : pf_comp[(size_t)item * nqp + q];
    pf[(size_t)w * nqp + q] = v;
    out[(size_t)off_pf + (size_t)item * nqp + q] = v;
  }
}
