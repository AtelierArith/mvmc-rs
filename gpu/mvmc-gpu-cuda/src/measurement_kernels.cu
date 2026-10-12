// Resident measurement kernels (issue #518 milestone 2), real (f64) normal mode.
//
// Ports of the measurement arithmetic of crates/mvmc-core:
//   * km_assemble       plane assembly of `stage_a_pfaffian` / `assemble_inv_m_real`,
//   * km_ratio_terms    the per-transfer-term `new_ip` ratio table of `CalHamiltonian1`
//                       (`transfer_new_ip` in `measurement_backend.rs`),
//   * km_slater_rows    per-(sector, row) orbital sums of the real `SlaterElmDiff` fast path,
//   * km_slater_fold    the QP/sector fold into the raw derivative accumulation.
//
// Compile with --fmad=false: every product/sum below is written in the order of the host
// kernels, so no contraction is fused.
//
// Layout (n = 2*Ne, n2 = 2*Nsite):
//   S     [qp][n2*n2]  Slater table, row-major per plane (`((qp*n2)+row)*n2+col`)
//   inv   [plane][n*n] mVMC invM = -X^-1 (plane = slot*nqp+qp), element (r,c) at r*n+c
//   pf    [slot][qp]
//   ele   [slot][n]    electron sites (spin of slot j: j >= ne)
//   planes[slot*nqp+qp][n*n] assembled skew plane -X (before the factorization)
//
// The km_assemble and pfinv_f64 kernels form the stage A pipeline; `pfinv_f64` lives in
// `pfaffian_batched.cu` and is launched by the same host module.

// ---------------------------------------------------------------------------------------------
// Stage A: assemble the skew planes X[msi,msj] = -S[qp][rsi][rsj] (assemble_inv_m_real).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void km_assemble(const double* __restrict__ S, const int* __restrict__ ele,
                                       double* __restrict__ planes, int n, int ne, int nsite,
                                       int nqp) {
  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int* e = ele + (size_t)item * n;
  const int n2 = 2 * nsite;
  const double* tab = S + (size_t)qp * n2 * n2;
  double* plane = planes + ((size_t)item * nqp + qp) * (size_t)n * n;
  for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) {
    const int msi = idx / n, j = idx % n;
    const int rsi = e[msi] + (msi >= ne ? nsite : 0);
    const int rsj = e[j] + (j >= ne ? nsite : 0);
    plane[idx] = -tab[(size_t)rsi * n2 + rsj];
  }
}

// ---------------------------------------------------------------------------------------------
// Local energy: new_ip[t] = sum_qp w[qp] * ( -( (sum_msj inv[msa,msj] * S[rsa][rsj(msj)]) * pf[qp] ) )
// with rsj(msj) = (msj == msa ? ri : ele[msj]) + (msj >= ne ? nsite : 0). One block per term;
// threads form the products, thread 0 sums them in msj order and folds qp in order (the exact
// host grouping). Dynamic shared memory: n doubles.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void km_ratio_terms(const double* __restrict__ S,
                                          const double* __restrict__ inv,
                                          const double* __restrict__ pf,
                                          const int* __restrict__ ele,
                                          const int* __restrict__ msa,
                                          const int* __restrict__ rsa,
                                          const int* __restrict__ ri,
                                          const double* __restrict__ w,
                                          double* __restrict__ out, int ele0, int plane0, int terms,
                                          int n, int ne, int nsite, int nqp) {
  extern __shared__ double prod[];
  const int t = blockIdx.x;
  if (t >= terms) return;
  const int n2 = 2 * nsite;
  const int a = msa[t], r = rsa[t], moved = ri[t];
  const int tid = threadIdx.x, nt = blockDim.x;
  const int* e = ele + ele0;
  double acc = 0.0;
  for (int qp = 0; qp < nqp; ++qp) {
    const double* tab = S + (size_t)qp * n2 * n2;
    const double* inv_row = inv + (size_t)(plane0 + qp) * n * n + (size_t)a * n;
    const double* S_row = tab + (size_t)r * n2;
    for (int j = tid; j < n; j += nt) {
      const int site = (j == a) ? moved : e[j];
      const int rsj = site + (j >= ne ? nsite : 0);
      // `inv` holds the true inverse X^-1 (the pfinv_f64 output); mVMC uses invM = -X^-1.
      prod[j] = -inv_row[j] * S_row[rsj];
    }
    __syncthreads();
    if (tid == 0) {
      double ratio = 0.0;
      for (int j = 0; j < n; ++j) ratio += prod[j];
      acc += w[qp] * (-(ratio * pf[plane0 + qp]));
    }
    __syncthreads();
  }
  if (tid == 0) out[t] = acc;
}

// ---------------------------------------------------------------------------------------------
// PhysCal Green functions (issue #517 milestone 5), two-body rank-2 reduction. One block per
// active two-body job; the block loops the QP planes and accumulates
//   new_ip = sum_qp w[qp] * (ratio_qp * pf[qp])
// with the C/Julia `calculate_new_pf_m_two2` ratio
//   ratio_qp = invM[ma,mb]*b[ma] + invM[ma,mb]*(b^T M a) + p_a*q_b - p_b*q_a,
//   p_v = sum_i invM[v,i]*a[i],  q_v = sum_i invM[v,i]*b[i],
//   invM = -X^-1 (the stored `inv` is X^-1), indices ma = msa, mb = msb.
// The p/q sums iterate the *rows* i (to share the bma row sweep): the kernel accumulates
// p'_v = sum_i invM[i,v]*a[i] = -p_v (invM is skew), so each p/q is negated, but only the
// invariant combination p_a*q_b - p_b*q_a is consumed and the negation of a product is exact.
// The vectors a/b gather the destination Slater rows `ra`/`rb` at the moved electron slots
// (slot `ma` -> `rsa_site`, `mb` -> `rsb_site`). Dynamic shared memory: 2*n + 5*blockDim doubles.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void km_green_two(const double* __restrict__ S,
                                        const double* __restrict__ inv,
                                        const double* __restrict__ pf,
                                        const int* __restrict__ ele,
                                        const int* __restrict__ msa,
                                        const int* __restrict__ msb,
                                        const int* __restrict__ rsa_site,
                                        const int* __restrict__ rsb_site,
                                        const double* __restrict__ w, double* __restrict__ out,
                                        int ele0, int plane0, int jobs, int n, int ne, int nsite,
                                        int nqp) {
  extern __shared__ double smem[];
  double* a = smem;
  double* b = smem + n;
  double* red = smem + 2 * n;
  const int t = blockIdx.x;
  if (t >= jobs) return;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int n2 = 2 * nsite;
  const int ma = msa[t], mb = msb[t];
  const int sa = rsa_site[t], sb = rsb_site[t];
  const int ra = sa + (ma >= ne ? nsite : 0);
  const int rb = sb + (mb >= ne ? nsite : 0);
  const int* e = ele + ele0;
  double acc = 0.0;
  for (int qp = 0; qp < nqp; ++qp) {
    const double* tab = S + (size_t)qp * n2 * n2;
    const double* rowa = tab + (size_t)ra * n2;
    const double* rowb = tab + (size_t)rb * n2;
    for (int i = tid; i < n; i += nt) {
      const int moved = (i == ma) ? sa : ((i == mb) ? sb : e[i]);
      const int rsi = moved + (i >= ne ? nsite : 0);
      a[i] = rowa[rsi];
      b[i] = rowb[rsi];
    }
    __syncthreads();
    const double* M = inv + (size_t)(plane0 + qp) * n * n;  // X^-1
    double pa = 0.0, pb = 0.0, qa = 0.0, qb = 0.0, bma = 0.0;
    for (int i = tid; i < n; i += nt) {
      const double* row = M + (size_t)i * n;
      double tmp = 0.0;
      for (int j = 0; j < n; ++j) tmp += (-row[j]) * a[j];
      bma += b[i] * tmp;
      const double ma_i = -row[ma], mb_i = -row[mb];
      pa += ma_i * a[i];
      pb += mb_i * a[i];
      qa += ma_i * b[i];
      qb += mb_i * b[i];
    }
    red[tid] = pa;
    red[nt + tid] = pb;
    red[2 * nt + tid] = qa;
    red[3 * nt + tid] = qb;
    red[4 * nt + tid] = bma;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
      if (tid < s) {
        red[tid] += red[tid + s];
        red[nt + tid] += red[nt + tid + s];
        red[2 * nt + tid] += red[2 * nt + tid + s];
        red[3 * nt + tid] += red[3 * nt + tid + s];
        red[4 * nt + tid] += red[4 * nt + tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      const double inv_ab = -M[(size_t)ma * n + mb];
      const double vec_ba = b[ma];
      const double ratio =
          inv_ab * vec_ba + inv_ab * red[4 * nt] + red[0] * red[3 * nt] - red[nt] * red[2 * nt];
      acc += w[qp] * ratio * pf[plane0 + qp];
    }
    __syncthreads();
  }
  if (tid == 0) out[t] = acc;
}

// ---------------------------------------------------------------------------------------------
// Slater derivative, kernel 1: one block per (trans, msi). The block writes the row's orbital
// sums rows[(trans*n+msi)*nslater + o] = sum over the sites of orbital o of sign * value:
//   value = sum_s coeff(block(msi,msj), s) * invM[qp0+s][msi*n + msj]
// with msj = inv_trans[v][trans*n_site + trj] (`v` = spin block of the electron slot; up and
// down electrons can share a site, so each translation site carries one slot per spin block),
// sign = sgni * inv_sgn[v][...] * list_sgn[tri*n_site + k]. The per-orbital site lists are
// contiguous in the CSR (offsets[tri*(nslater+1) + o]); a thread sums one orbital's list in
// order, so the result is deterministic. Dynamic shared memory: none.
extern "C" __global__ void km_slater_rows(const double* __restrict__ inv,
                                          const double* __restrict__ pf,
                                          const double* __restrict__ w,
                                          const double* __restrict__ cossin,
                                          const double* __restrict__ coscos,
                                          const double* __restrict__ sinsin,
                                          const int* __restrict__ offsets,
                                          const int* __restrict__ list_trj,
                                          const int* __restrict__ list_sgn,
                                          const int* __restrict__ tri,
                                          const int* __restrict__ trans_sgn,
                                          const int* __restrict__ inv_trans,
                                          const int* __restrict__ inv_sgn,
                                          double* __restrict__ rows, int plane0, int n, int ne,
                                          int nsite, int nqp, int nsp, int ntrans, int nslater) {
  const int trans = blockIdx.x / n;
  const int msi = blockIdx.x % n;
  const int t = tri[(size_t)trans * n + msi];
  const int row = (size_t)trans * n + msi;
  if (t < 0) {
    for (int o = threadIdx.x; o < nslater; o += blockDim.x)
      rows[((size_t)row) * nslater + o] = 0.0;
    return;
  }
  const int sgni = trans_sgn[(size_t)trans * n + msi];
  const int qp0 = trans * nsp;
  const int* off = offsets + (size_t)t * (nslater + 1);
  for (int o = threadIdx.x; o < nslater; o += blockDim.x) {
    double acc = 0.0;
    for (int k = off[o]; k < off[o + 1]; ++k) {
      const int trj = list_trj[(size_t)t * nsite + k];
      const int osgn = list_sgn[(size_t)t * nsite + k];
      for (int v = 0; v < 2; ++v) {
        const int msj = inv_trans[((size_t)v * ntrans + trans) * nsite + trj];
        if (msj < 0) continue;
        const int sgnj = inv_sgn[((size_t)v * ntrans + trans) * nsite + trj];
        double x = 0.0;
        for (int s = 0; s < nsp; ++s) {
          const int qp = qp0 + s;
          if (qp >= nqp) break;
          const double cs = w[qp] * pf[plane0 + qp] * cossin[s];
          const double cc = w[qp] * pf[plane0 + qp] * coscos[s];
          const double ss = w[qp] * pf[plane0 + qp] * sinsin[s];
          // `inv` holds X^-1; mVMC's invM = -X^-1.
          const double e = -inv[(size_t)(plane0 + qp) * n * n + (size_t)msi * n + msj];
          // Block coefficients of the C SlaterElmDiff fast path.
          if (msi < ne) {
            x += (v == 0 ? cs : -cc) * e;
          } else {
            x += (v == 0 ? ss : -cs) * e;
          }
        }
        acc += sgni * sgnj * osgn * x;
      }
    }
    rows[((size_t)row) * nslater + o] = acc;
  }
}

// ---------------------------------------------------------------------------------------------
// Slater derivative, kernel 2: acc[o] = sum_{trans, msi} rows[(trans*n+msi)*nslater+o] in
// trans-major then msi order (fixed, deterministic).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void km_slater_fold(const double* __restrict__ rows,
                                          double* __restrict__ acc, int n, int ntrans,
                                          int nslater) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= nslater) return;
  double sum = 0.0;
  for (int trans = 0; trans < ntrans; ++trans) {
    for (int msi = 0; msi < n; ++msi) {
      sum += rows[((size_t)trans * n + msi) * nslater + o];
    }
  }
  acc[o] = sum;
}
