// Complex normal-orbital resident Opt/PhysCal kernels (#678, #572 milestone 6).
//
// C's complex normal path: UpdateSlaterElm_fcmp (slater.c), CalculateIP_fcmp
// (qp.c), CalculateHamiltonian/CalculateGreenFunc (calham.c, calgrn.c) with
// complex coefficients, SlaterElmDiff_fcmp (slater.c) and the complex
// StochasticOpt (stcopt.c, stcopt_dposv.c) with 2*NPara real parameters.
// Parameters are interleaved (re, im). Projection factors use the real
// parts (C LogProjVal/LogProjRatio: creal(Proj)); `proj` is the real
// projection copy the sampler also reads.
//
// SR layout. C's real S/g are, for real parameter slots i, j in 0..2*NPara,
//   S_ij = Re<O_i^* O_j> - Re<O_i> Re<O_j>,  g_i = -2 dt (Re<e O_i> - Re<e> Re<O_i>),
// where slot 2k is d/dRe p_k (O_k) and slot 2k+1 is d/dIm p_k (i O_k; C
// writes 0 for projection parameters). With D = [Re O; Im O], the
// 2W x 2NPara stacked real derivative matrix (walker rows, then imaginary
// rows), Re<O_i^* O_j> = (D^T D)_ij / W, Re<O_i> is the mean of the first W
// rows, and Re<e O_i> = (D^T E)_i / W with E = [Re e; -Im e]. The real direct-SR
// kernels therefore solve the complex system unchanged on D, E and 2*NPara.

extern "C" __global__ void opt_slater_c(rcx* slater, const rcx* params, const int* pa,
    const int* pb, const int* sa, const int* sb, const double* cs, const double* cc,
    const double* ss, int ns, int nq) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns * ns * nq) return;
  int q = k / (ns * ns), r = (k / ns) % ns, c = k % ns, n2 = 2 * ns;
  const rcx a = rc_scale(params[pa[k]], (double)sa[k]);
  const rcx b = rc_scale(params[pb[k]], (double)sb[k]);
  const rcx d = rc_sub(a, b);
  rcx* s = slater + (size_t)q * n2 * n2;
  s[r * n2 + c] = rc_neg(rc_scale(d, cs[q]));
  s[r * n2 + c + ns] = rc_add(rc_scale(a, cc[q]), rc_scale(b, ss[q]));
  s[(r + ns) * n2 + c] = rc_sub(rc_neg(rc_scale(a, ss[q])), rc_scale(b, cc[q]));
  s[(r + ns) * n2 + c + ns] = rc_scale(d, cs[q]);
}

// opt_initial_commit with complex tables and weights.
extern "C" __global__ void opt_initial_commit_c(int* pending, const int* candidate,
    const int* status, const rcx* out, const rcx* newpf, const rcx* weights, const double* logs,
    rcx* inv, rcx* pf, double* logip, int* aw, int* remaining, int n, int nq) {
  int w = blockIdx.x;
  if (!pending[w]) return;
  __shared__ int good;
  __shared__ rcx ip;
  if (!threadIdx.x) {
    good = candidate[w] && isfinite(logs[w]);
    ip = rc(0.0, 0.0);
    for (int q = 0; q < nq; ++q) {
      const rcx p = newpf[w * nq + q];
      good &= status[w * nq + q] == 0 && rc_finite(p) && !rc_zero(p);
      ip = rc_add(ip, rc_mul(weights[q], p));
    }
    good &= rc_finite(ip) && !rc_zero(ip);
    if (!good) atomicAdd(remaining, 1);
  }
  __syncthreads();
  if (!good) return;
  size_t base = (size_t)w * nq * n * n;
  for (size_t k = threadIdx.x; k < (size_t)nq * n * n; k += blockDim.x)
    inv[base + k] = rc_neg(out[base + k]);
  for (int q = threadIdx.x; q < nq; q += blockDim.x) pf[w * nq + q] = newpf[w * nq + q];
  if (!threadIdx.x) {
    logip[w] = log(rc_abs(ip) + 1e-100);
    aw[w] = 0;
    pending[w] = 0;
  }
}

// Complex overlap ip = sum_q QPFullWeight[q] PfM[q] (C CalculateIP_fcmp)
// into overlaps[w] after opt_counts.
extern "C" __global__ void opt_overlaps_c(rcx* overlaps, const rcx* pf, const rcx* weights,
    int nq, int batch) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= batch) return;
  rcx ip = rc(0.0, 0.0);
  for (int q = 0; q < nq; ++q) ip = rc_add(ip, rc_mul(weights[q], pf[(size_t)w * nq + q]));
  overlaps[w] = ip;
}

// Stacked SR derivatives D (see the header): row w holds Re O, row
// batch + w holds Im O, columns 2p (d/dRe p) and 2p+1 (d/dIm p).
// Projection: O = count (real) in column 2p and 0 in column 2p+1 (C vmccal.c).
// Slater: C SlaterElmDiff_fcmp O_k = sum_q W_q sum_{(i,j)} coeff Pf_q dPf/Pf / ip,
// with the cofactor Pf_q invM[j][i] (an invalid plane stores its exact minor),
// and column 2k+1 = i O_k.
extern "C" __global__ void opt_derivatives_c(double* D, const int* counts, const int* lookup,
    const int* columns, const int* xyz, const double* coeff, const rcx* inv, const rcx* pf,
    const rcx* weights, const rcx* overlaps, int np, int nproj, int ns, int ne, int nq,
    int batch, int* error, const int* valid) {
  int lane = threadIdx.x % 32;
  unsigned int index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (index >= np * batch || *error) return;
  int w = index % batch, p = index / batch, n = 2 * ne;
  const size_t width = 2 * (size_t)np;
  double* re = D + (size_t)w * width + 2 * p;
  double* im = D + ((size_t)batch + w) * width + 2 * p;
  if (p < nproj) {
    if (!lane) {
      re[0] = counts[w * nproj + p];
      im[0] = 0.0;
      re[1] = 0.0;
      im[1] = 0.0;
    }
    return;
  }
  rcx sum = rc(0.0, 0.0);
  for (int k = columns[p] + lane; k < columns[p + 1]; k += 32) {
    size_t coordinate = (size_t)3 * k;
    int i = lookup[w * 2 * ns + xyz[coordinate + 1]], j = lookup[w * 2 * ns + xyz[coordinate]],
        q = xyz[coordinate + 2];
    if (i >= 0 && i < j) {
      const rcx stored = inv[((size_t)w * nq + q) * n * n + j * n + i];
      const rcx cofactor = valid[w * nq + q] ? rc_mul(pf[w * nq + q], stored) : stored;
      sum = rc_add(sum, rc_scale(rc_mul(weights[q], cofactor), coeff[k]));
    }
  }
  sum = rc_warp_sum(sum);
  if (!lane) {
    const rcx o = rc_div(rc_neg(sum), overlaps[w]);
    re[0] = o.re;
    im[0] = o.im;
    re[1] = -o.im;
    im[1] = o.re;
    if (!rc_finite(o)) atomicCAS(error, 0, RESIDENT_DERIVATIVES);
  }
}

// opt_energy_terms with complex tables. C locgrn.c GreenFunc1/GreenFunc2 return
// conj(ProjRatio ip_new / ip) = <psi|O|x>/<psi|x>, so value = conj(exp(dproj) newip / ip).
extern "C" __global__ void opt_energy_terms_c(rcx* values, const int* ops, const int* cfg,
    const int* occ, const int* lookup, const rcx* overlaps, const int* gutz, const int* jast,
    const double* proj, const rcx* slater, const rcx* weights, const rcx* inv, const rcx* pf,
    int ns, int ne, int nq, int ng, int nj, int terms, int batch, const int* error,
    const int* valid) {
  int lane = threadIdx.x % 32;
  unsigned int index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (index >= terms * batch || *error) return;
  int t = index % terms, w = index / terms, n = 2 * ne, n2 = 2 * ns;
  if (!opt_all_valid(valid, w, nq)) return;
  const int* op = ops + (size_t)7 * t;
  const int* map = lookup + w * n2;
  const int* e = cfg + w * n;
  int a, k, da, dk;
  int resolved = opt_resolve(op, map, e, ne, ns, a, k, da, dk);
  if (resolved) {
    if (!lane) values[index] = rc(resolved > 0 ? 1.0 : 0.0, 0.0);
    return;
  }
  double delta = opt_delta(occ + w * n2, gutz, jast, proj, op, ns, ng, nj, lane, nullptr, 0, 0,
                           nullptr, nullptr, nullptr, 0);
  rcx newip = rc(0.0, 0.0);
  for (int q = 0; q < nq; ++q) {
    const rcx* S = slater + (size_t)q * n2 * n2;
    const rcx* M = inv + ((size_t)w * nq + q) * n * n;
    int ra = da + (a >= ne ? ns : 0), rk = dk + (k >= ne ? ns : 0);
    rcx ratio;
    if (k < 0) {
      rcx sum = rc(0.0, 0.0);
      for (int j = lane; j < n; j += 32)
        sum = rc_add(sum, rc_mul(M[a * n + j], S[ra * n2 + e[j] + (j >= ne ? ns : 0)]));
      ratio = rc_neg(rc_warp_sum(sum));
    } else {
      rcx pa = rc(0, 0), pb = rc(0, 0), qa = rc(0, 0), qb = rc(0, 0), bma = rc(0, 0);
      for (int i = lane; i < n; i += 32) {
        int ri = opt_site(e, i, a, k, da, dk) + (i >= ne ? ns : 0);
        rcx av = S[ra * n2 + ri], bv = S[rk * n2 + ri], tmp = rc(0, 0);
        for (int j = 0; j < n; ++j) {
          int rj = opt_site(e, j, a, k, da, dk) + (j >= ne ? ns : 0);
          tmp = rc_add(tmp, rc_mul(M[i * n + j], S[ra * n2 + rj]));
        }
        bma = rc_add(bma, rc_mul(bv, tmp));
        pa = rc_add(pa, rc_mul(M[i * n + a], av));
        pb = rc_add(pb, rc_mul(M[i * n + k], av));
        qa = rc_add(qa, rc_mul(M[i * n + a], bv));
        qb = rc_add(qb, rc_mul(M[i * n + k], bv));
      }
      pa = rc_warp_sum(pa);
      pb = rc_warp_sum(pb);
      qa = rc_warp_sum(qa);
      qb = rc_warp_sum(qb);
      bma = rc_warp_sum(bma);
      ratio = rc_add(rc_add(rc_mul(M[a * n + k], S[rk * n2 + ra]), rc_mul(M[a * n + k], bma)),
                     rc_sub(rc_mul(pa, qb), rc_mul(pb, qa)));
    }
    newip = rc_add(newip, rc_mul(weights[q], rc_mul(ratio, pf[w * nq + q])));
  }
  if (!lane) values[index] = rc_conj(rc_scale(rc_div(newip, overlaps[w]), exp(delta)));
}

// opt_energy_fallback with complex full Pfaffians of the moved configuration.
// Dynamic shared memory: n ints (rounded to even), then 2n complex values.
extern "C" __global__ void opt_energy_fallback_c(rcx* values, const int* ops, const int* cfg,
    const int* occ, const int* lookup, const rcx* overlaps, const int* gutz, const int* jast,
    const double* proj, const rcx* slater, const rcx* weights, const int* valid, rcx* fac,
    int ns, int ne, int nq, int ng, int nj, int terms, int batch, const int* error) {
  int w = blockIdx.x, tid = threadIdx.x, n = 2 * ne, n2 = 2 * ns;
  if (w >= batch || *error || opt_all_valid(valid, w, nq)) return;
  extern __shared__ char fallback_mem[];
  int* rs = reinterpret_cast<int*>(fallback_mem);
  rcx* vec = reinterpret_cast<rcx*>(rs + ((n + 1) & ~1));
  __shared__ int s_resolved, s_a, s_k, s_da, s_dk;
  __shared__ double s_delta;
  __shared__ rcx s_ip;
  rcx* A = fac + (size_t)w * nq * n * n;
  const int* map = lookup + w * n2;
  const int* e = cfg + w * n;
  for (int t = 0; t < terms; ++t) {
    const int* op = ops + (size_t)7 * t;
    if (!tid) {
      int a, k, da, dk;
      s_resolved = opt_resolve(op, map, e, ne, ns, a, k, da, dk);
      s_a = a;
      s_k = k;
      s_da = da;
      s_dk = dk;
      s_ip = rc(0.0, 0.0);
    }
    __syncthreads();
    const int resolved = s_resolved;
    if (resolved) {
      if (!tid) values[(size_t)w * terms + t] = rc(resolved > 0 ? 1.0 : 0.0, 0.0);
      __syncthreads();
      continue;
    }
    if (tid < 32) {
      double d = opt_delta(occ + w * n2, gutz, jast, proj, op, ns, ng, nj, tid, nullptr, 0, 0,
                           nullptr, nullptr, nullptr, 0);
      if (!tid) s_delta = d;
    }
    for (int i = tid; i < n; i += blockDim.x)
      rs[i] = opt_site(e, i, s_a, s_k, s_da, s_dk) + (i >= ne ? ns : 0);
    __syncthreads();
    for (int q = 0; q < nq; ++q) {
      rc_gather(A, rs, n, -1, -1, slater, q, n2);
      const rcx p = rc_block_pfaffian(A, n, vec, vec + n);
      if (!tid) s_ip = rc_add(s_ip, rc_mul(weights[q], p));
    }
    if (!tid) values[(size_t)w * terms + t] = rc_conj(rc_scale(rc_div(s_ip, overlaps[w]), exp(s_delta)));
    __syncthreads();
  }
}

// opt_cofactors with complex minors: dPf(A)/dA(i,j) = (-1)^(i+j+1) Pf(minor).
extern "C" __global__ void opt_cofactors_c(rcx* inv, const int* valid, const int* cfg,
    const rcx* slater, rcx* fac, int ns, int ne, int nq, int batch, const int* error) {
  int plane = blockIdx.x, w = plane / nq, q = plane % nq, n = 2 * ne, n2 = 2 * ns;
  if (w >= batch || *error || valid[plane]) return;
  extern __shared__ char cofactor_mem[];
  int* rs = reinterpret_cast<int*>(cofactor_mem);
  rcx* vec = reinterpret_cast<rcx*>(rs + ((n + 1) & ~1));
  for (int i = threadIdx.x; i < n; i += blockDim.x) rs[i] = cfg[w * n + i] + (i >= ne ? ns : 0);
  __syncthreads();
  rcx* A = fac + (size_t)plane * n * n;
  rcx* C = inv + (size_t)plane * n * n;
  for (int j = 1; j < n; ++j)
    for (int i = 0; i < j; ++i) {
      const int m = rc_gather(A, rs, n, i, j, slater, q, n2);
      const rcx p = m ? rc_block_pfaffian(A, m, vec, vec + m) : rc(1.0, 0.0);
      if (!threadIdx.x) C[j * n + i] = ((i + j + 1) & 1) ? rc_neg(p) : p;
      __syncthreads();
    }
}

// Complex local energy e = sum_t coefficient_t value_t (C calham.c), stored
// as the SR vector E = [Re e; -Im e] of length 2*batch.
extern "C" __global__ void opt_energy_c(double* energies, const rcx* values,
    const rcx* coefficients, int terms, int batch, int* error) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= batch || *error) return;
  rcx e = rc(0.0, 0.0);
  for (int t = 0; t < terms; ++t)
    e = rc_add(e, rc_mul(coefficients[t], values[(size_t)w * terms + t]));
  energies[w] = e.re;
  energies[batch + w] = -e.im;
  if (!rc_finite(e)) atomicCAS(error, 0, RESIDENT_ENERGY);
}

// C SyncModifiedParameter for complex parameters (parameter.c:134-176),
// as the Rust C-order sync: shiftGJ subtracts the mean real Gutzwiller and
// Jastrow part from the real parts; Slater is scaled by D_AmpMax / max|z|
// (cabs). The real projection parts refresh the sampler copy.
extern "C" __global__ void opt_gauge_c(double* params, double* projection, int np, int ng,
    int nj, int nproj, int shift, const int* error) {
  if (threadIdx.x || blockIdx.x || *error) return;
  const int gj = ng + nj;
  if (shift) {
    double average = 0;
    for (int p = 0; p < gj; ++p) average = __dadd_rn(average, params[2 * p]);
    average /= gj;
    for (int p = 0; p < gj; ++p) params[2 * p] = __dsub_rn(params[2 * p], average);
  }
  double largest = 0;
  for (int p = nproj; p < np; ++p) {
    const double a = hypot(params[2 * p], params[2 * p + 1]);
    if (a > largest) largest = a;
  }
  if (largest > 0) {
    const double ratio = 4.0 / largest;
    for (int p = nproj; p < np; ++p) {
      params[2 * p] = __dmul_rn(params[2 * p], ratio);
      params[2 * p + 1] = __dmul_rn(params[2 * p + 1], ratio);
    }
  }
  for (int p = 0; p < nproj; ++p) projection[p] = params[2 * p];
}
