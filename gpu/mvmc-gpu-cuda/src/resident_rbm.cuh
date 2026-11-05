// RBM correlation factors on resident walkers (#667, #572 milestone 3).
//
// Physics: C rbm.c on the complex normal path (vmcmake.c, calham.c, calgrn.c,
// vmccal.c), restricted to real coefficients, where every quantity is real.
// C's real path (vmcmake_real.c, calham_real.c) omits the RBM factor from
// the Metropolis ratio and the local energy, and RBMDiff then reads RBMCnt
// that vmcmake_real.c never saves. The resident path keeps the factor in
// the wave function everywhere, as the complex C path does.
//
// `rbm` tables (built by opt/model.rs; indices relative to the RBM block,
// C RBM = Para + NProj; -1 means absent):
//   [0, ns)            ChargeRBM_PhysLayerIdx[r]
//   [ns, 2ns)          SpinRBM_PhysLayerIdx[r] + NChargeRBM_PhysLayerIdx
//   [2ns, 4ns)         GeneralRBM_PhysLayerIdx[r + s*ns] + NCharge + NSpin
//   [4ns, 4ns+H)       family of neuron h: 0 charge, 1 spin, 2 general
//   [4ns+H, 4ns+2H)    hidden-layer coefficient of neuron h
//   [4ns+2H, ...)      [h][rs] physical-hidden coefficient (charge/spin use
//                      rs < ns, general rs = r + s*ns)
// Neurons are ordered charge, spin, general (C rbmCnt layout). `wt` holds the
// NRBM coefficients and `theta[w*H + h]` the hidden field of walker w.

#ifndef RESIDENT_RBM_CUH
#define RESIDENT_RBM_CUH

__device__ inline int resident_rbm_family(const int* rbm, int ns, int h) {
  return rbm[4 * ns + h];
}

__device__ inline const int* resident_rbm_coupling(const int* rbm, int ns, int H, int h) {
  return rbm + 4 * ns + 2 * H + (size_t)h * 2 * ns;
}

// C MakeRBMCnt for neuron h: theta = 0 + bias, then += sum_r W[r] x_r with
// the coupling sum accumulated separately in site order (rbm.c:216-270).
// x_r = n_up + n_down - 1 (charge), n_up - n_down (spin), 2 n_rs - 1 (general).
__device__ inline double resident_rbm_make(const int* rbm, const double* wt, const int* occ,
                                           int ns, int H, int h) {
  const int family = resident_rbm_family(rbm, ns, h);
  const int b = rbm[4 * ns + H + h];
  const int* c = resident_rbm_coupling(rbm, ns, H, h);
  double theta = 0.0;
  if (b >= 0) theta = __dadd_rn(theta, wt[b]);
  if (c[0] >= 0) {
    double sum = 0.0;
    const int width = family == 2 ? 2 * ns : ns;
    for (int rs = 0; rs < width; ++rs) {
      const int x = family == 0 ? occ[rs] + occ[rs + ns] - 1
                  : family == 1 ? occ[rs] - occ[rs + ns]
                                : 2 * occ[rs] - 1;
      sum = __dadd_rn(sum, __dmul_rn(wt[c[rs]], (double)x));
    }
    theta = __dadd_rn(theta, sum);
  }
  return theta;
}

// C UpdateRBMCnt for neuron h of a hop ri -> rj with spin s (rbm.c:426-461):
// subtract the source coupling, then add the destination coupling, scaled by
// 1 (charge), 1-2s (spin) or 2 (general, at r + s*ns).
__device__ inline double resident_rbm_hop(const int* rbm, const double* wt, int ns, int H, int h,
                                          double theta, int ri, int rj, int s) {
  if (ri == rj) return theta;
  const int family = resident_rbm_family(rbm, ns, h);
  const int* c = resident_rbm_coupling(rbm, ns, H, h);
  if (c[0] < 0) return theta;
  if (family == 2) {
    theta = __dsub_rn(theta, __dmul_rn(2.0, wt[c[ri + s * ns]]));
    return __dadd_rn(theta, __dmul_rn(2.0, wt[c[rj + s * ns]]));
  }
  const double k = family == 1 ? (double)(1 - 2 * s) : 1.0;
  theta = __dsub_rn(theta, __dmul_rn(k, wt[c[ri]]));
  return __dadd_rn(theta, __dmul_rn(k, wt[c[rj]]));
}

// Real part of C LogRBMRatio's per-neuron terms (rbm.c:139-160) for real
// fields: fold to |theta|, add |new| - |old| and log((1+e^{-2|new|}) /
// (1+e^{-2|old|})). C multiplies the ratios of a block before one log; the
// resident path takes the log per neuron so neurons can be reduced in
// parallel (log of a product of positive factors, rounding only).
__device__ inline double resident_rbm_log_cosh_ratio(double tnew, double told) {
  const double a = fabs(tnew), b = fabs(told);
  const double factor = (1.0 + exp(-2.0 * a)) / (1.0 + exp(-2.0 * b));
  return __dadd_rn(__dsub_rn(a, b), log(factor));
}

// Physical-layer part of LogRBMRatio, sum_idx RBM[idx] (new - old), for up to
// two hops (ri[k] -> rj[k], spin s[k]). Counter changes follow UpdateRBMCnt;
// coinciding counters are merged and summed in ascending index order, as C
// sums over every idx (the unchanged ones contribute +-0).
__device__ inline double resident_rbm_phys_delta(const int* rbm, const double* wt, int ns,
                                                 const int* ri, const int* rj, const int* s,
                                                 int hops) {
  int idx[12];
  int d[12];
  int count = 0;
  auto add = [&](int i, int v) {
    if (i < 0) return;
    for (int k = 0; k < count; ++k)
      if (idx[k] == i) {
        d[k] += v;
        return;
      }
    idx[count] = i;
    d[count] = v;
    ++count;
  };
  for (int k = 0; k < hops; ++k) {
    if (ri[k] == rj[k]) continue;
    add(rbm[ri[k]], -1);
    add(rbm[rj[k]], 1);
    add(rbm[ns + ri[k]], 2 * s[k] - 1);
    add(rbm[ns + rj[k]], 1 - 2 * s[k]);
    add(rbm[2 * ns + ri[k] + s[k] * ns], -2);
    add(rbm[2 * ns + rj[k] + s[k] * ns], 2);
  }
  double z = 0.0;
  int last = -1;
  for (int n = 0; n < count; ++n) {
    int best = -1;
    for (int k = 0; k < count; ++k)
      if (idx[k] > last && (best < 0 || idx[k] < idx[best])) best = k;
    last = idx[best];
    if (d[best] != 0) z = __dadd_rn(z, __dmul_rn(wt[idx[best]], (double)d[best]));
  }
  return z;
}

// Hidden-layer part of LogRBMRatio for up to two hops, over the neurons
// h = first, first+stride, ... Callers reduce the partial sums in a fixed order.
__device__ inline double resident_rbm_hidden_delta(const int* rbm, const double* wt,
                                                   const double* theta_w, int ns, int H,
                                                   const int* ri, const int* rj, const int* s,
                                                   int hops, int first, int stride) {
  double z = 0.0;
  for (int h = first; h < H; h += stride) {
    const double told = theta_w[h];
    double tnew = told;
    for (int k = 0; k < hops; ++k) tnew = resident_rbm_hop(rbm, wt, ns, H, h, tnew, ri[k], rj[k], s[k]);
    z += resident_rbm_log_cosh_ratio(tnew, told);
  }
  return z;
}

#endif
