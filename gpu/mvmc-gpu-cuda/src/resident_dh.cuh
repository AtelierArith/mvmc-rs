// Doublon-holon correlation factors on resident walkers (#665, #572 milestone 2).
//
// C projection.c MakeProjCnt: a center site r of table x with occupation
// c = n_up(r) + n_down(r) contributes one count only when it is a holon
// (c = 0) or a doublon (c = 2). xm counts the doublons (holon center) or
// holons (doublon center) on its 2 (DH2) or 4 (DH4) neighbor sites, and the
// counter is offset + x + (c/2 + 2*xm) * N. Julia GPU has no DH support; this
// follows the C counters exactly, which are integers.
//
// `dh` holds the DH2 tables [n2][nsite][2], then the DH4 tables
// [n4][nsite][4]. Item k in [0, (n2+n4)*nsite) selects table k / nsite and
// center k % nsite. `base` is the first DH parameter (NGutzwiller+NJastrow).

#ifndef RESIDENT_DH_CUH
#define RESIDENT_DH_CUH

// Projection parameter index of item k, or -1 for a singly occupied center.
// `Occ` returns the occupation (0/1) of (site, spin).
template <class Occ>
__device__ inline int resident_dh_index(const Occ& occ, const int* dh, int n2, int n4,
                                        int nsite, int base, int k) {
  const int x = k / nsite, r = k % nsite;
  const bool four = x >= n2;
  const int width = four ? 4 : 2;
  const int* nb = four ? dh + (size_t)2 * n2 * nsite + ((size_t)(x - n2) * nsite + r) * 4
                       : dh + ((size_t)x * nsite + r) * 2;
  const int c = occ(r, 0) + occ(r, 1);
  if (c == 1) return -1;
  int xm = 0;
  for (int t = 0; t < width; ++t) {
    const int u = occ(nb[t], 0), d = occ(nb[t], 1);
    xm += c == 0 ? u * d : (1 - u) * (1 - d);
  }
  const int bin = c / 2 + 2 * xm;
  return four ? base + 6 * n2 + (x - n2) + bin * n4 : base + x + bin * n2;
}

// Occupation table `[site] up then [site] down`.
struct ResidentOcc {
  const int* o;
  int nsite;
  __device__ int operator()(int r, int s) const { return o[r + s * nsite]; }
};

// Occupation before one accepted-or-proposed hop ri -> rj of spin `spin`,
// read from the table after the hop.
struct ResidentOccBeforeHop {
  const int* o;
  int nsite, ri, rj, spin;
  __device__ int operator()(int r, int s) const {
    if (s == spin) {
      if (r == ri) return 1;
      if (r == rj) return 0;
    }
    return o[r + s * nsite];
  }
};

// Occupation after merged per-site changes (`du`, `dd`) on `count` sites.
struct ResidentOccMoved {
  const int* o;
  int nsite, count;
  const int* sites;
  const int* du;
  const int* dd;
  __device__ int operator()(int r, int s) const {
    int value = o[r + s * nsite];
    for (int i = 0; i < count; ++i)
      if (sites[i] == r) value += s ? dd[i] : du[i];
    return value;
  }
};

// Partial DH log-weight change sum_k P[after(k)] - P[before(k)] over the items
// k = first, first+stride, ... Unchanged items contribute nothing. Callers
// reduce the partial sums in a fixed order.
template <class Before, class After>
__device__ inline double resident_dh_delta(const Before& before, const After& after,
                                           const int* dh, int n2, int n4, int nsite, int base,
                                           const double* params, int first, int stride) {
  double delta = 0.0;
  const int items = (n2 + n4) * nsite;
  for (int k = first; k < items; k += stride) {
    const int a = resident_dh_index(after, dh, n2, n4, nsite, base, k);
    const int b = resident_dh_index(before, dh, n2, n4, nsite, base, k);
    if (a == b) continue;
    if (a >= 0) delta += params[a];
    if (b >= 0) delta -= params[b];
  }
  return delta;
}

#endif
