//! Engine state types + storage-order newtypes.
//!
//! Port targets: `MVMCOptimizers.jl/src/{types,workspace}.jl`.
//!
//! Phase 4.1 lands the following surface, sized exactly the way upstream
//! sizes it (so the next sub-step can hand pointers to the kernels
//! without re-shaping anything):
//!
//! | Rust item | Upstream Julia |
//! |---|---|
//! | [`SlaterElmFlat`] | `slater_elm` (row-major `[QPidx][ri+si*Nsite][rj+sj*Nsite]`) |
//! | [`InvMColMajor`] | `inv_m` (column-major `[QPidx][mi+si*Ne][mj+sj*Ne]` + per-QP pad slot) |
//! | [`EnergyData`] | `EnergyData` |
//! | [`SROptData`] | `SROptData` |
//! | [`ElectronConfiguration`] | `ElectronConfiguration` |
//! | [`SlaterMatrixData`] | `SlaterMatrixData` |
//! | [`PfaPackWorkspace`] | `PfaPackWorkspace` |
//! | [`ThreadedPfaPackWorkspace`] | `ThreadedPfaPackWorkspace` |
//! | [`SamplingWorkspace`] | `SamplingWorkspace` |
//! | [`PhysicalQuantities`] | `PhysicalQuantities` |
//! | [`VmcOptimizationState`] | `VMCOptimizationState` |
//!
//! The two newtypes wrap raw `Vec<T>` storage so that
//! `SlaterElmFlat<T>` (row-major) and `InvMColMajor<T>` (column-major)
//! never get accidentally exchanged. See `calculate_m_all.jl` and
//! `vmc_sampling.jl:3236` for the upstream comments that fix the
//! storage convention.

use std::sync::Mutex;

use num_complex::Complex64;
use pfapack::PivotIndex1Based;

// ---------------------------------------------------------------------------
// Storage-order newtypes
// ---------------------------------------------------------------------------

/// Slater-element table, **row-major** in `(ri + si * Nsite, rj + sj * Nsite)`
/// inside one QP plane. Upstream sizes it as `Vector{T}` of length
/// `n_qp_full * (2*n_site)^2`; we keep the same layout.
#[derive(Debug, Clone, PartialEq)]
pub struct SlaterElmFlat<T> {
    data: Vec<T>,
    n_qp_full: usize,
    n_site2: usize, // 2 * n_site
}

impl<T: Copy + Default> SlaterElmFlat<T> {
    /// Allocate a zero-filled table for `n_qp_full` QPs and `n_site` sites.
    pub fn zeros(n_qp_full: usize, n_site: usize) -> Self {
        let n_site2 = 2 * n_site;
        Self {
            data: vec![T::default(); n_qp_full * n_site2 * n_site2],
            n_qp_full,
            n_site2,
        }
    }
}

impl<T: Copy> SlaterElmFlat<T> {
    /// Number of QP planes.
    pub fn n_qp_full(&self) -> usize {
        self.n_qp_full
    }

    /// `2 * n_site`.
    pub fn n_site2(&self) -> usize {
        self.n_site2
    }

    /// Total entries (`n_qp_full * n_site2 * n_site2`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff the table holds zero entries (only when `n_qp_full == 0`).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row-major linear index, **0-based**. `row = ri + si * n_site`,
    /// `col = rj + sj * n_site` (same as upstream).
    #[inline]
    fn idx(&self, qp: usize, row: usize, col: usize) -> usize {
        debug_assert!(qp < self.n_qp_full);
        debug_assert!(row < self.n_site2);
        debug_assert!(col < self.n_site2);
        (qp * self.n_site2 + row) * self.n_site2 + col
    }

    /// Read `slater_elm[qp][row, col]`.
    #[inline]
    pub fn get(&self, qp: usize, row: usize, col: usize) -> T {
        self.data[self.idx(qp, row, col)]
    }

    /// Write `slater_elm[qp][row, col] = value`.
    #[inline]
    pub fn set(&mut self, qp: usize, row: usize, col: usize, value: T) {
        let k = self.idx(qp, row, col);
        self.data[k] = value;
    }

    /// Borrow the whole backing vector.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the whole backing vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow a single QP plane (length `n_site2 * n_site2`, row-major).
    pub fn qp_slice(&self, qp: usize) -> &[T] {
        let stride = self.n_site2 * self.n_site2;
        let start = qp * stride;
        &self.data[start..start + stride]
    }

    /// Mutably borrow a single QP plane.
    pub fn qp_slice_mut(&mut self, qp: usize) -> &mut [T] {
        let stride = self.n_site2 * self.n_site2;
        let start = qp * stride;
        &mut self.data[start..start + stride]
    }
}

/// Inverse-matrix table, **column-major** in `(mi + si * Ne, mj + sj * Ne)`
/// inside one QP plane, with the upstream `+1` pad slot per QP that
/// `vmc_sampling.jl:3236` reserves for the Pfaffian buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct InvMColMajor<T> {
    data: Vec<T>,
    n_qp_full: usize,
    n_size: usize, // 2 * n_elec
}

impl<T: Copy + Default> InvMColMajor<T> {
    /// Allocate a zero-filled table for `n_qp_full` QPs and `n_elec` electrons.
    pub fn zeros(n_qp_full: usize, n_elec: usize) -> Self {
        let n_size = 2 * n_elec;
        Self {
            // n_qp_full * (n_size * n_size + 1) -- the trailing slot is
            // the Pfaffian buffer pad upstream relies on.
            data: vec![T::default(); n_qp_full * (n_size * n_size + 1)],
            n_qp_full,
            n_size,
        }
    }
}

impl<T: Copy> InvMColMajor<T> {
    /// Number of QP planes.
    pub fn n_qp_full(&self) -> usize {
        self.n_qp_full
    }

    /// `2 * n_elec`.
    pub fn n_size(&self) -> usize {
        self.n_size
    }

    /// Total entries, including the per-QP pad slot.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff the table holds zero entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Column-major linear index, **0-based**. Within one QP plane the
    /// stride pattern is `data[row + col * n_size]`; the trailing pad
    /// slot (`n_size * n_size`) is reserved for Pfaffian scratch.
    #[inline]
    fn idx(&self, qp: usize, row: usize, col: usize) -> usize {
        debug_assert!(qp < self.n_qp_full);
        debug_assert!(row < self.n_size);
        debug_assert!(col < self.n_size);
        qp * (self.n_size * self.n_size + 1) + row + col * self.n_size
    }

    /// Read `inv_m[qp][row, col]`.
    #[inline]
    pub fn get(&self, qp: usize, row: usize, col: usize) -> T {
        self.data[self.idx(qp, row, col)]
    }

    /// Write `inv_m[qp][row, col] = value`.
    #[inline]
    pub fn set(&mut self, qp: usize, row: usize, col: usize, value: T) {
        let k = self.idx(qp, row, col);
        self.data[k] = value;
    }

    /// Borrow the whole backing vector (including the per-QP pad slots).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the whole backing vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow one QP plane as a contiguous column-major slice of length
    /// `n_size * n_size` (the pad slot is dropped).
    pub fn qp_matrix_slice(&self, qp: usize) -> &[T] {
        let stride = self.n_size * self.n_size + 1;
        let start = qp * stride;
        &self.data[start..start + self.n_size * self.n_size]
    }

    /// Mutably borrow one QP plane (matrix portion only).
    pub fn qp_matrix_slice_mut(&mut self, qp: usize) -> &mut [T] {
        let stride = self.n_size * self.n_size + 1;
        let start = qp * stride;
        &mut self.data[start..start + self.n_size * self.n_size]
    }

    /// Read the Pfaffian-pad slot for one QP.
    pub fn pad_slot(&self, qp: usize) -> T {
        let stride = self.n_size * self.n_size + 1;
        self.data[qp * stride + self.n_size * self.n_size]
    }

    /// Write the Pfaffian-pad slot for one QP.
    pub fn set_pad_slot(&mut self, qp: usize, value: T) {
        let stride = self.n_size * self.n_size + 1;
        let k = qp * stride + self.n_size * self.n_size;
        self.data[k] = value;
    }

    /// Borrow one QP plane as a contiguous column-major matrix view.
    pub fn qp_matrix(&self, qp: usize) -> InvMPlane<'_, T> {
        let n = self.n_size;
        InvMPlane::new(self.qp_matrix_slice(qp), n)
    }

    /// Mutably borrow one QP plane as a contiguous column-major matrix view.
    pub fn qp_matrix_mut(&mut self, qp: usize) -> InvMPlaneMut<'_, T> {
        let n = self.n_size;
        InvMPlaneMut::new(self.qp_matrix_slice_mut(qp), n)
    }
}

/// One `inv_m[qp]` plane borrowed as an immutable column-major slice.
#[derive(Debug, Clone, Copy)]
pub struct InvMPlane<'a, T> {
    data: &'a [T],
    n: usize,
}

impl<'a, T> InvMPlane<'a, T> {
    pub fn new(data: &'a [T], n: usize) -> Self {
        Self { data, n }
    }
}

impl<T: Copy> InvMPlane<'_, T> {
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row + col * self.n]
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

/// One `inv_m[qp]` plane borrowed as a mutable column-major slice.
#[derive(Debug)]
pub struct InvMPlaneMut<'a, T> {
    data: &'a mut [T],
    n: usize,
}

impl<'a, T> InvMPlaneMut<'a, T> {
    pub fn new(data: &'a mut [T], n: usize) -> Self {
        Self { data, n }
    }
}

impl<T: Copy> InvMPlaneMut<'_, T> {
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row + col * self.n]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row + col * self.n] = value;
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Scalar accumulators
// ---------------------------------------------------------------------------

/// Per-sample energy accumulator (`EnergyData` in upstream).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyData {
    /// Correlation-sampling weight `wc`.
    pub wc: Complex64,
    /// `<H>`.
    pub etot: Complex64,
    /// `<H^2>`.
    pub etot2: Complex64,
    /// `<Sz>`.
    pub sztot: Complex64,
    /// `<Sz^2>`.
    pub sztot2: Complex64,
}

impl EnergyData {
    /// Mirror of `EnergyData()` -- all fields zero.
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Stochastic-reconfiguration buffers
// ---------------------------------------------------------------------------

/// Stochastic-reconfiguration optimisation buffers (`SROptData`).
///
/// `sr_opt_size = 1 + n_para` (matches upstream). The complex blocks
/// are always allocated -- in real mode they serve as working buffers
/// (mirroring the Julia comment "Complex arrays are used as working
/// buffers even in real mode").
#[derive(Debug, Clone, PartialEq)]
pub struct SROptData {
    /// `1 + n_para`.
    pub sr_opt_size: usize,

    // Complex SR buffers (always allocated; serve as scratch in real mode).
    /// `<O† O>` matrix (length `2*sr_opt_size * (2*sr_opt_size + 2)`).
    pub sr_opt_oo: Vec<Complex64>,
    /// `<H O>` vector (length `2*sr_opt_size`).
    pub sr_opt_ho: Vec<Complex64>,
    /// Per-sample calculation buffer (length `2*sr_opt_size`).
    pub sr_opt_o: Vec<Complex64>,
    /// All-sample storage (length `2*sr_opt_size * n_vmc_sample`).
    pub sr_opt_o_store: Vec<Complex64>,

    // Real SR buffers (empty when `all_complex == true`).
    /// Real `<O† O>` (length `sr_opt_size * (sr_opt_size + 2)`).
    pub sr_opt_oo_real: Vec<f64>,
    /// Real `<H O>` (length `sr_opt_size`).
    pub sr_opt_ho_real: Vec<f64>,
    /// Real per-sample calculation buffer (length `sr_opt_size`).
    pub sr_opt_o_real: Vec<f64>,
    /// Real all-sample storage (length `sr_opt_size * n_vmc_sample`).
    pub sr_opt_o_store_real: Vec<f64>,
}

impl SROptData {
    /// Allocate matching upstream `SROptData(sr_opt_size, n_vmc_sample, all_complex)`.
    pub fn zeros(sr_opt_size: usize, n_vmc_sample: usize, all_complex: bool) -> Self {
        let cplx_oo_len = 2 * sr_opt_size * (2 * sr_opt_size + 2);
        let cplx_vec_len = 2 * sr_opt_size;
        let cplx_store_len = 2 * sr_opt_size * n_vmc_sample;

        let (oo_real, ho_real, o_real, o_store_real) = if all_complex {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        } else {
            (
                vec![0.0; sr_opt_size * (sr_opt_size + 2)],
                vec![0.0; sr_opt_size],
                vec![0.0; sr_opt_size],
                vec![0.0; sr_opt_size * n_vmc_sample],
            )
        };

        Self {
            sr_opt_size,
            sr_opt_oo: vec![Complex64::new(0.0, 0.0); cplx_oo_len],
            sr_opt_ho: vec![Complex64::new(0.0, 0.0); cplx_vec_len],
            sr_opt_o: vec![Complex64::new(0.0, 0.0); cplx_vec_len],
            sr_opt_o_store: vec![Complex64::new(0.0, 0.0); cplx_store_len],
            sr_opt_oo_real: oo_real,
            sr_opt_ho_real: ho_real,
            sr_opt_o_real: o_real,
            sr_opt_o_store_real: o_store_real,
        }
    }

    #[inline]
    pub fn sr_opt_o_store_slice(&self, sample: usize) -> &[Complex64] {
        let stride = 2 * self.sr_opt_size;
        let start = sample * stride;
        &self.sr_opt_o_store[start..start + stride]
    }

    #[inline]
    pub fn sr_opt_o_store_slice_mut(&mut self, sample: usize) -> &mut [Complex64] {
        let stride = 2 * self.sr_opt_size;
        let start = sample * stride;
        &mut self.sr_opt_o_store[start..start + stride]
    }

    #[inline]
    pub fn sr_opt_o_store_real_slice(&self, sample: usize) -> &[f64] {
        let stride = self.sr_opt_size;
        let start = sample * stride;
        &self.sr_opt_o_store_real[start..start + stride]
    }

    #[inline]
    pub fn sr_opt_o_store_real_slice_mut(&mut self, sample: usize) -> &mut [f64] {
        let stride = self.sr_opt_size;
        let start = sample * stride;
        &mut self.sr_opt_o_store_real[start..start + stride]
    }
}

// ---------------------------------------------------------------------------
// Optimisation history
// ---------------------------------------------------------------------------

/// One `(energy, parameters)` snapshot from the SR loop (`OptDataPoint`).
#[derive(Debug, Clone, PartialEq)]
pub struct OptDataPoint {
    /// Energy at this iteration.
    pub energy: Complex64,
    /// Parameter snapshot used to compute `energy`.
    pub parameters: Vec<Complex64>,
}

// ---------------------------------------------------------------------------
// Electron configuration buffers
// ---------------------------------------------------------------------------

/// Per-sample / per-walker electron configuration (`ElectronConfiguration`).
///
/// The buffer sizes match upstream exactly. The `*_spn` arrays stay
/// empty when `use_fsz == false`, mirroring the Julia `Int[]` sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectronConfiguration {
    n_sample: usize,
    n_size: usize,
    n_site2: usize,
    n_proj: usize,

    /// `[sample][mi + si * n_elec]`.
    pub ele_idx: Vec<i64>,
    /// `[sample][ri + si * n_site]`.
    pub ele_cfg: Vec<i64>,
    /// `[sample][ri + si * n_site]`.
    pub ele_num: Vec<i64>,
    /// `[sample][proj]`.
    pub ele_proj_cnt: Vec<i64>,
    /// `[sample][mi + si * n_elec]`, only for FSZ mode.
    pub ele_spn: Vec<i64>,

    /// Single-sample scratch index buffer.
    pub tmp_ele_idx: Vec<i64>,
    /// Single-sample scratch configuration.
    pub tmp_ele_cfg: Vec<i64>,
    /// Single-sample scratch electron count.
    pub tmp_ele_num: Vec<i64>,
    /// Single-sample scratch projection count.
    pub tmp_ele_proj_cnt: Vec<i64>,
    /// Single-sample scratch spin buffer (FSZ only).
    pub tmp_ele_spn: Vec<i64>,

    /// Combined burn-in storage (matches upstream layout).
    pub burn_ele_idx: Vec<i64>,
    /// Burn-in configuration.
    pub burn_ele_cfg: Vec<i64>,
    /// Burn-in electron-number buffer.
    pub burn_ele_num: Vec<i64>,
    /// Burn-in projection-count buffer.
    pub burn_ele_proj_cnt: Vec<i64>,
    /// Burn-in spin buffer (FSZ only).
    pub burn_ele_spn: Vec<i64>,

    /// 10 statistics counters (hopping attempts, accepts, ...).
    pub counter: [i64; 10],
}

impl ElectronConfiguration {
    /// Mirror of `ElectronConfiguration(n_sample, n_site, n_elec, n_proj, use_fsz)`.
    pub fn zeros(
        n_sample: usize,
        n_site: usize,
        n_elec: usize,
        n_proj: usize,
        use_fsz: bool,
    ) -> Self {
        let n_size = 2 * n_elec;
        let n_site2 = 2 * n_site;
        let (ele_spn, tmp_ele_spn, burn_ele_spn, burn_total) = if use_fsz {
            (
                vec![0i64; n_sample * n_size],
                vec![0i64; n_size],
                vec![0i64; n_size],
                n_size + n_site2 + n_site2 + n_proj + n_size,
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                n_size + n_site2 + n_site2 + n_proj,
            )
        };
        Self {
            n_sample,
            n_size,
            n_site2,
            n_proj,
            ele_idx: vec![0; n_sample * n_size],
            ele_cfg: vec![0; n_sample * n_site2],
            ele_num: vec![0; n_sample * n_site2],
            ele_proj_cnt: vec![0; n_sample * n_proj],
            ele_spn,
            tmp_ele_idx: vec![0; n_size],
            tmp_ele_cfg: vec![0; n_site2],
            tmp_ele_num: vec![0; n_site2],
            tmp_ele_proj_cnt: vec![0; n_proj],
            tmp_ele_spn,
            burn_ele_idx: vec![0; burn_total],
            burn_ele_cfg: vec![0; n_site2],
            burn_ele_num: vec![0; n_site2],
            burn_ele_proj_cnt: vec![0; n_proj],
            burn_ele_spn,
            counter: [0; 10],
        }
    }

    #[inline]
    pub fn ele_idx_slice(&self, sample: usize) -> &[i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_size;
        &self.ele_idx[start..start + self.n_size]
    }

    #[inline]
    pub fn ele_idx_slice_mut(&mut self, sample: usize) -> &mut [i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_size;
        &mut self.ele_idx[start..start + self.n_size]
    }

    #[inline]
    pub fn ele_cfg_slice(&self, sample: usize) -> &[i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_site2;
        &self.ele_cfg[start..start + self.n_site2]
    }

    #[inline]
    pub fn ele_cfg_slice_mut(&mut self, sample: usize) -> &mut [i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_site2;
        &mut self.ele_cfg[start..start + self.n_site2]
    }

    #[inline]
    pub fn ele_num_slice(&self, sample: usize) -> &[i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_site2;
        &self.ele_num[start..start + self.n_site2]
    }

    #[inline]
    pub fn ele_num_slice_mut(&mut self, sample: usize) -> &mut [i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_site2;
        &mut self.ele_num[start..start + self.n_site2]
    }

    #[inline]
    pub fn ele_proj_cnt_slice(&self, sample: usize) -> &[i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_proj;
        &self.ele_proj_cnt[start..start + self.n_proj]
    }

    #[inline]
    pub fn ele_proj_cnt_slice_mut(&mut self, sample: usize) -> &mut [i64] {
        debug_assert!(sample < self.n_sample);
        let start = sample * self.n_proj;
        &mut self.ele_proj_cnt[start..start + self.n_proj]
    }

    #[inline]
    pub fn ele_spn_slice(&self, sample: usize) -> &[i64] {
        if self.ele_spn.is_empty() {
            &self.ele_spn
        } else {
            debug_assert!(sample < self.n_sample);
            let start = sample * self.n_size;
            &self.ele_spn[start..start + self.n_size]
        }
    }

    #[inline]
    pub fn ele_spn_slice_mut(&mut self, sample: usize) -> &mut [i64] {
        if self.ele_spn.is_empty() {
            &mut self.ele_spn
        } else {
            debug_assert!(sample < self.n_sample);
            let start = sample * self.n_size;
            &mut self.ele_spn[start..start + self.n_size]
        }
    }
}

// ---------------------------------------------------------------------------
// Slater matrices
// ---------------------------------------------------------------------------

/// Slater-matrix / inverse / Pfaffian triple (`SlaterMatrixData`).
///
/// Both the complex and real blocks are always allocated when
/// `all_complex == false` (mirroring upstream: even in real mode the
/// optimizer keeps a complex working buffer). In all-complex mode the
/// real fields stay empty.
#[derive(Debug, Clone, PartialEq)]
pub struct SlaterMatrixData {
    /// Complex Slater elements.
    pub slater_elm: SlaterElmFlat<Complex64>,
    /// Complex inverse table.
    pub inv_m: InvMColMajor<Complex64>,
    /// One Pfaffian per QP plane.
    pub pf_m: Vec<Complex64>,

    /// Real Slater elements (empty in all-complex mode).
    pub slater_elm_real: SlaterElmFlat<f64>,
    /// Real inverse table (empty in all-complex mode).
    pub inv_m_real: InvMColMajor<f64>,
    /// Real Pfaffian buffer (empty in all-complex mode).
    pub pf_m_real: Vec<f64>,
}

impl SlaterMatrixData {
    /// Mirror `SlaterMatrixData(n_qp_full, n_site, n_elec, all_complex)`.
    pub fn zeros(n_qp_full: usize, n_site: usize, n_elec: usize, all_complex: bool) -> Self {
        let n_qp_full = n_qp_full.max(1);
        let n_site = n_site.max(1);
        let n_elec = n_elec.max(1);
        let complex_slater = SlaterElmFlat::<Complex64>::zeros(n_qp_full, n_site);
        let complex_inv = InvMColMajor::<Complex64>::zeros(n_qp_full, n_elec);
        let complex_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
        let (real_slater, real_inv, real_pf) = if all_complex {
            (
                SlaterElmFlat::<f64>::zeros(0, 0),
                InvMColMajor::<f64>::zeros(0, 0),
                Vec::new(),
            )
        } else {
            (
                SlaterElmFlat::<f64>::zeros(n_qp_full, n_site),
                InvMColMajor::<f64>::zeros(n_qp_full, n_elec),
                vec![0.0; n_qp_full],
            )
        };
        Self {
            slater_elm: complex_slater,
            inv_m: complex_inv,
            pf_m: complex_pf,
            slater_elm_real: real_slater,
            inv_m_real: real_inv,
            pf_m_real: real_pf,
        }
    }
}

// ---------------------------------------------------------------------------
// PfaPack workspaces
// ---------------------------------------------------------------------------

/// Storage selector for [`PfaPackWorkspace`] / [`ThreadedPfaPackWorkspace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PfaPackMode {
    /// Allocate both real and complex scratch (matches the Julia default).
    Both,
    /// Allocate complex scratch only (`complex_only=true` in upstream).
    ComplexOnly,
    /// Allocate real scratch only (`real_only=true` in upstream).
    RealOnly,
}

/// Pre-allocated workspace for Pfaffian + inverse (`PfaPackWorkspace`).
#[derive(Debug, Clone, PartialEq)]
pub struct PfaPackWorkspace {
    /// `n_size`-side square scratch (real).
    pub buf_m_real: Vec<f64>,
    /// `n_size - 1`-length tridiagonal scratch (real).
    pub v_t_real: Vec<f64>,
    /// `n_size`-side square scratch for `utu2inv!` (real).
    pub m_work_real: Vec<f64>,

    /// `n_size`-side square scratch (complex).
    pub buf_m_complex: Vec<Complex64>,
    /// `n_size - 1`-length tridiagonal scratch (complex).
    pub v_t_complex: Vec<Complex64>,
    /// `n_size`-side square scratch for `utu2inv!` (complex).
    pub m_work_complex: Vec<Complex64>,

    /// 1-based pivot indices shared by the real and complex paths.
    pub pivots: Vec<PivotIndex1Based>,

    /// Original `n_size` so the consumer can sanity-check.
    pub n_size: usize,
}

impl PfaPackWorkspace {
    /// Pre-allocate buffers for `n_size`-side matrices.
    pub fn new(n_size: usize, mode: PfaPackMode) -> Self {
        let n_size = n_size.max(2);
        let real_n2 = match mode {
            PfaPackMode::Both | PfaPackMode::RealOnly => n_size * n_size,
            PfaPackMode::ComplexOnly => 0,
        };
        let real_v_t = match mode {
            PfaPackMode::Both | PfaPackMode::RealOnly => n_size - 1,
            PfaPackMode::ComplexOnly => 0,
        };
        let cplx_n2 = match mode {
            PfaPackMode::Both | PfaPackMode::ComplexOnly => n_size * n_size,
            PfaPackMode::RealOnly => 0,
        };
        let cplx_v_t = match mode {
            PfaPackMode::Both | PfaPackMode::ComplexOnly => n_size - 1,
            PfaPackMode::RealOnly => 0,
        };
        Self {
            buf_m_real: vec![0.0; real_n2],
            v_t_real: vec![0.0; real_v_t],
            m_work_real: vec![0.0; real_n2],
            buf_m_complex: vec![Complex64::new(0.0, 0.0); cplx_n2],
            v_t_complex: vec![Complex64::new(0.0, 0.0); cplx_v_t],
            m_work_complex: vec![Complex64::new(0.0, 0.0); cplx_n2],
            pivots: vec![PivotIndex1Based(0); n_size],
            n_size,
        }
    }
}

/// Thread-local container for [`PfaPackWorkspace`].
///
/// Phase 4.1 ships a single-process implementation: one workspace per
/// rayon worker (or just one for the main thread when rayon is not
/// active). The lock guards the lazy push-back when more workers show
/// up than were originally allocated. Phase 6 swaps the `Mutex` for an
/// atomic free-list once we benchmark the contention.
pub struct ThreadedPfaPackWorkspace {
    workspaces: Mutex<Vec<PfaPackWorkspace>>,
    n_size: usize,
    mode: PfaPackMode,
}

impl ThreadedPfaPackWorkspace {
    /// Allocate one workspace per thread (`Both` mode).
    pub fn new(n_size: usize, n_threads: usize) -> Self {
        Self::with_mode(n_size, n_threads, PfaPackMode::Both)
    }

    /// Allocate one workspace per thread in `mode`.
    pub fn with_mode(n_size: usize, n_threads: usize, mode: PfaPackMode) -> Self {
        let n_threads = n_threads.max(1);
        let workspaces = (0..n_threads)
            .map(|_| PfaPackWorkspace::new(n_size, mode))
            .collect();
        Self {
            workspaces: Mutex::new(workspaces),
            n_size,
            mode,
        }
    }

    /// Current pool length.
    pub fn len(&self) -> usize {
        self.workspaces.lock().expect("pool lock poisoned").len()
    }

    /// Returns `true` iff the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.workspaces
            .lock()
            .expect("pool lock poisoned")
            .is_empty()
    }

    /// Underlying matrix side length (`n_size`).
    pub fn n_size(&self) -> usize {
        self.n_size
    }

    /// Active allocation mode.
    pub fn mode(&self) -> PfaPackMode {
        self.mode
    }

    /// Ensure the pool has at least `n_threads` workspaces. Newly
    /// allocated workspaces inherit `n_size` and `mode`.
    pub fn ensure_capacity(&self, n_threads: usize) {
        let mut guard = self.workspaces.lock().expect("pool lock poisoned");
        while guard.len() < n_threads {
            guard.push(PfaPackWorkspace::new(self.n_size, self.mode));
        }
    }

    /// Move out one workspace (panics if the pool is empty). The
    /// caller is expected to return it via [`Self::release`] when the
    /// kernel finishes.
    pub fn take(&self) -> PfaPackWorkspace {
        let mut guard = self.workspaces.lock().expect("pool lock poisoned");
        guard
            .pop()
            .unwrap_or_else(|| PfaPackWorkspace::new(self.n_size, self.mode))
    }

    /// Return a workspace to the pool.
    pub fn release(&self, ws: PfaPackWorkspace) {
        let mut guard = self.workspaces.lock().expect("pool lock poisoned");
        guard.push(ws);
    }
}

impl std::fmt::Debug for ThreadedPfaPackWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadedPfaPackWorkspace")
            .field("len", &self.len())
            .field("n_size", &self.n_size)
            .field("mode", &self.mode)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Sampling workspace
// ---------------------------------------------------------------------------

/// Pre-allocated scratch for the sampling hot loop (`SamplingWorkspace`).
#[derive(Debug)]
pub struct SamplingWorkspace {
    /// Real per-QP `inv_m` scratch shaped `[n_size, n_size, n_qp_full]`.
    pub inv_m_real_temp: Vec<f64>,
    /// Real per-QP Pfaffian scratch (length `n_qp_full`).
    pub pf_m_real_temp: Vec<f64>,

    /// Complex per-QP `inv_m` scratch shaped `[n_size, n_size, n_qp_full]`.
    pub inv_m_temp: Vec<Complex64>,
    /// Complex per-QP Pfaffian scratch.
    pub pf_m_temp: Vec<Complex64>,

    /// Per-sample projection scratch (length `n_proj`).
    pub proj_cnt_new: Vec<i64>,
    /// Per-sample new real Pfaffian scratch (length `n_qp_full`).
    pub pf_m_new_real: Vec<f64>,
    /// Per-sample new complex Pfaffian scratch (length `n_qp_full`).
    pub pf_m_new: Vec<Complex64>,

    /// Cached local-spin array (length `n_site`).
    pub loc_spn: Vec<i64>,

    /// Thread-local PfaPack pool.
    pub pfapack: ThreadedPfaPackWorkspace,

    /// Sizes captured for sanity checks downstream.
    pub n_size: usize,
    /// `n_qp_full` captured at construction.
    pub n_qp_full: usize,
}

impl SamplingWorkspace {
    /// Mirror of `SamplingWorkspace(n_size, n_qp_full, n_proj, n_site)`.
    pub fn zeros(n_size: usize, n_qp_full: usize, n_proj: usize, n_site: usize) -> Self {
        let scratch_real = vec![0.0; n_size * n_size * n_qp_full];
        let scratch_cplx = vec![Complex64::new(0.0, 0.0); n_size * n_size * n_qp_full];
        Self {
            inv_m_real_temp: scratch_real,
            pf_m_real_temp: vec![0.0; n_qp_full],
            inv_m_temp: scratch_cplx,
            pf_m_temp: vec![Complex64::new(0.0, 0.0); n_qp_full],
            proj_cnt_new: vec![0; n_proj],
            pf_m_new_real: vec![0.0; n_qp_full],
            pf_m_new: vec![Complex64::new(0.0, 0.0); n_qp_full],
            loc_spn: vec![0; n_site],
            pfapack: ThreadedPfaPackWorkspace::new(n_size, 1),
            n_size,
            n_qp_full,
        }
    }
}

// ---------------------------------------------------------------------------
// Physical-quantities accumulator
// ---------------------------------------------------------------------------

/// Physical-observable accumulators for `NVMCCalMode=1` (`PhysicalQuantities`).
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalQuantities {
    /// Per-sample 1-body Green function (length `n_cis_ajs`).
    pub local_cis_ajs: Vec<Complex64>,
    /// Weighted-average 1-body Green function (length `n_cis_ajs`).
    pub phys_cis_ajs: Vec<Complex64>,

    /// Weighted-average 2-body product Green function (length `n_cis_ajs_ckt_alt`).
    pub phys_cis_ajs_ckt_alt: Vec<Complex64>,

    /// Per-sample direct 2-body Green function (length `n_cis_ajs_ckt_alt_dc`).
    pub local_cis_ajs_ckt_alt_dc: Vec<Complex64>,
    /// Weighted-average direct 2-body Green function (length `n_cis_ajs_ckt_alt_dc`).
    pub phys_cis_ajs_ckt_alt_dc: Vec<Complex64>,
}

impl PhysicalQuantities {
    /// Mirror of `PhysicalQuantities(n_cis_ajs, n_cis_ajs_ckt_alt, n_cis_ajs_ckt_alt_dc)`.
    pub fn zeros(n_cis_ajs: usize, n_cis_ajs_ckt_alt: usize, n_cis_ajs_ckt_alt_dc: usize) -> Self {
        Self {
            local_cis_ajs: vec![Complex64::new(0.0, 0.0); n_cis_ajs],
            phys_cis_ajs: vec![Complex64::new(0.0, 0.0); n_cis_ajs],
            phys_cis_ajs_ckt_alt: vec![Complex64::new(0.0, 0.0); n_cis_ajs_ckt_alt],
            local_cis_ajs_ckt_alt_dc: vec![Complex64::new(0.0, 0.0); n_cis_ajs_ckt_alt_dc],
            phys_cis_ajs_ckt_alt_dc: vec![Complex64::new(0.0, 0.0); n_cis_ajs_ckt_alt_dc],
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level container
// ---------------------------------------------------------------------------

/// Top-level VMC optimisation state (`VMCOptimizationState`).
///
/// Holds every per-iteration buffer the sampler / SR loop needs.
/// `phys_quantities` stays `None` for `NVMCCalMode=0` (optimisation)
/// and gets populated for `NVMCCalMode=1` (physics calc).
#[derive(Debug)]
pub struct VmcOptimizationState {
    /// Per-iteration energy accumulator.
    pub energy: EnergyData,
    /// Slater matrices.
    pub slater_matrix: SlaterMatrixData,
    /// Electron-configuration buffers.
    pub electron_config: ElectronConfiguration,
    /// SR buffers.
    pub sr_opt: SROptData,
    /// Optimisation history.
    pub opt_data: Vec<OptDataPoint>,
    /// Sampling workspace.
    pub workspace: SamplingWorkspace,
    /// Physical quantities (`Some` in measurement mode).
    pub phys_quantities: Option<PhysicalQuantities>,
}

impl VmcOptimizationState {
    /// Mirror of `VMCOptimizationState(n_site, n_elec, n_proj, n_para,
    /// n_qp_full, n_vmc_sample, all_complex, use_fsz)`.
    #[allow(clippy::too_many_arguments)]
    pub fn zeros(
        n_site: usize,
        n_elec: usize,
        n_proj: usize,
        n_para: usize,
        n_qp_full: usize,
        n_vmc_sample: usize,
        all_complex: bool,
        use_fsz: bool,
    ) -> Self {
        let n_size = 2 * n_elec;
        Self {
            energy: EnergyData::new(),
            slater_matrix: SlaterMatrixData::zeros(n_qp_full, n_site, n_elec, all_complex),
            electron_config: ElectronConfiguration::zeros(
                n_vmc_sample,
                n_site,
                n_elec,
                n_proj,
                use_fsz,
            ),
            sr_opt: SROptData::zeros(1 + n_para, n_vmc_sample, all_complex),
            opt_data: Vec::new(),
            workspace: SamplingWorkspace::zeros(n_size, n_qp_full, n_proj, n_site),
            phys_quantities: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slater_elm_flat_row_major_indexing() {
        let n_qp_full = 2;
        let n_site = 3;
        let mut a = SlaterElmFlat::<f64>::zeros(n_qp_full, n_site);
        a.set(0, 1, 4, 11.0);
        a.set(1, 5, 2, 22.0);
        // Row-major linearisation matches `(qp * n_site2 + row) * n_site2 + col`.
        assert_eq!(a.as_slice()[(0 * 6 + 1) * 6 + 4], 11.0);
        assert_eq!(a.as_slice()[(1 * 6 + 5) * 6 + 2], 22.0);
        assert_eq!(a.get(0, 1, 4), 11.0);
        assert_eq!(a.get(1, 5, 2), 22.0);
        assert_eq!(a.qp_slice(0).len(), 6 * 6);
        assert_eq!(a.qp_slice(1).len(), 6 * 6);
    }

    #[test]
    fn inv_m_column_major_indexing_with_pad() {
        let n_qp_full = 2;
        let n_elec = 3;
        let n_size = 2 * n_elec;
        let mut a = InvMColMajor::<f64>::zeros(n_qp_full, n_elec);
        a.set(0, 1, 4, 11.0);
        a.set(1, 5, 2, 22.0);
        a.set_pad_slot(1, 99.0);
        // Column-major linearisation matches `qp * (n_size^2 + 1) + row + col * n_size`.
        let stride = n_size * n_size + 1;
        assert_eq!(a.as_slice()[0 * stride + 1 + 4 * n_size], 11.0);
        assert_eq!(a.as_slice()[1 * stride + 5 + 2 * n_size], 22.0);
        assert_eq!(a.pad_slot(1), 99.0);
        assert_eq!(a.qp_matrix_slice(0).len(), n_size * n_size);
        assert_eq!(a.qp_matrix_slice(1).len(), n_size * n_size);
    }

    #[test]
    fn slater_elm_vec_layout_preserves_qp_row_major_planes() {
        let mut a = SlaterElmFlat::<f64>::zeros(2, 3);
        a.set(0, 1, 4, 11.0);
        a.set(1, 5, 2, 22.0);

        assert_eq!(a.as_slice()[(0 * 6 + 1) * 6 + 4], 11.0);
        assert_eq!(a.as_slice()[(1 * 6 + 5) * 6 + 2], 22.0);
        assert_eq!(a.qp_slice(0)[1 * 6 + 4], 11.0);
        assert_eq!(a.qp_slice(1)[5 * 6 + 2], 22.0);
        assert_eq!(a.get(0, 1, 4), 11.0);
        assert_eq!(a.get(1, 5, 2), 22.0);
    }

    #[test]
    fn inv_m_vec_layout_preserves_qp_matrix_layout_and_pad_slot() {
        let mut a = InvMColMajor::<f64>::zeros(2, 3);
        a.set(0, 1, 4, 11.0);
        a.set(1, 5, 2, 22.0);
        a.set_pad_slot(0, 7.0);
        a.set_pad_slot(1, 8.0);

        let n_size = 6;
        assert_eq!(a.qp_matrix_slice(0)[1 + 4 * n_size], 11.0);
        assert_eq!(a.qp_matrix_slice(1)[5 + 2 * n_size], 22.0);
        assert_eq!(a.pad_slot(0), 7.0);
        assert_eq!(a.pad_slot(1), 8.0);
        assert_eq!(a.qp_matrix_slice(0).len(), n_size * n_size);
        assert_eq!(a.qp_matrix_slice(1).len(), n_size * n_size);
    }

    #[test]
    fn electron_config_sample_accessors_preserve_legacy_flat_order() {
        let mut cfg = ElectronConfiguration::zeros(3, 2, 2, 4, true);

        cfg.ele_idx_slice_mut(1)[0] = 11;
        cfg.ele_cfg_slice_mut(1)[3] = 22;
        cfg.ele_num_slice_mut(2)[1] = 33;
        cfg.ele_proj_cnt_slice_mut(0)[2] = 44;
        cfg.ele_spn_slice_mut(2)[1] = 55;

        assert_eq!(cfg.ele_idx_slice(1).len(), 4);
        assert_eq!(cfg.ele_cfg_slice(1).len(), 4);
        assert_eq!(cfg.ele_num_slice(2).len(), 4);
        assert_eq!(cfg.ele_proj_cnt_slice(0).len(), 4);
        assert_eq!(cfg.ele_spn_slice(2).len(), 4);
        assert_eq!(cfg.ele_idx[1 * 4], 11);
        assert_eq!(cfg.ele_cfg[1 * 4 + 3], 22);
        assert_eq!(cfg.ele_num[2 * 4 + 1], 33);
        assert_eq!(cfg.ele_proj_cnt[2], 44);
        assert_eq!(cfg.ele_spn[2 * 4 + 1], 55);
    }

    #[test]
    fn sr_store_sample_accessors_preserve_component_major_order() {
        let mut sro = SROptData::zeros(4, 3, false);
        let sample = 2;
        sro.sr_opt_o_store_slice_mut(sample)[1] = Complex64::new(1.5, -2.5);
        sro.sr_opt_o_store_real_slice_mut(sample)[3] = 7.25;

        assert_eq!(sro.sr_opt_o_store_slice(sample).len(), 2 * 4);
        assert_eq!(sro.sr_opt_o_store_real_slice(sample).len(), 4);
        assert_eq!(
            sro.sr_opt_o_store[2 * (2 * 4) + 1],
            Complex64::new(1.5, -2.5)
        );
        assert_eq!(sro.sr_opt_o_store_real[2 * 4 + 3], 7.25);
    }

    #[test]
    fn sr_opt_real_mode_allocates_both_blocks() {
        let sro = SROptData::zeros(5, 7, false);
        assert_eq!(sro.sr_opt_size, 5);
        assert_eq!(sro.sr_opt_oo.len(), 2 * 5 * (2 * 5 + 2));
        assert_eq!(sro.sr_opt_ho.len(), 2 * 5);
        assert_eq!(sro.sr_opt_o.len(), 2 * 5);
        assert_eq!(sro.sr_opt_o_store.len(), 2 * 5 * 7);
        assert_eq!(sro.sr_opt_oo_real.len(), 5 * (5 + 2));
        assert_eq!(sro.sr_opt_ho_real.len(), 5);
        assert_eq!(sro.sr_opt_o_real.len(), 5);
        assert_eq!(sro.sr_opt_o_store_real.len(), 5 * 7);
    }

    #[test]
    fn sr_opt_all_complex_mode_drops_real_blocks() {
        let sro = SROptData::zeros(5, 7, true);
        assert!(sro.sr_opt_oo_real.is_empty());
        assert!(sro.sr_opt_ho_real.is_empty());
        assert!(sro.sr_opt_o_real.is_empty());
        assert!(sro.sr_opt_o_store_real.is_empty());
    }

    #[test]
    fn electron_config_fsz_allocates_spin_blocks() {
        let cfg = ElectronConfiguration::zeros(4, 3, 2, 5, true);
        let n_size = 2 * 2;
        let n_site2 = 2 * 3;
        assert_eq!(cfg.ele_idx.len(), 4 * n_size);
        assert_eq!(cfg.ele_cfg.len(), 4 * n_site2);
        assert_eq!(cfg.ele_spn.len(), 4 * n_size);
        assert_eq!(cfg.tmp_ele_spn.len(), n_size);
        // FSZ burn-in storage adds an extra n_size for the spin buffer.
        assert_eq!(
            cfg.burn_ele_idx.len(),
            n_size + n_site2 + n_site2 + 5 + n_size,
        );
    }

    #[test]
    fn electron_config_non_fsz_skips_spin_blocks() {
        let cfg = ElectronConfiguration::zeros(4, 3, 2, 5, false);
        assert!(cfg.ele_spn.is_empty());
        assert!(cfg.tmp_ele_spn.is_empty());
        assert!(cfg.burn_ele_spn.is_empty());
    }

    #[test]
    fn slater_matrix_real_mode_allocates_real_blocks() {
        let sm = SlaterMatrixData::zeros(2, 3, 4, false);
        assert!(!sm.slater_elm_real.is_empty());
        assert!(!sm.inv_m_real.is_empty());
        assert_eq!(sm.pf_m_real.len(), 2);
        // Sizes match Julia
        assert_eq!(sm.slater_elm.len(), 2 * 6 * 6);
        assert_eq!(sm.inv_m.len(), 2 * (8 * 8 + 1));
    }

    #[test]
    fn slater_matrix_all_complex_mode_drops_real_blocks() {
        let sm = SlaterMatrixData::zeros(2, 3, 4, true);
        assert!(sm.slater_elm_real.is_empty());
        assert!(sm.inv_m_real.is_empty());
        assert!(sm.pf_m_real.is_empty());
    }

    #[test]
    fn pfapack_workspace_lengths_match_upstream() {
        let ws = PfaPackWorkspace::new(8, PfaPackMode::Both);
        assert_eq!(ws.buf_m_real.len(), 64);
        assert_eq!(ws.v_t_real.len(), 7);
        assert_eq!(ws.m_work_real.len(), 64);
        assert_eq!(ws.buf_m_complex.len(), 64);
        assert_eq!(ws.v_t_complex.len(), 7);
        assert_eq!(ws.m_work_complex.len(), 64);
        assert_eq!(ws.pivots.len(), 8);

        let ws_real = PfaPackWorkspace::new(8, PfaPackMode::RealOnly);
        assert_eq!(ws_real.buf_m_complex.len(), 0);
        let ws_cplx = PfaPackWorkspace::new(8, PfaPackMode::ComplexOnly);
        assert_eq!(ws_cplx.buf_m_real.len(), 0);
    }

    #[test]
    fn threaded_pool_take_release_and_capacity() {
        let pool = ThreadedPfaPackWorkspace::new(4, 2);
        assert_eq!(pool.len(), 2);
        let ws = pool.take();
        assert_eq!(pool.len(), 1);
        pool.release(ws);
        assert_eq!(pool.len(), 2);
        pool.ensure_capacity(5);
        assert_eq!(pool.len(), 5);
    }

    #[test]
    fn vmc_state_top_level_sizing() {
        let state = VmcOptimizationState::zeros(
            /* n_site */ 6, /* n_elec */ 3, /* n_proj */ 2, /* n_para */ 7,
            /* n_qp_full */ 4, /* n_vmc_sample */ 10, /* all_complex */ false,
            /* use_fsz */ false,
        );
        assert_eq!(state.sr_opt.sr_opt_size, 1 + 7);
        assert_eq!(state.slater_matrix.slater_elm.n_qp_full(), 4);
        assert_eq!(state.slater_matrix.inv_m.n_size(), 6);
        assert_eq!(state.electron_config.ele_idx.len(), 10 * 6);
        assert_eq!(state.workspace.n_qp_full, 4);
        assert_eq!(state.workspace.n_size, 6);
        assert!(state.phys_quantities.is_none());
    }
}
