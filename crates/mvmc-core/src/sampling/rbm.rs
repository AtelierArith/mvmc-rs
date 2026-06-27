//! Phase 4.3.2 — RBM counters and log-cosh ratios.
//!
//! Port target: `make_rbm_cnt`, `update_rbm_cnt_hopping!`,
//! `log_rbm_ratio`, `log_rbm_val`, `_rbm_log_cosh_stable` in
//! `MVMCOptimizers.jl/src/vmc_sampling.jl`.
//!
//! ## Scope
//!
//! This module ships the bit-faithful kernels, but it does **not** yet
//! wire them through [`mvmc_expert_parsers::ExpertModeData`] -- the
//! Phase 3 RBM parsers are still pending, so the parsed-data carrier
//! has no RBM term vectors to feed in. The 4 upstream namelists used
//! by the round-trip suite contain no RBM blocks, so this is safe: the
//! kernels return the same `0` / no-op as Julia's
//! `has_rbm_terms(data) == false` branch.
//!
//! Callers (and the parity dumper at
//! `extern/Julia-mVMC/tools/dump_rbm_reference.jl`) construct an
//! [`RbmConfig`] manually — the borrow-only view bundles the 9 RBM
//! term slices, the three `nneuron_*` widths, and the global `n_site`
//! / `nblock_size_rbm_ratio`. Once Phase 3 RBM parsers land, an
//! `impl<'a> From<&'a ExpertModeData> for RbmConfig<'a>` will close the
//! loop without touching any of these kernels.

use num_complex::Complex64;

/// Charge / spin / general RBM physical-layer term (one parameter per
/// (site, idx) pair).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbmPhysLayerTerm {
    /// Physical site index `ri`.
    pub site: i64,
    /// Parameter slot inside the family (0-based).
    pub idx: i64,
    /// Parameter value `RBM[idx]`.
    pub value: Complex64,
}

/// General-RBM physical-layer term (carries an explicit spin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbmGeneralPhysLayerTerm {
    /// Physical site `ri`.
    pub site: i64,
    /// 0 = up, 1 = down.
    pub spin: u8,
    /// Parameter slot inside the family (0-based).
    pub idx: i64,
    /// Parameter value `RBM[idx]`.
    pub value: Complex64,
}

/// Charge / spin / general RBM hidden-layer term (one parameter per
/// hidden neuron).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbmHiddenLayerTerm {
    /// Hidden-neuron index `hi`.
    pub site: i64,
    /// Parameter value.
    pub value: Complex64,
}

/// Charge / spin physical-hidden coupling term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbmPhysHiddenTerm {
    /// Physical site `ri`.
    pub site1: i64,
    /// Hidden neuron `hi`.
    pub site2: i64,
    /// Coupling parameter.
    pub value: Complex64,
}

/// General physical-hidden coupling term (carries a spin index).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbmGeneralPhysHiddenTerm {
    /// Physical site `ri`.
    pub site1: i64,
    /// 0 = up, 1 = down.
    pub spin: u8,
    /// Hidden neuron `hi`.
    pub site2: i64,
    /// Coupling parameter.
    pub value: Complex64,
}

/// Borrow-only view of the RBM terms + scalar config needed by the
/// kernels in this module.
///
/// All slices are allowed to be empty (mirrors Julia's empty-vector
/// branches). When every slice is empty and every `nneuron_*` is zero
/// the kernels short-circuit to the same zero / no-op behaviour as
/// Julia's `has_rbm_terms(data) == false` branch.
#[derive(Debug, Clone, Copy)]
pub struct RbmConfig<'a> {
    /// `data.modpara.nsite`.
    pub n_site: usize,
    /// Block size for the hidden-layer principal-log batching
    /// (`data.modpara.nblock_size_rbm_ratio`).
    pub nblock_size_rbm_ratio: usize,
    /// Lower bound on the per-family hidden width (`data.modpara.nneuron_*`).
    pub nneuron_charge: usize,
    /// Lower bound on the per-family hidden width.
    pub nneuron_spin: usize,
    /// Lower bound on the per-family hidden width.
    pub nneuron_general: usize,

    /// Charge RBM physical-layer terms.
    pub charge_phys: &'a [RbmPhysLayerTerm],
    /// Spin RBM physical-layer terms.
    pub spin_phys: &'a [RbmPhysLayerTerm],
    /// General RBM physical-layer terms.
    pub general_phys: &'a [RbmGeneralPhysLayerTerm],

    /// Charge RBM hidden-layer terms.
    pub charge_hidden: &'a [RbmHiddenLayerTerm],
    /// Spin RBM hidden-layer terms.
    pub spin_hidden: &'a [RbmHiddenLayerTerm],
    /// General RBM hidden-layer terms.
    pub general_hidden: &'a [RbmHiddenLayerTerm],

    /// Charge phys-hidden coupling terms.
    pub charge_phys_hidden: &'a [RbmPhysHiddenTerm],
    /// Spin phys-hidden coupling terms.
    pub spin_phys_hidden: &'a [RbmPhysHiddenTerm],
    /// General phys-hidden coupling terms.
    pub general_phys_hidden: &'a [RbmGeneralPhysHiddenTerm],
}

impl<'a> RbmConfig<'a> {
    /// True iff any RBM term family is non-empty (mirrors `has_rbm_terms`).
    pub fn has_terms(&self) -> bool {
        !self.charge_phys.is_empty()
            || !self.spin_phys.is_empty()
            || !self.general_phys.is_empty()
            || !self.charge_hidden.is_empty()
            || !self.spin_hidden.is_empty()
            || !self.general_hidden.is_empty()
            || !self.charge_phys_hidden.is_empty()
            || !self.spin_phys_hidden.is_empty()
            || !self.general_phys_hidden.is_empty()
    }

    fn n_charge_phys(&self) -> usize {
        max_idx_plus_one(self.charge_phys.iter().map(|t| t.idx))
    }
    fn n_spin_phys(&self) -> usize {
        max_idx_plus_one(self.spin_phys.iter().map(|t| t.idx))
    }
    fn n_general_phys(&self) -> usize {
        max_idx_plus_one(self.general_phys.iter().map(|t| t.idx))
    }

    fn n_charge_neuron(&self) -> usize {
        rbm_neuron_width(
            self.nneuron_charge,
            self.charge_hidden.iter().map(|t| t.site),
            self.charge_phys_hidden.iter().map(|t| t.site2),
        )
    }
    fn n_spin_neuron(&self) -> usize {
        rbm_neuron_width(
            self.nneuron_spin,
            self.spin_hidden.iter().map(|t| t.site),
            self.spin_phys_hidden.iter().map(|t| t.site2),
        )
    }
    fn n_general_neuron(&self) -> usize {
        rbm_neuron_width(
            self.nneuron_general,
            self.general_hidden.iter().map(|t| t.site),
            self.general_phys_hidden.iter().map(|t| t.site2),
        )
    }
}

fn max_idx_plus_one<I: IntoIterator<Item = i64>>(it: I) -> usize {
    let mut max = -1i64;
    for v in it {
        if v > max {
            max = v;
        }
    }
    if max < 0 {
        0
    } else {
        (max as usize) + 1
    }
}

fn rbm_neuron_width<H, P>(modpara_n: usize, hidden_sites: H, phys_hidden_sites: P) -> usize
where
    H: IntoIterator<Item = i64>,
    P: IntoIterator<Item = i64>,
{
    let nh = max_idx_plus_one(hidden_sites);
    let nph = max_idx_plus_one(phys_hidden_sites);
    modpara_n.max(nh).max(nph)
}

/// `make_rbm_cnt(ele_num, data)` -- build RBM counters from `ele_num`.
/// Returns the empty vector when `cfg.has_terms()` is false (matches
/// upstream `has_rbm_terms == false`).
pub fn make_rbm_cnt(ele_num: &[i64], cfg: &RbmConfig<'_>) -> Vec<Complex64> {
    if !cfg.has_terms() {
        return Vec::new();
    }
    let n_site = cfg.n_site;
    let n_charge_phys = cfg.n_charge_phys();
    let n_spin_phys = cfg.n_spin_phys();
    let n_general_phys = cfg.n_general_phys();
    let n_phys = n_charge_phys + n_spin_phys + n_general_phys;
    let n_charge_neuron = cfg.n_charge_neuron();
    let n_spin_neuron = cfg.n_spin_neuron();
    let n_general_neuron = cfg.n_general_neuron();

    let len = n_phys + n_charge_neuron + n_spin_neuron + n_general_neuron;
    let mut rbm = vec![Complex64::new(0.0, 0.0); len];
    if n_site == 0 || ele_num.len() < 2 * n_site {
        return rbm;
    }

    let spin_phys_offset = n_charge_phys;
    let general_phys_offset = n_charge_phys + n_spin_phys;
    let hidden_offset = n_phys;
    let spin_hidden_offset = hidden_offset + n_charge_neuron;
    let general_hidden_offset = hidden_offset + n_charge_neuron + n_spin_neuron;

    // Physical layer.
    for term in cfg.charge_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site {
            continue;
        }
        let idx = term.idx;
        if idx < 0 || (idx as usize) >= n_charge_phys {
            continue;
        }
        let ri = ri as usize;
        rbm[idx as usize] += Complex64::new((ele_num[ri] + ele_num[ri + n_site] - 1) as f64, 0.0);
    }
    for term in cfg.spin_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site {
            continue;
        }
        let idx = term.idx;
        if idx < 0 || (idx as usize) >= n_spin_phys {
            continue;
        }
        let ri = ri as usize;
        rbm[spin_phys_offset + idx as usize] +=
            Complex64::new((ele_num[ri] - ele_num[ri + n_site]) as f64, 0.0);
    }
    for term in cfg.general_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site || term.spin > 1 {
            continue;
        }
        let idx = term.idx;
        if idx < 0 || (idx as usize) >= n_general_phys {
            continue;
        }
        let rsi = (ri as usize) + (term.spin as usize) * n_site;
        rbm[general_phys_offset + idx as usize] +=
            Complex64::new((2 * ele_num[rsi] - 1) as f64, 0.0);
    }

    // Hidden-layer biases.
    for term in cfg.charge_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_charge_neuron {
            continue;
        }
        rbm[hidden_offset + hi as usize] += term.value;
    }
    for term in cfg.spin_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_spin_neuron {
            continue;
        }
        rbm[spin_hidden_offset + hi as usize] += term.value;
    }
    for term in cfg.general_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_general_neuron {
            continue;
        }
        rbm[general_hidden_offset + hi as usize] += term.value;
    }

    // Physical <-> hidden couplings.
    for term in cfg.charge_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0 || (ri as usize) >= n_site || hi < 0 || (hi as usize) >= n_charge_neuron {
            continue;
        }
        let ri = ri as usize;
        let xi = (ele_num[ri] + ele_num[ri + n_site] - 1) as f64;
        rbm[hidden_offset + hi as usize] += term.value * Complex64::new(xi, 0.0);
    }
    for term in cfg.spin_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0 || (ri as usize) >= n_site || hi < 0 || (hi as usize) >= n_spin_neuron {
            continue;
        }
        let ri = ri as usize;
        let xi = (ele_num[ri] - ele_num[ri + n_site]) as f64;
        rbm[spin_hidden_offset + hi as usize] += term.value * Complex64::new(xi, 0.0);
    }
    for term in cfg.general_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0
            || (ri as usize) >= n_site
            || term.spin > 1
            || hi < 0
            || (hi as usize) >= n_general_neuron
        {
            continue;
        }
        let rsi = (ri as usize) + (term.spin as usize) * n_site;
        let xi = (2 * ele_num[rsi] - 1) as f64;
        rbm[general_hidden_offset + hi as usize] += term.value * Complex64::new(xi, 0.0);
    }

    rbm
}

/// `update_rbm_cnt_hopping!(rbm_cnt_new, rbm_cnt_old, ri, rj, s, data)`.
///
/// Unlike [`crate::sampling::projection::update_proj_cnt`], this update
/// does **not** read `ele_num`, so the call ordering vs. `update_ele_config`
/// is irrelevant.
pub fn update_rbm_cnt_hopping(
    rbm_cnt_new: &mut [Complex64],
    rbm_cnt_old: &[Complex64],
    ri: i64,
    rj: i64,
    spin: u8,
    cfg: &RbmConfig<'_>,
) {
    let n = rbm_cnt_new.len().min(rbm_cnt_old.len());
    if !std::ptr::eq(rbm_cnt_new.as_ptr(), rbm_cnt_old.as_ptr()) {
        rbm_cnt_new[..n].copy_from_slice(&rbm_cnt_old[..n]);
    }
    if ri == rj {
        return;
    }
    let n_site = cfg.n_site;
    let rsi = ri + (spin as i64) * (n_site as i64);
    let rsj = rj + (spin as i64) * (n_site as i64);

    let n_charge_phys = cfg.n_charge_phys();
    let n_spin_phys = cfg.n_spin_phys();
    let n_general_phys = cfg.n_general_phys();
    let n_phys = n_charge_phys + n_spin_phys + n_general_phys;
    let n_charge_neuron = cfg.n_charge_neuron();
    let n_spin_neuron = cfg.n_spin_neuron();
    let _ = n_spin_neuron; // used below via offsets

    let spin_phys_offset = n_charge_phys;
    let general_phys_offset = n_charge_phys + n_spin_phys;
    let charge_cnt_offset = n_phys;
    let spin_cnt_offset = n_phys + n_charge_neuron;
    let general_cnt_offset = n_phys + n_charge_neuron + n_spin_neuron;

    let spin_delta = (1i64 - 2 * (spin as i64)) as f64;

    // Physical layer.
    for term in cfg.charge_phys {
        let idx = term.idx;
        if idx < 0 {
            continue;
        }
        let pos = idx as usize;
        if pos >= rbm_cnt_new.len() {
            continue;
        }
        if term.site == ri {
            rbm_cnt_new[pos] -= Complex64::new(1.0, 0.0);
        } else if term.site == rj {
            rbm_cnt_new[pos] += Complex64::new(1.0, 0.0);
        }
    }
    for term in cfg.spin_phys {
        let idx = term.idx;
        if idx < 0 {
            continue;
        }
        let pos = spin_phys_offset + idx as usize;
        if pos >= rbm_cnt_new.len() {
            continue;
        }
        if term.site == ri {
            rbm_cnt_new[pos] -= Complex64::new(spin_delta, 0.0);
        } else if term.site == rj {
            rbm_cnt_new[pos] += Complex64::new(spin_delta, 0.0);
        }
    }
    for term in cfg.general_phys {
        let idx = term.idx;
        if idx < 0 {
            continue;
        }
        let pos = general_phys_offset + idx as usize;
        if pos >= rbm_cnt_new.len() {
            continue;
        }
        let r = term.site + (term.spin as i64) * (n_site as i64);
        if r == rsi {
            rbm_cnt_new[pos] -= Complex64::new(2.0, 0.0);
        } else if r == rsj {
            rbm_cnt_new[pos] += Complex64::new(2.0, 0.0);
        }
    }

    // Physical <-> hidden coupling.
    for term in cfg.charge_phys_hidden {
        if term.site1 == ri || term.site1 == rj {
            let pos = charge_cnt_offset + term.site2 as usize;
            if pos >= rbm_cnt_new.len() {
                continue;
            }
            if term.site1 == ri {
                rbm_cnt_new[pos] -= term.value;
            } else {
                rbm_cnt_new[pos] += term.value;
            }
        }
    }
    for term in cfg.spin_phys_hidden {
        if term.site1 == ri || term.site1 == rj {
            let pos = spin_cnt_offset + term.site2 as usize;
            if pos >= rbm_cnt_new.len() {
                continue;
            }
            let delta = Complex64::new(spin_delta, 0.0) * term.value;
            if term.site1 == ri {
                rbm_cnt_new[pos] -= delta;
            } else {
                rbm_cnt_new[pos] += delta;
            }
        }
    }
    for term in cfg.general_phys_hidden {
        let pos = general_cnt_offset + term.site2 as usize;
        if pos >= rbm_cnt_new.len() {
            continue;
        }
        let r = term.site1 + (term.spin as i64) * (n_site as i64);
        if r == rsi {
            rbm_cnt_new[pos] -= Complex64::new(2.0, 0.0) * term.value;
        } else if r == rsj {
            rbm_cnt_new[pos] += Complex64::new(2.0, 0.0) * term.value;
        }
    }
}

/// Numerically stable `log(2 cosh(z))` with the upstream sign flip.
#[inline]
pub fn log_cosh_stable(z: Complex64) -> Complex64 {
    let zp = if z.re <= 0.0 { -z } else { z };
    // log(2*cosh(z)) = z + log(1 + exp(-2z)) - log(2),
    // but upstream calls `log1p`, so we do the same and drop a final `- log(2)`
    // since the C reference subtracts `log(2.0)` too.
    let two = Complex64::new(2.0, 0.0);
    let log2 = Complex64::new(std::f64::consts::LN_2, 0.0);
    zp + (Complex64::new(1.0, 0.0) + (-two * zp).exp()).ln() - log2
}

/// `log_rbm_ratio(rbm_cnt_new, rbm_cnt_old, data)` -- principal-log
/// + block-product algorithm, matching Julia.
pub fn log_rbm_ratio(
    rbm_cnt_new: &[Complex64],
    rbm_cnt_old: &[Complex64],
    cfg: &RbmConfig<'_>,
) -> Complex64 {
    if !cfg.has_terms() {
        return Complex64::new(0.0, 0.0);
    }
    let n_charge_phys = cfg.n_charge_phys();
    let n_spin_phys = cfg.n_spin_phys();
    let n_general_phys = cfg.n_general_phys();
    let n_phys = n_charge_phys + n_spin_phys + n_general_phys;

    // Materialise the physical-layer parameter vector with the same
    // family ordering as `make_rbm_cnt`.
    let mut phys_params = vec![Complex64::new(0.0, 0.0); n_phys];
    for term in cfg.charge_phys {
        let idx = term.idx;
        if idx >= 0 && (idx as usize) < n_charge_phys {
            phys_params[idx as usize] = term.value;
        }
    }
    for term in cfg.spin_phys {
        let idx = term.idx;
        if idx >= 0 && (idx as usize) < n_spin_phys {
            phys_params[n_charge_phys + idx as usize] = term.value;
        }
    }
    for term in cfg.general_phys {
        let idx = term.idx;
        if idx >= 0 && (idx as usize) < n_general_phys {
            phys_params[n_charge_phys + n_spin_phys + idx as usize] = term.value;
        }
    }

    let n = rbm_cnt_new.len().min(rbm_cnt_old.len());
    let n_phys_eff = n_phys.min(n);

    let mut z = Complex64::new(0.0, 0.0);
    for i in 0..n_phys_eff {
        z += phys_params[i] * (rbm_cnt_new[i] - rbm_cnt_old[i]);
    }

    let n_hidden = n.saturating_sub(n_phys_eff);
    if n_hidden == 0 {
        return z;
    }

    let block_size = cfg.nblock_size_rbm_ratio.max(1);
    let n_blk = (n_hidden - 1) / block_size + 1;

    let two = Complex64::new(2.0, 0.0);
    for iblk in 0..n_blk {
        let hist = iblk * block_size;
        let hiend = (hist + block_size).min(n_hidden);
        let mut zz = Complex64::new(1.0, 0.0);
        for hi in hist..hiend {
            let idx = n_phys_eff + hi;
            let mut rbm_new = rbm_cnt_new[idx];
            let mut rbm_old = rbm_cnt_old[idx];
            if rbm_new.re <= 0.0 {
                rbm_new = -rbm_new;
            }
            if rbm_old.re <= 0.0 {
                rbm_old = -rbm_old;
            }
            z += rbm_new - rbm_old;
            zz *= (Complex64::new(1.0, 0.0) + (-two * rbm_new).exp())
                / (Complex64::new(1.0, 0.0) + (-two * rbm_old).exp());
        }
        z += zz.ln();
    }
    z
}

/// `log_rbm_val(ele_num, data)` -- principal log of the RBM weight.
pub fn log_rbm_val(ele_num: &[i64], cfg: &RbmConfig<'_>) -> Complex64 {
    if !cfg.has_terms() {
        return Complex64::new(0.0, 0.0);
    }
    let n_site = cfg.n_site;
    if n_site == 0 || ele_num.len() < 2 * n_site {
        return Complex64::new(0.0, 0.0);
    }
    let n_charge = cfg.nneuron_charge;
    let n_spin = cfg.nneuron_spin;
    let n_general = cfg.nneuron_general;

    let mut hidden_charge = vec![Complex64::new(0.0, 0.0); n_charge];
    let mut hidden_spin = vec![Complex64::new(0.0, 0.0); n_spin];
    let mut hidden_general = vec![Complex64::new(0.0, 0.0); n_general];
    let mut z = Complex64::new(0.0, 0.0);

    // Physical layer biases.
    for term in cfg.charge_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site {
            continue;
        }
        let ri = ri as usize;
        z += term.value * Complex64::new((ele_num[ri] + ele_num[ri + n_site] - 1) as f64, 0.0);
    }
    for term in cfg.spin_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site {
            continue;
        }
        let ri = ri as usize;
        z += term.value * Complex64::new((ele_num[ri] - ele_num[ri + n_site]) as f64, 0.0);
    }
    for term in cfg.general_phys {
        let ri = term.site;
        if ri < 0 || (ri as usize) >= n_site || term.spin > 1 {
            continue;
        }
        let rsi = (ri as usize) + (term.spin as usize) * n_site;
        z += term.value * Complex64::new((2 * ele_num[rsi] - 1) as f64, 0.0);
    }

    // Hidden-layer biases.
    for term in cfg.charge_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_charge {
            continue;
        }
        hidden_charge[hi as usize] += term.value;
    }
    for term in cfg.spin_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_spin {
            continue;
        }
        hidden_spin[hi as usize] += term.value;
    }
    for term in cfg.general_hidden {
        let hi = term.site;
        if hi < 0 || (hi as usize) >= n_general {
            continue;
        }
        hidden_general[hi as usize] += term.value;
    }

    // Physical <-> hidden couplings.
    for term in cfg.charge_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0 || (ri as usize) >= n_site || hi < 0 || (hi as usize) >= n_charge {
            continue;
        }
        let ri = ri as usize;
        let xi = (ele_num[ri] + ele_num[ri + n_site] - 1) as f64;
        hidden_charge[hi as usize] += term.value * Complex64::new(xi, 0.0);
    }
    for term in cfg.spin_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0 || (ri as usize) >= n_site || hi < 0 || (hi as usize) >= n_spin {
            continue;
        }
        let ri = ri as usize;
        let xi = (ele_num[ri] - ele_num[ri + n_site]) as f64;
        hidden_spin[hi as usize] += term.value * Complex64::new(xi, 0.0);
    }
    for term in cfg.general_phys_hidden {
        let ri = term.site1;
        let hi = term.site2;
        if ri < 0
            || (ri as usize) >= n_site
            || term.spin > 1
            || hi < 0
            || (hi as usize) >= n_general
        {
            continue;
        }
        let rsi = (ri as usize) + (term.spin as usize) * n_site;
        let xi = (2 * ele_num[rsi] - 1) as f64;
        hidden_general[hi as usize] += term.value * Complex64::new(xi, 0.0);
    }

    for v in hidden_charge {
        z += log_cosh_stable(v);
    }
    for v in hidden_spin {
        z += log_cosh_stable(v);
    }
    for v in hidden_general {
        z += log_cosh_stable(v);
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_cfg<'a>() -> RbmConfig<'a> {
        RbmConfig {
            n_site: 4,
            nblock_size_rbm_ratio: 16,
            nneuron_charge: 0,
            nneuron_spin: 0,
            nneuron_general: 0,
            charge_phys: &[],
            spin_phys: &[],
            general_phys: &[],
            charge_hidden: &[],
            spin_hidden: &[],
            general_hidden: &[],
            charge_phys_hidden: &[],
            spin_phys_hidden: &[],
            general_phys_hidden: &[],
        }
    }

    #[test]
    fn make_rbm_cnt_no_terms_returns_empty() {
        let cfg = empty_cfg();
        let ele_num = vec![1, 0, 0, 0, 0, 1, 0, 0];
        assert!(make_rbm_cnt(&ele_num, &cfg).is_empty());
    }

    #[test]
    fn log_rbm_val_no_terms_is_zero() {
        let cfg = empty_cfg();
        let ele_num = vec![1, 0, 0, 0, 0, 1, 0, 0];
        assert_eq!(log_rbm_val(&ele_num, &cfg), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn log_rbm_ratio_no_terms_is_zero() {
        let cfg = empty_cfg();
        let new = vec![Complex64::new(1.0, 0.0); 3];
        let old = vec![Complex64::new(0.5, 0.0); 3];
        assert_eq!(log_rbm_ratio(&new, &old, &cfg), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn make_rbm_cnt_charge_phys_only() {
        // 2 sites, 2 charge-phys terms (one per site, both idx=0 so the
        // single counter sums (n0+n1-1) over both sites).
        let terms = vec![
            RbmPhysLayerTerm {
                site: 0,
                idx: 0,
                value: Complex64::new(0.3, 0.0),
            },
            RbmPhysLayerTerm {
                site: 1,
                idx: 0,
                value: Complex64::new(0.7, 0.0),
            },
        ];
        let cfg = RbmConfig {
            n_site: 2,
            nblock_size_rbm_ratio: 16,
            nneuron_charge: 0,
            nneuron_spin: 0,
            nneuron_general: 0,
            charge_phys: &terms,
            spin_phys: &[],
            general_phys: &[],
            charge_hidden: &[],
            spin_hidden: &[],
            general_hidden: &[],
            charge_phys_hidden: &[],
            spin_phys_hidden: &[],
            general_phys_hidden: &[],
        };
        // Up at 0, down at 1 -> site 0: 1+0-1=0, site 1: 0+1-1=0.
        let ele_num = vec![1, 0, 0, 1];
        let cnt = make_rbm_cnt(&ele_num, &cfg);
        assert_eq!(cnt, vec![Complex64::new(0.0, 0.0)]);
        // Doublon at 0, empty at 1 -> 2-1 + 0-1 = 0.
        let ele_num = vec![1, 0, 1, 0];
        let cnt = make_rbm_cnt(&ele_num, &cfg);
        assert_eq!(cnt, vec![Complex64::new(0.0, 0.0)]);
        // Doublon at 0, doublon at 1 -> 1 + 1 = 2.
        let ele_num = vec![1, 1, 1, 1];
        let cnt = make_rbm_cnt(&ele_num, &cfg);
        assert_eq!(cnt, vec![Complex64::new(2.0, 0.0)]);
    }

    #[test]
    fn update_then_make_match_for_charge_phys() {
        let terms = vec![
            RbmPhysLayerTerm {
                site: 0,
                idx: 0,
                value: Complex64::new(0.3, 0.0),
            },
            RbmPhysLayerTerm {
                site: 1,
                idx: 0,
                value: Complex64::new(0.7, 0.0),
            },
        ];
        let cfg = RbmConfig {
            n_site: 2,
            nblock_size_rbm_ratio: 16,
            nneuron_charge: 0,
            nneuron_spin: 0,
            nneuron_general: 0,
            charge_phys: &terms,
            spin_phys: &[],
            general_phys: &[],
            charge_hidden: &[],
            spin_hidden: &[],
            general_hidden: &[],
            charge_phys_hidden: &[],
            spin_phys_hidden: &[],
            general_phys_hidden: &[],
        };
        // Start: up at 0, down at 0 (doublon at site 0, hole at site 1).
        let mut ele_num = vec![1, 0, 1, 0];
        let old = make_rbm_cnt(&ele_num, &cfg);
        // Hop up from site 0 -> 1 (post-hop ele_num for projection updates,
        // but RBM update only reads ri/rj/spin so we don't actually need
        // ele_num to match -- we just update it for the parity check).
        let mut new = old.clone();
        update_rbm_cnt_hopping(&mut new, &old, 0, 1, 0, &cfg);
        ele_num[0] = 0;
        ele_num[1] = 1;
        let fresh = make_rbm_cnt(&ele_num, &cfg);
        assert_eq!(new, fresh);
    }

    #[test]
    fn log_cosh_stable_matches_log_cosh_for_simple_args() {
        use approx::assert_relative_eq;
        let z = Complex64::new(0.5, 0.25);
        let direct = z.cosh().ln();
        let ours = log_cosh_stable(z);
        assert_relative_eq!(direct.re, ours.re, max_relative = 1e-12);
        // Imaginary parts can disagree by 2π because of the principal-log
        // branch flip; just verify the modulus difference.
        let diff = (direct - ours).norm();
        assert!(diff < 1e-12);
    }
}
