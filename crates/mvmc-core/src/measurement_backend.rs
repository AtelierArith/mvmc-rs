//! tenferro measurement stage (issue #518 milestone 1): local energy and the
//! Slater-parameter derivative of one real, non-FSZ sample as tensor contractions.
//!
//! The trait and its C-order oracle live in [`crate::stage_backend`]
//! ([`MeasurementStages`](crate::stage_backend::MeasurementStages),
//! [`COrderMeasurement`](crate::stage_backend::COrderMeasurement)). This module provides the
//! tenferro implementation on an [`EagerRuntime`], so the same code runs with the CPU
//! runtime (`Placement::Host`, the milestone gate) and with the CUDA runtime
//! (`Placement::Device`, `gpu/mvmc-gpu-cuda`).
//!
//! # Formulation
//!
//! * Local energy: the diagonal Hamiltonian terms stay scalar on the host; the transfer
//!   (one-body) energy follows the C `CalHamiltonian1` fast path. Its per-term `new_ip`
//!   ratio is the C-order expression
//!   `new_ip = -sum_qp w_qp * pf_qp * (sum_msj inv_qp[msa,msj] * S_qp[rsa,rsj])`,
//!   evaluated for every Hamiltonian term at once as batched dot products over the
//!   electron-slot axis (`dot_general` with the term and QP axes as batch dimensions). The
//!   per-term projection ratio and the term-ordered `-value * proj_ratio * new_ip / ip`
//!   accumulation are unchanged host code.
//! * Slater derivative: `SlaterElmDiff`'s per-QP scatter
//!   `qp_orbital[qp,o] += (+-inv_qp[msj,msi]) * pf_qp * sp_qp * sgn`, followed by the QP
//!   fold `acc[o] = sum_qp weight_qp * qp_orbital[qp,o]`, is re-associated into per-sector
//!   QP reductions `A[block,msi,msj] = sum_qp weight*pf*sp * inv_qp[msi,msj]` (one
//!   `dot_general` of `[4, sp]` with `[sp, n, n]` per translation sector) plus the host
//!   scatter over the orbital index table. The final `acc * reciprocal(ip)` arithmetic is
//!   unchanged.
//!
//! # Supported class
//!
//! Real (not all-complex), non-FSZ, no RBM, no two-body Green terms
//! (PairHop/Exchange/InterAll), with the real `CalH1` fast path available (direct
//! projection tables). Everything else is a hard [`StageError::Unsupported`], never a
//! silent CPU fallback. Complex and FSZ inputs are refused here even if a runner routed
//! them.
//!
//! # Numerical contract
//!
//! The tensor contractions re-associate sums that the C-order kernels evaluate as
//! sequential chains: the QP fold of the Slater derivative and the electron-slot dot
//! product of `new_ip`. Deviations are bounded by the reordering terms
//! (`n * eps * sum |a b|`), validated against [`COrderMeasurement`] with explicit
//! tolerances in `crates/mvmc-core/tests/measurement_tenferro_518.rs`. The C order stays
//! authoritative and default.

use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::EagerRuntime;
use tenferro_tensor::DotGeneralConfig;

use crate::c_timer::CTimer;
use crate::observables;
use crate::slater_derivative;
use crate::sr_backend::{EagerTransfer, Placement};
use crate::stage_backend::{
    MeasurementInput, MeasurementOutput, MeasurementStages, MeasurementTiming, StageError,
};

fn err(e: impl std::fmt::Display) -> StageError {
    StageError::Failed(e.to_string())
}

fn unsupported<T>(reason: impl std::fmt::Display) -> Result<T, StageError> {
    Err(StageError::Unsupported(reason.to_string()))
}

/// Batched contraction over the trailing axis of `[batch..., k]` operands.
fn batched_dot(lhs_dim: usize, batch: usize) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![lhs_dim],
        rhs_contracting_dims: vec![lhs_dim],
        lhs_batch_dims: (0..batch).collect(),
        rhs_batch_dims: (0..batch).collect(),
    }
}

/// Contraction of a trailing axis into a leading axis.
fn contract_into(contract_l: usize, contract_r: usize) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![contract_l],
        rhs_contracting_dims: vec![contract_r],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

/// One transfer term whose Pfaffian ratio must be evaluated by the tensor contraction.
struct ActiveTerm {
    /// Electron slot `mj + spin * Ne` being moved.
    msa: usize,
    /// Destination Slater row `ri + spin * Nsite`.
    rsa: usize,
    /// Destination site `ri` (the special column of the moved slot).
    ri: usize,
    /// Direct projection ratio of the hop.
    proj_ratio: f64,
}

/// How a transfer term contributes to the energy: an immediate real `g1` (density or zero)
/// or a deferred tensor-table lookup.
enum TermOutcome {
    Immediate(f64),
    Deferred(usize),
}

/// Measurement stage (local energy + Slater derivative) through tenferro eager ops on one
/// runtime. Host tensors are uploaded and results downloaded explicitly; tenferro never
/// transfers implicitly, so the same object serves `Placement::Host` and `Placement::Device`.
pub struct TenferroMeasurement {
    io: EagerTransfer,
    label: String,
}

impl std::fmt::Debug for TenferroMeasurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenferroMeasurement")
            .field("label", &self.label)
            .field("placement", &self.io.placement)
            .finish()
    }
}

impl TenferroMeasurement {
    /// CPU runtime with the `cpu-faer` provider.
    pub fn new_cpu() -> Result<Self, StageError> {
        let ctx = EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new()).map_err(err)?;
        Ok(Self::with_runtime(
            ctx,
            Placement::Host,
            "tenferro-measurement cpu-faer".to_string(),
        ))
    }

    /// Any eager runtime (for example CUDA, built by `gpu/mvmc-gpu-cuda`).
    pub fn with_runtime(ctx: Arc<EagerRuntime>, placement: Placement, label: String) -> Self {
        Self {
            io: EagerTransfer::new(ctx, placement),
            label,
        }
    }

    /// A stage-level reason why this input is outside the tensor formulation.
    fn check_supported(input: &MeasurementInput<'_>) -> Result<(), StageError> {
        if input.data.i_flg_orbital_general != 0 {
            return unsupported("the tenferro measurement stage does not provide FSZ mode");
        }
        if input.state.all_complex || input.state.slater_matrix.pf_m_real.is_empty() {
            return unsupported("the tenferro measurement stage does not provide complex mode");
        }
        if input.data.has_rbm_terms() {
            return unsupported("the tenferro measurement stage does not provide RBM terms");
        }
        if !input.data.pair_hop_terms.is_empty()
            || !input.data.exchange_terms.is_empty()
            || !input.data.inter_all_terms.is_empty()
        {
            return unsupported(
                "the tenferro measurement stage does not provide two-body Green terms \
                 (PairHop/Exchange/InterAll)",
            );
        }
        Ok(())
    }

    fn run<const TIMED: bool>(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timer: &mut CTimer<TIMED>,
    ) -> Result<MeasurementOutput, StageError> {
        let local_energy = self.local_energy(input, timer)?;
        let slater_diff = if slater_diff_len > 0 {
            self.slater_diff(input, slater_diff_len, timer)?
        } else {
            Vec::new()
        };
        Ok(MeasurementOutput {
            local_energy,
            slater_diff,
        })
    }

    /// Local energy: host diagonal terms plus the tensor-formulated real `CalH1` transfer
    /// energy, in the C `calculate_local_energy` section order (70 = diagonal, 71 = transfer).
    fn local_energy<const TIMED: bool>(
        &mut self,
        input: &mut MeasurementInput<'_>,
        timer: &mut CTimer<TIMED>,
    ) -> Result<Complex64, StageError> {
        let data = input.data;
        timer.start(70);
        let mut e = observables::calculate_hamiltonian_diagonal(input.ele_num, data);
        timer.stop(70);
        timer.start(71);
        let ip = input.ip;
        if data.transfer_terms.is_empty() || ip.norm() == 0.0 {
            timer.stop(71);
            return Ok(e);
        }
        observables::refresh_transfer_cache(data, input.state);
        if !observables::calh1_fast_path_available(input.state) {
            timer.stop(71);
            return unsupported(
                "the tenferro measurement stage requires the real CalH1 fast path \
                 (real transfer coefficients, no RBM, direct projection tables)",
            );
        }
        let n_site = data.modpara.nsite.max(0) as usize;
        let n_elec = data.modpara.nelec.max(0) as usize;
        let n_size = 2 * n_elec;
        let pf_m_real = &input.state.slater_matrix.pf_m_real;
        let n_qp = data
            .qp_weights
            .as_ref()
            .map_or(0, |w| w.qp_full_weight.len().min(pf_m_real.len()));
        if input.ele_idx.len() < n_size
            || input
                .ele_idx
                .iter()
                .any(|&site| site < 0 || site as usize >= n_site)
        {
            timer.stop(71);
            return unsupported("the tenferro measurement stage requires a valid ele_idx");
        }
        let tables = input
            .state
            .transfer_cache
            .direct_tables
            .as_ref()
            .expect("checked by calh1_fast_path_available");
        // Term classification in the exact order of `calh1_real_fast`: a term either has an
        // immediate real g1 (density or zero) or defers to the batched new_ip table.
        let mut outcomes = Vec::with_capacity(input.state.transfer_cache.terms.len());
        let mut active: Vec<ActiveTerm> = Vec::new();
        for term in &input.state.transfer_cache.terms {
            let (ri, rj) = (term.site1, term.site2);
            let (spin_create, spin_annihilate) = (term.spin1, term.spin2);
            let dst = ri + spin_create as usize * n_site;
            let src = rj + spin_annihilate as usize * n_site;
            if spin_create == spin_annihilate && ri == rj {
                outcomes.push(TermOutcome::Immediate(input.ele_num[src] as f64));
            } else if spin_create != spin_annihilate
                || input.ele_num[dst] == 1
                || input.ele_num[src] == 0
                || input.ele_cfg[src] < 0
            {
                outcomes.push(TermOutcome::Immediate(0.0));
            } else {
                let mj = input.ele_cfg[src] as usize;
                let msa = mj + spin_annihilate as usize * n_elec;
                if msa >= n_size {
                    timer.stop(71);
                    return unsupported(
                        "the tenferro measurement stage requires a valid electron configuration",
                    );
                }
                let proj_ratio = observables::calh1_direct_projection_ratio_moved(
                    rj,
                    ri,
                    spin_annihilate as usize,
                    input.ele_num,
                    tables,
                );
                outcomes.push(TermOutcome::Deferred(active.len()));
                active.push(ActiveTerm {
                    msa,
                    rsa: ri + spin_annihilate as usize * n_site,
                    ri,
                    proj_ratio,
                });
            }
        }
        let new_ip = self.transfer_new_ip(input, &active, n_qp, n_size, n_elec, n_site)?;
        let mut transfer_energy = 0.0;
        for (term, outcome) in input.state.transfer_cache.terms.iter().zip(&outcomes) {
            let g1 = match *outcome {
                TermOutcome::Immediate(value) => value,
                TermOutcome::Deferred(index) => {
                    let act = &active[index];
                    act.proj_ratio * new_ip[index] / ip.re
                }
            };
            transfer_energy -= term.value.re * g1;
        }
        timer.stop(71);
        e += Complex64::new(transfer_energy, 0.0);
        Ok(e)
    }

    /// Batched `new_ip` ratio table: for every active transfer term `t`,
    /// `new_ip[t] = -sum_qp w_qp * pf_qp * (sum_msj inv_qp[msa_t,msj] * S_qp[rsa_t,rsj])`,
    /// with the moved slot's column `msj == msa_t` reading the destination row `rsa_t` and
    /// every other column `msj` reading the unmoved `ele_idx`.
    fn transfer_new_ip(
        &mut self,
        input: &MeasurementInput<'_>,
        active: &[ActiveTerm],
        n_qp: usize,
        n_size: usize,
        n_elec: usize,
        n_site: usize,
    ) -> Result<Vec<f64>, StageError> {
        if active.is_empty() || n_qp == 0 {
            return Ok(vec![0.0; active.len()]);
        }
        let terms = active.len();
        // Column-major [terms, n_qp, n_size]: index = t + terms * (qp + n_qp * msj).
        let mut a = vec![0.0; terms * n_qp * n_size];
        let mut b = vec![0.0; terms * n_qp * n_size];
        let slater = &input.state.slater_matrix.slater_elm_real;
        let inv = &input.state.slater_matrix.inv_m_real;
        for (t, act) in active.iter().enumerate() {
            let (msa, rsa, ri) = (act.msa, act.rsa, act.ri);
            for qp in 0..n_qp {
                let inv_plane = inv.qp_matrix_slice(qp);
                for msj in 0..n_size {
                    let site = if msj == msa {
                        ri
                    } else {
                        input.ele_idx[msj] as usize
                    };
                    let rsj = site + if msj < n_elec { 0 } else { n_site };
                    a[t + terms * (qp + n_qp * msj)] = inv_plane[msa * n_size + msj];
                    b[t + terms * (qp + n_qp * msj)] = slater.get(qp, rsa, rsj);
                }
            }
        }
        let a_t = self.io.upload_real(vec![terms, n_qp, n_size], a)?;
        let b_t = self.io.upload_real(vec![terms, n_qp, n_size], b)?;
        let ratio = a_t.dot_general(&b_t, batched_dot(2, 2)).map_err(err)?;
        let pf = self.io.upload_real(
            vec![n_qp],
            input.state.slater_matrix.pf_m_real[..n_qp].to_vec(),
        )?;
        let weights = input
            .data
            .qp_weights
            .as_ref()
            .expect("n_qp > 0 requires QP weights");
        let w: Vec<f64> = weights.qp_full_weight[..n_qp]
            .iter()
            .map(|value| value.re)
            .collect();
        let w_t = self.io.upload_real(vec![n_qp], w)?;
        // `new_pf = -(ratio * pf)` then `new_ip = sum_qp w * new_pf`: the C grouping,
        // with only the QP summation order left to the tensor reduction.
        let new_pf = ratio.mul(&pf).map_err(err)?.neg().map_err(err)?;
        let new_ip_t = new_pf.dot_general(&w_t, contract_into(1, 0)).map_err(err)?;
        self.io.download_f64(&new_ip_t)
    }

    /// Slater-parameter derivative vector (the real QP-folded `SlaterElmDiff`).
    fn slater_diff<const TIMED: bool>(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timer: &mut CTimer<TIMED>,
    ) -> Result<Vec<Complex64>, StageError> {
        let mut out = vec![Complex64::new(0.0, 0.0); slater_diff_len];
        let data = input.data;
        let ip = input.ip;
        let n_site = data.modpara.nsite.max(0) as usize;
        let n_elec = data.modpara.nelec.max(0) as usize;
        if n_site == 0 || n_elec == 0 || ip.norm() == 0.0 {
            return Ok(out);
        }
        let weights = match data.qp_weights.as_ref() {
            Some(weights) => weights,
            None => return Ok(out),
        };
        let n_qp_full = weights
            .qp_full_weight
            .len()
            .min(input.state.slater_matrix.pf_m.len());
        let n_slater = mvmc_expert_parsers::utils::parameter_init::n_slater(data);
        if n_qp_full == 0 || n_slater == 0 || out.len() < 2 * n_slater {
            return Ok(out);
        }
        let n_sp_gauss_leg = data.modpara.nsp_gauss_leg.max(1) as usize;
        if !slater_derivative::real_slater_weights(weights, n_qp_full, n_sp_gauss_leg) {
            return unsupported(
                "the tenferro measurement stage requires real Slater-derivative weights",
            );
        }
        if input.state.slater_matrix.pf_m_real.is_empty() {
            return unsupported("the tenferro measurement stage requires the real Slater tables");
        }
        timer.start_diag(930, timer.diagnostics.slater);
        // The translation/orbital tables are the exact construction of the C-order kernel.
        let tables = slater_derivative::build_trans_orb_tables(data, input.ele_idx);
        let complete_maps = slater_derivative::complete_translation_maps(data);
        let n_size = tables.n_size;
        let n_sp = tables.n_sp_gauss_leg;
        let n_slater = tables.n_slater;
        let nn = n_size * n_size;
        let mut acc = vec![0.0_f64; n_slater];
        let pf_m = &input.state.slater_matrix.pf_m_real;
        let inv = &input.state.slater_matrix.inv_m_real;
        for trans in 0..tables.n_trans {
            if !complete_maps && trans >= tables.n_mp_trans {
                continue;
            }
            let qp_start = trans * n_sp;
            let qp_end = ((trans + 1) * n_sp).min(n_qp_full);
            if qp_start >= qp_end {
                continue;
            }
            let sp_len = qp_end - qp_start;
            // Column-major [4, sp_len]: block = (msi block) * 2 + (msj block) with the C
            // `SlaterElmDiff` coefficients cs / -cc / ss / -cs.
            let mut w = vec![0.0_f64; 4 * sp_len];
            for s in 0..sp_len {
                let qp = qp_start + s;
                let spidx = qp % n_sp;
                if spidx >= weights.spgl_cos_sin.len()
                    || spidx >= weights.spgl_cos_cos.len()
                    || spidx >= weights.spgl_sin_sin.len()
                {
                    continue;
                }
                let pf = pf_m[qp];
                let wr = weights.qp_full_weight[qp].re;
                let cs = pf * weights.spgl_cos_sin[spidx].re;
                let cc = pf * weights.spgl_cos_cos[spidx].re;
                let ss = pf * weights.spgl_sin_sin[spidx].re;
                w[4 * s] = wr * cs;
                w[1 + 4 * s] = -wr * cc;
                w[2 + 4 * s] = wr * ss;
                w[3 + 4 * s] = -wr * cs;
            }
            // Column-major [sp_len, n_size, n_size]: element (s, d0, d1) sits at
            // `s + sp_len * (d0 + n_size * d1)`; each plane is column-major too, so the
            // copy strides the plane axis by `sp_len`.
            let mut inv_sub = vec![0.0_f64; sp_len * nn];
            for s in 0..sp_len {
                let plane = inv.qp_matrix_slice(qp_start + s);
                for (k, &value) in plane.iter().enumerate() {
                    inv_sub[s + sp_len * k] = value;
                }
            }
            let w_t = self.io.upload_real(vec![4, sp_len], w)?;
            let inv_t = self.io.upload_real(vec![sp_len, n_size, n_size], inv_sub)?;
            let a_t = w_t.dot_general(&inv_t, contract_into(1, 0)).map_err(err)?;
            let a = self.io.download_f64(&a_t)?;
            let base = trans * nn;
            for msi in 0..n_size {
                for msj in 0..n_size {
                    let orbidx = tables.trans_orb_idx[base + msi * n_size + msj];
                    if orbidx < 0 || orbidx as usize >= n_slater {
                        continue;
                    }
                    let block = usize::from(msi >= n_elec) * 2 + usize::from(msj >= n_elec);
                    let sign = tables.trans_orb_sgn[base + msi * n_size + msj] as f64;
                    // The plane copy is column-major, so `plane[msj + msi * n_size]` (the
                    // C kernel's index) sits at tensor index `msj + n_size * msi`.
                    acc[orbidx as usize] += a[block + 4 * (msj + n_size * msi)] * sign;
                }
            }
        }
        // The final normalization of `reduce_qp_weighted_real_into`, unchanged.
        let inv_ip = crate::julia_complex::reciprocal(ip);
        for (orbidx, &value) in acc.iter().enumerate() {
            let value = Complex64::new(value, 0.0) * inv_ip;
            out[2 * orbidx] = value;
            out[2 * orbidx + 1] = value * Complex64::new(0.0, 1.0);
        }
        timer.stop_diag(930, timer.diagnostics.slater);
        Ok(out)
    }
}

impl MeasurementStages for TenferroMeasurement {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn provider(&self) -> String {
        match self.io.placement {
            Placement::Host => {
                "tenferro-ad EagerRuntime + CpuBackend (cpu-faer); tensor ratio tables".to_string()
            }
            Placement::Device => {
                "tenferro-ad EagerRuntime + tenferro-gpu CudaBackend (cuBLAS); tensor ratio tables"
                    .to_string()
            }
        }
    }

    fn measure_sample(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        Self::check_supported(input)?;
        if timing.enabled() {
            let mut timer = CTimer::<true>::new();
            timer.diagnostics = timing.diagnostics();
            let out = self.run(input, slater_diff_len, &mut timer)?;
            timing.merge_elapsed(&timer.elapsed_ns);
            Ok(out)
        } else {
            let mut timer = CTimer::<false>::new();
            timer.diagnostics = timing.diagnostics();
            self.run(input, slater_diff_len, &mut timer)
        }
    }
}
#[cfg(test)]
mod layout_tests {
    use super::*;

    /// The derivative contraction relies on tenferro's `dot_general` keeping the operand's
    /// free dimensions in order: `[b, s] x [s, m, n] -> [b, m, n]` with column-major storage.
    #[test]
    fn contract_into_keeps_free_dimension_order() {
        for (b, s, m, n) in [(2usize, 3usize, 5usize, 7usize), (4, 8, 16, 16)] {
            contract_case(b, s, m, n);
        }
    }

    fn contract_case(b: usize, s: usize, m: usize, n: usize) {
        let mut state = 987_654_321_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        // lhs[b, s], column-major [b, s].
        let lhs: Vec<f64> = (0..b * s).map(|_| next()).collect();
        // rhs[s, m, n], column-major [s, m, n].
        let rhs: Vec<f64> = (0..s * m * n).map(|_| next()).collect();
        let ctx = EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new()).unwrap();
        let io = EagerTransfer::new(ctx, Placement::Host);
        let lhs_t = io.upload_real(vec![b, s], lhs.clone()).unwrap();
        let rhs_t = io.upload_real(vec![s, m, n], rhs.clone()).unwrap();
        let out = io
            .download_f64(&lhs_t.dot_general(&rhs_t, contract_into(1, 0)).unwrap())
            .unwrap();
        assert_eq!(out.len(), b * m * n);
        for nn_ in 0..n {
            for mm in 0..m {
                for bb in 0..b {
                    let mut expected = 0.0;
                    for ss in 0..s {
                        let l = lhs[bb + b * ss];
                        let r = rhs[ss + s * (mm + m * nn_)];
                        expected += l * r;
                    }
                    let actual = out[bb + b * (mm + m * nn_)];
                    assert!(
                        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                        "b={bb} m={mm} n={nn_}: actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    /// The Slater-derivative contraction against the explicit `SlaterElmDiff` sum, pinning
    /// the column-major `[sp, n, n]` plane-axis stride of the stage's `inv_sub`.
    #[test]
    fn plane_axis_stride_matches_the_slater_elm_diff_sum() {
        let (sp, n) = (3usize, 4usize);
        let n_elec = n / 2;
        let nn = n * n;
        let mut state = 202_605_180_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let w: Vec<f64> = (0..4 * sp).map(|_| next()).collect();
        let planes: Vec<Vec<f64>> = (0..sp).map(|_| (0..nn).map(|_| next()).collect()).collect();
        // The stage's upload layout.
        let mut inv_sub = vec![0.0_f64; sp * nn];
        for (s, plane) in planes.iter().enumerate() {
            for (k, &value) in plane.iter().enumerate() {
                inv_sub[s + sp * k] = value;
            }
        }
        let ctx = EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new()).unwrap();
        let io = EagerTransfer::new(ctx, Placement::Host);
        let w_t = io.upload_real(vec![4, sp], w.clone()).unwrap();
        let inv_t = io.upload_real(vec![sp, n, n], inv_sub).unwrap();
        let a = io
            .download_f64(&w_t.dot_general(&inv_t, contract_into(1, 0)).unwrap())
            .unwrap();
        // The C-order scatter reads `plane[msj + msi * n]` for the block/coefficient pair
        // of `(msi, msj)`; the tensor index is `blk + 4 * (msj + n * msi)`.
        for msi in 0..n {
            for msj in 0..n {
                let block = usize::from(msi >= n_elec) * 2 + usize::from(msj >= n_elec);
                let mut expected = 0.0;
                for (s, plane) in planes.iter().enumerate() {
                    expected += w[block + 4 * s] * plane[msj + msi * n];
                }
                let actual = a[block + 4 * (msj + n * msi)];
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "msi={msi} msj={msj} block={block}: actual={actual} expected={expected}"
                );
            }
        }
    }
}
