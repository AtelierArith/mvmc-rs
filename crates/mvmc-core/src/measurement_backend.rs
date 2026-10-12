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
use crate::sampling::projection::{log_proj_ratio, update_proj_cnt};
use crate::slater_derivative;
use crate::sr_backend::{EagerTransfer, Placement};
use crate::stage_backend::{
    GreenOutput, MeasurementInput, MeasurementOutput, MeasurementStages, MeasurementTiming,
    StageError,
};
use crate::state::VmcOptimizationState;
use mvmc_expert_parsers::utils::c_math::exp as c_exp;
use mvmc_expert_parsers::ExpertModeData;

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
        check_measurement_supported(input)
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
        let ctx = match transfer_context(
            data,
            input.state,
            input.ele_idx,
            input.ele_cfg,
            input.ele_num,
        ) {
            Ok(ctx) => ctx,
            Err(error) => {
                timer.stop(71);
                return Err(error);
            }
        };
        let new_ip = self.transfer_new_ip(input, &ctx)?;
        let transfer_energy = fold_transfer_energy(input, &ctx, &new_ip);
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
        ctx: &TransferContext,
    ) -> Result<Vec<f64>, StageError> {
        let active = &ctx.active;
        if active.is_empty() || ctx.n_qp == 0 {
            return Ok(vec![0.0; active.len()]);
        }
        let (terms, n_qp, n_size, n_elec, n_site) =
            (active.len(), ctx.n_qp, ctx.n_size, ctx.n_elec, ctx.n_site);
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

// ---------------------------------------------------------------------------------------
// Shared host-side classification (tenferro and resident paths)
// ---------------------------------------------------------------------------------------

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

/// Term classification and dimensions of one sample's transfer energy.
struct TransferContext {
    n_site: usize,
    n_elec: usize,
    n_size: usize,
    n_qp: usize,
    outcomes: Vec<TermOutcome>,
    active: Vec<ActiveTerm>,
}

/// A stage-level reason why this input is outside the measurement stage's supported class
/// (real, non-FSZ, no RBM, no two-body Green terms).
pub fn check_measurement_supported(input: &MeasurementInput<'_>) -> Result<(), StageError> {
    if input.data.i_flg_orbital_general != 0 {
        return unsupported("the measurement stage does not provide FSZ mode");
    }
    if input.state.all_complex || input.state.slater_matrix.pf_m_real.is_empty() {
        return unsupported("the measurement stage does not provide complex mode");
    }
    if input.data.has_rbm_terms() {
        return unsupported("the measurement stage does not provide RBM terms");
    }
    if !input.data.pair_hop_terms.is_empty()
        || !input.data.exchange_terms.is_empty()
        || !input.data.inter_all_terms.is_empty()
    {
        return unsupported(
            "the measurement stage does not provide two-body Green terms \
             (PairHop/Exchange/InterAll)",
        );
    }
    Ok(())
}

/// Classify the transfer terms of one sample in the exact order of `calh1_real_fast`: a term
/// either has an immediate real `g1` (density or zero) or defers to the `new_ip` ratio table.
fn transfer_context(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
) -> Result<TransferContext, StageError> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let pf_m_real = &state.slater_matrix.pf_m_real;
    let n_qp = data
        .qp_weights
        .as_ref()
        .map_or(0, |w| w.qp_full_weight.len().min(pf_m_real.len()));
    if ele_idx.len() < n_size
        || ele_idx
            .iter()
            .any(|&site| site < 0 || site as usize >= n_site)
    {
        return unsupported("the measurement stage requires a valid ele_idx");
    }
    let tables = state.transfer_cache.direct_tables.as_ref().ok_or_else(|| {
        StageError::Unsupported(
            "the measurement stage requires the real CalH1 fast path \
                 (real transfer coefficients, no RBM, direct projection tables)"
                .to_string(),
        )
    })?;
    let mut outcomes = Vec::with_capacity(state.transfer_cache.terms.len());
    let mut active: Vec<ActiveTerm> = Vec::new();
    for term in &state.transfer_cache.terms {
        let (ri, rj) = (term.site1, term.site2);
        let (spin_create, spin_annihilate) = (term.spin1, term.spin2);
        let dst = ri + spin_create as usize * n_site;
        let src = rj + spin_annihilate as usize * n_site;
        if spin_create == spin_annihilate && ri == rj {
            outcomes.push(TermOutcome::Immediate(ele_num[src] as f64));
        } else if spin_create != spin_annihilate
            || ele_num[dst] == 1
            || ele_num[src] == 0
            || ele_cfg[src] < 0
        {
            outcomes.push(TermOutcome::Immediate(0.0));
        } else {
            let mj = ele_cfg[src] as usize;
            let msa = mj + spin_annihilate as usize * n_elec;
            if msa >= n_size {
                return unsupported(
                    "the measurement stage requires a valid electron configuration",
                );
            }
            let proj_ratio = observables::calh1_direct_projection_ratio_moved(
                rj,
                ri,
                spin_annihilate as usize,
                ele_num,
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
    Ok(TransferContext {
        n_site,
        n_elec,
        n_size,
        n_qp,
        outcomes,
        active,
    })
}

/// The term-ordered transfer energy from a `new_ip` ratio table (the C fold):
/// `-sum_terms value * (immediate | proj_ratio * new_ip / ip)`.
fn fold_transfer_energy(
    input: &MeasurementInput<'_>,
    ctx: &TransferContext,
    new_ip: &[f64],
) -> f64 {
    let mut transfer_energy = 0.0;
    for (term, outcome) in input.state.transfer_cache.terms.iter().zip(&ctx.outcomes) {
        let g1 = match *outcome {
            TermOutcome::Immediate(value) => value,
            TermOutcome::Deferred(index) => {
                ctx.active[index].proj_ratio * new_ip[index] / input.ip.re
            }
        };
        transfer_energy -= term.value.re * g1;
    }
    transfer_energy
}

// ---------------------------------------------------------------------------------------
// Device-resident measurement (issue #518 milestone 2)
// ---------------------------------------------------------------------------------------

/// Device operations of the resident measurement path.
///
/// Implemented by the CUDA provider (`gpu/mvmc-gpu-cuda`), and by a host mock in the tests of
/// this crate. The host orchestration (term classification, projection ratios, the term-ordered
/// energy fold and the final Slater-derivative normalization) stays in [`resident_measure_sample`]
/// so that the resident and the eager paths share it operation for operation.
pub trait ResidentMeasurementOps: Send {
    /// Upload the input-fixed tables (Slater-table geometry is checked by `stage_a`).
    fn prepare_input(
        &mut self,
        _request: &crate::stage_backend::ResidentInputRequest<'_>,
    ) -> Result<(), StageError> {
        Ok(())
    }

    /// Device-resident stage A ([`PfaffianStages::stage_a_resident`]).
    fn stage_a(
        &mut self,
        input: &crate::stage_backend::ResidentStageAInput<'_>,
    ) -> Result<crate::stage_backend::ResidentStageAOutput, StageError>;

    /// Whether the tables of `slot` are resident (produced by [`Self::stage_a`]).
    fn has_slot(&self, slot: usize) -> bool;

    /// `new_ip[t]` of the active transfer terms `t` for `slot`: the device computes
    /// `sum_qp w_qp * (-(ratio_qp * pf_qp))` from the resident inverse planes, Slater table
    /// and Pfaffians.
    fn ratio_terms(
        &mut self,
        slot: usize,
        input: &ResidentRatioTerms<'_>,
    ) -> Result<Vec<f64>, StageError>;

    /// Raw Slater-derivative accumulation `acc[n_slater]` for `slot` (before the `inv(ip)`
    /// normalization): the device folds the QP weights, per-sector coefficients and the
    /// orbital scatter of the C-order kernel.
    fn slater_diff(
        &mut self,
        slot: usize,
        input: &ResidentSlaterDiff<'_>,
    ) -> Result<Vec<f64>, StageError>;

    /// `new_ip[t]` of the active two-body (rank-2) Green jobs `t` for `slot` (issue #517
    /// milestone 5): the device computes the `b^T M a` bilinear of `calculate_new_pf_m_two2`
    /// from the resident inverse planes and Slater table. `msa`/`msb` address the two moved
    /// electron slots and `rsa_site`/`rsb_site` their destination sites.
    fn green_two(
        &mut self,
        slot: usize,
        input: &ResidentGreenTwo<'_>,
    ) -> Result<Vec<f64>, StageError> {
        let _ = (slot, input);
        unsupported("this backend does not provide the device two-body Green reduction")
    }

    /// Counters, `None` when the implementation does not report them.
    fn stats(&self) -> Option<crate::stage_backend::ResidentStats> {
        None
    }
}

/// Per-sample input of [`ResidentMeasurementOps::ratio_terms`].
pub struct ResidentRatioTerms<'a> {
    /// Electron sites `[n]` of the sample (the sample's own changed rows).
    pub ele_idx: &'a [i64],
    /// Moved electron slot `msa` per active term.
    pub msa: &'a [i64],
    /// Destination Slater row `rsa` per active term.
    pub rsa: &'a [i64],
    /// Destination site `ri` per active term.
    pub ri: &'a [i64],
}

/// Per-sample input of [`ResidentMeasurementOps::green_two`] (issue #517 milestone 5).
pub struct ResidentGreenTwo<'a> {
    /// Electron sites `[n]` of the sample (the sample's own changed rows).
    pub ele_idx: &'a [i64],
    /// Moved electron slot of each active two-body job (moves to `rsa_site`).
    pub msa: &'a [i64],
    /// Second moved electron slot of each active two-body job (moves to `rsb_site`).
    pub msb: &'a [i64],
    /// Destination site of `msa` (with its own spin block).
    pub rsa_site: &'a [i64],
    /// Destination site of `msb` (with its own spin block).
    pub rsb_site: &'a [i64],
}

/// Per-sample input of [`ResidentMeasurementOps::slater_diff`].
pub struct ResidentSlaterDiff<'a> {
    /// QP points per translation sector (`NSPGaussLeg`).
    pub n_sp_gauss_leg: usize,
    /// Translation sectors of the tables.
    pub n_trans: usize,
    /// `tri[trans * n + msi]` translated site (`-1` invalid), [`build_sector_translation`].
    pub tri: &'a [i64],
    /// Translation signs, same layout.
    pub sgn: &'a [i64],
}

/// Upload the input-fixed tables of the resident measurement path (no-op for a backend that
/// does not need them). The orbital matrices are taken from the data cache when present.
pub fn resident_prepare(
    ops: &mut dyn ResidentMeasurementOps,
    data: &ExpertModeData,
    state: &VmcOptimizationState,
) -> Result<(), StageError> {
    let weights = match data.qp_weights.as_ref() {
        Some(weights) => weights,
        None => return Ok(()),
    };
    let n_qp = weights
        .qp_full_weight
        .len()
        .min(state.slater_matrix.pf_m_real.len());
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    if n_qp == 0 {
        return Ok(());
    }
    let built;
    let (orbital_idx, orbital_sgn): (&[Vec<i64>], &[Vec<i64>]) =
        match (&data.orbital_idx_matrix, &data.orbital_sgn_matrix) {
            (Some(idx), Some(sgn)) => (idx, sgn),
            _ => {
                built = data.build_orbital_matrices();
                (&built.0, &built.1)
            }
        };
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_slater = mvmc_expert_parsers::utils::parameter_init::n_slater(data);
    let qp_weight: Vec<f64> = weights.qp_full_weight[..n_qp]
        .iter()
        .map(|v| v.re)
        .collect();
    let real_take =
        |values: &[Complex64]| -> Vec<f64> { values.iter().take(n_sp).map(|v| v.re).collect() };
    ops.prepare_input(&crate::stage_backend::ResidentInputRequest {
        n_site,
        n_elec,
        n_qp,
        n_slater,
        n_sp_gauss_leg: n_sp,
        orbital_idx,
        orbital_sgn,
        qp_weight: &qp_weight,
        spgl_cos_sin: &real_take(&weights.spgl_cos_sin),
        spgl_cos_cos: &real_take(&weights.spgl_cos_cos),
        spgl_sin_sin: &real_take(&weights.spgl_sin_sin),
    })
}

/// Resident measurement of one sample: local energy and Slater derivative from the device
/// tables produced by [`ResidentMeasurementOps::stage_a`]. The host performs the same term
/// classification, projection ratios and folds as the C-order kernel; only the table arithmetic
/// runs on the device.
pub fn resident_measure_sample<const TIMED: bool>(
    ops: &mut dyn ResidentMeasurementOps,
    input: &mut MeasurementInput<'_>,
    slater_diff_len: usize,
    timer: &mut CTimer<TIMED>,
) -> Result<MeasurementOutput, StageError> {
    check_measurement_supported(input)?;
    let local_energy = resident_local_energy::<TIMED>(ops, input, timer)?;
    let slater_diff = if slater_diff_len > 0 {
        resident_slater_diff::<TIMED>(ops, input, slater_diff_len, timer)?
    } else {
        Vec::new()
    };
    Ok(MeasurementOutput {
        local_energy,
        slater_diff,
    })
}

/// Resident local energy: host diagonal terms and the unchanged host-side term fold around the
/// device `new_ip` table (sections 70/71 of `calculate_local_energy`).
fn resident_local_energy<const TIMED: bool>(
    ops: &mut dyn ResidentMeasurementOps,
    input: &mut MeasurementInput<'_>,
    timer: &mut CTimer<TIMED>,
) -> Result<Complex64, StageError> {
    let data = input.data;
    timer.start(70);
    let mut e = observables::calculate_hamiltonian_diagonal(input.ele_num, data);
    timer.stop(70);
    timer.start(71);
    if data.transfer_terms.is_empty() || input.ip.norm() == 0.0 {
        timer.stop(71);
        return Ok(e);
    }
    observables::refresh_transfer_cache(data, input.state);
    if !observables::calh1_fast_path_available(input.state) {
        timer.stop(71);
        return unsupported(
            "the resident measurement stage requires the real CalH1 fast path \
             (real transfer coefficients, no RBM, direct projection tables)",
        );
    }
    let ctx = match transfer_context(
        data,
        input.state,
        input.ele_idx,
        input.ele_cfg,
        input.ele_num,
    ) {
        Ok(ctx) => ctx,
        Err(error) => {
            timer.stop(71);
            return Err(error);
        }
    };
    let new_ip = if ctx.active.is_empty() || ctx.n_qp == 0 {
        vec![0.0; ctx.active.len()]
    } else {
        let msa: Vec<i64> = ctx.active.iter().map(|a| a.msa as i64).collect();
        let rsa: Vec<i64> = ctx.active.iter().map(|a| a.rsa as i64).collect();
        let ri: Vec<i64> = ctx.active.iter().map(|a| a.ri as i64).collect();
        match ops.ratio_terms(
            input.batch_slot,
            &ResidentRatioTerms {
                ele_idx: input.ele_idx,
                msa: &msa,
                rsa: &rsa,
                ri: &ri,
            },
        ) {
            Ok(values) => values,
            Err(error) => {
                timer.stop(71);
                return Err(error);
            }
        }
    };
    let transfer_energy = fold_transfer_energy(input, &ctx, &new_ip);
    timer.stop(71);
    e += Complex64::new(transfer_energy, 0.0);
    Ok(e)
}

/// Resident Slater derivative: the device folds the QP weights, the per-sector block
/// coefficients and the orbital scatter of the C `SlaterElmDiff` fast path; the host performs
/// the final `* reciprocal(ip)` normalization (section 930, the call site of
/// `reduce_qp_weighted_real_into`).
fn resident_slater_diff<const TIMED: bool>(
    ops: &mut dyn ResidentMeasurementOps,
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
            "the resident measurement stage requires real Slater-derivative weights",
        );
    }
    timer.start_diag(930, timer.diagnostics.slater);
    // The device tables hold one sector per `n_sp_gauss_leg` QP points; a partial last sector
    // is not representable in the resident kernel (n_qp_full is a multiple of it in every
    // supported input).
    if n_sp_gauss_leg == 0 || !n_qp_full.is_multiple_of(n_sp_gauss_leg) {
        timer.stop_diag(930, timer.diagnostics.slater);
        return unsupported(
            "the resident measurement stage requires NQP to be a multiple of NSPGaussLeg",
        );
    }
    let n_trans = n_qp_full / n_sp_gauss_leg;
    let (tri, sgn) = slater_derivative::build_sector_translation(data, input.ele_idx);
    let acc = match ops.slater_diff(
        input.batch_slot,
        &ResidentSlaterDiff {
            n_sp_gauss_leg,
            n_trans,
            tri: &tri,
            sgn: &sgn,
        },
    ) {
        Ok(acc) => acc,
        Err(error) => {
            timer.stop_diag(930, timer.diagnostics.slater);
            return Err(error);
        }
    };
    if acc.len() != n_slater {
        timer.stop_diag(930, timer.diagnostics.slater);
        return Err(StageError::Failed(format!(
            "resident Slater derivative returned {} values for {n_slater} orbitals",
            acc.len()
        )));
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

// ---------------------------------------------------------------------------------------
// PhysCal Green functions on the device (issue #517 milestone 5)
// ---------------------------------------------------------------------------------------

/// One deferred one-body Pfaffian reduction (the `calculate_new_pf_m2` path).
#[derive(Debug, Clone, Copy)]
struct GreenOneJob {
    /// Electron slot being moved (moves to `ri`).
    msa: usize,
    /// Destination Slater row `ri + spin * Nsite`.
    rsa: usize,
    /// Destination site `ri`.
    ri: usize,
    /// Projection ratio of the hop.
    proj_ratio: f64,
}

/// One deferred two-body (rank-2) reduction (the `calculate_new_pf_m_two2` path).
#[derive(Debug, Clone, Copy)]
struct GreenTwoJob {
    /// First moved electron slot (moves to `rsa_site`, its own spin block).
    msa: usize,
    /// Second moved electron slot (moves to `rsb_site`, its own spin block).
    msb: usize,
    /// Destination site of `msa`.
    rsa_site: usize,
    /// Destination site of `msb`.
    rsb_site: usize,
    /// Projection ratio of the two sequential hops.
    proj_ratio: f64,
}

/// How one `green_func1`/`green_func2` term resolves.
#[derive(Debug, Clone, Copy)]
enum GreenOutcome {
    /// Already-final value (density or zero, no projection ratio or `1/ip`).
    Value(f64),
    /// `sign * proj_ratio * new_ip / ip` with `new_ip` from `one_jobs[index]`.
    One { index: usize, sign: f64 },
    /// `proj_ratio * new_ip / ip` with `new_ip` from `two_jobs[index]`.
    Two { index: usize },
}

/// Term classification of one sample's Green functions: the outcomes in term order plus the
/// deferred jobs gathered for the batched device reductions.
struct GreenContext {
    one_jobs: Vec<GreenOneJob>,
    two_jobs: Vec<GreenTwoJob>,
    one_outcomes: Vec<GreenOutcome>,
    two_outcomes: Vec<GreenOutcome>,
}

/// Host classification of the Green terms. It reproduces the branch structure of
/// `green_func1_impl`/`green_func2_impl` (real, non-FSZ, no RBM) and defers only the
/// per-term Pfaffian reduction to the device.
struct GreenClassifier<'a> {
    n_site: usize,
    n_elec: usize,
    n_proj: usize,
    ip: Complex64,
    data: &'a ExpertModeData,
    ele_cfg: &'a [i64],
    ele_num: &'a [i64],
    ele_proj_cnt: &'a [i64],
    ctx: GreenContext,
    moved_num: Vec<i64>,
    proj_new: Vec<i64>,
    proj_mid: Vec<i64>,
    proj_final: Vec<i64>,
}

impl<'a> GreenClassifier<'a> {
    fn new(
        data: &'a ExpertModeData,
        ip: Complex64,
        ele_cfg: &'a [i64],
        ele_num: &'a [i64],
        ele_proj_cnt: &'a [i64],
    ) -> Self {
        let n_site = data.modpara.nsite.max(0) as usize;
        let n_elec = data.modpara.nelec.max(0) as usize;
        let n_proj = data.projection_layout().n_proj;
        Self {
            n_site,
            n_elec,
            n_proj,
            ip,
            data,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            ctx: GreenContext {
                one_jobs: Vec::new(),
                two_jobs: Vec::new(),
                one_outcomes: Vec::new(),
                two_outcomes: Vec::new(),
            },
            moved_num: vec![0; 2 * n_site],
            proj_new: vec![0; n_proj],
            proj_mid: vec![0; n_proj],
            proj_final: vec![0; n_proj],
        }
    }

    /// `proj_ratio` of one hop `rj -> ri` with the moved occupations in `self.moved_num`:
    /// `update_proj_cnt(rj, ri, ...)` and `exp(log_proj_ratio(new, old))`, matching
    /// `green_func1_impl`.
    fn hop_proj_ratio(&mut self, ri: usize, rj: usize, spin: u8, out: &mut Vec<i64>) {
        if self.n_proj == 0 {
            return;
        }
        out.clear();
        out.resize(self.n_proj, 0);
        update_proj_cnt(
            rj as i64,
            ri as i64,
            spin,
            out,
            self.ele_proj_cnt,
            &self.moved_num,
            self.data,
        );
    }

    /// `green_func1_impl::<false, false, false>` classification.
    fn classify_one(
        &mut self,
        ri: usize,
        rj: usize,
        spin_create: u8,
        spin_annihilate: u8,
    ) -> GreenOutcome {
        if self.ip.norm() == 0.0 {
            return GreenOutcome::Value(0.0);
        }
        let dst = ri + spin_create as usize * self.n_site;
        let src = rj + spin_annihilate as usize * self.n_site;
        if spin_create == spin_annihilate && ri == rj {
            return GreenOutcome::Value(self.ele_num[src] as f64);
        }
        if spin_create != spin_annihilate {
            return GreenOutcome::Value(0.0);
        }
        if self.ele_num[dst] == 1 || self.ele_num[src] == 0 {
            return GreenOutcome::Value(0.0);
        }
        let mj_raw = self.ele_cfg[src];
        if mj_raw < 0 {
            return GreenOutcome::Value(0.0);
        }
        let mj = mj_raw as usize;
        let msj = mj + spin_annihilate as usize * self.n_elec;
        self.moved_num.copy_from_slice(self.ele_num);
        self.moved_num[dst] = 1;
        self.moved_num[src] = 0;
        let proj_ratio = if self.n_proj > 0 {
            let mut proj = std::mem::take(&mut self.proj_new);
            self.hop_proj_ratio(ri, rj, spin_create, &mut proj);
            let ratio = c_exp(log_proj_ratio(&proj, self.ele_proj_cnt, self.data));
            self.proj_new = proj;
            ratio
        } else {
            1.0
        };
        // No RBM terms in the supported class, so `with_rbm_ratio` is the identity.
        let index = self.ctx.one_jobs.len();
        self.ctx.one_jobs.push(GreenOneJob {
            msa: msj,
            rsa: ri + spin_annihilate as usize * self.n_site,
            ri,
            proj_ratio,
        });
        GreenOutcome::One { index, sign: 1.0 }
    }

    /// `green_func2_impl::<false>` classification.
    fn classify_two(
        &mut self,
        ri: usize,
        rj: usize,
        rk: usize,
        rl: usize,
        spin: u8,
        spin_other: u8,
    ) -> GreenOutcome {
        if self.n_elec == 0 || self.n_site == 0 {
            return GreenOutcome::Value(0.0);
        }
        let rsi = ri + spin as usize * self.n_site;
        let rsj = rj + spin as usize * self.n_site;
        let rtk = rk + spin_other as usize * self.n_site;
        if spin == spin_other {
            if rk == rl {
                return if self.ele_num[rtk] == 0 {
                    GreenOutcome::Value(0.0)
                } else {
                    self.classify_one(ri, rj, spin, spin)
                };
            } else if rj == rl {
                return GreenOutcome::Value(0.0);
            } else if ri == rl {
                return if self.ele_num[rsi] == 0 {
                    GreenOutcome::Value(0.0)
                } else if rj == rk {
                    GreenOutcome::Value((1 - self.ele_num[rsj]) as f64)
                } else {
                    match self.classify_one(rk, rj, spin, spin) {
                        GreenOutcome::Value(v) => GreenOutcome::Value(-v),
                        GreenOutcome::One { index, sign } => {
                            GreenOutcome::One { index, sign: -sign }
                        }
                        other => other,
                    }
                };
            } else if rj == rk {
                return if self.ele_num[rsj] == 1 {
                    GreenOutcome::Value(0.0)
                } else {
                    self.classify_one(ri, rl, spin, spin)
                };
            } else if ri == rk {
                return GreenOutcome::Value(0.0);
            } else if ri == rj {
                return if self.ele_num[rsi] == 0 {
                    GreenOutcome::Value(0.0)
                } else {
                    self.classify_one(rk, rl, spin, spin)
                };
            }
        } else if rk == rl {
            return if self.ele_num[rtk] == 0 {
                GreenOutcome::Value(0.0)
            } else if ri == rj {
                GreenOutcome::Value(self.ele_num[rsi] as f64)
            } else {
                self.classify_one(ri, rj, spin, spin)
            };
        } else if ri == rj {
            return if self.ele_num[rsi] == 0 {
                GreenOutcome::Value(0.0)
            } else {
                self.classify_one(rk, rl, spin_other, spin_other)
            };
        }
        // General case: rightmost hop rl -> rk (spin_other) then rj -> ri (spin).
        if self.ele_num[ri + spin as usize * self.n_site] != 0
            || self.ele_num[rj + spin as usize * self.n_site] != 1
            || self.ele_num[rk + spin_other as usize * self.n_site] != 0
            || self.ele_num[rl + spin_other as usize * self.n_site] != 1
        {
            return GreenOutcome::Value(0.0);
        }
        let mj_raw = self.ele_cfg[rj + spin as usize * self.n_site];
        let mi_raw = self.ele_cfg[rl + spin_other as usize * self.n_site];
        if mj_raw < 0 || mi_raw < 0 {
            return GreenOutcome::Value(0.0);
        }
        let mj = mj_raw as usize;
        let mi = mi_raw as usize;
        let msj = mj + spin as usize * self.n_elec;
        let msi = mi + spin_other as usize * self.n_elec;
        self.moved_num.copy_from_slice(self.ele_num);
        self.moved_num[rl + spin_other as usize * self.n_site] = 0;
        self.moved_num[rk + spin_other as usize * self.n_site] = 1;
        let mut proj_mid = std::mem::take(&mut self.proj_mid);
        self.hop_proj_ratio(rk, rl, spin_other, &mut proj_mid);
        self.moved_num[rj + spin as usize * self.n_site] = 0;
        self.moved_num[ri + spin as usize * self.n_site] = 1;
        let mut proj_final = std::mem::take(&mut self.proj_final);
        if self.n_proj > 0 {
            proj_final.clear();
            proj_final.resize(self.n_proj, 0);
            update_proj_cnt(
                rj as i64,
                ri as i64,
                spin,
                &mut proj_final,
                &proj_mid,
                &self.moved_num,
                self.data,
            );
        }
        let proj_ratio = if self.n_proj > 0 {
            c_exp(log_proj_ratio(&proj_final, self.ele_proj_cnt, self.data))
        } else {
            1.0
        };
        self.proj_mid = proj_mid;
        self.proj_final = proj_final;
        let index = self.ctx.two_jobs.len();
        self.ctx.two_jobs.push(GreenTwoJob {
            msa: msi,
            msb: msj,
            rsa_site: rk,
            rsb_site: ri,
            proj_ratio,
        });
        GreenOutcome::Two { index }
    }
}

/// Classify the PhysCal Green functions of one sample; only the real, non-FSZ, no-RBM class is
/// supported (the same class as the resident measurement).
fn green_context(
    data: &ExpertModeData,
    ip: Complex64,
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
) -> Result<GreenContext, StageError> {
    if data.has_rbm_terms() {
        return unsupported("the resident Green stage does not provide RBM terms");
    }
    let mut classifier = GreenClassifier::new(data, ip, ele_cfg, ele_num, ele_proj_cnt);
    for term in &data.green_one_terms {
        let outcome = classifier.classify_one(
            term.site1 as usize,
            term.site2 as usize,
            observables::spin_code(term.spin1),
            observables::spin_code(term.spin2),
        );
        classifier.ctx.one_outcomes.push(outcome);
    }
    for term in &data.green_two_terms {
        let outcome = classifier.classify_two(
            term.site1 as usize,
            term.site2 as usize,
            term.site3 as usize,
            term.site4 as usize,
            observables::spin_code(term.spin1),
            observables::spin_code(term.spin3),
        );
        classifier.ctx.two_outcomes.push(outcome);
    }
    Ok(classifier.ctx)
}

/// Device PhysCal Green functions of one sample: the host classifies the terms and computes the
/// projection ratios, the device evaluates the deferred Pfaffian reductions, and the host folds
/// them in term order exactly as [`observables::ordinary_green_values`].
pub fn resident_green_values(
    ops: &mut dyn ResidentMeasurementOps,
    input: &mut MeasurementInput<'_>,
) -> Result<GreenOutput, StageError> {
    let one_len = input.data.green_one_terms.len();
    let direct_len = input.data.green_two_terms.len();
    let ctx = green_context(
        input.data,
        input.ip,
        input.ele_cfg,
        input.ele_num,
        input.ele_proj_cnt,
    )?;
    let new_ip_one = if ctx.one_jobs.is_empty() {
        Vec::new()
    } else {
        let msa: Vec<i64> = ctx.one_jobs.iter().map(|j| j.msa as i64).collect();
        let rsa: Vec<i64> = ctx.one_jobs.iter().map(|j| j.rsa as i64).collect();
        let ri: Vec<i64> = ctx.one_jobs.iter().map(|j| j.ri as i64).collect();
        ops.ratio_terms(
            input.batch_slot,
            &ResidentRatioTerms {
                ele_idx: input.ele_idx,
                msa: &msa,
                rsa: &rsa,
                ri: &ri,
            },
        )?
    };
    let new_ip_two = if ctx.two_jobs.is_empty() {
        Vec::new()
    } else {
        let msa: Vec<i64> = ctx.two_jobs.iter().map(|j| j.msa as i64).collect();
        let msb: Vec<i64> = ctx.two_jobs.iter().map(|j| j.msb as i64).collect();
        let rsa: Vec<i64> = ctx.two_jobs.iter().map(|j| j.rsa_site as i64).collect();
        let rsb: Vec<i64> = ctx.two_jobs.iter().map(|j| j.rsb_site as i64).collect();
        ops.green_two(
            input.batch_slot,
            &ResidentGreenTwo {
                ele_idx: input.ele_idx,
                msa: &msa,
                msb: &msb,
                rsa_site: &rsa,
                rsb_site: &rsb,
            },
        )?
    };
    let ip = input.ip;
    let resolve = |outcome: GreenOutcome| -> Complex64 {
        match outcome {
            GreenOutcome::Value(v) => Complex64::new(v, 0.0),
            GreenOutcome::One { index, sign } => {
                let job = ctx.one_jobs[index];
                let value = crate::julia_complex::divide(
                    Complex64::new(job.proj_ratio * new_ip_one[index], 0.0),
                    ip,
                )
                .conj();
                if sign < 0.0 {
                    -value
                } else {
                    value
                }
            }
            GreenOutcome::Two { index } => {
                let job = ctx.two_jobs[index];
                crate::julia_complex::divide(
                    Complex64::new(job.proj_ratio * new_ip_two[index], 0.0),
                    ip,
                )
                .conj()
            }
        }
    };
    let one: Vec<Complex64> = ctx.one_outcomes.iter().map(|o| resolve(*o)).collect();
    let direct: Vec<Complex64> = ctx.two_outcomes.iter().map(|o| resolve(*o)).collect();
    debug_assert_eq!(one.len(), one_len);
    debug_assert_eq!(direct.len(), direct_len);
    Ok(GreenOutput { one, direct })
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
