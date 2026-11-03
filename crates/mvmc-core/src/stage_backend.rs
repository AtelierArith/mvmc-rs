//! One layered backend abstraction for the accelerated stages (issue #437, design
//! `docs/design/gpu-readiness.md` section 5.2 and the section "Unified stage backend").
//!
//! # Layering
//!
//! A *stage trait* describes one family of tensor-shaped work with host-slice inputs and
//! outputs:
//!
//! * [`SrStages`](crate::sr_backend::SrStages): Gram product, S/g assembly, Cholesky solve, CG
//!   matrix-vector product and the composite `sr_s_g` stage (issues #421, #424);
//! * [`PfaffianStages`]: batched Pfaffian and inverse of skew-symmetric planes (issue #423);
//! * [`MeasurementStages`]: local energy and Slater-parameter derivative of one real, non-FSZ
//!   measurement sample (issue #519).
//!
//! A [`StageBackend`] is the single object that composes one implementation of every stage
//! trait. It is what production code acquires ([`acquire`]), what the validation harness of
//! `accel_validation` drives, and what the GPU gates construct, so *validated means deployed*:
//! the harness exercises the very objects (`COrderSr`, `TenferroSr`, the batched Pfaffian
//! engines) that production selects. A future stage (batched local energy, #426; the
//! device-resident sampler, #434) is added as one more stage trait and one more slot of
//! [`StageBackend`]; existing implementations answer [`StageError::Unsupported`] until they
//! provide it, never a silent CPU result.
//!
//! # Selection
//!
//! [`StageBackendKind`]: C-order CPU (default, the parity oracle), tenferro CPU, or CUDA.
//! `MVMC_RS_SR_BACKEND=c-order|tenferro|cuda[:N]` selects it for production SR; an invalid
//! value or an unavailable device is an error, never a fallback. CUDA is only available when
//! the `gpu-cuda` feature is enabled and a provider (`gpu/mvmc-gpu-cuda`) is registered.
//!
//! Production routes the SR stages through the selected backend, and a backend that keeps the
//! SR step on a device (`SrStages::resident_direct`/`resident_cg`, issue #452) takes the real
//! direct step and the CG solve as whole composite stages (`sr.rs`, `sr_cg.rs`). The measurement
//! routes stage A (Pfaffian/inverse tables) through [`PfaffianStages`] and stage B (local
//! energy and Slater derivative) through [`MeasurementStages`] when the corresponding selector
//! is non-default (issues #422, #519). Since issue #518 milestone 2 a backend can additionally
//! run both measurement stages device-resident: [`PfaffianStages::stage_a_resident`] fills
//! tables that [`MeasurementStages::measure_sample`] consumes by `batch_slot`, so only the
//! sample's own rows and the small outputs cross the link inside a series.

use std::sync::{Mutex, MutexGuard};

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use pfapack::{dsktf2, utu2inv_real, utu2pfa_real, PivotIndex1Based, SqMat};

use crate::c_timer::{CTimer, TimerEnv, CTIMER_N};
use crate::measurement_backend::TenferroMeasurement;
use crate::slater_derivative::SlaterDerivativeScratch;
use crate::sr_backend::{COrderSr, SrStages, TenferroSr, TenferroStats};
use crate::state::{SlaterElmFlat, VmcOptimizationState};

/// Why a stage produced no result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// The backend does not implement this stage or dtype (typed capability failure; no CPU
    /// fallback).
    Unsupported(String),
    /// The backend failed while executing the stage.
    Failed(String),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Failed(s) => write!(f, "failed: {s}"),
        }
    }
}

impl std::error::Error for StageError {}

/// Pfaffian and inverse of one skew-symmetric matrix (column-major, `n x n`).
#[derive(Debug, Clone, PartialEq)]
pub struct PfInv {
    /// Pfaffian.
    pub pf: f64,
    /// Inverse `X^-1` (`X * inv = I`), column-major `n x n`. mVMC's `invM` is `-inv`
    /// (`calc_m_all_child_real` applies the sign flip); `pf` is unchanged.
    pub inv: Vec<f64>,
}

/// SR matrix and force.
#[derive(Debug, Clone, PartialEq)]
pub struct SrSg {
    /// `S`, column-major `npara x npara`.
    pub s: Vec<f64>,
    /// `g`, length `npara`.
    pub g: Vec<f64>,
}

/// Per-plane outcome of the batched Pfaffian stage (pfapack semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneOutcome {
    /// Factorization succeeded and the Pfaffian is finite.
    Ok,
    /// `dsktf2` found an exactly zero pivot column (1-based row of pfapack's `INFO`).
    ZeroPivot {
        /// 1-based row.
        row: usize,
    },
    /// The Pfaffian is not finite.
    NonFinite,
}

/// Result of [`PfaffianStages::pfaffian_inverse_batch`], planes in order `p = 0..planes`.
#[derive(Debug, Clone, PartialEq)]
pub struct PfInvBatch {
    /// Pfaffian per plane.
    pub pf: Vec<f64>,
    /// Inverses, `[n, n, planes]` column-major (`inv[p*n*n..]` is plane `p`).
    pub inv: Vec<f64>,
    /// Outcome per plane.
    pub outcome: Vec<PlaneOutcome>,
}

/// Per-plane results when inverses overwrite caller-owned matrix storage.
#[derive(Debug, Clone, PartialEq)]
pub struct PfBatch {
    /// Pfaffian per plane, in input order.
    pub pf: Vec<f64>,
    /// Numerical outcome per plane, in input order.
    pub outcome: Vec<PlaneOutcome>,
}

/// Device-resident measurement stage A request (issue #518 milestone 2, `stage_a_pfaffian`).
///
/// The caller lists the batch slots it needs tables for by their electron sites; the backend
/// assembles and factorizes the planes on the device and keeps them there for the resident
/// measurement stage B. Only the Pfaffians cross back by default; the inverse planes are
/// returned only when the caller states that host consumers need them.
pub struct ResidentStageAInput<'a> {
    /// The working Slater table of the current wavefunction.
    pub slater: &'a SlaterElmFlat<f64>,
    /// `[slots * n]` electron sites of the requested samples (slot-major, `n = 2*Ne`).
    pub ele_idx: &'a [i64],
    /// Number of samples in the request.
    pub slots: usize,
    /// Lattice sites.
    pub n_site: usize,
    /// Electrons per spin.
    pub n_elec: usize,
    /// QP planes per sample.
    pub n_qp: usize,
    /// When true the caller needs the inverse planes on the host as well (PhysCal Green
    /// functions, Lanczos, or a host measurement stage B).
    pub download_inv: bool,
}

/// Result of [`PfaffianStages::stage_a_resident`], planes ordered `slot * n_qp + qp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentStageAOutput {
    /// Pfaffians, `[slots * n_qp]`.
    pub pf: Vec<f64>,
    /// mVMC inverse planes `invM = -X^-1`, `[n, n, slots * n_qp]` column-major, when the
    /// request asked for them.
    pub inv: Option<Vec<f64>>,
    /// Per-plane outcome (pfapack semantics).
    pub outcome: Vec<PlaneOutcome>,
}

/// Batched Pfaffian and inverse of real skew-symmetric planes.
pub trait PfaffianStages: Send {
    /// Short label, for example `c-order-cpu`.
    fn label(&self) -> String;
    /// Provider description recorded in the benchmark metadata.
    fn provider(&self) -> String;

    /// Pfaffian and inverse of `planes` column-major `n x n` matrices stored back to back
    /// (`x.len() == n * n * planes`). Numerical failures of a plane are reported in
    /// [`PfInvBatch::outcome`]; a backend that cannot run the stage returns
    /// [`StageError::Unsupported`].
    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError>;

    /// Optionally replace each input plane by its inverse, avoiding a full
    /// batch allocation/copy. `None` must leave `x` unchanged; the caller then
    /// uses this same provider's ordinary batch method, never another backend.
    fn pfaffian_inverse_batch_in_place(
        &mut self,
        _x: &mut [f64],
        _n: usize,
        _planes: usize,
    ) -> Result<Option<PfBatch>, StageError> {
        Ok(None)
    }

    /// Whether this backend provides the device-resident stage A protocol
    /// ([`Self::stage_a_resident`]). The C-order oracle and every host-only backend leave the
    /// default `false`, so the caller keeps the host-plane path.
    fn resident_stage_a(&self) -> bool {
        false
    }

    /// Device-resident stage A (issue #518 milestone 2): assemble and factorize the requested
    /// samples without host plane buffers, keeping the tables on the device for a resident
    /// measurement stage B. A backend without the protocol returns [`StageError::Unsupported`];
    /// `resident_stage_a` is the capability query the caller checks first.
    fn stage_a_resident(
        &mut self,
        _input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        Err(StageError::Unsupported(format!(
            "{} does not provide the device-resident measurement stage A",
            self.label()
        )))
    }

    /// One plane; a numerical failure is a [`StageError::Failed`].
    fn pfaffian_inverse(&mut self, x: &[f64], n: usize) -> Result<PfInv, StageError> {
        let out = self.pfaffian_inverse_batch(x, n, 1)?;
        match out.outcome[0] {
            PlaneOutcome::Ok => Ok(PfInv {
                pf: out.pf[0],
                inv: out.inv,
            }),
            PlaneOutcome::ZeroPivot { row } => {
                Err(StageError::Failed(format!("zero pivot at {row}")))
            }
            PlaneOutcome::NonFinite => Err(StageError::Failed("non-finite Pfaffian".into())),
        }
    }
}

/// Weighted centering used by every backend (a definition of the SR stage, in C order).
pub fn centered(o: &[f64], ns: usize, np: usize, e: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut d = o.to_vec();
    for p in 0..np {
        let mut mean = 0.0;
        for k in 0..ns {
            mean += w[k] * o[k + p * ns];
        }
        for k in 0..ns {
            d[k + p * ns] -= mean;
        }
    }
    let mut emean = 0.0;
    for k in 0..ns {
        emean += w[k] * e[k];
    }
    let de = e.iter().map(|v| v - emean).collect();
    (d, de)
}

/// C-order CPU Pfaffian stage: the production PfaPack sequence (`dsktf2`, `utu2pfa_real`,
/// `utu2inv_real`) per plane. This is the oracle of the validation harness.
#[derive(Debug, Default, Clone, Copy)]
pub struct COrderPfaffian;

type COrderPlaneScratch = (Vec<PivotIndex1Based>, Vec<f64>, Vec<f64>);
thread_local! {
    // Each worker retains only its pivots and inversion scratch. dsktf2 resets
    // pivots and utu2inv resets its full workspace before reading it, so failed
    // planes and changes of matrix size cannot supply numerical input later.
    static C_ORDER_PLANE_SCRATCH: std::cell::RefCell<COrderPlaneScratch> = const {
        std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new()))
    };
}

impl PfaffianStages for COrderPfaffian {
    fn label(&self) -> String {
        "c-order-cpu".to_string()
    }

    fn provider(&self) -> String {
        "pfapack dsktf2/utu2pfa_real/utu2inv_real".to_string()
    }

    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        let mut inv = x.to_vec();
        let out = self
            .pfaffian_inverse_batch_in_place(&mut inv, n, planes)?
            .expect("C-order implements the in-place batch method");
        Ok(PfInvBatch {
            pf: out.pf,
            inv,
            outcome: out.outcome,
        })
    }

    fn pfaffian_inverse_batch_in_place(
        &mut self,
        x: &mut [f64],
        n: usize,
        planes: usize,
    ) -> Result<Option<PfBatch>, StageError> {
        if x.len() != n * n * planes {
            return Err(StageError::Failed(format!(
                "buffer length {} != n*n*planes = {}",
                x.len(),
                n * n * planes
            )));
        }
        let plane_len = n * n.max(1);
        assert_ne!(plane_len, 0, "chunk size must be non-zero");
        // Each plane has its own factorization, inverse and status. Keep the
        // scalar PfaPack sequence, but distribute independent planes across the
        // same bounded inner pool used by calc_m_all (#586). The disjoint output
        // windows retain input order, including failed/non-finite planes.
        let mut results = vec![(0.0, PlaneOutcome::Ok); planes];
        crate::threading::qp_update(
            &mut results,
            x,
            plane_len,
            plane_len,
            0,
            planes,
            crate::pfaffian::pfaffian_qp_cost_ns(n),
            || (),
            |_, _, result_slot, a| {
                C_ORDER_PLANE_SCRATCH.with_borrow_mut(|(pivots, vt, work)| {
                    pivots.resize(n, PivotIndex1Based(0));
                    let result = {
                        let mut view = SqMat::new(&mut *a, n);
                        dsktf2(&mut view, pivots).map(|_| utu2pfa_real(&view, pivots))
                    };
                    match result {
                        Err(info) => {
                            a.fill(0.0);
                            *result_slot = (0.0, PlaneOutcome::ZeroPivot { row: info });
                        }
                        Ok(value) => {
                            // Worker-local scratch is reused across batches.
                            // utu2inv resets its full matrix workspace;
                            // failed/non-finite planes cannot leak into the next one.
                            vt.resize(n.saturating_sub(1), 0.0);
                            work.resize(n * n, 0.0);
                            {
                                let mut view = SqMat::new(&mut *a, n);
                                let mut m_work = SqMat::new(work, n);
                                utu2inv_real(&mut view, pivots, vt, &mut m_work);
                            }
                            // `a` now holds X^-1 (X * invM = I) for the matrix as given
                            *result_slot = (
                                value,
                                if value.is_finite() {
                                    PlaneOutcome::Ok
                                } else {
                                    PlaneOutcome::NonFinite
                                },
                            );
                        }
                    }
                })
            },
        );
        let (pf, outcome) = results.into_iter().unzip();
        Ok(Some(PfBatch { pf, outcome }))
    }
}

pub(crate) const TENFERRO_PFAFFIAN_UNAVAILABLE: &str =
    "tenferro 0.8 has no skew-symmetric factorization or Pfaffian";

/// A Pfaffian stage that always reports [`StageError::Unsupported`] with the stated reason
/// (tenferro 0.8 has no skew-symmetric factorization or Pfaffian).
#[derive(Debug, Clone)]
pub struct UnsupportedPfaffian {
    label: String,
    reason: String,
}

impl UnsupportedPfaffian {
    /// Stage named `label` that is unsupported because of `reason`.
    pub fn new(label: &str, reason: &str) -> Self {
        Self {
            label: label.to_string(),
            reason: reason.to_string(),
        }
    }

    /// The tenferro 0.8 gap.
    pub fn tenferro() -> Self {
        Self::new("none", TENFERRO_PFAFFIAN_UNAVAILABLE)
    }
}

impl PfaffianStages for UnsupportedPfaffian {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn provider(&self) -> String {
        "unsupported".to_string()
    }

    fn pfaffian_inverse_batch(
        &mut self,
        _x: &[f64],
        _n: usize,
        _planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        Err(StageError::Unsupported(self.reason.clone()))
    }
}

// ---------------------------------------------------------------------------------------
// Measurement stage B (issue #519): local energy and Slater derivative
// ---------------------------------------------------------------------------------------

/// Timing sink of the routed measurement stage. The runner passes its own [`CTimer`] and the
/// C-order stage records the same section IDs by running the kernels against a fresh internal
/// timer and merging its elapsed counters here, so routing does not change the timing report.
pub trait MeasurementTiming {
    /// Whether the runner's timer records elapsed sections.
    fn enabled(&self) -> bool;
    /// The runner's diagnostic switches (they select the numerical path of the kernels).
    fn diagnostics(&self) -> TimerEnv;
    /// Add a worker timer's elapsed counters into this timer.
    fn merge_elapsed(&mut self, elapsed_ns: &[u64; CTIMER_N]);
}

impl<const ENABLED: bool> MeasurementTiming for CTimer<ENABLED> {
    fn enabled(&self) -> bool {
        ENABLED
    }

    fn diagnostics(&self) -> TimerEnv {
        self.diagnostics
    }

    fn merge_elapsed(&mut self, elapsed_ns: &[u64; CTIMER_N]) {
        if ENABLED {
            for (slot, worker) in self.elapsed_ns.iter_mut().zip(elapsed_ns) {
                *slot = slot.wrapping_add(*worker);
            }
        }
    }
}

/// Inputs of one real, non-FSZ saved-sample measurement. The runner's working Slater tables
/// (already swapped in) hold the sample.
///
/// This is the host contract of milestone 1 (issue #519): the runner passes its
/// `ExpertModeData`/`VmcOptimizationState`, so a device backend that cannot reconstruct them
/// answers [`StageError::Unsupported`]. A flat device-facing layout is a later milestone.
///
/// `sample` and `batch_slot` are the residency keys of milestone 2 (issue #518): the runner
/// tells the stage which saved configuration of the chain and which slot of the stage A chunk
/// the sample is, so a device-resident backend can address the tables produced by
/// [`PfaffianStages::stage_a_resident`] without any per-sample table transfer.
pub struct MeasurementInput<'a> {
    /// Immutable physics configuration.
    pub data: &'a ExpertModeData,
    /// Mutable optimizer state whose working Slater tables hold this sample.
    pub state: &'a mut VmcOptimizationState,
    /// Overlap `<phi|Psi>` of the sample.
    pub ip: Complex64,
    /// Electron sites `[2 * Ne]` of the sample.
    pub ele_idx: &'a [i64],
    /// Electron-configuration descriptors.
    pub ele_cfg: &'a [i64],
    /// Site occupation numbers.
    pub ele_num: &'a [i64],
    /// Projection counts.
    pub ele_proj_cnt: &'a [i64],
    /// Index of the saved configuration in the chain (`0..NVMCSample`).
    pub sample: usize,
    /// Slot of the sample inside the stage A chunk (the residency key).
    pub batch_slot: usize,
}

/// Result of [`MeasurementStages::measure_sample`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementOutput {
    /// Local energy `<phi|H|Psi>/<phi|Psi>`.
    pub local_energy: Complex64,
    /// Slater-parameter derivative block (`2 * n_slater` real/imaginary pairs); empty when the
    /// runner did not request it.
    pub slater_diff: Vec<Complex64>,
}

/// PhysCal Green-function values of one sample (issue #517 milestone 5), in the term order of
/// `data.green_one_terms` / `data.green_two_terms`.
///
/// The one-body values are `<c^dag_{ri,s1} c_{rj,s2}>` and the direct two-body values are
/// `<c^dag_{ri,s1} c_{rj,s1} c^dag_{rk,s3} c_{rl,s3}>`; the runner accumulates them exactly as
/// it accumulates the host [`crate::observables`] consumer. A device backend returns this so
/// the host never needs the inverse planes of a PhysCal sample.
#[derive(Debug, Clone, PartialEq)]
pub struct GreenOutput {
    /// One-body values, one per `data.green_one_terms` entry.
    pub one: Vec<Complex64>,
    /// Direct two-body values, one per `data.green_two_terms` entry.
    pub direct: Vec<Complex64>,
}

/// Host-visible counters of the device-resident measurement path (issue #518 milestone 2),
/// for tests and benchmark metadata.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ResidentStats {
    /// Slater-table uploads (one per changed content version).
    pub slater_uploads: u64,
    /// Bytes uploaded for Slater tables.
    pub slater_upload_bytes: u64,
    /// Resident stage A chunks.
    pub stage_a_calls: u64,
    /// Planes factorized on the device.
    pub planes_factorized: u64,
    /// Per-plane inverse-table downloads (`slots * n_qp` planes each).
    pub inv_download_planes: u64,
    /// Bytes downloaded for inverse tables.
    pub inv_download_bytes: u64,
    /// Bytes downloaded for Pfaffians (one `NQP` vector per sample plus status).
    pub pf_download_bytes: u64,
    /// Resident local-energy ratio tables.
    pub ratio_tables: u64,
    /// Resident Slater-derivative evaluations.
    pub slater_diffs: u64,
}

impl ResidentStats {
    /// Total bytes transferred host-to-device for tables (configuration/rows excluded).
    pub fn table_upload_bytes(&self) -> u64 {
        self.slater_upload_bytes
    }

    /// Total bytes transferred device-to-host for computed tables.
    pub fn table_download_bytes(&self) -> u64 {
        self.inv_download_bytes + self.pf_download_bytes
    }
}

/// Input-fixed tables of the resident measurement path: uploaded once per `(geometry, data)`
/// and reused across samples and steps.
pub struct ResidentInputRequest<'a> {
    /// Lattice sites.
    pub n_site: usize,
    /// Electrons per spin.
    pub n_elec: usize,
    /// QP planes per sample.
    pub n_qp: usize,
    /// Number of unique Slater parameters (`orbital_idx` values are `< n_slater`).
    pub n_slater: usize,
    /// QP points per translation sector (`NSPGaussLeg`).
    pub n_sp_gauss_leg: usize,
    /// Site-to-orbital index matrix (`orbital_idx[site][site]`, `-1` = none).
    pub orbital_idx: &'a [Vec<i64>],
    /// Site-to-orbital sign matrix.
    pub orbital_sgn: &'a [Vec<i64>],
    /// QP weights `[n_qp]`.
    pub qp_weight: &'a [f64],
    /// `spgl_cos_sin[spidx]` weight factors `[n_sp_gauss_leg]`.
    pub spgl_cos_sin: &'a [f64],
    /// `spgl_cos_cos[spidx]`.
    pub spgl_cos_cos: &'a [f64],
    /// `spgl_sin_sin[spidx]`.
    pub spgl_sin_sin: &'a [f64],
}

/// Local energy and Slater-parameter derivative of one real, non-FSZ sample (issue #519).
///
/// A backend that cannot run the stage returns [`StageError::Unsupported`]; the runner reports
/// that as a hard error and never falls back to the inline C-order calls.
pub trait MeasurementStages: Send {
    /// Short label, for example `c-order-cpu`.
    fn label(&self) -> String;
    /// Provider description recorded in the benchmark metadata.
    fn provider(&self) -> String;

    /// Whether this backend keeps the measurement tables device-resident (issue #518
    /// milestone 2). When true, `batch_slot` addresses tables produced by
    /// [`PfaffianStages::stage_a_resident`] and no per-sample Slater/inverse table transfer
    /// happens.
    fn resident_measurement(&self) -> bool {
        false
    }

    /// Upload the input-fixed tables of the resident path (no-op otherwise). Called once per
    /// measurement sweep before the first stage A chunk.
    fn resident_prepare(
        &mut self,
        _data: &ExpertModeData,
        _state: &VmcOptimizationState,
    ) -> Result<(), StageError> {
        Ok(())
    }

    /// Counters of the resident path, `None` when the backend is not resident.
    fn resident_stats(&self) -> Option<ResidentStats> {
        None
    }

    /// Whether this backend computes the PhysCal Green functions on the device (issue #517
    /// milestone 5). When true the runner uses [`Self::measure_green`] for the host Green
    /// consumer and may keep the inverse planes resident.
    fn resident_green(&self) -> bool {
        false
    }

    /// Device PhysCal Green functions of one sample (issue #517 milestone 5). The default
    /// `None` keeps the host C-order consumer; a backend that reports [`Self::resident_green`]
    /// must return `Some`. `one_len`/`direct_len` size the outputs at the caller's term counts.
    fn measure_green(
        &mut self,
        _input: &mut MeasurementInput<'_>,
        _one_len: usize,
        _direct_len: usize,
        _timing: &mut dyn MeasurementTiming,
    ) -> Result<Option<GreenOutput>, StageError> {
        Ok(None)
    }

    /// Compute the local energy and (when `slater_diff_len > 0`) the Slater-parameter
    /// derivative of one sample. `slater_diff_len` is the number of complex slots the runner
    /// reserves for the derivative (the truncated length of the `SROptO` Slater block).
    fn measure_sample(
        &mut self,
        _input: &mut MeasurementInput<'_>,
        _slater_diff_len: usize,
        _timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        Err(StageError::Unsupported(format!(
            "{} does not provide the measurement stage",
            self.label()
        )))
    }
}

/// C-order CPU measurement stage: the production `calculate_local_energy` and
/// `slater_elm_diff` kernels, operation for operation. This is the oracle.
#[derive(Default)]
pub struct COrderMeasurement {
    scratch: SlaterDerivativeScratch,
}

impl COrderMeasurement {
    fn run<const TIMED: bool>(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timer: &mut CTimer<TIMED>,
    ) -> MeasurementOutput {
        let local_energy = crate::observables::calculate_local_energy_timed::<TIMED>(
            input.ip,
            input.data,
            input.state,
            input.ele_idx,
            input.ele_cfg,
            input.ele_num,
            input.ele_proj_cnt,
            timer,
        );
        let mut slater_diff = Vec::new();
        if slater_diff_len > 0 {
            slater_diff.resize(slater_diff_len, Complex64::new(0.0, 0.0));
            // Section 930 is the runner's Slater-derivative wrapper around the inline call;
            // record it here too so the routed path keeps the timing report.
            timer.start_diag(930, timer.diagnostics.slater);
            crate::slater_derivative::slater_elm_diff_with_scratch_timed::<TIMED>(
                &mut slater_diff,
                input.ip,
                input.ele_idx,
                input.data,
                &input.state.slater_matrix,
                true,
                &mut self.scratch,
                timer,
            );
            timer.stop_diag(930, timer.diagnostics.slater);
        }
        MeasurementOutput {
            local_energy,
            slater_diff,
        }
    }
}

impl MeasurementStages for COrderMeasurement {
    fn label(&self) -> String {
        "c-order-cpu".to_string()
    }

    fn provider(&self) -> String {
        "observables calculate_local_energy + slater_elm_diff (C order)".to_string()
    }

    fn measure_sample(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        if timing.enabled() {
            let mut timer = CTimer::<true>::new();
            timer.diagnostics = timing.diagnostics();
            let out = self.run(input, slater_diff_len, &mut timer);
            timing.merge_elapsed(&timer.elapsed_ns);
            Ok(out)
        } else {
            let mut timer = CTimer::<false>::new();
            timer.diagnostics = timing.diagnostics();
            Ok(self.run(input, slater_diff_len, &mut timer))
        }
    }
}

/// A measurement stage that always reports [`StageError::Unsupported`] with the stated reason
/// (a backend without a device or host measurement implementation).
#[derive(Debug, Clone)]
pub struct UnsupportedMeasurement {
    label: String,
    reason: String,
}

impl UnsupportedMeasurement {
    /// Stage named `label` that is unsupported because of `reason`.
    pub fn new(label: &str, reason: &str) -> Self {
        Self {
            label: label.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl MeasurementStages for UnsupportedMeasurement {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn provider(&self) -> String {
        "unsupported".to_string()
    }

    fn measure_sample(
        &mut self,
        _input: &mut MeasurementInput<'_>,
        _slater_diff_len: usize,
        _timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        Err(StageError::Unsupported(self.reason.clone()))
    }
}

type CpuEngineHandle = std::sync::Arc<Mutex<crate::engine_resource::EngineResource>>;

// Count stage handles and solve leases separately from the stage mutex. A solve lease may
// span MPI collectives, but it never carries the stage mutex across those collectives.
// Concurrent walkers share the team until the last lease ends.
struct ScopedCpuTeam {
    engine: CpuEngineHandle,
    leases: Mutex<usize>,
    settings: crate::sr_backend::CpuEngineSettings,
}

impl ScopedCpuTeam {
    fn acquire(&self) -> Result<(), StageError> {
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        if *leases == 0 {
            let fresh = crate::sr_backend::cpu_engine_with_settings(self.settings)?;
            let replacement = std::mem::replace(
                &mut *fresh.lock().unwrap_or_else(|e| e.into_inner()),
                crate::engine_resource::EngineResource::new(Box::new(
                    tenferro_cpu::CpuEngine::serial(),
                )),
            );
            let old = std::mem::replace(
                &mut *self.engine.lock().unwrap_or_else(|e| e.into_inner()),
                replacement,
            );
            drop(old);
        }
        *leases += 1;
        Ok(())
    }

    fn release(&self) {
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        *leases -= 1;
        if *leases == 0 {
            let old = std::mem::replace(
                &mut *self.engine.lock().unwrap_or_else(|e| e.into_inner()),
                crate::engine_resource::EngineResource::new(Box::new(
                    tenferro_cpu::CpuEngine::serial(),
                )),
            );
            drop(old);
        }
    }
}

pub(crate) struct CpuTeamLease(std::sync::Arc<ScopedCpuTeam>);

impl Drop for CpuTeamLease {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// The single backend object: one implementation of every stage trait.
pub struct StageBackend<'a> {
    label: String,
    sr: Box<dyn SrStages + 'a>,
    pfaffian: Box<dyn PfaffianStages + 'a>,
    measurement: Box<dyn MeasurementStages + 'a>,
    transient_cpu: Option<std::sync::Arc<ScopedCpuTeam>>,
    probe_cpu: Option<CpuEngineHandle>,
}

impl<'a> StageBackend<'a> {
    /// Compose a backend labelled `label` from one implementation per stage trait. The
    /// measurement slot defaults to [`UnsupportedMeasurement`]; add it with
    /// [`StageBackend::with_measurement`].
    pub fn new(
        label: &str,
        sr: Box<dyn SrStages + 'a>,
        pfaffian: Box<dyn PfaffianStages + 'a>,
    ) -> Self {
        Self {
            label: label.to_string(),
            sr,
            pfaffian,
            transient_cpu: None,
            probe_cpu: None,
            measurement: Box::new(UnsupportedMeasurement::new(
                label,
                "this backend provides no measurement stage",
            )),
        }
    }

    /// C-order CPU: BLAS/LAPACK SR stages, the PfaPack Pfaffian and the inline measurement
    /// kernels (default, parity oracle).
    pub fn c_order() -> StageBackend<'static> {
        StageBackend::new(
            "c-order-cpu",
            Box::new(COrderSr::default()),
            Box::new(COrderPfaffian),
        )
        .with_measurement(Box::new(COrderMeasurement::default()))
    }

    /// tenferro CPU (`native`): SR stages through eager ops and the real, non-FSZ
    /// measurement stage (local energy + Slater derivative) through tensor contractions;
    /// the Pfaffian stage is unsupported by tenferro 0.8.
    pub fn tenferro_cpu() -> Result<StageBackend<'static>, StageError> {
        Self::tenferro_cpu_with_transient(false)
    }

    fn tenferro_cpu_with_transient(transient: bool) -> Result<StageBackend<'static>, StageError> {
        let ctx =
            tenferro_ad::EagerRuntime::new().map_err(|e| StageError::Failed(e.to_string()))?;
        let settings = crate::sr_backend::cpu_engine_settings()?;
        let engine = crate::sr_backend::cpu_engine_with_settings(settings)?;
        let mut backend = StageBackend::new(
            "tenferro-cpu",
            Box::new(TenferroSr::with_runtime(
                ctx.clone(),
                engine.clone(),
                crate::sr_backend::Placement::Host,
                "tenferro native".to_string(),
            )),
            Box::new(UnsupportedPfaffian::tenferro()),
        )
        .with_measurement(Box::new(TenferroMeasurement::with_runtime(
            ctx,
            engine.clone(),
            crate::sr_backend::Placement::Host,
            "tenferro-measurement native".to_string(),
        )));
        if !transient {
            backend.probe_cpu = Some(engine.clone());
        }
        if transient {
            let old = std::mem::replace(
                &mut *engine.lock().unwrap_or_else(|e| e.into_inner()),
                crate::engine_resource::EngineResource::new(Box::new(
                    tenferro_cpu::CpuEngine::serial(),
                )),
            );
            drop(old);
            backend.transient_cpu = Some(std::sync::Arc::new(ScopedCpuTeam {
                engine,
                leases: Mutex::new(0),
                settings,
            }));
        }
        Ok(backend)
    }

    /// Replace the Pfaffian stage (for example by the batched CUDA kernel), relabelled.
    pub fn with_pfaffian(mut self, label: &str, pfaffian: Box<dyn PfaffianStages + 'a>) -> Self {
        self.label = label.to_string();
        self.pfaffian = pfaffian;
        self
    }

    /// Replace the measurement stage (for example by a device stage).
    pub fn with_measurement(mut self, measurement: Box<dyn MeasurementStages + 'a>) -> Self {
        self.measurement = measurement;
        self
    }

    /// Backend label.
    pub fn label(&self) -> String {
        self.label.clone()
    }

    /// Provider description: one entry per stage trait.
    pub fn provider(&self) -> String {
        format!(
            "sr: {}; pfaffian: {}; measurement: {}",
            self.sr.provider(),
            self.pfaffian.provider(),
            self.measurement.provider()
        )
    }

    /// SR stages.
    pub fn sr(&mut self) -> &mut dyn SrStages {
        &mut *self.sr
    }

    /// Batched Pfaffian and inverse stage.
    pub fn pfaffian(&mut self) -> &mut dyn PfaffianStages {
        &mut *self.pfaffian
    }

    /// Local-energy and Slater-derivative measurement stage.
    pub fn measurement(&mut self) -> &mut dyn MeasurementStages {
        &mut *self.measurement
    }
}

impl std::fmt::Debug for StageBackend<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageBackend")
            .field("label", &self.label)
            .field("provider", &self.provider())
            .finish()
    }
}

// ---------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------

/// Selectable backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageBackendKind {
    /// C-order CPU (default, parity oracle).
    COrder,
    /// tenferro eager ops on the CPU backend (`native`).
    TenferroCpu,
    /// tenferro CUDA on the given device ordinal (feature `gpu-cuda` and a provider).
    Cuda(usize),
}

/// Environment variable selecting the production backend (`c-order`, `tenferro`, `cuda[:N]`).
pub const SR_BACKEND_VARIABLE: &str = "MVMC_RS_SR_BACKEND";

/// Parse a selector value.
pub fn parse_stage_backend(value: &str) -> Result<StageBackendKind, String> {
    let v = value.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "c" | "c-order" | "corder" | "default" => Ok(StageBackendKind::COrder),
        "tenferro" | "tenferro-cpu" => Ok(StageBackendKind::TenferroCpu),
        "cuda" => Ok(StageBackendKind::Cuda(0)),
        other => match other.strip_prefix("cuda:").map(str::parse::<usize>) {
            Some(Ok(ordinal)) => Ok(StageBackendKind::Cuda(ordinal)),
            _ => Err(format!(
                "{SR_BACKEND_VARIABLE}={other:?} is not one of c-order, tenferro, cuda[:N]"
            )),
        },
    }
}

static OVERRIDE: Mutex<Option<StageBackendKind>> = Mutex::new(None);

/// Process-wide override of the backend choice for tests (`None` restores env/default).
pub fn set_stage_backend_override(kind: Option<StageBackendKind>) {
    *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = kind;
}

/// The backend chosen by the override, then the environment, then the default (C order).
///
/// An invalid environment value is an error, never a silent fallback.
pub fn try_selected_stage_backend() -> Result<StageBackendKind, String> {
    if let Some(kind) = *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) {
        return Ok(kind);
    }
    match std::env::var(SR_BACKEND_VARIABLE) {
        Ok(v) => parse_stage_backend(&v),
        Err(std::env::VarError::NotPresent) => Ok(StageBackendKind::COrder),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{SR_BACKEND_VARIABLE} is not valid UTF-8"))
        }
    }
}

/// Like [`try_selected_stage_backend`], but panics on an invalid value (library callers that
/// reached production code without the CLI preflight; the CLI uses
/// [`validate_selected_stage_backend`] before any IO).
pub fn selected_stage_backend() -> StageBackendKind {
    try_selected_stage_backend().unwrap_or_else(|e| panic!("{e}"))
}

/// Startup preflight: resolve the selected backend and, for a non-default kind, open (and
/// cache) it, so that an invalid selector or an unavailable backend (no `gpu-cuda` feature, no
/// registered provider, device ordinal out of range) is reported as an error before any IO
/// instead of a panic in the first SR step. The default C-order backend needs no opening.
///
/// # Errors
///
/// A message that names [`SR_BACKEND_VARIABLE`].
pub fn validate_selected_stage_backend() -> Result<StageBackendKind, String> {
    let kind = try_selected_stage_backend()?;
    preflight_stage_backend(kind, SR_BACKEND_VARIABLE)?;
    Ok(kind)
}

/// Open (and cache) a non-default backend `kind` selected through `variable`, reporting every
/// failure, including a provider panic, as an error message that names the variable. The
/// C-order backend needs no opening. Used by the startup preflight of every selector that
/// resolves to a [`StageBackendKind`].
///
/// # Errors
///
/// A message that names `variable`.
pub fn preflight_stage_backend(kind: StageBackendKind, variable: &str) -> Result<(), String> {
    if kind != StageBackendKind::COrder {
        // A provider may panic while loading a driver library (cudarc panics when `libcudart`
        // is missing); report that as an error too, with the panic message and without the
        // default hook's output.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let opened = std::panic::catch_unwind(|| shared(kind).map(|_| ()));
        std::panic::set_hook(previous);
        match opened {
            Ok(result) => result.map_err(|e| format!("{variable}: {e}"))?,
            Err(payload) => {
                let text = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "backend initialization panicked".to_string());
                return Err(format!("{variable}: backend unavailable: {text}"));
            }
        }
    }
    Ok(())
}

/// Construct a fresh backend of the given kind.
///
/// CUDA is an error unless the build has the `gpu-cuda` feature and a provider is registered
/// (`crate::backend::register_cuda_provider`): there is no CPU fallback.
pub fn open_stage_backend(kind: StageBackendKind) -> Result<StageBackend<'static>, StageError> {
    match kind {
        StageBackendKind::COrder => Ok(StageBackend::c_order()),
        StageBackendKind::TenferroCpu => StageBackend::tenferro_cpu(),
        StageBackendKind::Cuda(ordinal) => crate::backend::open_cuda_stage_backend(ordinal)
            .map_err(|e| StageError::Unsupported(e.to_string())),
    }
}

type Shared = &'static Mutex<StageBackend<'static>>;

/// Backends other than C order are built once per process and shared (a tenferro runtime or a
/// CUDA context is expensive; eager calls cost tens of microseconds).
static SHARED: Mutex<Vec<(StageBackendKind, Shared)>> = Mutex::new(Vec::new());
static PROBE_CPU_ENGINE: std::sync::OnceLock<CpuEngineHandle> = std::sync::OnceLock::new();
#[cfg(test)]
static PROBE_CPU_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROBE_ATTEMPTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROBE_SUCCESSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROBE_BUSY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROBE_UNAVAILABLE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROBE_PARTICIPANTS: std::sync::OnceLock<
    Mutex<std::collections::HashSet<std::thread::ThreadId>>,
> = std::sync::OnceLock::new();

fn probe_profile() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE").as_deref() == Ok("1")
    })
}

#[doc(hidden)]
pub fn probe_cpu_snapshot() -> (usize, usize, usize, usize, usize) {
    use std::sync::atomic::Ordering;
    let participants = PROBE_PARTICIPANTS
        .get()
        .map_or(0, |ids| ids.lock().unwrap().len());
    (
        PROBE_ATTEMPTS.load(Ordering::Relaxed),
        PROBE_SUCCESSES.load(Ordering::Relaxed),
        PROBE_BUSY.load(Ordering::Relaxed),
        PROBE_UNAVAILABLE.load(Ordering::Relaxed),
        participants,
    )
}

#[derive(Clone, Copy)]
pub(crate) enum ProbeCpuDispatch {
    Complete,
    Unavailable,
    Busy,
}

fn probe_outcome(outcome: ProbeCpuDispatch, profile: bool) -> ProbeCpuDispatch {
    if profile {
        let counter = match outcome {
            ProbeCpuDispatch::Complete => &PROBE_SUCCESSES,
            ProbeCpuDispatch::Busy => &PROBE_BUSY,
            ProbeCpuDispatch::Unavailable => &PROBE_UNAVAILABLE,
        };
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    outcome
}

pub(crate) fn try_probe_cpu_blocks(blocks: usize, body: impl Fn(usize) + Sync) -> ProbeCpuDispatch {
    let profile = probe_profile();
    if profile {
        PROBE_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(engine) = PROBE_CPU_ENGINE.get() else {
        return probe_outcome(ProbeCpuDispatch::Unavailable, profile);
    };
    let mut engine = match engine.try_lock() {
        Ok(engine) => engine,
        Err(std::sync::TryLockError::WouldBlock | std::sync::TryLockError::Poisoned(_)) => {
            return probe_outcome(ProbeCpuDispatch::Busy, profile);
        }
    };
    let outcome = tenferro_cpu::with_cpu_engine(&mut ***engine, |cpu| {
        if !cpu.has_team() {
            return ProbeCpuDispatch::Unavailable;
        }
        let Ok(mut exec) = cpu.exec() else {
            return ProbeCpuDispatch::Busy;
        };
        exec.token()
            .for_ranges(blocks, std::num::NonZeroUsize::MIN, |range| {
                if profile {
                    PROBE_PARTICIPANTS
                        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
                        .lock()
                        .unwrap()
                        .insert(std::thread::current().id());
                }
                for block in range {
                    body(block);
                }
            });
        #[cfg(test)]
        PROBE_CPU_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ProbeCpuDispatch::Complete
    })
    .unwrap_or(ProbeCpuDispatch::Unavailable);
    probe_outcome(outcome, profile)
}

fn scoped_cpu_team() -> Result<bool, StageError> {
    const VARIABLE: &str = "MVMC_RS_TENFERRO_TEAM_LIFETIME";
    match std::env::var(VARIABLE) {
        Ok(value) if value == "held" => Ok(false),
        Ok(value) if value == "stage" => Ok(true),
        Ok(_) => Err(StageError::Failed(format!(
            "{VARIABLE} must be held or stage"
        ))),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(StageError::Failed(format!("{VARIABLE}: {error}"))),
    }
}

fn shared(kind: StageBackendKind) -> Result<Shared, StageError> {
    let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, backend)) = registry.iter().find(|(k, _)| *k == kind) {
        return Ok(backend);
    }
    let instance = if kind == StageBackendKind::TenferroCpu {
        let transient = scoped_cpu_team()?;
        StageBackend::tenferro_cpu_with_transient(transient)?
    } else {
        open_stage_backend(kind)?
    };
    if let Some(engine) = &instance.probe_cpu {
        let _ = PROBE_CPU_ENGINE.set(engine.clone());
    }
    let backend: Shared = Box::leak(Box::new(Mutex::new(instance)));
    registry.push((kind, backend));
    Ok(backend)
}

/// Pre-register the process-wide backend of `kind` (tests and programs that supply their own
/// stages). Returns `false`, leaving the existing backend, when `kind` was already opened or
/// registered; `StageBackendKind::COrder` is never shared and cannot be registered.
pub fn register_stage_backend(kind: StageBackendKind, backend: StageBackend<'static>) -> bool {
    if kind == StageBackendKind::COrder {
        return false;
    }
    let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if registry.iter().any(|(k, _)| *k == kind) {
        return false;
    }
    let leaked: Shared = Box::leak(Box::new(Mutex::new(backend)));
    registry.push((kind, leaked));
    true
}

/// True when the selected backend keeps the direct SR step resident on a device (the selected
/// kind is not C order and its SR stages report [`SrStages::resident_direct`]). A backend that
/// cannot be opened is not resident here; the SR step reports that error itself.
pub fn selected_resident_direct() -> bool {
    let kind = selected_stage_backend();
    if kind == StageBackendKind::COrder {
        return false;
    }
    match shared(kind) {
        Ok(m) => m
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sr()
            .resident_direct(),
        Err(_) => false,
    }
}

enum StageStorage {
    Owned(StageBackend<'static>),
    Shared(MutexGuard<'static, StageBackend<'static>>),
}

/// Handle to the selected backend for one SR solve or one stage call.
pub struct StageHandle {
    storage: StageStorage,
    // Keep the acquired team independent of later replacement of the stage object.
    _team: Option<CpuTeamLease>,
}

impl StageHandle {
    /// The backend object.
    pub fn backend(&mut self) -> &mut StageBackend<'static> {
        match &mut self.storage {
            StageStorage::Owned(b) => b,
            StageStorage::Shared(b) => b,
        }
    }

    /// Shorthand for `backend().sr()`.
    pub fn sr(&mut self) -> &mut dyn SrStages {
        self.backend().sr()
    }

    pub(crate) fn is_owned(&self) -> bool {
        matches!(self.storage, StageStorage::Owned(_))
    }
}

/// Acquire the selected backend. An unavailable backend is a hard error.
pub fn acquire() -> StageHandle {
    acquire_kind(selected_stage_backend())
}

/// Acquire a backend of an explicit kind (shared instance for non-C-order kinds).
pub fn acquire_kind(kind: StageBackendKind) -> StageHandle {
    match kind {
        StageBackendKind::COrder => StageHandle {
            storage: StageStorage::Owned(StageBackend::c_order()),
            _team: None,
        },
        kind => match shared(kind) {
            Ok(m) => {
                let backend = m.lock().unwrap_or_else(|e| e.into_inner());
                let team_lease = if let Some(team) = &backend.transient_cpu {
                    team.acquire()
                        .unwrap_or_else(|e| panic!("{SR_BACKEND_VARIABLE}: {e}"));
                    Some(CpuTeamLease(team.clone()))
                } else {
                    None
                };
                StageHandle {
                    storage: StageStorage::Shared(backend),
                    _team: team_lease,
                }
            }
            Err(e) => panic!("{SR_BACKEND_VARIABLE}: {e}"),
        },
    }
}

// The returned guard owns only a team lease, not a stage mutex guard.
pub(crate) fn retain_cpu_team(kind: StageBackendKind) -> Result<Option<CpuTeamLease>, StageError> {
    if kind != StageBackendKind::TenferroCpu {
        return Ok(None);
    }
    let stage = shared(kind)?;
    let stage = stage.lock().unwrap_or_else(|e| e.into_inner());
    let Some(team) = &stage.transient_cpu else {
        return Ok(None);
    };
    team.acquire()?;
    Ok(Some(CpuTeamLease(team.clone())))
}

/// Fresh operand version for the constant-operand cache.
pub fn next_operand_version() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Counters of the shared tenferro CPU backend, `None` if it was never built.
pub fn tenferro_stats() -> Option<TenferroStats> {
    let registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    let (_, backend) = registry
        .iter()
        .find(|(k, _)| *k == StageBackendKind::TenferroCpu)?;
    let stats = backend
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sr()
        .stats();
    stats
}

#[cfg(test)]
mod transient_probe_tests {
    use super::*;

    #[test]
    fn shared_cpu_team_probe_dispatches_fixed_blocks_and_falls_back_when_busy() {
        if std::env::var("MVMC_SHARED_TEAM_TEST_CHILD").as_deref() != Ok("1") {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "stage_backend::transient_probe_tests::shared_cpu_team_probe_dispatches_fixed_blocks_and_falls_back_when_busy",
                ])
                .env("MVMC_SHARED_TEAM_TEST_CHILD", "1")
                .env("MVMC_EXPERIMENT_SHARED_CPU_TEAM", "1")
                .env("MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE", "1")
                .env("MVMC_RS_TENFERRO_TEAM_LIFETIME", "held")
                .env("MVMC_RS_TENFERRO_THREADS", "4")
                .env("MVMC_RS_INNER_THREADS", "4")
                .env("MVMC_RS_INNER_THRESHOLD", "1")
                .spawn()
                .unwrap();
            let start = std::time::Instant::now();
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    return;
                }
                if start.elapsed() >= std::time::Duration::from_secs(30) {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("shared team probe did not complete");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        assert!(matches!(
            try_probe_cpu_blocks(4, |_| panic!("unavailable team cannot execute")),
            ProbeCpuDispatch::Unavailable
        ));
        let mut initial = [0usize; 32];
        let rayon_entries = std::sync::atomic::AtomicUsize::new(0);
        crate::threading::for_each_mut(&mut initial, 1, |i, value| {
            if rayon::current_thread_index().is_some() {
                rayon_entries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            *value = i + 1;
        });
        assert!(rayon_entries.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(initial.iter().enumerate().all(|(i, value)| *value == i + 1));
        shared(StageBackendKind::TenferroCpu).unwrap();
        let mut values = [0usize; 64];
        let before = PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        crate::threading::for_each_mut(&mut values, 1, |i, value| *value = i + 1);
        assert_eq!(
            PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            before + 1
        );
        assert!(values.iter().enumerate().all(|(i, value)| *value == i + 1));
        let participants = Mutex::new(std::collections::HashSet::new());
        assert!(matches!(
            try_probe_cpu_blocks(4, |_| {
                participants
                    .lock()
                    .unwrap()
                    .insert(std::thread::current().id());
                std::thread::sleep(std::time::Duration::from_millis(2));
            }),
            ProbeCpuDispatch::Complete
        ));
        assert!(participants.lock().unwrap().len() >= 2);
        let before_nested = PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        crate::threading::for_each_mut(&mut values, 1, |i, value| {
            let mut nested = [0usize; 4];
            crate::threading::for_each_mut(&mut nested, 1, |j, entry| *entry = i + j);
            assert_eq!(nested, [i, i + 1, i + 2, i + 3]);
            *value = i;
        });
        assert_eq!(
            PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            before_nested + 1
        );
        let before_bounded = PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        crate::threading::install(|| {
            crate::threading::with_qp_region_team(2, usize::MAX, || {
                assert_eq!(spindle::max_num_threads(), 2);
                crate::threading::for_each_mut(&mut values, 1, |i, value| *value = i + 3);
            });
        });
        assert_eq!(
            PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            before_bounded,
            "a bounded saved-sample team must not enter the separate CPU Engine team"
        );
        assert!(values.iter().enumerate().all(|(i, value)| *value == i + 3));
        let engine = PROBE_CPU_ENGINE.get().unwrap();
        let _busy = engine.lock().unwrap();
        assert!(matches!(
            try_probe_cpu_blocks(4, |_| panic!("busy engine must fall back")),
            ProbeCpuDispatch::Busy
        ));
        let before_busy = PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let caller = std::thread::current().id();
        crate::threading::for_each_mut(&mut values, 1, |i, value| {
            assert_eq!(std::thread::current().id(), caller);
            *value = i * 2;
        });
        assert_eq!(
            PROBE_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            before_busy
        );
        assert!(values.iter().enumerate().all(|(i, value)| *value == i * 2));
        let (attempts, successes, busy, unavailable, participants) = probe_cpu_snapshot();
        assert_eq!(attempts, successes + busy + unavailable);
        assert!(successes > 0 && busy >= 2 && unavailable >= 1 && participants >= 2);
    }

    #[test]
    fn invalid_team_lifetime_is_rejected() {
        if std::env::var("MVMC_TEAM_INVALID_CHILD").as_deref() != Ok("1") {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "stage_backend::transient_probe_tests::invalid_team_lifetime_is_rejected",
                ])
                .env("MVMC_TEAM_INVALID_CHILD", "1")
                .env("MVMC_RS_TENFERRO_TEAM_LIFETIME", "invalid")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let error = match shared(StageBackendKind::TenferroCpu) {
            Ok(_) => panic!("invalid team lifetime accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("MVMC_RS_TENFERRO_TEAM_LIFETIME"));
    }

    fn threads(backend: &mut StageBackend<'_>) -> usize {
        let mut engine = backend
            .transient_cpu
            .as_ref()
            .unwrap()
            .engine
            .lock()
            .unwrap();
        tenferro_cpu::with_cpu_engine(&mut ***engine, |cpu| cpu.num_threads()).unwrap()
    }

    #[test]
    fn transient_stage_releases_and_reacquires_team() {
        if std::env::var("MVMC_TRANSIENT_TEST_CHILD").as_deref() != Ok("1") {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "stage_backend::transient_probe_tests::transient_stage_releases_and_reacquires_team",
                ])
                .env("MVMC_TRANSIENT_TEST_CHILD", "1")
                .env("MVMC_RS_TENFERRO_TEAM_LIFETIME", "stage")
                .env("MVMC_RS_TENFERRO_THREADS", "2")
                .spawn()
                .unwrap();
            let start = std::time::Instant::now();
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    break;
                }
                if start.elapsed() >= std::time::Duration::from_secs(30) {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("concurrent team leases did not complete");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            return;
        }
        assert_eq!(
            std::env::var("MVMC_RS_TENFERRO_TEAM_LIFETIME").as_deref(),
            Ok("stage")
        );
        assert_eq!(
            std::env::var("MVMC_RS_TENFERRO_THREADS").as_deref(),
            Ok("2")
        );
        assert!(StageBackend::tenferro_cpu()
            .unwrap()
            .transient_cpu
            .is_none());
        let registered = shared(StageBackendKind::TenferroCpu).unwrap();
        assert_eq!(threads(&mut registered.lock().unwrap()), 1);
        // Reacquisition uses the validated startup settings, not later environment changes.
        std::env::set_var("MVMC_RS_TENFERRO_THREADS", "1");
        let lease = retain_cpu_team(StageBackendKind::TenferroCpu)
            .unwrap()
            .unwrap();
        let second_lease = retain_cpu_team(StageBackendKind::TenferroCpu)
            .unwrap()
            .unwrap();
        assert_eq!(threads(&mut registered.lock().unwrap()), 2);
        {
            let mut handle = acquire_kind(StageBackendKind::TenferroCpu);
            assert_eq!(threads(handle.backend()), 2);
        }
        assert_eq!(threads(&mut registered.lock().unwrap()), 2);
        drop(lease);
        assert_eq!(threads(&mut registered.lock().unwrap()), 2);
        drop(second_lease);
        assert_eq!(threads(&mut registered.lock().unwrap()), 1);
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..8 {
                        let lease = retain_cpu_team(StageBackendKind::TenferroCpu)
                            .unwrap()
                            .unwrap();
                        barrier.wait();
                        let mut handle = acquire_kind(StageBackendKind::TenferroCpu);
                        assert_eq!(threads(handle.backend()), 2);
                        drop(handle);
                        barrier.wait();
                        drop(lease);
                    }
                });
            }
        });
        assert_eq!(threads(&mut registered.lock().unwrap()), 1);
        for _ in 0..2 {
            let mut handle = acquire_kind(StageBackendKind::TenferroCpu);
            assert_eq!(threads(handle.backend()), 2);
            drop(handle);
            assert_eq!(threads(&mut registered.lock().unwrap()), 1);
        }
        let samples = crate::sr_backend::CgSamples {
            real: &[1.0, 2.0, 3.0, 4.0],
            imag: &[],
            components: 2,
            samples: 2,
            version: 42,
        };
        for _ in 0..3 {
            let mut handle = acquire_kind(StageBackendKind::TenferroCpu);
            let mut z = [0.0; 2];
            handle
                .sr()
                .cg_local_product(&samples, &[5.0, 6.0], &mut z)
                .unwrap();
            assert!((z[0] - 134.0).abs() < 1e-12);
            assert!((z[1] - 190.0).abs() < 1e-12);
        }
        let stats = tenferro_stats().unwrap();
        assert_eq!(stats.cg_products, 3);
        assert_eq!(stats.cg_uploads, 1);
        let panic = std::panic::catch_unwind(|| {
            let _handle = acquire_kind(StageBackendKind::TenferroCpu);
            panic!("exercise cleanup");
        });
        assert!(panic.is_err());
        assert_eq!(
            threads(&mut registered.lock().unwrap_or_else(|e| e.into_inner())),
            1
        );
        let mut handle = acquire_kind(StageBackendKind::TenferroCpu);
        assert_eq!(threads(handle.backend()), 2);
        let mut original = std::mem::replace(handle.backend(), StageBackend::c_order());
        drop(handle);
        assert_eq!(threads(&mut original), 1);
    }
}
