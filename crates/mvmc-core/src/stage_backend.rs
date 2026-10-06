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
//! * [`PfaffianStages`]: batched Pfaffian and inverse of skew-symmetric planes (issue #423).
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
//! Production currently routes the SR stages through the selected backend. The sampler and the
//! measurement Pfaffian still use the C-order kernels directly (`calc_m_all_*`); the
//! [`PfaffianStages`] slot is exercised by the harness and the gates and is where #422/#423
//! wiring plugs in.

use std::sync::{Mutex, MutexGuard};

use pfapack::{dsktf2, utu2inv_real, utu2pfa_real, PivotIndex1Based, SqMat};

use crate::sr_backend::{COrderSr, SrStages, TenferroSr, TenferroStats};

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

/// C-order CPU Pfaffian stage: the production PfaPack sequence (`dsktf2`, `utu2pfa_real`,
/// `utu2inv_real`) per plane. This is the oracle of the validation harness.
#[derive(Debug, Default, Clone, Copy)]
pub struct COrderPfaffian;

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
        if x.len() != n * n * planes {
            return Err(StageError::Failed(format!(
                "buffer length {} != n*n*planes = {}",
                x.len(),
                n * n * planes
            )));
        }
        let mut pf = Vec::with_capacity(planes);
        let mut inv = Vec::with_capacity(x.len());
        let mut outcome = Vec::with_capacity(planes);
        for plane in x.chunks(n * n.max(1)).take(planes) {
            let mut a = plane.to_vec();
            let mut pivots = vec![PivotIndex1Based(0); n];
            let result = {
                let mut view = SqMat::new(&mut a, n);
                dsktf2(&mut view, &mut pivots).map(|_| utu2pfa_real(&view, &pivots))
            };
            match result {
                Err(info) => {
                    pf.push(0.0);
                    inv.extend(std::iter::repeat_n(0.0, n * n));
                    outcome.push(PlaneOutcome::ZeroPivot { row: info });
                }
                Ok(value) => {
                    let mut vt = vec![0.0; n.saturating_sub(1)];
                    let mut work = vec![0.0; n * n];
                    {
                        let mut view = SqMat::new(&mut a, n);
                        let mut m_work = SqMat::new(&mut work, n);
                        utu2inv_real(&mut view, &pivots, &mut vt, &mut m_work);
                    }
                    // `a` now holds X^-1 (X * invM = I) for the matrix as given
                    pf.push(value);
                    inv.extend_from_slice(&a);
                    outcome.push(if value.is_finite() {
                        PlaneOutcome::Ok
                    } else {
                        PlaneOutcome::NonFinite
                    });
                }
            }
        }
        Ok(PfInvBatch { pf, inv, outcome })
    }
}

/// A Pfaffian stage that always reports [`StageError::Unsupported`] with the stated reason
/// (tenferro 0.7.1 has no skew-symmetric factorization or Pfaffian).
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

    /// The tenferro 0.7.1 gap.
    pub fn tenferro() -> Self {
        Self::new(
            "none",
            "tenferro 0.7.1 has no skew-symmetric factorization or Pfaffian",
        )
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

/// The single backend object: one implementation of every stage trait.
pub struct StageBackend<'a> {
    label: String,
    sr: Box<dyn SrStages + 'a>,
    pfaffian: Box<dyn PfaffianStages + 'a>,
}

impl<'a> StageBackend<'a> {
    /// Compose a backend labelled `label` from one implementation per stage trait.
    pub fn new(
        label: &str,
        sr: Box<dyn SrStages + 'a>,
        pfaffian: Box<dyn PfaffianStages + 'a>,
    ) -> Self {
        Self {
            label: label.to_string(),
            sr,
            pfaffian,
        }
    }

    /// C-order CPU: BLAS/LAPACK SR stages and the PfaPack Pfaffian (default, parity oracle).
    pub fn c_order() -> StageBackend<'static> {
        StageBackend::new(
            "c-order-cpu",
            Box::new(COrderSr::default()),
            Box::new(COrderPfaffian),
        )
    }

    /// tenferro CPU (`cpu-faer`): SR stages through eager ops; the Pfaffian stage is
    /// unsupported by tenferro 0.7.1.
    pub fn tenferro_cpu() -> Result<StageBackend<'static>, StageError> {
        Ok(StageBackend::new(
            "tenferro-cpu",
            Box::new(TenferroSr::new_cpu()?),
            Box::new(UnsupportedPfaffian::tenferro()),
        ))
    }

    /// Replace the Pfaffian stage (for example by the batched CUDA kernel), relabelled.
    pub fn with_pfaffian(mut self, label: &str, pfaffian: Box<dyn PfaffianStages + 'a>) -> Self {
        self.label = label.to_string();
        self.pfaffian = pfaffian;
        self
    }

    /// Backend label.
    pub fn label(&self) -> String {
        self.label.clone()
    }

    /// Provider description: one entry per stage trait.
    pub fn provider(&self) -> String {
        format!(
            "sr: {}; pfaffian: {}",
            self.sr.provider(),
            self.pfaffian.provider()
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
    /// tenferro eager ops on the CPU backend (`cpu-faer`).
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

fn shared(kind: StageBackendKind) -> Result<Shared, StageError> {
    let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, backend)) = registry.iter().find(|(k, _)| *k == kind) {
        return Ok(backend);
    }
    let backend: Shared = Box::leak(Box::new(Mutex::new(open_stage_backend(kind)?)));
    registry.push((kind, backend));
    Ok(backend)
}

/// Handle to the selected backend for one SR solve or one stage call.
pub enum StageHandle {
    /// C-order backend owned by the caller (cheap, holds scratch vectors).
    Owned(StageBackend<'static>),
    /// Process-wide shared backend, locked while the handle lives.
    Shared(MutexGuard<'static, StageBackend<'static>>),
}

impl StageHandle {
    /// The backend object.
    pub fn backend(&mut self) -> &mut StageBackend<'static> {
        match self {
            Self::Owned(b) => b,
            Self::Shared(b) => b,
        }
    }

    /// Shorthand for `backend().sr()`.
    pub fn sr(&mut self) -> &mut dyn SrStages {
        self.backend().sr()
    }
}

/// Acquire the selected backend. An unavailable backend is a hard error.
pub fn acquire() -> StageHandle {
    acquire_kind(selected_stage_backend())
}

/// Acquire a backend of an explicit kind (shared instance for non-C-order kinds).
pub fn acquire_kind(kind: StageBackendKind) -> StageHandle {
    match kind {
        StageBackendKind::COrder => StageHandle::Owned(StageBackend::c_order()),
        kind => match shared(kind) {
            Ok(m) => StageHandle::Shared(m.lock().unwrap_or_else(|e| e.into_inner())),
            Err(e) => panic!("{SR_BACKEND_VARIABLE}: {e}"),
        },
    }
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
