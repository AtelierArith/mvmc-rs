//! Production sampler stage seam (issue #517 milestone 1).
//!
//! The production real, non-FSZ normal-mode sampler (`VMCMakeSample`) of the VMC optimization
//! and PhysCal loops is one swappable stage: [`SamplerStages::sample_real`]. The C-order
//! implementation is the unchanged inline driver
//! ([`vmc_make_sample_real_with_reducer_timed`](crate::sampling::driver::vmc_make_sample_real_with_reducer_timed)),
//! byte for byte the path production used before. The CUDA implementation (provided by
//! `gpu/mvmc-gpu-cuda`) drives the device-resident lock-step sampler of
//! [`crate::device_sampler`] on a `CudaSamplerService` with one production chain.
//!
//! # Selection
//!
//! `MVMC_RS_SAMPLER_BACKEND=c-order|cuda[:N]` (default `c-order`). An invalid value or an
//! unavailable device is an error, never a fallback. The CUDA sampler supports only the real,
//! non-FSZ normal mode: complex and FSZ runs selected onto it are a hard error
//! ([`check_supported_mode`]), not a silent C-order execution. The same applies to a
//! multi-rank reducer ([`SamplerStages::sample_real`] implementations reject it).
//!
//! # RNG contract
//!
//! The stage runs the unchanged host driver around the device Pfaffian operations, so the SFMT
//! draw order and count and the sampled configurations are the C path's. Production runs exactly
//! one chain seeded like the C path; multi-walker runs (`device_sampler`) keep the
//! `RndSeed + group` contract of the `#450` runner.

use std::sync::{Mutex, MutexGuard};

use mvmc_expert_parsers::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::c_timer::CTimer;
use crate::reducer::Reducer;
use crate::sampling::driver::{vmc_make_sample_real_with_reducer_timed, SampleStats};
use crate::stage_backend::StageError;
use crate::state::VmcOptimizationState;

/// Timing sink of the routed sampler stage.
///
/// This is the same `enabled`/`diagnostics`/`merge_elapsed` bridge the measurement stage uses:
/// the C-order implementation runs the driver against a fresh internal [`CTimer`] and merges its
/// per-section counters, so routing does not change the `zvo_CalcTimer.dat` report.
pub use crate::stage_backend::MeasurementTiming as SamplerTiming;

/// The production real, non-FSZ normal-mode sampler of one chain (`VMCMakeSample`).
///
/// A backend that cannot run the stage returns [`StageError::Unsupported`]; the runner reports
/// that as a hard error and never falls back to the inline C-order call.
pub trait SamplerStages: Send {
    /// Short label, for example `c-order` or `cuda:0`.
    fn label(&self) -> String;
    /// Provider description recorded in the benchmark metadata.
    fn provider(&self) -> String;

    /// Begin a complete public optimization or PhysCal run. Shared stages keep their
    /// compiled service, but run-owned random streams must restart from the new run's seed.
    /// Stateless stages and caller-owned SFMT sources need no reset.
    fn begin_run(&mut self) {}

    /// Run one sample series on `(data, state, rng)`, exactly like the inline
    /// [`vmc_make_sample_real_with_reducer_timed`](crate::sampling::driver::vmc_make_sample_real_with_reducer_timed).
    ///
    /// `reducer` is the chain's reducer: a CUDA backend must reject anything but a single-process
    /// chain before consuming RNG.
    fn sample_real(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError>;

    /// Whether this stage batches every walker of one run into a single device call through
    /// [`Self::sample_real_group`] and the reducer rendezvous (`rank`/`world_size`/`barrier`).
    /// The single-chain lock-step and C-order stages are not group-capable.
    fn group_capable(&self) -> bool {
        false
    }

    /// Batched sampling entry point: every walker of the run calls this exactly once per sample
    /// series and rendezvous through `reducer`. The walker index is `reducer.rank()` and the
    /// walker count is `reducer.world_size()`. The default runs the single-walker
    /// [`Self::sample_real`], so a stage that does not override it is unchanged.
    fn sample_real_group(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        self.sample_real(data, state, rng, timing, reducer)
    }
}

/// C-order CPU sampler: the unchanged inline driver of the production path (the parity oracle).
#[derive(Debug, Default, Clone, Copy)]
pub struct COrderSampler;

impl SamplerStages for COrderSampler {
    fn label(&self) -> String {
        "c-order".to_string()
    }

    fn provider(&self) -> String {
        "vmc_make_sample_real_with_reducer_timed (C order)".to_string()
    }

    fn sample_real(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        if timing.enabled() {
            let mut timer = CTimer::<true>::new();
            timer.diagnostics = timing.diagnostics();
            let result =
                vmc_make_sample_real_with_reducer_timed(data, state, rng, &mut timer, reducer);
            timing.merge_elapsed(&timer.elapsed_ns);
            result.map_err(|error| StageError::Failed(error.to_string()))
        } else {
            let mut timer = CTimer::<false>::new();
            timer.diagnostics = timing.diagnostics();
            vmc_make_sample_real_with_reducer_timed(data, state, rng, &mut timer, reducer)
                .map_err(|error| StageError::Failed(error.to_string()))
        }
    }
}

/// A sampler stage that always reports [`StageError::Unsupported`] with the stated reason.
#[derive(Debug, Clone)]
pub struct UnsupportedSampler {
    label: String,
    reason: String,
}

impl UnsupportedSampler {
    /// Stage named `label` that is unsupported because of `reason`.
    pub fn new(label: &str, reason: &str) -> Self {
        Self {
            label: label.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl SamplerStages for UnsupportedSampler {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn provider(&self) -> String {
        "unsupported".to_string()
    }

    fn sample_real(
        &mut self,
        _data: &ExpertModeData,
        _state: &mut VmcOptimizationState,
        _rng: &mut Sfmt19937Rng,
        _timing: &mut dyn SamplerTiming,
        _reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        Err(StageError::Unsupported(self.reason.clone()))
    }
}

// ---------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------

/// Selectable production sampler backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerBackendKind {
    /// C-order CPU driver (default, parity oracle).
    COrder,
    /// Device-resident lock-step sampler on the given CUDA device ordinal.
    Cuda(usize),
    /// Device-resident interval (group) sampler on the given CUDA device ordinal.
    CudaInterval(usize),
}

impl SamplerBackendKind {
    /// Group-capable backends batch every walker into one device call and rendezvous through
    /// the reducer, so they are acquired per walker, never as one run-long shared instance.
    pub fn is_group(&self) -> bool {
        matches!(self, Self::CudaInterval(_))
    }
}

/// Environment variable selecting the production sampler backend (`c-order`, `cuda[:N]`).
pub const SAMPLER_BACKEND_VARIABLE: &str = "MVMC_RS_SAMPLER_BACKEND";

/// Parse a selector value.
pub fn parse_sampler_backend(value: &str) -> Result<SamplerBackendKind, String> {
    let v = value.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "c" | "c-order" | "corder" | "default" => Ok(SamplerBackendKind::COrder),
        "cuda" => Ok(SamplerBackendKind::Cuda(0)),
        "cuda-interval" => Ok(SamplerBackendKind::CudaInterval(0)),
        other => match other.strip_prefix("cuda-interval:").map(str::parse::<usize>) {
            Some(Ok(ordinal)) => Ok(SamplerBackendKind::CudaInterval(ordinal)),
            _ => match other.strip_prefix("cuda:").map(str::parse::<usize>) {
                Some(Ok(ordinal)) => Ok(SamplerBackendKind::Cuda(ordinal)),
                _ => Err(format!(
                    "{SAMPLER_BACKEND_VARIABLE}={other:?} is not one of c-order, cuda[:N], cuda-interval[:N]"
                )),
            },
        },
    }
}

static OVERRIDE: Mutex<Option<SamplerBackendKind>> = Mutex::new(None);

/// Process-wide override of the sampler choice for tests (`None` restores env/default).
pub fn set_sampler_backend_override(kind: Option<SamplerBackendKind>) {
    *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = kind;
}

/// The sampler chosen by the override, then the environment, then the default (C order).
///
/// An invalid environment value is an error, never a silent fallback.
pub fn try_selected_sampler_backend() -> Result<SamplerBackendKind, String> {
    if let Some(kind) = *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) {
        return Ok(kind);
    }
    match std::env::var(SAMPLER_BACKEND_VARIABLE) {
        Ok(v) => parse_sampler_backend(&v),
        Err(std::env::VarError::NotPresent) => Ok(SamplerBackendKind::COrder),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{SAMPLER_BACKEND_VARIABLE} is not valid UTF-8"))
        }
    }
}

/// Like [`try_selected_sampler_backend`], but panics on an invalid value (library callers that
/// reached production code without the CLI preflight; the CLI uses
/// [`validate_selected_sampler_backend`] before any IO).
pub fn selected_sampler_backend() -> SamplerBackendKind {
    try_selected_sampler_backend().unwrap_or_else(|e| panic!("{e}"))
}

/// The CUDA sampler has no complex or FSZ implementation: selecting it for such a mode is a hard
/// error, never a silent C-order execution.
pub fn check_supported_mode(
    kind: SamplerBackendKind,
    all_complex: bool,
    use_fsz: bool,
) -> Result<(), String> {
    if kind != SamplerBackendKind::COrder && (all_complex || use_fsz) {
        return Err(format!(
            "{SAMPLER_BACKEND_VARIABLE}: the {} sampler supports only the real, non-FSZ normal mode",
            match kind {
                SamplerBackendKind::COrder => "C-order",
                SamplerBackendKind::Cuda(_) | SamplerBackendKind::CudaInterval(_) => "CUDA",
            }
        ));
    }
    Ok(())
}

/// The CUDA sampler contract: one serial chain.
///
/// MPI QP splits and collective sampling sums have no device implementation. A multi-process
/// reducer is a hard [`StageError::Unsupported`], never a silent C-order execution.
pub fn check_serial_reducer(reducer: &dyn Reducer) -> Result<(), StageError> {
    if reducer.world_size() != 1 || reducer.reduction_size() != 1 {
        return Err(StageError::Unsupported(
            "the CUDA sampler supports a single-process chain (world/reduction size 1)".to_string(),
        ));
    }
    Ok(())
}

/// Reducer contract of the selected sampler.
///
/// Both CUDA samplers reject a multi-process QP split (`reduction_size != 1`). The lock-step
/// sampler is one serial chain (`world_size == 1`); the interval sampler is group-capable and
/// expects every walker of the run (`world_size == W`) to participate.
pub fn check_sampler_reducer(
    kind: SamplerBackendKind,
    reducer: &dyn Reducer,
) -> Result<(), StageError> {
    if reducer.reduction_size() != 1 {
        return Err(StageError::Unsupported(
            "the CUDA sampler supports a single-process chain (reduction size 1)".to_string(),
        ));
    }
    if !kind.is_group() && reducer.world_size() != 1 {
        return Err(StageError::Unsupported(
            "the CUDA lock-step sampler supports a single-process chain (world size 1)".to_string(),
        ));
    }
    Ok(())
}

/// Startup preflight: resolve the selected sampler and, for a non-default kind, open (and cache)
/// it, so that an invalid selector or an unavailable device is reported before any IO instead of
/// a panic in the first sampling step. The default C-order sampler needs no opening.
///
/// # Errors
///
/// A message that names [`SAMPLER_BACKEND_VARIABLE`].
pub fn validate_selected_sampler_backend() -> Result<SamplerBackendKind, String> {
    let kind = try_selected_sampler_backend()?;
    preflight_sampler_backend(kind, SAMPLER_BACKEND_VARIABLE)?;
    Ok(kind)
}

/// Open (and cache) a non-default sampler `kind` selected through `variable`, reporting every
/// failure, including a provider panic, as an error message that names the variable.
///
/// # Errors
///
/// A message that names `variable`.
pub fn preflight_sampler_backend(kind: SamplerBackendKind, variable: &str) -> Result<(), String> {
    if kind != SamplerBackendKind::COrder {
        // A provider may panic while loading a driver library (cudarc panics when `libcudart`
        // is missing); report that as an error too, with the panic message and without the
        // default hook's output.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let opened = std::panic::catch_unwind(|| {
            if kind.is_group() {
                open_sampler_backend(kind).map(|_| ())
            } else {
                shared(kind).map(|_| ())
            }
        });
        std::panic::set_hook(previous);
        match opened {
            Ok(result) => result.map_err(|e| format!("{variable}: {e}"))?,
            Err(payload) => {
                let text = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "sampler initialization panicked".to_string());
                return Err(format!("{variable}: sampler unavailable: {text}"));
            }
        }
    }
    Ok(())
}

/// Construct a fresh sampler of the given kind.
///
/// CUDA is an error unless the build has the `gpu-cuda` feature and a provider is registered
/// (`crate::backend::register_cuda_provider`): there is no CPU fallback.
pub fn open_sampler_backend(
    kind: SamplerBackendKind,
) -> Result<Box<dyn SamplerStages>, StageError> {
    match kind {
        SamplerBackendKind::COrder => Ok(Box::new(COrderSampler)),
        SamplerBackendKind::Cuda(ordinal) => crate::backend::open_cuda_sampler_stages(ordinal)
            .map_err(|e| StageError::Unsupported(e.to_string())),
        SamplerBackendKind::CudaInterval(ordinal) => {
            crate::backend::open_cuda_interval_sampler_stages(ordinal)
                .map_err(|e| StageError::Unsupported(e.to_string()))
        }
    }
}

type Shared = &'static Mutex<Box<dyn SamplerStages>>;

/// Samplers other than C order are built once per process and shared (a CUDA device context and
/// its compiled kernels are expensive; they must survive every sample series of a run).
static SHARED: Mutex<Vec<(SamplerBackendKind, Shared)>> = Mutex::new(Vec::new());

fn shared(kind: SamplerBackendKind) -> Result<Shared, StageError> {
    let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, sampler)) = registry.iter().find(|(k, _)| *k == kind) {
        return Ok(sampler);
    }
    let sampler: Shared = Box::leak(Box::new(Mutex::new(open_sampler_backend(kind)?)));
    registry.push((kind, sampler));
    Ok(sampler)
}

/// Pre-register the process-wide sampler of `kind` (tests and programs that supply their own
/// stages). Returns `false`, leaving the existing sampler, when `kind` was already opened or
/// registered; `SamplerBackendKind::COrder` is never shared and cannot be registered.
pub fn register_sampler_backend(kind: SamplerBackendKind, sampler: Box<dyn SamplerStages>) -> bool {
    if kind == SamplerBackendKind::COrder {
        return false;
    }
    let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if registry.iter().any(|(k, _)| *k == kind) {
        return false;
    }
    let leaked: Shared = Box::leak(Box::new(Mutex::new(sampler)));
    registry.push((kind, leaked));
    true
}

/// Handle to the selected sampler for one run.
pub enum SamplerHandle {
    /// C-order sampler owned by the caller (cheap, holds no state).
    Owned(Box<dyn SamplerStages>),
    /// Process-wide shared sampler, locked while the handle lives.
    Shared(MutexGuard<'static, Box<dyn SamplerStages>>),
}

impl SamplerHandle {
    /// The sampler object.
    pub fn sampler(&mut self) -> &mut dyn SamplerStages {
        match self {
            Self::Owned(s) => &mut **s,
            Self::Shared(s) => &mut ***s,
        }
    }
}

/// Acquire a sampler for one complete run (shared instance for non-C-order kinds).
/// Calls [`SamplerStages::begin_run`] once; retain the handle across all sample series.
///
/// Panics when a non-C-order sampler cannot be opened, like the stage backend's `acquire`; the
/// CLI preflight ([`validate_selected_sampler_backend`]) reports that as a usage error instead.
pub fn acquire_sampler_kind(kind: SamplerBackendKind) -> SamplerHandle {
    let mut handle = match kind {
        SamplerBackendKind::COrder => {
            SamplerHandle::Owned(open_sampler_backend(kind).expect("c-order"))
        }
        kind if kind.is_group() => SamplerHandle::Owned(
            open_sampler_backend(kind)
                .unwrap_or_else(|e| panic!("{SAMPLER_BACKEND_VARIABLE}: {e}")),
        ),
        kind => match shared(kind) {
            Ok(m) => SamplerHandle::Shared(m.lock().unwrap_or_else(|e| e.into_inner())),
            Err(e) => panic!("{SAMPLER_BACKEND_VARIABLE}: {e}"),
        },
    };
    handle.sampler().begin_run();
    handle
}
