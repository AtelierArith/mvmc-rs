//! Device-selection abstraction for tensor-shaped (tenferro) work (issue #420).
//!
//! The tenferro CPU backend is always available and remains the numerical reference
//! (`docs/design/gpu-readiness.md`). A CUDA device is selectable only when the `gpu-cuda`
//! feature is enabled **and** a provider has been registered. The provider lives in the
//! standalone workspace `gpu/mvmc-gpu-cuda`, which owns the `tenferro-gpu` dependency, its
//! `lru` patch and its own lock file, so default builds and `Cargo.lock` never see the CUDA
//! dependency tree (see `docs/design/gpu-readiness.md`, section "gpu-cuda build").
//!
//! Selecting an unavailable backend is always an error: there is no silent CPU fallback.

use std::fmt;

/// Compute backend requested for tensor-shaped work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// tenferro CPU backend (always available; the parity reference).
    Cpu,
    /// tenferro CUDA backend on the given device ordinal (feature `gpu-cuda`).
    Cuda(usize),
}

/// Errors from backend selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// This build was compiled without the `gpu-cuda` feature.
    CudaNotCompiled,
    /// `gpu-cuda` is enabled but no CUDA provider was registered.
    NoCudaProvider,
    /// The provider reported no usable device or failed.
    Device(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaNotCompiled => write!(f, "built without the `gpu-cuda` feature"),
            Self::NoCudaProvider => write!(f, "no CUDA provider registered"),
            Self::Device(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Device and library versions recorded with every accelerated result.
///
/// Fields that do not apply (CPU) or could not be queried are `None`; the rendering marks
/// them `unavailable` explicitly instead of omitting them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceReport {
    /// Backend label (`cpu` or `cuda:<ordinal>`).
    pub backend: String,
    /// Device name.
    pub device: Option<String>,
    /// Compute capability, `major.minor`.
    pub compute_capability: Option<String>,
    /// Total device memory in bytes.
    pub total_memory_bytes: Option<u64>,
    /// NVIDIA kernel driver version (for example `580.178.04`).
    pub driver_version: Option<String>,
    /// Highest CUDA version supported by the driver (`cuDriverGetVersion`).
    pub cuda_driver_api: Option<String>,
    /// NVRTC version (CUDA toolkit used for runtime compilation).
    pub nvrtc_version: Option<String>,
    /// cuBLAS version.
    pub cublas_version: Option<String>,
    /// cuSOLVER version.
    pub cusolver_version: Option<String>,
    /// tenferro crate version used for the run.
    pub tenferro_version: Option<String>,
}

impl DeviceReport {
    /// Render as stable `key=value` lines for result files.
    pub fn render(&self) -> String {
        fn text(value: &Option<String>) -> &str {
            value.as_deref().unwrap_or("unavailable")
        }
        let memory = self
            .total_memory_bytes
            .map_or_else(|| "unavailable".to_string(), |bytes| bytes.to_string());
        format!(
            "backend={}\ndevice={}\ncompute_capability={}\ntotal_memory_bytes={}\n\
             driver_version={}\ncuda_driver_api={}\nnvrtc_version={}\ncublas_version={}\n\
             cusolver_version={}\ntenferro_version={}\n",
            self.backend,
            text(&self.device),
            text(&self.compute_capability),
            memory,
            text(&self.driver_version),
            text(&self.cuda_driver_api),
            text(&self.nvrtc_version),
            text(&self.cublas_version),
            text(&self.cusolver_version),
            text(&self.tenferro_version),
        )
    }
}

/// Report for the always-available tenferro CPU backend.
pub fn cpu_report() -> DeviceReport {
    DeviceReport {
        backend: "cpu".to_string(),
        device: Some("tenferro-cpu (cpu-faer)".to_string()),
        tenferro_version: Some("0.7.1".to_string()),
        ..DeviceReport::default()
    }
}

/// Source of CUDA devices, implemented by the standalone `gpu/mvmc-gpu-cuda` crate.
#[cfg(feature = "gpu-cuda")]
pub trait CudaProvider: Send + Sync {
    /// Number of usable CUDA devices (0 when none).
    fn device_count(&self) -> Result<usize, String>;
    /// Device and library versions for one device ordinal.
    fn report(&self, ordinal: usize) -> Result<DeviceReport, String>;
}

#[cfg(feature = "gpu-cuda")]
static CUDA_PROVIDER: std::sync::OnceLock<Box<dyn CudaProvider>> = std::sync::OnceLock::new();

/// Register the process-wide CUDA provider. Returns `false` if one is already registered.
#[cfg(feature = "gpu-cuda")]
pub fn register_cuda_provider(provider: Box<dyn CudaProvider>) -> bool {
    CUDA_PROVIDER.set(provider).is_ok()
}

#[cfg(feature = "gpu-cuda")]
fn cuda_provider() -> Result<&'static dyn CudaProvider, BackendError> {
    CUDA_PROVIDER
        .get()
        .map(|p| p.as_ref())
        .ok_or(BackendError::NoCudaProvider)
}

/// Number of CUDA devices. `Err` when CUDA support is not compiled in or not registered.
#[cfg(not(feature = "gpu-cuda"))]
pub fn cuda_device_count() -> Result<usize, BackendError> {
    Err(BackendError::CudaNotCompiled)
}

/// Number of CUDA devices. `Err` when CUDA support is not compiled in or not registered.
#[cfg(feature = "gpu-cuda")]
pub fn cuda_device_count() -> Result<usize, BackendError> {
    cuda_provider()?
        .device_count()
        .map_err(BackendError::Device)
}

/// Report device and library versions for the requested backend.
///
/// Never falls back to the CPU: a CUDA request that cannot be satisfied is an error.
pub fn device_report(kind: BackendKind) -> Result<DeviceReport, BackendError> {
    match kind {
        BackendKind::Cpu => Ok(cpu_report()),
        BackendKind::Cuda(ordinal) => cuda_report(ordinal),
    }
}

#[cfg(not(feature = "gpu-cuda"))]
fn cuda_report(_ordinal: usize) -> Result<DeviceReport, BackendError> {
    Err(BackendError::CudaNotCompiled)
}

#[cfg(feature = "gpu-cuda")]
fn cuda_report(ordinal: usize) -> Result<DeviceReport, BackendError> {
    let count = cuda_device_count()?;
    if ordinal >= count {
        return Err(BackendError::Device(format!(
            "CUDA device {ordinal} requested but {count} device(s) found"
        )));
    }
    cuda_provider()?
        .report(ordinal)
        .map_err(BackendError::Device)
}

/// Environment variable that requests the optional CUDA gate.
pub const CUDA_GATE_VARIABLE: &str = "MVMC_RS_CUDA_GATE";

/// Decision of the optional CUDA gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaGateDecision {
    /// A device is present: run the gate.
    Run,
    /// Not requested explicitly and no device: report "skipped, no device" (not a pass).
    SkippedNoDevice(String),
    /// Requested explicitly (`MVMC_RS_CUDA_GATE=1`) but no device: hard failure.
    FailNoDevice(String),
}

/// Decide the CUDA gate from the selector value and the device count result.
///
/// `requested` is the value of [`CUDA_GATE_VARIABLE`]. Only `1` (or `require`) requests the
/// gate; any other value, including unset, allows an explicit "skipped, no device".
pub fn cuda_gate_decision(
    requested: Option<&str>,
    devices: Result<usize, BackendError>,
) -> CudaGateDecision {
    let required = matches!(requested.map(str::trim), Some("1") | Some("require"));
    match devices {
        Ok(count) if count > 0 => CudaGateDecision::Run,
        Ok(_) if required => CudaGateDecision::FailNoDevice("no CUDA device found".to_string()),
        Ok(_) => CudaGateDecision::SkippedNoDevice("no CUDA device found".to_string()),
        Err(e) if required => CudaGateDecision::FailNoDevice(e.to_string()),
        Err(e) => CudaGateDecision::SkippedNoDevice(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_always_selectable() {
        let report = device_report(BackendKind::Cpu).unwrap();
        assert_eq!(report.backend, "cpu");
        assert!(report.render().contains("cublas_version=unavailable"));
    }

    #[cfg(not(feature = "gpu-cuda"))]
    #[test]
    fn cuda_is_rejected_without_feature_and_never_falls_back() {
        assert_eq!(
            device_report(BackendKind::Cuda(0)),
            Err(BackendError::CudaNotCompiled)
        );
    }

    #[test]
    fn gate_requested_without_device_fails_closed() {
        let none = || Err(BackendError::CudaNotCompiled);
        assert!(matches!(
            cuda_gate_decision(Some("1"), none()),
            CudaGateDecision::FailNoDevice(_)
        ));
        assert!(matches!(
            cuda_gate_decision(Some("1"), Ok(0)),
            CudaGateDecision::FailNoDevice(_)
        ));
    }

    #[test]
    fn gate_not_requested_without_device_is_explicit_skip() {
        assert!(matches!(
            cuda_gate_decision(None, Ok(0)),
            CudaGateDecision::SkippedNoDevice(_)
        ));
        assert!(matches!(
            cuda_gate_decision(Some("auto"), Err(BackendError::NoCudaProvider)),
            CudaGateDecision::SkippedNoDevice(_)
        ));
    }

    #[test]
    fn gate_runs_when_a_device_exists() {
        assert_eq!(cuda_gate_decision(None, Ok(2)), CudaGateDecision::Run);
        assert_eq!(cuda_gate_decision(Some("1"), Ok(1)), CudaGateDecision::Run);
    }
}
