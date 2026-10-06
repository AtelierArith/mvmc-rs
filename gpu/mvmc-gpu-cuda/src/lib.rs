//! tenferro CUDA provider for `mvmc-core::backend` (issue #420).
//!
//! * [`install`] registers the provider (device discovery and version report).
//! * [`versions`] queries driver / CUDA / NVRTC / cuBLAS / cuSOLVER versions by `dlopen`.
//! * [`bench`] is the FP64/C64 `dot_general` and Cholesky micro-benchmark used by the gate.
//!
//! The CPU C-order path stays the parity reference; nothing here changes RNG draw order.

pub mod bench;
pub mod device_sampler;
pub mod pfaffian;
pub mod sr_bench;
pub mod sr_device;
pub mod sr_problem;
pub mod stages;
pub mod transfer;
pub mod transfer_bench;
pub mod versions;

use mvmc_core::backend::{register_cuda_provider, CudaProvider, DeviceReport};
use tenferro_gpu::cuda::cuda_devices;

/// Version string of the tenferro crates this provider is built against.
pub const TENFERRO_VERSION: &str = "0.7.1";

struct Provider;

impl CudaProvider for Provider {
    fn device_count(&self) -> Result<usize, String> {
        // `cuda_devices` loads libcuda lazily and returns an error (or panics inside
        // cudarc when the library is absent); both mean "no usable device".
        match std::panic::catch_unwind(cuda_devices) {
            Ok(Ok(devices)) => Ok(devices.len()),
            Ok(Err(e)) => Err(format!("CUDA device discovery failed: {e}")),
            Err(_) => Err("CUDA driver library (libcuda) not loadable".to_string()),
        }
    }

    fn open_stage_backend(
        &self,
        ordinal: usize,
    ) -> Result<mvmc_core::stage_backend::StageBackend<'static>, String> {
        // Production SR (`MVMC_RS_SR_BACKEND=cuda[:N]`) runs the device-resident direct and CG
        // steps (issue #452); the per-stage backend stays available as `stages::cuda_stage_backend`.
        stages::cuda_resident_stage_backend(ordinal)
    }

    fn report(&self, ordinal: usize) -> Result<DeviceReport, String> {
        let devices = cuda_devices().map_err(|e| e.to_string())?;
        let device = devices
            .get(ordinal)
            .ok_or_else(|| format!("no CUDA device with ordinal {ordinal}"))?;
        let capability = device.compute_capability();
        let v = versions::query();
        Ok(DeviceReport {
            backend: format!("cuda:{ordinal}"),
            device: Some(device.name().to_string()),
            compute_capability: Some(format!("{capability:?}")),
            total_memory_bytes: Some(device.total_memory_bytes()),
            driver_version: v.driver_version,
            cuda_driver_api: v.cuda_driver_api,
            nvrtc_version: v.nvrtc,
            cublas_version: v.cublas,
            cusolver_version: v.cusolver,
            tenferro_version: Some(TENFERRO_VERSION.to_string()),
        })
    }
}

/// Register the CUDA provider with `mvmc-core`. Idempotent.
pub fn install() {
    let _ = register_cuda_provider(Box::new(Provider));
}
