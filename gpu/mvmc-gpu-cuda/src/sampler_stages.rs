//! Production [`SamplerStages`] on the CUDA device-resident lock-step sampler (issue #517).
//!
//! The service is the standalone [`CudaSamplerService`] of issue #434 with the pinned
//! asynchronous transfer path. `sample_real` runs the single production chain `(data, state,
//! rng)` through [`run_lockstep_real_one`]: the host driver, the SFMT draws and the sampled
//! configurations are the C path's, only the Pfaffian stages live on the device.
//!
//! The device chain is the serial production chain. A multi-process reducer (MPI QP split or a
//! collective sampling sum) has no device implementation and is a hard [`StageError::Unsupported`],
//! never a silent C-order execution. Complex and FSZ modes are rejected by the runner before this
//! stage is reached (`sampler_backend::check_supported_mode`).

use mvmc_core::device_sampler::{run_lockstep_real_one, LockstepOptions};
use mvmc_core::reducer::Reducer;
use mvmc_core::sampler_backend::{SamplerStages, SamplerTiming};
use mvmc_core::sampling::driver::SampleStats;
use mvmc_core::stage_backend::StageError;
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::device_sampler::{CudaSamplerService, PinnedTransfer};

/// [`SamplerStages`] implementation on one CUDA device.
pub struct CudaSamplerStages {
    service: CudaSamplerService<PinnedTransfer>,
}

impl CudaSamplerStages {
    /// Compile the kernels and open the service on `device`.
    pub fn new(device: usize) -> Result<Self, String> {
        Ok(Self {
            service: CudaSamplerService::new_pinned(device)?,
        })
    }

    /// The underlying service (transfer path label, resident-table downloads for the gates).
    pub fn service(&self) -> &CudaSamplerService<PinnedTransfer> {
        &self.service
    }

    /// Mutable service access (diagnostics and gate downloads).
    pub fn service_mut(&mut self) -> &mut CudaSamplerService<PinnedTransfer> {
        &mut self.service
    }
}

impl SamplerStages for CudaSamplerStages {
    fn label(&self) -> String {
        format!("cuda-sampler:{}", self.service.transfer_name())
    }

    fn provider(&self) -> String {
        "device-resident lock-step sampler (issue #434) with pinned-async transfers".to_string()
    }

    fn sample_real(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        _timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        // The device chain is the serial production chain: MPI QP splits and collective sampling
        // sums have no device implementation. Hard error, never a CPU fallback.
        mvmc_core::sampler_backend::check_serial_reducer(reducer)?;
        let (run, _stats) = run_lockstep_real_one(
            data,
            state,
            rng,
            reducer,
            &mut self.service,
            LockstepOptions::default(),
        )
        .map_err(StageError::Failed)?;
        run.stats.map_err(StageError::Failed)
    }
}
