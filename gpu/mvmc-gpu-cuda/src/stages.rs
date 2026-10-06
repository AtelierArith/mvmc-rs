//! The unified stage backend (`mvmc_core::stage_backend::StageBackend`, issue #437) on a
//! tenferro `EagerRuntime`, CPU or CUDA.
//!
//! * SR stages (`S`/`g`, Gram, Cholesky, CG product) run through
//!   `mvmc_core::sr_backend::TenferroSr` on the runtime, the same object the production SR
//!   path selects with `MVMC_RS_SR_BACKEND=tenferro|cuda`. Upload and download are part of
//!   every call (`transfers_included = true` in the benchmark metadata).
//! * The Pfaffian stage is the batched CUDA kernel ([`crate::pfaffian::PersistentCudaEngine`] through
//!   `mvmc_gpu::stages::BatchedStages`) on CUDA, and `Unsupported` on the tenferro CPU eager
//!   runtime (tenferro 0.7.1 has no skew-symmetric factorization or Pfaffian). No CPU
//!   fallback.

use mvmc_core::sr_backend::{
    CgSamples, CgStepInput, CgStepResult, DirectStepInput, Placement, SrAssembleInput, SrStages,
    TenferroSr,
};
use mvmc_core::stage_backend::{StageBackend, StageError, UnsupportedPfaffian};
use mvmc_gpu::stages::BatchedStages;
use mvmc_gpu::Backend;

use crate::pfaffian::PersistentCudaEngine;
use crate::sr_device::{DeviceSr, SrDeviceError};

/// tenferro CPU eager runtime: SR stages through eager ops, Pfaffian unsupported.
pub fn cpu_eager_stage_backend() -> Result<StageBackend<'static>, String> {
    Ok(StageBackend::new(
        "tenferro-eager-cpu",
        Box::new(TenferroSr::with_runtime(
            crate::bench::cpu_runtime()?,
            Placement::Host,
            "tenferro eager cpu".to_string(),
        )),
        Box::new(UnsupportedPfaffian::tenferro()),
    ))
}

/// CUDA stage backend on `ordinal`: tenferro SR stages on the device plus the batched CUDA
/// Pfaffian kernel.
pub fn cuda_stage_backend(ordinal: usize) -> Result<StageBackend<'static>, String> {
    let sr = TenferroSr::with_runtime(
        crate::bench::cuda_runtime(ordinal)?,
        Placement::Device,
        format!("tenferro cuda:{ordinal}"),
    );
    // One persistent session per opened backend (the compiled module lives on a worker
    // thread); leaking the engine gives the `'static` borrow `Backend::Engine` needs for the
    // shared production instance (opened once per process).
    let engine: &'static PersistentCudaEngine =
        Box::leak(Box::new(PersistentCudaEngine::new(ordinal)));
    let pfaffian = BatchedStages::new(Backend::Engine(engine), &format!("cuda:{ordinal}"));
    Ok(StageBackend::new(
        &format!("cuda:{ordinal}"),
        Box::new(sr),
        Box::new(pfaffian),
    ))
}

/// SR stages with a device-resident fused step (issue #447): every per-stage method delegates to
/// the tenferro CUDA stages (unchanged), while [`SrStages::direct_step`] and
/// [`SrStages::cg_step`] run on the resident [`DeviceSr`] pipeline (cuBLAS/cuSOLVER, no host
/// round trips between stages). Opt-in: [`cuda_resident_stage_backend`]; the default
/// [`cuda_stage_backend`] and the C-order path are unchanged.
pub struct ResidentCudaSr {
    inner: TenferroSr,
    device: usize,
    dev: Option<DeviceSr>,
}

impl ResidentCudaSr {
    /// Resident SR stages on `ordinal` (the device pipeline is created on first use).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        Ok(Self {
            inner: TenferroSr::with_runtime(
                crate::bench::cuda_runtime(ordinal)?,
                Placement::Device,
                format!("tenferro cuda:{ordinal}"),
            ),
            device: ordinal,
            dev: None,
        })
    }

    fn pipeline(&mut self) -> Result<&mut DeviceSr, StageError> {
        if self.dev.is_none() {
            self.dev = Some(
                DeviceSr::new(self.device).map_err(|e| StageError::Unsupported(e.to_string()))?,
            );
        }
        Ok(self.dev.as_mut().expect("pipeline"))
    }
}

fn stage_err(e: SrDeviceError) -> StageError {
    match e {
        SrDeviceError::Unavailable(s) => StageError::Unsupported(s),
        other => StageError::Failed(other.to_string()),
    }
}

impl SrStages for ResidentCudaSr {
    fn label(&self) -> String {
        format!("{} + resident SR step", self.inner.label())
    }
    fn provider(&self) -> String {
        format!(
            "{}; direct_step/cg_step: DeviceSr (cuBLAS dsyrk/dgemv, cuSOLVER dpotrf/dpotrs)",
            self.inner.provider()
        )
    }
    fn stats(&self) -> Option<mvmc_core::sr_backend::TenferroStats> {
        SrStages::stats(&self.inner)
    }
    fn sr_s_g(
        &mut self,
        o: &[f64],
        nsample: usize,
        npara: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<mvmc_core::stage_backend::SrSg, StageError> {
        self.inner.sr_s_g(o, nsample, npara, e, w)
    }
    fn gram_real(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
        out: &mut [f64],
    ) -> Result<(), StageError> {
        self.inner.gram_real(store, n, samples, out)
    }
    fn gram_complex(
        &mut self,
        store: &[num_complex::Complex64],
        n: usize,
        samples: usize,
    ) -> Result<Vec<num_complex::Complex64>, StageError> {
        self.inner.gram_complex(store, n, samples)
    }
    fn assemble_s_g(
        &mut self,
        input: &SrAssembleInput<'_>,
        s: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), StageError> {
        self.inner.assemble_s_g(input, s, g)
    }
    fn cholesky_solve(&mut self, s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
        self.inner.cholesky_solve(s, rhs, n)
    }
    fn cg_local_product(
        &mut self,
        samples: &CgSamples<'_>,
        x: &[f64],
        z: &mut [f64],
    ) -> Result<(), StageError> {
        self.inner.cg_local_product(samples, x, z)
    }

    fn direct_step(&mut self, input: &DirectStepInput<'_>) -> Result<Vec<f64>, StageError> {
        let dev = self.pipeline()?;
        dev.upload_store(input.store, input.n, input.samples)
            .map_err(stage_err)?;
        dev.solve_direct(
            input.ho,
            input.map,
            input.offset,
            input.sta_del,
            input.step_dt,
        )
        .map_err(stage_err)
    }

    fn cg_step(&mut self, input: &CgStepInput<'_>) -> Result<CgStepResult, StageError> {
        let dev = self.pipeline()?;
        dev.set_cg_operand(input.real, input.imag, input.components, input.samples)
            .map_err(stage_err)?;
        let out = dev
            .solve_cg(
                input.gradient,
                input.mean,
                input.diagonal,
                input.inv_weight,
                input.shift,
                input.tolerance,
                input.max_iterations,
            )
            .map_err(stage_err)?;
        Ok(CgStepResult {
            solution: out.solution,
            iterations: out.iterations,
        })
    }
}

/// CUDA stage backend with the resident SR step (opt-in; SR stages through
/// [`ResidentCudaSr`], Pfaffian as in [`cuda_stage_backend`]).
pub fn cuda_resident_stage_backend(ordinal: usize) -> Result<StageBackend<'static>, String> {
    let engine: &'static PersistentCudaEngine =
        Box::leak(Box::new(PersistentCudaEngine::new(ordinal)));
    let pfaffian = BatchedStages::new(Backend::Engine(engine), &format!("cuda:{ordinal}"));
    Ok(StageBackend::new(
        &format!("cuda-resident:{ordinal}"),
        Box::new(ResidentCudaSr::new(ordinal)?),
        Box::new(pfaffian),
    ))
}
