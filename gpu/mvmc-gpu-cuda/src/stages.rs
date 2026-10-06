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

use mvmc_core::sr_backend::{Placement, TenferroSr};
use mvmc_core::stage_backend::{StageBackend, UnsupportedPfaffian};
use mvmc_gpu::stages::BatchedStages;
use mvmc_gpu::Backend;

use crate::pfaffian::PersistentCudaEngine;

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
