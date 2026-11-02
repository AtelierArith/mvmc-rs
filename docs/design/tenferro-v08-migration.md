# tenferro v0.8 migration (work in progress)

The root and standalone CUDA workspaces use the caller-owned tenferro Engine API. The root workspace selects the `native` CPU provider and remains independent of CUDA. The benchmark and CUDA workspaces maintain separate lockfiles. See the migration PR for the final tested versions and numerical evidence; this document does not itself establish completion of the migration.

## CPU ownership

The SR and measurement stages in one `StageBackend` share an eager context and a mutex-protected caller-owned Engine. The engine is borrowed exclusively for a complete operation or transfer. CPU construction uses `RAYON_NUM_THREADS`, falling back to the process's available parallelism when it is absent; one thread selects a serial Engine and larger values create a `threadteam` pool. The mVMC Rayon pool and the tenferro pool are separate, so callers running multiple walkers must still control their combined thread budget. Parallel utilization, nested pool behavior, and wait policies need measured validation; the existence of multiple threads does not establish a speedup.

The normal non-C-order stage acquisition uses one process-wide shared stage, so concurrent acquisitions serialize on its mutex. Direct construction of multiple `StageBackend::tenferro_cpu()` instances can create multiple independent pools. MPI processes also have independent pools. `RAYON_NUM_THREADS` is therefore not a global limit on all runnable threads: a Rayon worker, a tenferro team, other walkers and BLAS threads may overlap. Use `RAYON_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` where a single-threaded tenferro stage is required; broader team budgeting and multi-walker/MPI behavior remain open in #581–#584.

An idle-only probe on the migration host, sampled three seconds after stage creation, observed one OS thread and 0.0% CPU with `RAYON_NUM_THREADS=1`, and five OS threads and 0.3% CPU with `RAYON_NUM_THREADS=4`. The ten-second `crates/mvmc-core/examples/tenferro_idle_probe.rs` program reproduces that observation when inspected with `ps`. This single idle sample does not measure active worker overlap, nesting, affinity, MPI or performance.

## CUDA ownership

Each CUDA Engine owns a runtime and stream. The eager context and its Engine are explicitly paired; uploads, eager operations and downloads borrow that same Engine. A raw-execution scope visits the actual CUDA Engine and uses its stream. Downstream specialized kernels are compiled by NVRTC and loaded by the CUDA driver because tenferro 0.8's raw session exposes no downstream kernel compilation or launch surface. Compilation flags are part of the module cache key; the standalone Pfaffian kernel disables fused multiply-add as before.

Raw device buffers are retained on the CUDA primary context, associated with their originating runtime identity, and rejected by other runtimes even on the same device. Device buffers, loaded modules, pinned staging and native handles must remain live until their stream work completes. Raw scopes synchronize on both successful and returned-error paths; this does not by itself establish panic safety or remove the caller's responsibility for asynchronous buffer lifetimes. Keep transfer and synchronization counts visible in validation. No cross-device pointer borrowing or silent CPU fallback is permitted.

The production Pfaffian stage owns its persistent CUDA worker instead of leaking it. Dropping the stage closes the work channel and joins the worker. This changes resource lifetime; GPU tests must exercise repeated creation, use and destruction before merge.

## Numerical and security boundaries

CPU C-order execution retains the C numerical and SFMT contract. The resident GPU Opt path has a separate Julia GPU numerical and RNG contract; passing one path's tests does not validate the other. Review singular solve behavior and reduction-order changes when validating tenferro 0.8. Preserve explicit tolerances, first-divergence evidence and independent RNG checks.

The old tenferro 0.7.1 patches for `lru` are not active in the three migrated workspace locks. Verify each dependency tree and retain the historical provenance in `docs/security/dependabot-alert-5.md`.

## Shared Pfaffian extension

`mvmc_gpu::tenferro_ext::PfaffianInverseOp` defines a multi-output tenferro operation on compact column-major `[n,n,NQP,B]` tensors. It accepts real `f64` and `Complex64`; matrix side `n` must be even and at least 2. It returns Pfaffians `[NQP,B]`, true inverses `[n,n,NQP,B]` (the mVMC stored `invM` is their negation), and i32 per-plane status `[NQP,B]`: zero for success, a positive one-based zero-pivot row, and `-1` for a non-finite Pfaffian. Failed inverse planes are zero-filled. Neither provider silently falls back to another device.

Register `cpu_extension_module` with a caller-held CPU Engine, or `mvmc_gpu_cuda::pfaffian_extension::cuda_extension_module` with a caller-held CUDA Engine, using the matching tenferro Engine ID. Use `TraceContext::add_extension` and `GraphCompiler::compile_traced_graph` to produce a reusable multi-output program. `gpu/mvmc-gpu-cuda/tests/pfaffian_extension_gate.rs` is an executable end-to-end example, including explicit upload and download outside the timed extension execution.

The CUDA executor accepts owned, compact tensors in the same allocation domain and supports `n <= 1024`; it explicitly rejects tensor views and unsupported shapes or scalar types. Empty batches are accepted. It keeps input and outputs on that engine, runs the specialized CUDA kernel, and retains at most 64 MiB of intermediate scratch through the engine's bounded extension cache. Larger scratch allocations are scoped to the call. The compiled module is stored in the same engine-owned cache, with its host PTX length used as the retained-byte estimate; this is not a measurement of driver-side module memory. When the cache cannot retain the module, the executor synchronizes before releasing it. The executor synchronizes its raw scope before returning; it does not claim asynchronous behavior. Existing production sampler kernels remain unchanged and are not automatically replaced by this API.
