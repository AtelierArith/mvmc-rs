# tenferro-ext (adapted vendored copy)

This crate is a **heavily modified copy** of `tenferro-ext` from
[`AtelierArith/tenferro-decision-rs`](https://github.com/AtelierArith/tenferro-decision-rs),
brought into mvmc-rs to reuse its single-stream raw CUDA execution helper for the
device-resident interval sampler (issue #547).

## Provenance

| field | value |
| --- | --- |
| origin repository | `https://github.com/AtelierArith/tenferro-decision-rs` |
| origin commit | `5b78172f78ff8cc523276233a0e869542dbd9e8a` |
| origin path | `crates/tenferro-ext/` |
| upstream license | MIT OR Apache-2.0 (kept) |

## What is compiled

Only `src/raw_exec.rs` (via `src/lib.rs`). It provides:

* `with_raw_exec(session, label, |exec| ...)` — runs a whole workload inside one
  `CudaExecSession::with_raw` scope on tenferro's captured stream.
* `RawExec` — `module()` (NVRTC compile, thread-local cache), `launch(function, grid,
  block, shared, args)` (dynamic shared memory), `alloc` / `ptr` / `upload` /
  `download` / `zero` / `synchronize`, `gemm` / `gemm_batched`.
* `DeviceBuffer` — plain driver allocations on tenferro's primary context, bypassing
  CubeCL's process-global pool.
* `ArenaBuilder` — aligned host staging for resident weight uploads.

## What is not compiled

The upstream decision-engine extension ops (`erf`, GELU, `linear`/`gemm_bias`,
LayerNorm, GeGLU, gated SiLU, RMSNorm, the Laya/Jeff attention blocks, the fused CUDA
scope and `device.rs`) are preserved under `upstream/` for reference. They depend on the
origin workspace's `cpu-kernels` / `tenferro-infer` crates and are not built here.

## Local modifications

* `Cargo.toml` materialized as a standalone manifest (the upstream used
  `version.workspace`/`edition.workspace`/... and workspace dependencies) and reduced to
  the dependencies `raw_exec` needs.
* `src/lib.rs` reduced to `pub mod raw_exec;`.
* Upstream tests/examples removed (they exercise the decision-engine ops).
* Subsequent adaptations (for example a `f64` element type for mvmc's sampler) are
  recorded in the commit history.

Keep this file in sync when updating the copy.
