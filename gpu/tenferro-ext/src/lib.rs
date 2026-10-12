//! Adapted vendored copy of `tenferro-ext` from `AtelierArith/tenferro-decision-rs`.
//!
//! Only [`raw_exec`] is compiled here: the single-stream raw CUDA execution helper
//! (`with_raw_exec` / `RawExec`) that runs a whole workload of NVRTC kernels and cuBLAS
//! GEMMs inside one `CudaExecSession::with_raw` scope on tenferro's captured stream, with
//! resident device buffers and a single host synchronization. It is the template for
//! mvmc-rs' own device-resident work (issue #547).
//!
//! See `README.md` for provenance and the list of local modifications. The upstream
//! decision-engine extension ops are kept under `upstream/` for reference and are not
//! part of this build.

pub mod raw_exec;
