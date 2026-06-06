//! Shared helpers for `mvmc-cli` example binaries.
//!
//! Mirrors the role of `extern/Julia-mVMC/examples/*.jl` preamble logic:
//! * Locate the bundled `Julia-mVMC` checkout so examples find their input files.
//! * Read `JULIA_MVMC_EXAMPLE_STEPS` / `MVMC_NSTEPS` for the step-count override.
//!
//! Include in each example with:
//! ```rust,ignore
//! #[path = "support.rs"]
//! mod support;
//! ```

use std::path::PathBuf;

/// Locate the `Julia-mVMC` checkout that ships the example inputs.
///
/// Resolution order:
/// 1. `JULIA_MVMC_ROOT` env var — explicit override.
/// 2. Paths relative to this crate's `CARGO_MANIFEST_DIR`
///    (`rust/crates/mvmc-cli/`):
///    * `../../../extern/Julia-mVMC` — canonical layout inside
///      `ManyVariableVariationalMonteCarlo.jl/`.
///    * additional fallbacks for alternate checkout locations.
///
/// Returns `None` when no candidate directory exists.
pub fn julia_mvmc_root() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("JULIA_MVMC_ROOT") {
        let p = PathBuf::from(custom);
        if p.is_dir() {
            return Some(p);
        }
    }
    // CARGO_MANIFEST_DIR = …/rust/crates/mvmc-cli
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        // canonical: ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC
        manifest.join("../../../extern/Julia-mVMC"),
        // sibling checkout at the workspace root level
        manifest.join("../../../../extern/Julia-mVMC"),
        // running from within rust/ directly
        manifest.join("../../extern/Julia-mVMC"),
        // legacy / alternate layouts
        manifest.join("../Julia-mVMC"),
        manifest.join("../../Julia-mVMC"),
    ];
    for c in &candidates {
        if c.is_dir() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

/// Read the SR step count from the environment.
///
/// Checks (in order):
/// * `JULIA_MVMC_EXAMPLE_STEPS` — matches the Julia examples' env var name.
/// * `MVMC_NSTEPS` — Rust CLI convention.
///
/// Falls back to `50` (same default as the Julia examples).
pub fn example_nsteps() -> usize {
    std::env::var("JULIA_MVMC_EXAMPLE_STEPS")
        .or_else(|_| std::env::var("MVMC_NSTEPS"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50)
}

/// Return `rust/output/<model>/` as the canonical output directory for
/// example runs and create it if it does not yet exist.
///
/// The path is fixed (not PID-suffixed) so successive runs overwrite the
/// previous outputs, exactly like the Julia examples which write to a
/// caller-supplied directory. Results accumulate in the Rust workspace at:
///
/// ```text
/// rust/
/// └── output/
///     ├── heisenberg_chain_real/
///     │   ├── zvo_out.dat
///     │   ├── zvo_var.dat
///     │   └── zqp_opt.dat
///     ├── heisenberg_chain_cmp/
///     ├── heisenberg_chain_fsz/
///     └── hubbard_chain/
/// ```
///
/// Override with the `MVMC_OUT_DIR` env var when a different root is wanted
/// (e.g. in CI: `MVMC_OUT_DIR=/tmp/mvmc-out cargo run --example ...`).
pub fn make_output_dir(model: &str) -> PathBuf {
    let base = if let Ok(custom) = std::env::var("MVMC_OUT_DIR") {
        PathBuf::from(custom)
    } else {
        // CARGO_MANIFEST_DIR = …/rust/crates/mvmc-cli
        // ../../output        = …/rust/output
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../output")
    };
    let dir = base.join(model);
    std::fs::create_dir_all(&dir).expect("create output dir");
    // Canonicalize to resolve `..` components so printed paths are clean.
    dir.canonicalize().unwrap_or(dir)
}
