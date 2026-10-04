#![allow(dead_code)]
//! Locate the bundled `Julia-mVMC` checkout. The Rust workspace may live
//! either inside `ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC-rs/`
//! (legacy layout) or as a sibling of `ManyVariableVariationalMonteCarlo.jl`
//! (current layout). Tests call into this helper instead of hard-coding
//! `../../../Julia-mVMC` paths so both layouts work without edits.

use std::path::PathBuf;

pub fn julia_mvmc_root() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("JULIA_MVMC_ROOT") {
        let p = PathBuf::from(custom);
        if p.is_dir() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../../../Julia-mVMC"),
        manifest.join("../../../ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC"),
        manifest.join("../../../../ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC"),
        manifest.join("../../extern/Julia-mVMC"),
        manifest.join("../../Julia-mVMC"),
        manifest.join("../Julia-mVMC"),
    ];
    candidates.into_iter().find(|c| c.is_dir())
}
