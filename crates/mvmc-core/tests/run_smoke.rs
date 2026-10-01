//! Phase 4.8 smoke test — runs one SR step on the upstream Heisenberg
//! chain fixture and verifies the run pipeline (Slater rebuild,
//! sampling driver, observables, weighted averages, SR, IO writer).
//!
//! Bit-parity with C-mVMC is the eventual Phase-4 gate; for the Phase-4
//! `作業完了` boundary it is enough that the driver produces a
//! `zvo_out.dat` row with a finite energy.

use std::fs;
use std::path::PathBuf;

mod support;
use support::julia_mvmc_root;

#[test]
fn heisenberg_chain_real_runs_one_sr_step() {
    let Some(julia) = julia_mvmc_root() else {
        eprintln!("skipping run_smoke: Julia-mVMC checkout not found");
        return;
    };
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join("heisenberg_chain_real")
        .join("namelist.def");
    if !namelist.is_file() {
        eprintln!(
            "skipping run_smoke test: fixture missing at {}",
            namelist.display()
        );
        return;
    }

    let tmp = tempdir_in_target();
    let out_dir = tmp.join("out");
    let summary = mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(1),
            seed: Some(1),
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(1, "real")
        },
    )
    .expect("run completes");
    let zvo = out_dir.join("zvo_out.dat");
    let content = fs::read_to_string(&zvo).expect("zvo_out.dat written");
    let line = content.lines().next().expect("at least one line");
    let first_token = line
        .split_whitespace()
        .next()
        .expect("first column present");
    let energy: f64 = first_token.parse().expect("energy column is f64");
    assert!(energy.is_finite(), "energy must be finite, got {energy}");
    assert_eq!(summary.effective_nsteps, 1);
}

fn tempdir_in_target() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("mvmc-core-run-smoke-{}", std::process::id(),));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp dir");
    path
}
