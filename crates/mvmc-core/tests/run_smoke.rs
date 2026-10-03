//! Phase 4.8 smoke test — runs one SR step on the upstream Heisenberg
//! chain fixture and verifies the run pipeline (Slater rebuild,
//! sampling driver, observables, weighted averages, SR, IO writer).
//!
//! This smoke test checks a `zvo_out.dat` row with finite energy.
//! Separate numerical kernel and runner regressions compare against independent
//! references with explicit bounds and exact RNG/sampling controls.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

mod support;
use support::julia_mvmc_root;

#[test]
fn heisenberg_chain_real_runs_one_sr_step() {
    let julia = julia_mvmc_root()
        .unwrap_or_else(|| support::missing_fixture("run-smoke", "Julia-mVMC checkout not found"));
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join("heisenberg_chain_real")
        .join("namelist.def");
    if !namelist.is_file() {
        support::missing_fixture(
            "run-smoke",
            format!("namelist missing at {}", namelist.display()),
        );
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

#[test]
fn mandatory_smoke_missing_checkout_or_namelist_fails_before_running() {
    let empty_checkout = tempdir_in_target();
    for root in [
        empty_checkout.join("absent-checkout"),
        empty_checkout.clone(),
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "heisenberg_chain_real_runs_one_sr_step",
                "--nocapture",
            ])
            .env("JULIA_MVMC_ROOT", root)
            .output()
            .unwrap();
        assert!(!output.status.success(), "missing fixture must not pass");
        assert!(String::from_utf8_lossy(&output.stderr).contains("MissingFixture"));
    }
    fs::remove_dir(empty_checkout).unwrap();
}

fn tempdir_in_target() -> PathBuf {
    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    loop {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mvmc-core-run-smoke-{}-{sequence}",
            std::process::id(),
        ));
        match fs::create_dir(&path) {
            Ok(()) => return path,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("cannot create {}: {error}", path.display()),
        }
    }
}
