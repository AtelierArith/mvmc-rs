//! Phase 5 gate: 10-step `zvo_out.dat` parity for `hubbard_chain_real`.
//!
//! Hubbard differs from the Heisenberg gates in three ways:
//!   - Uses `CoulombIntra` (on-site U=4) instead of Exchange/CoulombInter/Hund.
//!   - Carries an `initial.def` overlay that seeds the variational parameters.
//!   - `NExUpdatePath = 0` (single-electron hopping only, no exchange path).
//!
//! The gate confirms that `read_initial_def`, the CoulombIntra energy path,
//! and the NExUpdatePath=0 sampler all reproduce the C-mVMC reference to
//! within 1e-10 (absolute or relative).

use std::fs;
use std::path::PathBuf;

mod support;
use support::julia_mvmc_root;

#[test]
fn hubbard_chain_real_zvo_out_first10_matches_reference() {
    let Some(julia) = julia_mvmc_root() else {
        eprintln!("skipping Phase 5 Hubbard gate: Julia-mVMC checkout not found");
        return;
    };
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join("hubbard_chain_real")
        .join("namelist.def");
    let reference = julia
        .join("test")
        .join("integration")
        .join("reference")
        .join("hubbard_chain_real")
        .join("zvo_out_first10.dat");
    if !namelist.is_file() || !reference.is_file() {
        eprintln!("skipping Phase 5 Hubbard gate: missing fixture/reference");
        return;
    }

    let out_dir = tempdir_in_target();
    mvmc_core::run_para_opt_from_namelist(&namelist, 10, Some(1), Some(&out_dir))
        .expect("10-step Hubbard run completes");

    let got = fs::read_to_string(out_dir.join("zvo_out.dat")).expect("zvo_out.dat written");
    let expected = fs::read_to_string(&reference).expect("reference readable");
    let got_rows = parse_rows(&got);
    let expected_rows = parse_rows(&expected);
    assert_eq!(got_rows.len(), 10, "expected 10 output rows");
    assert_eq!(expected_rows.len(), 10, "reference must have 10 rows");
    for (row, (got_row, expected_row)) in got_rows.iter().zip(expected_rows.iter()).enumerate() {
        assert_eq!(got_row.len(), expected_row.len(), "row {row} column count");
        for (col, (&g, &e)) in got_row.iter().zip(expected_row.iter()).enumerate() {
            let tol = 1.0e-10_f64.max(e.abs() * 1.0e-10);
            assert!(
                (g - e).abs() <= tol,
                "row {row} col {col}: got {g:.18e}, expected {e:.18e}, tol {tol:.3e}"
            );
        }
    }
}

fn parse_rows(content: &str) -> Vec<Vec<f64>> {
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| token.parse::<f64>().expect("numeric zvo token"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn tempdir_in_target() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "mvmc-core-phase5-gate-hubbard-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp dir");
    path
}
