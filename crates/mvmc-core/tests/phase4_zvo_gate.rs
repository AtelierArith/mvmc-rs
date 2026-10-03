//! Phase 4 gate: 10-step `zvo_out.dat` parity for `heisenberg_chain_real`.

use std::fs;
use std::path::PathBuf;

mod support;
use support::julia_mvmc_root;

#[test]
#[ignore = "optional parity gate: MVMC_RS_PHASE4_REAL_ZVO required"]
fn heisenberg_chain_real_zvo_out_first10_matches_reference() {
    support::require_gate("phase4-real", "MVMC_RS_PHASE4_REAL_ZVO");
    let julia = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("phase4-real", "Julia-mVMC checkout not found")
    });
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join("heisenberg_chain_real")
        .join("namelist.def");
    let reference = julia
        .join("test")
        .join("integration")
        .join("reference")
        .join("heisenberg_chain_real")
        .join("zvo_out_first10.dat");
    if !namelist.is_file() || !reference.is_file() {
        support::missing_fixture(
            "phase4-real",
            format!(
                "required input {} or reference {} is missing",
                namelist.display(),
                reference.display()
            ),
        );
    }

    let tmp = tempdir_in_target();
    let out_dir = tmp.join("out");
    mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(10),
            seed: Some(1),
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(10, "real")
        },
    )
    .expect("10-step run completes");

    let got = fs::read_to_string(out_dir.join("zvo_out.dat")).expect("zvo_out.dat written");
    let expected = fs::read_to_string(reference).expect("reference readable");
    let got_rows = parse_rows(&got);
    let expected_rows = parse_rows(&expected);
    assert_eq!(got_rows.len(), 10);
    assert_eq!(expected_rows.len(), 10);
    for (row, (got_row, expected_row)) in got_rows.iter().zip(expected_rows.iter()).enumerate() {
        assert_eq!(got_row.len(), expected_row.len(), "row {row} width");
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
    path.push(format!("mvmc-core-phase4-gate-{}", std::process::id(),));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("create temp dir");
    path
}
