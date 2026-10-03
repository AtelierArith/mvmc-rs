//! Phase 7 stretch: 10-step `zvo_out.dat` parity for `heisenberg_chain_cmp`.
//!
//! The complex driver exercises the rank-2 `update_m_all_two_complex_flat`
//! pathway as well as `green_func2`'s `conj(z/ip)` sign convention.

use std::fs;

mod support;
use support::{julia_mvmc_root, report_gate, require_gate, GateStatus};

#[test]
#[ignore = "optional parity gate; set MVMC_RS_PHASE4_CMP_ZVO=1 and explicitly run ignored tests"]
fn heisenberg_chain_cmp_zvo_out_first10_matches_reference() {
    require_gate("phase4-cmp-zvo", "MVMC_RS_PHASE4_CMP_ZVO");
    let Some(julia) = julia_mvmc_root() else {
        support::missing_fixture("phase4-cmp-zvo", "Julia-mVMC checkout not found");
    };
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join("heisenberg_chain_cmp")
        .join("namelist.def");
    let reference = julia
        .join("test")
        .join("integration")
        .join("reference")
        .join("heisenberg_chain_cmp")
        .join("zvo_out_first10.dat");
    if !namelist.is_file() || !reference.is_file() {
        support::missing_fixture("phase4-cmp-zvo", "namelist or reference is missing");
    }

    let out_dir =
        std::env::temp_dir().join(format!("mvmc-core-phase4-gate-cmp-{}", std::process::id(),));
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir).expect("create temp dir");
    mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(10),
            seed: Some(1),
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(10, "cmp")
        },
    )
    .expect("10-step cmp run completes");

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
    report_gate(
        "phase4-cmp-zvo",
        GateStatus::Pass,
        "10-step reference matched",
    );
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
