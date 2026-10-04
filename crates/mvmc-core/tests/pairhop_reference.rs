//! Port of canonical pairhop_equivalent.jl; exact Julia trajectories have separate gates.
use std::fs;
use std::path::Path;

use mvmc_core::{run_para_opt_from_namelist, RunConfig};

#[test]
fn real_and_fsz_pairhop_match_the_canonical_one_step_c_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/test/integration/reference");
    for mode in ["real", "fsz"] {
        let case = root.join(format!("hubbard_chain_pairhop_{mode}"));
        let result = run_para_opt_from_namelist(
            case.join("inputs/namelist.def"),
            RunConfig {
                nsmp: Some(1),
                ..RunConfig::new(1, mode)
            },
        )
        .unwrap();
        let expected = fs::read_to_string(case.join("zvo_out_first1.dat")).unwrap();
        let row: Vec<f64> = expected
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let actual: Vec<f64> = result.zvo_first_n[0]
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(row.len(), 6);
        assert_eq!(actual.len(), 6);
        for (column, (&actual, &expected)) in actual.iter().zip(&row).enumerate() {
            // Exactly the canonical tolerances: <H²> and variance are columns 3/4.
            let tolerance = if matches!(column, 2 | 3) { 1e-9 } else { 1e-10 };
            assert!(
                (actual - expected).abs() <= tolerance,
                "{mode} column {column}: {actual} vs {expected}, tolerance={tolerance}"
            );
        }
        fs::remove_dir_all(result.output_dir).unwrap();
    }
}
