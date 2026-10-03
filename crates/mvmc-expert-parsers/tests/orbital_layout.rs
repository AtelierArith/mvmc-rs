//! Declared widths reserve parameter slots even when mappings are sparse.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::fs;
use std::path::PathBuf;

use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::parsers::orbital::{parse_orbital_content, OrbitalKind};
use mvmc_expert_parsers::utils::parameter_init::init_parameter;
use sfmt19937::Sfmt19937Rng;

fn definition(header: &str, width: usize, rows: &str) -> String {
    format!("===\n{header} {width}\nComplexType 0\n===\n===\n{rows}")
}

fn fixture(name: &str, namelist: &str, ap: &str, parallel: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-orbital-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("modpara.def"), "Nsite 2\nNElec 1\n").unwrap();
    fs::write(
        dir.join("namelist.def"),
        format!("ModPara modpara.def\n{namelist}"),
    )
    .unwrap();
    fs::write(dir.join("ap.def"), ap).unwrap();
    fs::write(dir.join("p.def"), parallel).unwrap();
    dir
}

#[test]
fn sparse_ap_and_parallel_use_declared_widths_and_rng_consumption() {
    let dir = fixture(
        "sparse",
        "OrbitalAntiParallel ap.def\nOrbitalParallel p.def\n",
        &definition(
            "NOrbitalAntiParallel",
            7,
            "0 0 0\n0 1 1\n1 0 1\n1 1 0\n0 1\n1 1\n2 1\n3 1\n4 1\n5 1\n6 1\n",
        ),
        &definition("NOrbitalParallel", 3, "0 1 0\n0 1\n1 1\n2 1\n"),
    );
    let mut data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    assert_eq!(data.n_orbital_anti_parallel, 7);
    assert_eq!(data.modpara.n_orbital_idx, 13);
    assert_eq!(
        data.orbital_terms.iter().map(|t| t.idx).collect::<Vec<_>>(),
        [0, 1, 1, 0, 7, 8]
    );
    let mut rng = Sfmt19937Rng::new(1);
    let mut probe = Sfmt19937Rng::new(1);
    init_parameter(&mut data, &mut rng);
    // Live Julia v0.5.0 / SFMT v0.1.0 reference, seed 1. Compare
    // numerical values with rounding bounds and the exact next RNG word.
    for (idx, expected) in [0, 1, 7, 8].into_iter().zip([
        -0.3232123088091612_f64,
        0.201519466470927,
        -0.2580129294656217,
        -0.6150796245783567,
    ]) {
        // Only a scale/subtraction chain is computed; draw count stays exact.
        numerical_comparison::assert_close(
            data.slater_params[idx].re,
            expected,
            1e-14,
            1e-14,
            format!("Slater index {idx}"),
        );
        assert_eq!(data.slater_params[idx].im, 0.0);
    }
    // C initializes all thirteen declared active slots.
    for _ in 0..13 {
        probe.genrand_real2();
    }
    assert_eq!(rng.gen_rand32(), 1_738_325_211);
    assert_eq!(probe.gen_rand32(), 1_738_325_211);
    for _ in 0..624 {
        assert_eq!(rng.gen_rand32(), probe.gen_rand32());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_positive_width_ap_is_rejected_instead_of_reserving_a_valid_definition() {
    let dir = fixture(
        "empty",
        "Orbital ap.def\nOrbitalParallel p.def\n",
        &definition("NOrbitalIdx", 7, ""),
        &definition("NOrbitalParallel", 3, "0 1 0\n0 1\n1 1\n2 1\n"),
    );
    let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    assert!(data
        .input_errors
        .iter()
        .any(|error| error.contains("incomplete orbital mapping")));
    assert_eq!(data.i_flg_orbital_anti_parallel, 0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn complete_general_rows_retain_unused_declared_slots_but_headerless_definitions_fail() {
    let general = parse_orbital_content(
        &definition("NOrbitalGeneral", 9, "0 0 1 0 1 1\n0 0 0 1 1 1\n0 0 1 1 1 1\n1 0 0 1 1 1\n1 0 1 1 1 1\n0 1 1 1 1 1\n0 1\n1 1\n2 1\n3 1\n4 1\n5 1\n6 1\n7 1\n8 1\n"),
        2,
        OrbitalKind::General,
    )
    .unwrap();
    assert_eq!(general.n_orbital_idx, 9);
    assert_eq!(general.terms.len(), 6);
    assert!(parse_orbital_content("0 1 2\n1 0 0\n", 2, OrbitalKind::AntiParallel).is_err());
}
