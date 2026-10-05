//! C `OutputOptData` output contracts, including declared RBM blocks.
use mvmc_core::{
    io::{format_c_double, output_opt_data},
    ExpertModeData, OptDataPoint, VmcOptimizationState,
};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
use num_complex::Complex64;
use std::fs;

fn test_output_dir(label: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    for _ in 0..10_000 {
        let index = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mvmc-output-blocks-{label}-{}-{index}",
            std::process::id(),
        ));
        match fs::create_dir(&path) {
            Ok(()) => return path,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create exclusive output test directory: {error}"),
        }
    }
    panic!("cannot allocate exclusive output test directory");
}

#[test]
fn optimization_var_full_declared_storage_matches_fixed_c_bytes() {
    let coefficient =
        |index: usize| Complex64::new((index + 1) as f64 * 0.125, -((index + 1) as f64) * 0.0625);
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 2;
    data.n_jastrow_idx = 2;
    // Declared but unmapped second Gutz/Jast coefficients remain zero.
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: coefficient(0),
        is_complex: true,
    });
    data.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: coefficient(2),
        is_complex: true,
    });
    data.doublon_holon_2site_indices
        .push(mvmc_expert_parsers::DoublonHolon2SiteIndex { neighbors: vec![] });
    data.doublon_holon_4site_indices
        .push(mvmc_expert_parsers::DoublonHolon4SiteIndex { neighbors: vec![] });
    data.doublon_holon_2site_params = (4..10).map(coefficient).collect();
    data.doublon_holon_4site_params = (10..20).map(coefficient).collect();
    data.rbm_section_widths = [1; 9];
    data.rbm_params = (20..29).map(coefficient).collect();
    data.modpara.n_orbital_idx = 2;
    data.slater_params = (29..31).map(coefficient).collect();
    // Both stored Slater/OptTrans slots are emitted without requiring mappings.
    data.opt_trans = (31..33).map(coefficient).collect();
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, true, false);
    state.energy.etot = Complex64::new(-3.0, 0.0);
    state.energy.etot2 = Complex64::new(9.0, 0.0);
    let dir = test_output_dir("c-var-bytes");
    mvmc_core::io::output_data(&data, &state, 0, Some(&dir), false).unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/formatting/var-full.dat");
    assert_eq!(
        fs::read(dir.join("zvo_var.dat")).unwrap(),
        fs::read(fixture).unwrap(),
        "C full-storage slot order, exponential format, trailing spaces and single newline"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn optimization_var_writes_complete_c_parameter_storage_and_truncates_on_first_step() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/physcal_181");
    for name in [
        "hubbard_chain_dh_overlays",
        "hubbard_chain_dh_opttrans",
        "hubbard_chain_dh_rbm_opttrans",
        "unnormalized-reserved-slater",
    ] {
        let fixture = root.join(name);
        let namelist = fixture.join("inputs/namelist.def");
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist, true).unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(1);
        mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng).unwrap();
        mvmc_core::read_opt_para_file(&mut data, fixture.join("zqp_opt.dat")).unwrap();
        mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
            &mut data, &namelist,
        )
        .unwrap();
        mvmc_core::sync::sync_modified_parameter_local(&mut data, false);
        // Independently packed by the actual C parameter-stage probe, not
        // this writer or a Rust-before snapshot. Includes every declared slot.
        let native =
            fs::read_to_string(fixture.join("native-c-stages/synchronized-parameters.txt"))
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
        let (parameters, remainder) = native.as_chunks::<2>();
        assert!(remainder.is_empty(), "complete native C complex parameters");
        if name == "hubbard_chain_dh_overlays" {
            assert_eq!(parameters.len(), 35);
            assert_eq!(6 + 3 * parameters.len(), 111, "C DH2/DH4 var width");
        }
        let dir = test_output_dir(&format!("full-var-{name}"));
        let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
        state.energy.etot = Complex64::new(-3.0, 0.0);
        state.energy.etot2 = Complex64::new(9.0, 0.0);
        for step in [0, 1, 0] {
            mvmc_core::io::output_data(&data, &state, step, Some(&dir), false).unwrap();
            let text = fs::read_to_string(dir.join("zvo_var.dat")).unwrap();
            assert_eq!(text.lines().count(), if step == 1 { 2 } else { 1 });
            for line in text.lines() {
                let values = line
                    .split_whitespace()
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    values.len(),
                    6 + 3 * parameters.len(),
                    "{name} full C var shape"
                );
                assert_eq!(&values[..6], &[-3.0, 0.0, 0.0, 9.0, 0.0, 0.0]);
                let (triples, remainder) = values[6..].as_chunks::<3>();
                assert!(remainder.is_empty());
                for (index, (actual, expected)) in triples.iter().zip(parameters).enumerate() {
                    assert_eq!(actual[2], 0.0);
                    for component in 0..2 {
                        let (a, e) = (actual[component], expected[component]);
                        assert!(
                            (a - e).abs() <= 1e-12_f64.max(1e-10 * a.abs().max(e.abs())),
                            "{name} C slot {index} component {component}: {a} != {e}"
                        );
                    }
                }
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }
}

// Independent constant-history case of C avevar.c: two completed records,
// measured energies -3 and 9, identical post-SR coefficients, zero deviations.
// Full nonconstant C-window fixtures are covered by ctest_model_prefixes.
fn constant_window(parameters: Vec<Complex64>) -> VmcOptimizationState {
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    state.opt_data = vec![
        OptDataPoint {
            energy: Complex64::new(-3.0, 0.0),
            energy_squared: Complex64::new(9.0, 0.0),
            parameters,
        };
        2
    ];
    state
}

fn c_header(name: &str, count: usize) -> String {
    // C WriteHeader: 22 equals, two spaces before count, four separators.
    format!("======================\n{name}  {count}\n======================\n======================\n======================\n")
}

#[test]
fn block_files_keep_declared_slater_order_and_unmapped_slots_with_existing_headers() {
    let dir = test_output_dir("block-output");
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 4;
    data.slater_params = vec![
        Complex64::new(-2.0, 1.5),
        Complex64::new(1.5, -2.0),
        Complex64::new(-2.0, 1.5),
        Complex64::new(1.5, -2.0),
    ];
    data.modpara.c_para_file_head = "custom".into();
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 9,
        value: Complex64::new(1.5, -2.0),
        is_complex: true,
    });
    data.jastrow_terms.push(JastrowTerm {
        site1: 3,
        site2: 7,
        value: Complex64::new(-2.0, 1.5),
        is_complex: true,
    });
    for idx in [3, 3] {
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx,
            is_complex: true,
            sign: 1,
        });
    }
    let state = constant_window(vec![
        Complex64::new(1.5, -2.0),
        Complex64::new(-2.0, 1.5),
        Complex64::new(-2.0, 1.5),
        Complex64::new(1.5, -2.0),
        Complex64::new(-2.0, 1.5),
        Complex64::new(1.5, -2.0),
    ]);
    output_opt_data(&data, &state, Some(&dir)).unwrap();
    let row = " 1.500000000000000000e+00 -2.000000000000000000e+00 \n";
    let reverse = "-2.000000000000000000e+00  1.500000000000000000e+00 \n";
    let header = c_header;
    let zero = " 0.000000000000000000e+00 ";
    let triple = |pair: &str| format!("{}{zero}", pair.trim_end_matches('\n'));
    assert_eq!(
        fs::read_to_string(dir.join("custom_opt.dat")).unwrap(),
        format!("-3.000000000000000000e+00 {zero}{zero} 9.000000000000000000e+00 {zero}{zero}{}{}{}{}{}{}\n",
            triple(row), triple(reverse), triple(reverse), triple(row), triple(reverse), triple(row))
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_gutzwiller_opt.dat")).unwrap(),
        format!("{}0 {row}", header("NGutzwillerIdx", 1))
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_jastrow_opt.dat")).unwrap(),
        format!("{}0 {reverse}", header("NJastrowIdx", 1))
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_orbital_opt.dat")).unwrap(),
        format!(
            "{}0 {reverse}1 {row}2 {reverse}3 {row}",
            header("NOrbitalIdx", 4)
        )
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_blocks_are_skipped_and_no_directory_uses_the_given_prefix() {
    let dir = test_output_dir("empty-output");
    let mut data = ExpertModeData::new();
    data.modpara.c_para_file_head = dir.join("prefix").to_str().unwrap().to_owned();
    output_opt_data(&data, &constant_window(Vec::new()), None).unwrap();
    assert_eq!(fs::read_to_string(dir.join("prefix_opt.dat")).unwrap(),
        "-3.000000000000000000e+00  0.000000000000000000e+00  0.000000000000000000e+00  9.000000000000000000e+00  0.000000000000000000e+00  0.000000000000000000e+00 \n");
    assert!(!dir.join("prefix_gutzwiller_opt.dat").exists());
    assert!(!dir.join("prefix_jastrow_opt.dat").exists());
    assert!(!dir.join("prefix_orbital_opt.dat").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rbm_blocks_follow_c_declared_section_order() {
    let dir = test_output_dir("rbm-output");
    let mut data = ExpertModeData::new();
    data.modpara.c_para_file_head = "rbm".into();
    data.rbm_section_widths = [1; 9];
    data.rbm_params = (0..9)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let state = constant_window(
        (0..9)
            .map(|value| Complex64::new(value as f64, -(value as f64)))
            .collect(),
    );
    output_opt_data(&data, &state, Some(&dir)).unwrap();

    let names = [
        ("chargeRBM_physlayer", "NChargeRBM_PhysLayerIdx", 0),
        ("spinRBM_physlayer", "NSpinRBM_PhysLayerIdx", 1),
        ("generalRBM_physlayer", "NGeneralRBM_PhysLayerIdx", 2),
        ("chargeRBM_hiddenlayer", "NChargeRBM_HiddenLayerIdx", 3),
        ("spinRBM_hiddenlayer", "NSpinRBM_HiddenLayerIdx", 4),
        ("generalRBM_hiddenlayer", "NGeneralRBM_HiddenLayerIdx", 5),
        ("chargeRBM_physhidden", "NChargeRBM_PhysHiddenIdx", 6),
        ("spinRBM_physhidden", "NSpinRBM_PhysHiddenIdx", 7),
        ("generalRBM_physhidden", "NGeneralRBM_PhysHiddenIdx", 8),
    ];
    for (suffix, label, value) in names {
        let content = fs::read_to_string(dir.join(format!("rbm_{suffix}_opt.dat"))).unwrap();
        assert!(content.starts_with(&c_header(label, 1)));
        assert!(content.ends_with(&format!(
            "0 {} {} \n",
            format_c_double(value as f64),
            // C's sequential sum starts at +0; averaging -0 yields +0.
            format_c_double(if value == 0 { 0.0 } else { -(value as f64) })
        )));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn single_record_uses_c_real_zero_pairs_and_incomplete_windows_create_no_files() {
    let dir = test_output_dir("single-window");
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 1;
    data.slater_params = vec![Complex64::new(99.0, 99.0)];
    let mut state = constant_window(vec![Complex64::new(1.5, -2.0)]);
    state.opt_data.truncate(1);
    state.opt_data[0].energy = Complex64::new(-3.0, 7.0);
    state.opt_data[0].energy_squared = Complex64::new(9.0, -8.0);
    output_opt_data(&data, &state, Some(&dir)).unwrap();
    // avevar.c:106–109 ignores imaginary components and emits one terminal
    // newline, not a legacy Green-file blank line. Stored post-SR Para wins
    // over the current data snapshot (99+99i).
    assert_eq!(fs::read(dir.join("zqp_opt.dat")).unwrap(),
        b"-3.000000000000000000e+00  0.000000000000000000e+00  9.000000000000000000e+00  0.000000000000000000e+00  1.500000000000000000e+00  0.000000000000000000e+00 \n");
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    let before = fs::read(dir.join("zqp_opt.dat")).unwrap();
    state.opt_data[0].parameters.clear();
    assert_eq!(
        output_opt_data(&data, &state, Some(&dir))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
    assert_eq!(fs::read(dir.join("zqp_opt.dat")).unwrap(), before);
    fs::remove_dir_all(dir).unwrap();
}
