use mvmc_core::{qp, read_initial_def, read_opt_para_file, ExpertModeData};
use num_complex::Complex64;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans")
}

#[test]
fn full_record_loads_opttrans_after_all_other_factors() {
    let mut data = ExpertModeData::new();
    data.opt_trans = vec![Complex64::new(1.0, 0.0); 2];
    let path = std::env::temp_dir().join(format!("mvmc-opttrans-red-{}.def", std::process::id()));
    std::fs::write(&path, "1 2 3 4 5 6 0.7 -0.8 9.9 0.9 0.2 9.9").unwrap();
    let result = read_opt_para_file(&mut data, &path);
    std::fs::remove_file(path).unwrap();
    assert_eq!(result.unwrap(), 2);
    assert_eq!(
        data.opt_trans,
        vec![Complex64::new(0.7, -0.8), Complex64::new(0.9, 0.2)]
    );
}

#[test]
fn data_weight_initialization_and_refresh_include_opttrans_sectors() {
    let mut data = ExpertModeData::new();
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.nmp_trans = 2;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
    data.opt_trans = vec![Complex64::new(0.25, 0.0), Complex64::new(0.75, 0.0)];
    qp::init_qp_weight(&mut data);
    assert_eq!(data.qp_weights.as_ref().unwrap().qp_full_weight.len(), 4);
    data.opt_trans[1] = Complex64::new(0.5, -0.25);
    qp::update_qp_weight_for(&mut data);
    assert_eq!(
        data.qp_weights.unwrap().qp_full_weight[2],
        data.opt_trans[1]
    );
}

fn model(name: &str) -> ExpertModeData {
    let base = if matches!(name, "short_opt" | "long_opt" | "empty_opt") {
        "layout"
    } else {
        name
    };
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files(root().join(format!("namelist_{base}.def")))
            .unwrap();
    match name {
        "empty_opt" => data.opt_trans.clear(),
        "short_opt" => data.opt_trans.truncate(1),
        "long_opt" => data.opt_trans.push(Complex64::new(0.5, -0.25)),
        _ => {}
    }
    data
}

fn bits(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
    let actual: Vec<_> = values
        .into_iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect();
    let expected: Vec<_> = expected
        .split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect();
    assert_eq!(actual.len(), expected.len(), "{label}: component count");
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{label}: component {index}");
    }
}

fn values(data: &mut ExpertModeData) -> Vec<Complex64> {
    let mut values = data.projection_parameters();
    data.visit_rbm_terms_mut(|_, term| values.push(term.value()));
    values.extend(data.orbital_terms.iter().map(|t| t.value));
    values.extend(data.opt_trans.iter().copied());
    values
}

#[test]
fn full_record_loader_counts_errors_values_and_atomicity_match_julia() {
    let text = std::fs::read_to_string(root().join("loaders.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let mut data = model(fields[0]);
        let file = format!("load_{}_{}.def", fields[0], fields[1]);
        let path = root().join(&file);
        let (result, error) = match fields[2] {
            "initial" => (
                i64::from(read_initial_def(&mut data, &path).unwrap()),
                String::new(),
            ),
            "optimized" => match read_opt_para_file(&mut data, &path) {
                Ok(n) => (n as i64, String::new()),
                Err(e) => (-1, e.replace(path.to_str().unwrap(), &file)),
            },
            _ => panic!("unknown loader: {header}"),
        };
        assert_eq!(result, fields[3].parse::<i64>().unwrap(), "{header}");
        assert_eq!(error, lines.next().unwrap(), "{header}");
        bits(values(&mut data), lines.next().unwrap(), header);
        bits(data.para_qp_opt_trans, lines.next().unwrap(), header);
        cases += 1;
    }
    assert_eq!(cases, 132);
}

#[test]
fn initialized_updated_and_resized_sector_weights_match_julia_bits() {
    let text = std::fs::read_to_string(root().join("weights.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut data = ExpertModeData::new();
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        match fields[4] {
            "initial" => {
                data = ExpertModeData::new();
                data.modpara.nsp_gauss_leg = fields[0].parse().unwrap();
                data.modpara.nsp_stot = fields[1].parse().unwrap();
                data.modpara.nmp_trans = fields[2].parse().unwrap();
                data.para_qp_trans = vec![Complex64::new(1.0, 0.25), Complex64::new(-0.5, -0.125)];
                data.opt_trans = match fields[3] {
                    "empty" => vec![],
                    "one" => vec![Complex64::new(0.75, -0.0)],
                    "zero" => vec![Complex64::new(0.0, 0.0), Complex64::new(-0.0, -0.0)],
                    "complex" => vec![Complex64::new(0.25, -0.125), Complex64::new(-0.75, 0.5)],
                    _ => panic!("unknown mode: {header}"),
                };
                qp::init_qp_weight(&mut data);
            }
            "replace" => {
                data.opt_trans = vec![Complex64::new(0.5, 0.25), Complex64::new(-0.125, -0.75)]
            }
            "grow" => data.opt_trans.push(Complex64::new(1.5, -0.5)),
            "shrink" => data.opt_trans.truncate(1),
            "clear" => data.opt_trans.clear(),
            "repeat" => {}
            _ => panic!("unknown phase: {header}"),
        }
        if fields[4] != "initial" {
            qp::update_qp_weight_for(&mut data);
        }
        let weights = data.qp_weights.as_ref().unwrap();
        for values in [
            &weights.qp_full_weight,
            &weights.qp_fix_weight,
            &weights.spgl_cos,
            &weights.spgl_sin,
            &weights.spgl_cos_sin,
            &weights.spgl_cos_cos,
            &weights.spgl_sin_sin,
        ] {
            bits(values.iter().copied(), lines.next().unwrap(), header);
        }
        cases += 1;
    }
    assert_eq!(cases, 576);
}

fn projection_model(mode: &str, leg: i64, boundary: i64, mapping: &str) -> ExpertModeData {
    use mvmc_expert_parsers::{OrbitalTerm, QPTransEntry};
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.nelec = 1;
    data.modpara.nmp_trans = 2 * boundary;
    data.modpara.nsp_gauss_leg = leg;
    data.i_flg_orbital_general = i64::from(mode == "fsz");
    data.n_qp_trans = 2;
    data.n_qp_opt_trans = 3;
    let dim = if mode == "fsz" { 8 } else { 4 };
    data.modpara.n_orbital_idx = dim * dim;
    for i in 0..dim {
        for j in 0..dim {
            let idx = i * dim + j;
            data.orbital_terms.push(OrbitalTerm {
                site1: i,
                site2: j,
                idx,
                value: Complex64::new(
                    (idx + 1) as f64 / 16.0,
                    if mode == "real" {
                        0.0
                    } else {
                        (-idx) as f64 / 32.0
                    },
                ),
                is_complex: mode != "real",
                sign: if (i + j) % 2 == 1 { -1 } else { 1 },
            });
        }
    }
    data.ensure_orbital_idx_matrix();
    data.qp_trans_entries = vec![
        QPTransEntry {
            weight: Complex64::new(1.0, 0.25),
            site_map: vec![0, 1, 2, 3],
            site_sign: if boundary == -1 {
                vec![1, -1, -1, 1]
            } else {
                vec![1; 4]
            },
        },
        QPTransEntry {
            weight: Complex64::new(-0.5, -0.125),
            site_map: vec![3, 2, 1, 0],
            site_sign: if boundary == -1 {
                vec![-1, 1, -1, 1]
            } else {
                vec![1; 4]
            },
        },
    ];
    data.para_qp_trans = data.qp_trans_entries.iter().map(|t| t.weight).collect();
    data.qp_opt_trans = vec![vec![0, 1, 2, 3], vec![1, 2, 3, 0], vec![3, 2, 1, 0]];
    data.qp_opt_trans_sgn = if boundary == -1 {
        vec![vec![1; 4], vec![-1, 1, -1, 1], vec![1, -1, -1, 1]]
    } else {
        vec![vec![1; 4]; 3]
    };
    data.opt_trans = vec![
        Complex64::new(0.25, -0.125),
        Complex64::new(-0.75, 0.5),
        Complex64::new(0.5, 0.25),
    ];
    match mapping {
        "absent" => {
            data.qp_opt_trans.clear();
            data.qp_opt_trans_sgn.clear();
        }
        "short_maps" => {
            data.qp_opt_trans.truncate(1);
            data.qp_opt_trans[0] = vec![2, 3, 0, 1];
        }
        "short_signs" => {
            data.qp_opt_trans_sgn.truncate(1);
            data.qp_opt_trans_sgn[0] = vec![-1, 1, 1, -1];
        }
        "missing_maps" => data.qp_opt_trans.clear(),
        "missing_signs" => data.qp_opt_trans_sgn.clear(),
        "complete" => {}
        _ => panic!("unknown mapping {mapping}"),
    }
    qp::init_qp_weight(&mut data);
    data
}

#[test]
fn nonidentity_slater_matrices_and_derivatives_match_canonical_julia() {
    let text = std::fs::read_to_string(root().join("projection.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut data = ExpertModeData::new();
    let mut state = mvmc_core::VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        if fields[4] == "initial" {
            data = projection_model(
                fields[0],
                fields[1].parse().unwrap(),
                fields[2].parse().unwrap(),
                fields[3],
            );
            let nq = data.qp_weights.as_ref().unwrap().qp_full_weight.len();
            state = mvmc_core::VmcOptimizationState::zeros(
                4,
                1,
                0,
                data.modpara.n_orbital_idx as usize + 3,
                nq,
                1,
                fields[0] != "real",
                fields[0] == "fsz",
            );
        } else {
            for term in &mut data.orbital_terms {
                term.value *= 0.5;
            }
            data.opt_trans = vec![
                Complex64::new(0.5, 0.25),
                Complex64::new(-0.125, -0.75),
                Complex64::new(1.5, -0.5),
            ];
            qp::update_qp_weight_for(&mut data);
        }
        if fields[0] == "fsz" {
            mvmc_core::slater_update::update_slater_elm_fsz(&mut data, &mut state);
        } else {
            mvmc_core::slater_update::update_slater_elm(&mut data, &mut state);
        }
        bits(
            state.slater_matrix.slater_elm.as_slice().iter().copied(),
            lines.next().unwrap(),
            header,
        );
        for (q, pf) in state.slater_matrix.pf_m.iter_mut().enumerate() {
            *pf = Complex64::new((q + 1) as f64 / 8.0, (2.0 - q as f64) / 16.0);
            let z = Complex64::new((q + 3) as f64 / 16.0, (1.0 - q as f64) / 32.0);
            state
                .slater_matrix
                .inv_m
                .qp_matrix_slice_mut(q)
                .copy_from_slice(&[Complex64::new(0.0, 0.0), z, -z, Complex64::new(0.0, 0.0)]);
        }
        let mut sr = vec![Complex64::new(7.0, -9.0); 2 * data.modpara.n_orbital_idx as usize];
        let ip = Complex64::new(1.25, -0.75);
        if fields[0] == "fsz" {
            mvmc_core::observables::slater_elm_diff_fsz(
                &mut sr,
                ip,
                &[0, 3],
                &[0, 1],
                &data,
                &state,
            );
        } else {
            mvmc_core::observables::slater_elm_diff(&mut sr, ip, &[0, 3], &data, &state);
        }
        bits(sr, lines.next().unwrap(), header);
        let mut opt = vec![Complex64::new(7.0, -9.0); 6];
        mvmc_core::observables::opt_trans_diff(&mut opt, ip, &data, &state.slater_matrix.pf_m);
        bits(opt, lines.next().unwrap(), header);
        cases += 1;
    }
    assert_eq!(cases, 120);
}

#[test]
fn opttrans_derivative_bounds_empty_inputs_and_partial_pfaffians_match_julia() {
    let text = std::fs::read_to_string(root().join("opt_derivatives.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<i64> = header
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let mut data = ExpertModeData::new();
        data.opt_trans = vec![Complex64::new(1.0, 0.0); fields[0] as usize];
        if fields[1] >= 0 {
            let mut weights = mvmc_expert_parsers::QuantumProjectionWeights::new();
            if fields[1] > 0 {
                weights.qp_fix_weight =
                    vec![Complex64::new(1.0, 0.25), Complex64::new(-0.5, -0.125)];
            }
            data.qp_weights = Some(weights);
        }
        let mut state = mvmc_core::VmcOptimizationState::zeros(4, 1, 0, 3, 6, 1, true, false);
        state.slater_matrix.pf_m = (0..fields[2])
            .map(|q| Complex64::new((q + 1) as f64 / 8.0, (2 - q) as f64 / 16.0))
            .collect();
        let mut opt = vec![Complex64::new(7.0, -9.0); fields[3] as usize];
        mvmc_core::observables::opt_trans_diff(
            &mut opt,
            Complex64::new(1.25, -0.75),
            &data,
            &state.slater_matrix.pf_m,
        );
        bits(opt, lines.next().unwrap(), header);
        cases += 1;
    }
    assert_eq!(cases, 162);
}

#[test]
fn slater_amplitude_cutoff_and_duplicate_values_match_canonical_julia() {
    let text = std::fs::read_to_string(root().join("slater_threshold.txt")).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let f: Vec<_> = header.split_whitespace().collect();
        let mut data = projection_model(f[0], 1, f[1].parse().unwrap(), "complete");
        let cutoff = 1e-14_f64;
        let value = match f[2] {
            "negative_zero" => Complex64::new(-0.0, -0.0),
            "below" => Complex64::new(f64::from_bits(cutoff.to_bits() - 1), 0.0),
            "equal" => Complex64::new(cutoff, 0.0),
            "above" => Complex64::new(f64::from_bits(cutoff.to_bits() + 1), 0.0),
            "complex_below" => Complex64::new(1e-15, -1e-15),
            "complex_above" => Complex64::new(1e-14, -1e-14),
            "duplicate_zero" | "duplicate_tiny" => Complex64::new(2e-14, 0.0),
            "zero" => Complex64::new(0.0, 0.0),
            _ => panic!("{header}"),
        };
        for t in &mut data.orbital_terms {
            t.value = value;
        }
        if f[2].starts_with("duplicate") {
            let mut term = data.orbital_terms[1];
            term.value = if f[2] == "duplicate_zero" {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(1e-15, 0.0)
            };
            data.orbital_terms.push(term);
        }
        let mut state = mvmc_core::VmcOptimizationState::zeros(
            4,
            1,
            0,
            data.modpara.n_orbital_idx as usize + 3,
            6,
            1,
            f[0] != "real",
            f[0] == "fsz",
        );
        if f[0] == "fsz" {
            mvmc_core::slater_update::update_slater_elm_fsz(&mut data, &mut state);
        } else {
            mvmc_core::slater_update::update_slater_elm(&mut data, &mut state);
        }
        bits(
            state.slater_matrix.slater_elm.as_slice().iter().copied(),
            lines.next().unwrap(),
            header,
        );
        cases += 1;
    }
    assert_eq!(cases, 54);
}
