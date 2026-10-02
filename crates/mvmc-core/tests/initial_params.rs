//! Port of Julia's test_unit_read_opt_para.jl: strict, atomic triples loading.
use mvmc_core::{read_initial_def, read_opt_para_file, ExpertModeData};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
use num_complex::Complex64;
use std::fs;

const RECORD: &str = "1 2 3 4 5 6 0.10 0 9.9 0.20 0 9.9 0.30 0 9.9 0.40 -0.10 9.9 0.50 -0.20 9.9";

fn data() -> ExpertModeData {
    let mut d = ExpertModeData::new();
    for site in 0..2 {
        d.gutzwiller_terms.push(GutzwillerTerm {
            site,
            value: Complex64::new(7.0, 8.0),
            is_complex: true,
        });
    }
    d.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(7.0, 8.0),
        is_complex: true,
    });
    // Repeated mappings must receive the same value, not consume extra triples.
    for idx in [0, 1, 0] {
        d.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx,
            value: Complex64::new(7.0, 8.0),
            is_complex: true,
            sign: 1,
        });
    }
    d
}

fn values(d: &ExpertModeData) -> Vec<Complex64> {
    d.gutzwiller_terms
        .iter()
        .map(|t| t.value)
        .chain(d.jastrow_terms.iter().map(|t| t.value))
        .chain(d.orbital_terms.iter().map(|t| t.value))
        .collect()
}

fn file(name: &str, text: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("mvmc-loader-{}-{name}.def", std::process::id()));
    fs::write(&path, text).unwrap();
    path
}

#[test]
fn strict_loader_consumes_unique_parameters_verbatim_and_is_idempotent() {
    let path = file("valid", RECORD);
    let mut d = data();
    assert_eq!(read_opt_para_file(&mut d, &path).unwrap(), 5);
    let expected = vec![
        Complex64::new(0.1, 0.0),
        Complex64::new(0.2, 0.0),
        Complex64::new(0.3, 0.0),
        Complex64::new(0.4, -0.1),
        Complex64::new(0.5, -0.2),
        Complex64::new(0.4, -0.1),
    ];
    assert_eq!(values(&d), expected);
    assert_eq!(read_opt_para_file(&mut d, &path).unwrap(), 5);
    assert!(read_initial_def(&mut d, &path).unwrap());
    assert_eq!(values(&d), expected);
    fs::remove_file(path).unwrap();
}

#[test]
fn every_malformed_record_is_rejected_before_mutation() {
    let tokens: Vec<_> = RECORD.split_whitespace().collect();
    let mut cases = vec![
        String::new(),
        tokens[..20].join(" "),
        format!("{RECORD} 0.123"),
        format!("{RECORD} 1 2 3"),
        format!("{RECORD} garbage"),
    ];
    for position in [0, 7, 20] {
        for bad in ["garbled", "NaN", "Inf", "-Inf"] {
            let mut altered = tokens.clone();
            altered[position] = bad;
            cases.push(altered.join(" "));
        }
    }
    for (index, text) in cases.iter().enumerate() {
        let path = file(&format!("bad-{index}"), text);
        let mut d = data();
        let before = values(&d);
        assert!(read_opt_para_file(&mut d, &path).is_err(), "{text}");
        assert_eq!(values(&d), before);
        assert!(!read_initial_def(&mut d, &path).unwrap(), "{text}");
        assert_eq!(values(&d), before);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn missing_file_and_empty_parameter_model_have_distinct_contracts() {
    let path = std::env::temp_dir().join(format!("mvmc-loader-missing-{}.def", std::process::id()));
    let mut d = data();
    assert!(!read_initial_def(&mut d, &path).unwrap());
    assert!(read_opt_para_file(&mut d, &path)
        .unwrap_err()
        .contains("file not found"));
    let path = file("zero", "1 2 3 4 5 6");
    let mut empty = ExpertModeData::new();
    assert!(read_initial_def(&mut empty, &path).unwrap());
    assert!(read_opt_para_file(&mut empty, &path)
        .unwrap_err()
        .contains("no parameters consumed"));
    fs::remove_file(path).unwrap();
}

#[test]
fn declared_sparse_blocks_and_reserved_orbitals_keep_their_offsets() {
    let mut d = data();
    d.n_gutzwiller_idx = 3;
    d.n_jastrow_idx = 2;
    d.modpara.n_orbital_idx = 4;
    let mut record = "1 2 3 4 5 6".to_owned();
    for slot in 0..9 {
        record.push_str(&format!(" {} {} 99", slot + 10, -(slot + 10)));
    }
    let path = file("sparse", &record);
    assert_eq!(read_opt_para_file(&mut d, &path).unwrap(), 9);
    assert_eq!(
        values(&d),
        [
            Complex64::new(10.0, -10.0),
            Complex64::new(11.0, -11.0),
            Complex64::new(13.0, -13.0),
            Complex64::new(15.0, -15.0),
            Complex64::new(16.0, -16.0),
            Complex64::new(15.0, -15.0)
        ]
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn errors_report_the_first_bad_field_even_in_unused_diagnostics_and_gradients() {
    for (name, record, reason) in [
        (
            "diagnostic",
            RECORD.replacen('1', "garbled", 1),
            "non-numeric token 'garbled' at field 1",
        ),
        (
            "gradient",
            RECORD.replacen("9.9", "Inf", 1),
            "non-finite token 'Inf' at field 9",
        ),
        (
            "tail",
            format!("{RECORD} junk"),
            "non-numeric token 'junk' at field 22",
        ),
        (
            "extra",
            format!("{RECORD} 1 2 3"),
            "OptTrans-style block of 1 triples but OptTrans is not active",
        ),
    ] {
        let path = file(name, &record);
        let mut d = data();
        let before = values(&d);
        assert!(read_opt_para_file(&mut d, &path)
            .unwrap_err()
            .contains(reason));
        assert_eq!(values(&d), before);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn parsed_fixed_correlations_and_rng_match_three_canonical_sr_sync_steps() {
    use mvmc_core::{sr, sync::sync_modified_parameter_local, VmcOptimizationState};
    use mvmc_expert_parsers::{parse_expert_mode_files, utils::parameter_init::init_parameter};
    use sfmt19937::Sfmt19937Rng;
    let dir = std::env::temp_dir().join(format!("mvmc-fixed-sr-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let definition = |name: &str, width: usize, complex: usize, rows: &str| {
        format!("===\n{name} {width}\nComplexType {complex}\n===\n===\n{rows}")
    };
    fs::write(dir.join("modpara.def"), "Nsite 3\nNElec 1\n").unwrap();
    fs::write(
        dir.join("g.def"),
        definition("NGutzwillerIdx", 2, 0, "0 0\n1 0\n2 1\n0 1\n1 0\n"),
    )
    .unwrap();
    fs::write(
        dir.join("o.def"),
        definition("NOrbitalIdx", 2, 0, "0 1 0\n1 0 1\n0 0\n1 1\n"),
    )
    .unwrap();
    fs::write(
        dir.join("namelist.def"),
        "ModPara modpara.def\nOrbitalAntiParallel o.def\nGutzwiller g.def\nJastrow j.def\n",
    )
    .unwrap();
    let text = include_str!("../../../tests/fixtures/sr_failure/fixed_flag_steps.txt");
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    for complex_jastrow in [0, 1] {
        fs::write(
            dir.join("j.def"),
            definition(
                "NJastrowIdx",
                2,
                complex_jastrow,
                "0 1 0\n1 2 1\n0 0\n1 1\n",
            ),
        )
        .unwrap();
        let mut data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        let header = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            header,
            [complex_jastrow, data.count_variational_parameters()]
        );
        let flags = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s == "1")
            .collect::<Vec<_>>();
        assert_eq!(data.optimization_flags, flags);
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng);
        for (i, t) in data.gutzwiller_terms.iter_mut().enumerate() {
            t.value = Complex64::new((i + 1) as f64, 0.2);
        }
        for (i, t) in data.jastrow_terms.iter_mut().enumerate() {
            t.value = Complex64::new((i + 3) as f64, 0.5);
        }
        data.orbital_terms[0].value = Complex64::new(1.0, 0.0);
        data.orbital_terms[1].value = Complex64::new(2.0, 0.0);
        for _ in 0..3 {
            sync_modified_parameter_local(&mut data, true);
        }
        data.modpara.dsr_opt_sta_del = 0.0;
        data.modpara.dsr_opt_step_dt = 0.125;
        let n = data.count_variational_parameters();
        let complex = complex_jastrow != 0;
        data.complex_flags = vec![i64::from(complex)];
        let off = if complex { 2 } else { 1 };
        let size = off * (n + 1);
        for step in 1..=3 {
            let mut state = VmcOptimizationState::zeros(3, 1, 4, n, 1, 1, complex, false);
            state.energy.wc = Complex64::new(1.0, 0.0);
            for component in off..size {
                let covariance = 1.0 + step as f64 / 4.0;
                let gradient = (component + 1) as f64 / 16.0;
                if complex {
                    state.sr_opt.sr_opt_oo[component * size + component] =
                        Complex64::new(covariance, 0.0);
                    state.sr_opt.sr_opt_ho[component] = Complex64::new(gradient, 0.0);
                } else {
                    state.sr_opt.sr_opt_oo_real[component * size + component] = covariance;
                    state.sr_opt.sr_opt_ho_real[component] = gradient;
                }
            }
            let status = if complex {
                sr::stochastic_opt_complex(&mut data, &mut state)
            } else {
                sr::stochastic_opt_real(&mut data, &mut state)
            };
            assert_eq!(status, 0);
            sync_modified_parameter_local(&mut data, true);
            assert_eq!(data.gutzwiller_terms[1].value, Complex64::new(2.0, 0.2));
            assert_eq!(data.jastrow_terms[0].value, Complex64::new(3.0, 0.5));
            let expected = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect::<Vec<_>>();
            let actual = values(&data)
                .into_iter()
                .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "complex={complex}, step={step}");
            assert_eq!(data.optimization_flags, flags);
        }
        let expected = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse::<u32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
            expected
        );
    }
    assert!(lines.next().is_none());
    fs::remove_dir_all(dir).unwrap();
}
