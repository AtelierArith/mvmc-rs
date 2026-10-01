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
