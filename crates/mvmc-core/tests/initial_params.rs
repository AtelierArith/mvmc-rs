//! C complete records plus bounded, atomic diagnostics for malformed inputs.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::{read_initial_def, read_opt_para_file, ExpertModeData};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
use num_complex::Complex64;
use std::fs;

const RECORD: &str = "1 2 3 4 5 6 0.10 0 9.9 0.20 0 9.9 0.30 0 9.9 0.40 -0.10 9.9 0.50 -0.20 9.9";

fn data() -> ExpertModeData {
    let mut d = ExpertModeData::new();
    d.modpara.n_orbital_idx = 2;
    d.slater_params = vec![Complex64::new(7.0, 8.0); 2];
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
        .chain(
            d.orbital_terms
                .iter()
                .map(|t| d.slater_params[t.idx as usize]),
        )
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
fn incomplete_or_non_numeric_records_are_rejected_before_mutation() {
    let tokens: Vec<_> = RECORD.split_whitespace().collect();
    let mut cases = vec![
        tokens[..20].join(" "),
        format!("{RECORD} 0.123"),
        format!("{RECORD} 1 2 3"),
        format!("{RECORD} garbage"),
        format!("{RECORD} {RECORD} 0.123"),
        format!("{RECORD} {RECORD} garbage"),
    ];
    for position in [0, 7, 20] {
        let mut altered = tokens.clone();
        altered[position] = "garbled";
        cases.push(altered.join(" "));
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
    assert_eq!(read_opt_para_file(&mut empty, &path).unwrap(), 0);
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
            RECORD.replacen("9.9", "garbled", 1),
            "non-numeric token 'garbled' at field 9",
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
fn full_declared_slater_load_and_sync_match_c_without_consuming_rng() {
    use mvmc_core::sync::sync_modified_parameter_local;
    use sfmt19937::Sfmt19937Rng;
    let rows: Vec<_> = include_str!("../../../tests/fixtures/orbital_general/c_loaded.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 8);
    for record in rows.as_chunks::<4>().0.iter() {
        let complex = record[0] == "1";
        let mut payload = "0 0 0 0 0 0".to_owned();
        for index in 0..13 {
            let real = if index == 12 {
                8.0
            } else {
                (index + 1) as f64 / 16.0
            };
            let imag = if complex {
                -(index as f64 + 1.0) / 32.0
            } else {
                0.0
            };
            payload.push_str(&format!(" {real} {imag} 99"));
        }
        let path = file(&format!("c-full-{}", record[0]), &payload);
        for optional in [false, true] {
            let mut data = ExpertModeData::new();
            data.modpara.n_orbital_idx = 13;
            data.slater_params = vec![Complex64::new(99.0, 99.0); 13];
            data.orbital_terms = [0, 1, 1, 7, 8]
                .into_iter()
                .map(|idx| OrbitalTerm {
                    site1: 0,
                    site2: 1,
                    idx,
                    sign: 1,
                    is_complex: complex,
                })
                .collect();
            let mappings = data.orbital_terms.clone();
            let mut rng = Sfmt19937Rng::new(1);
            if optional {
                assert!(read_initial_def(&mut data, &path).unwrap());
            } else {
                assert_eq!(read_opt_para_file(&mut data, &path).unwrap(), 13);
            }
            for expected_row in [record[1], record[2]] {
                let expected: Vec<_> = expected_row
                    .split_whitespace()
                    .map(|s| u64::from_str_radix(s, 16).unwrap())
                    .collect();
                let actual: Vec<_> = data
                    .slater_params
                    .iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect();
                numerical_comparison::assert_values_close(
                    actual,
                    expected.into_iter().map(f64::from_bits),
                    16.0 * f64::EPSILON,
                    16.0 * f64::EPSILON,
                    format!("C mode={}, optional={optional}", record[0]),
                );
                assert_eq!(data.orbital_terms, mappings);
                sync_modified_parameter_local(&mut data, false);
            }
            let expected: Vec<_> = record[3]
                .split_whitespace()
                .map(|s| s.parse::<u32>().unwrap())
                .collect();
            assert_eq!(expected.len(), 624);
            assert_eq!(
                (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
                expected
            );
        }
        fs::remove_file(path).unwrap();
    }
}
