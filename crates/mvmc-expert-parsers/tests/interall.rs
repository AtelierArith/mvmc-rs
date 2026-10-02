//! Exact Julia parser contract; Hamiltonian support has a separate runtime gate.
use std::path::{Path, PathBuf};

use mvmc_expert_parsers::parsers::interall::{parse_interall_content, parse_interall_def};
use mvmc_expert_parsers::utils::parameter_init::{all_complex_flag, init_parameter};
use mvmc_expert_parsers::{parse_expert_mode_files, InterAllTerm};
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/interall")
}

fn indices(t: &InterAllTerm) -> [i64; 8] {
    [
        t.site0, t.spin0, t.site1, t.spin1, t.site2, t.spin2, t.site3, t.spin3,
    ]
}

#[test]
fn retains_all_four_pairs_coupling_bits_duplicates_and_input_order_like_julia() {
    let fixture = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for name in ["parser_cases", "raw", "kitaev", "replace"] {
        let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(header[0], name);
        let n: usize = header[1].parse().unwrap();
        let terms = parse_interall_def(root().join(format!("{name}.def"))).unwrap();
        assert_eq!(terms.len(), n, "{name}");
        for (i, term) in terms.iter().enumerate() {
            let row: Vec<_> = lines.next().unwrap().split_whitespace().collect();
            let expected: Vec<i64> = row[..8].iter().map(|s| s.parse().unwrap()).collect();
            assert_eq!(indices(term).as_slice(), expected, "{name} term {i}");
            assert_eq!(
                term.value.re.to_bits(),
                u64::from_str_radix(row[8], 16).unwrap()
            );
            assert_eq!(
                term.value.im.to_bits(),
                u64::from_str_radix(row[9], 16).unwrap()
            );
            assert_eq!(term.is_complex, row[10] == "1", "{name} term {i}");
        }
    }
    assert!(lines.next().is_none());
    assert!(parse_interall_content("# comment\n===\nNInterAll 1\n").is_empty());
    assert!(parse_interall_def(root().join("absent.def")).is_err());
}

#[test]
fn orchestration_replaces_repeated_sections_and_records_missing_files() {
    let data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(
        data.inter_all_terms,
        parse_interall_def(root().join("parser_cases.def")).unwrap()
    );
    let replaced = parse_expert_mode_files(root().join("namelist_replace.def")).unwrap();
    assert!(replaced.input_errors.is_empty());
    assert_eq!(
        replaced.inter_all_terms,
        parse_interall_def(root().join("replace.def")).unwrap()
    );
    let missing = parse_expert_mode_files(root().join("namelist_missing.def")).unwrap();
    assert!(missing.inter_all_terms.is_empty());
    assert_eq!(missing.input_errors.len(), 1);
    assert!(missing.input_errors[0].contains("InterAll file not found"));
}

#[test]
fn complex_hamiltonian_coefficients_leave_initialization_mode_values_and_rng_unchanged() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.inter_all_terms.iter().any(|term| term.is_complex));
    assert!(!all_complex_flag(&data));
    let flags: Vec<bool> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s == "1")
        .collect();
    assert_eq!(data.optimization_flags, flags);
    let original = data.inter_all_terms.clone();
    let mut plain = data.clone();
    plain.inter_all_terms.clear();
    let mut rng = Sfmt19937Rng::new(11272);
    let mut plain_rng = Sfmt19937Rng::new(11272);
    init_parameter(&mut data, &mut rng);
    init_parameter(&mut plain, &mut plain_rng);
    assert_eq!(data.inter_all_terms, original);
    assert_eq!(data.orbital_terms, plain.orbital_terms);
    assert_eq!(data.slater_params, plain.slater_params);
    let bits: Vec<u64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    let actual: Vec<u64> = data
        .orbital_terms
        .iter()
        .flat_map(|t| {
            [
                data.slater_params[t.idx as usize].re.to_bits(),
                data.slater_params[t.idx as usize].im.to_bits(),
            ]
        })
        .collect();
    assert_eq!(actual, bits);
    let words: Vec<u32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(words.len(), 624);
    for expected in words {
        assert_eq!(rng.gen_rand32(), expected);
        assert_eq!(plain_rng.gen_rand32(), expected);
    }
    assert!(lines.next().is_none());
}
