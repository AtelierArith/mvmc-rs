//! Julia's directed expansion, strict section errors, and initialization contract.
use std::path::{Path, PathBuf};

use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::parsers::pairhop::{parse_pairhop_content, parse_pairhop_def};
use mvmc_expert_parsers::utils::parameter_init::{all_complex_flag, init_parameter};
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pairhop")
}

#[test]
fn expands_each_input_row_in_order_and_retains_partial_results_and_julia_errors() {
    let fixture = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for name in ["parser_cases", "raw", "replace", "invalid"] {
        let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(header[0], name);
        let result = parse_pairhop_def(root().join(format!("{name}.def"))).unwrap();
        assert_eq!(result.is_success(), header[1] == "1", "{name}");
        assert_eq!(result.terms.len(), header[2].parse::<usize>().unwrap());
        for term in result.terms {
            let row: Vec<_> = lines.next().unwrap().split_whitespace().collect();
            assert_eq!(term.site1, row[0].parse::<i64>().unwrap());
            assert_eq!(term.site2, row[1].parse::<i64>().unwrap());
            assert_eq!(
                term.value.to_bits(),
                u64::from_str_radix(row[2], 16).unwrap()
            );
        }
        assert_eq!(result.errors.join("; "), lines.next().unwrap(), "{name}");
    }
    assert!(lines.next().is_none());
    let empty = parse_pairhop_content("# comment\n// comment\n");
    assert!(empty.is_success() && empty.terms.is_empty());
    assert!(parse_pairhop_def(root().join("absent.def")).is_err());
}

#[test]
fn replaces_successful_sections_and_retains_previous_payload_when_a_later_section_fails() {
    let data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(
        data.pair_hop_terms,
        parse_pairhop_def(root().join("parser_cases.def"))
            .unwrap()
            .terms
    );
    let replacement = parse_pairhop_def(root().join("replace.def")).unwrap().terms;
    let replaced = parse_expert_mode_files(root().join("namelist_replace.def")).unwrap();
    assert!(replaced.input_errors.is_empty());
    assert_eq!(replaced.pair_hop_terms, replacement);
    let invalid = parse_expert_mode_files(root().join("namelist_invalid.def")).unwrap();
    assert_eq!(invalid.pair_hop_terms, replacement);
    assert_eq!(invalid.input_errors.len(), 1);
    assert!(invalid.input_errors[0].contains("error parsing PairHop"));
    assert!(invalid.input_errors[0].contains("Line 6: Site1 number must be non-negative"));
    let missing = parse_expert_mode_files(root().join("namelist_missing.def")).unwrap();
    assert!(missing.pair_hop_terms.is_empty());
    assert_eq!(missing.input_errors.len(), 1);
    assert!(missing.input_errors[0].contains("PairHop file not found"));
}

#[test]
fn pairhop_retains_wavefunction_flags_initialized_values_and_the_next_rng_block() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(!all_complex_flag(&data));
    let flags: Vec<bool> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s == "1")
        .collect();
    assert_eq!(data.optimization_flags, flags);
    let original = data.pair_hop_terms.clone();
    let mut plain = data.clone();
    plain.pair_hop_terms.clear();
    let mut rng = Sfmt19937Rng::new(11272);
    let mut plain_rng = Sfmt19937Rng::new(11272);
    init_parameter(&mut data, &mut rng);
    init_parameter(&mut plain, &mut plain_rng);
    assert_eq!(data.pair_hop_terms, original);
    assert_eq!(data.orbital_terms, plain.orbital_terms);
    assert_eq!(data.slater_params, plain.slater_params);
    let bits: Vec<u64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    let actual: Vec<_> = data
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
