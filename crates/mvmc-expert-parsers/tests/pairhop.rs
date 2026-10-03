//! Julia's directed expansion, strict section errors, and initialization contract.
#[path = "../../../tests/support/historical_optimization_flags.rs"]
mod historical_optimization_flags;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
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
fn archived_component_replacement_and_single_entry_loader_errors_are_separate() {
    let data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(
        data.pair_hop_terms,
        parse_pairhop_def(root().join("parser_cases.def"))
            .unwrap()
            .terms
    );
    let replacement = parse_pairhop_def(root().join("replace.def")).unwrap().terms;
    // These archived multi-entry namelists describe Julia component sequences,
    // not supported C filename lists. Preserve the original sources but reject
    // them at the public loader; compose public component results explicitly.
    for name in ["namelist_replace.def", "namelist_invalid.def"] {
        let error = mvmc_expert_parsers::parse_expert_mode_files(root().join(name)).unwrap_err();
        let mvmc_expert_parsers::ParseError::InvalidInput { message } = error else {
            panic!("duplicate keyword must be InvalidInput: {error:?}");
        };
        assert!(message.contains("duplicate keyword PairHop"), "{message}");
    }
    let mut replaced = data.clone();
    let replacement_section = parse_pairhop_def(root().join("replace.def")).unwrap();
    assert!(replacement_section.is_success());
    replaced.pair_hop_terms = replacement_section.terms;
    assert!(replaced.input_errors.is_empty());
    assert_eq!(replaced.pair_hop_terms, replacement);
    let retained = replaced.clone();
    let invalid_section = parse_pairhop_def(root().join("invalid.def")).unwrap();
    assert!(!invalid_section.is_success());
    // This is explicitly test-side composition, not mutation by the component
    // API: an unsuccessful result is not assigned to the previous payload.
    assert_eq!(retained.pair_hop_terms, replacement);
    let directory = (0..)
        .find_map(|attempt| {
            let directory = std::env::temp_dir().join(format!(
                "issue184-pairhop-invalid-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&directory) {
                Ok(()) => Some(directory),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("exclusive invalid PairHop input: {error}"),
            }
        })
        .unwrap();
    std::fs::write(
        directory.join("namelist.def"),
        format!(
            "ModPara {}\nPairHop {}\nOrbital {}\n",
            root().join("../interall/modpara.def").display(),
            root().join("invalid.def").display(),
            root().join("../interall/orbital.def").display()
        ),
    )
    .unwrap();
    // Actual loader diagnostics, from a valid single-keyword filename list.
    let invalid = parse_expert_mode_files(directory.join("namelist.def")).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    assert!(invalid.pair_hop_terms.is_empty());
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
    let flags: Vec<i64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse::<i64>().unwrap())
        .collect();
    let flags = historical_optimization_flags::c_orbital_representation(&data, flags);
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
                data.slater_params[t.idx as usize].re,
                data.slater_params[t.idx as usize].im,
            ]
        })
        .collect();
    // Initialization arithmetic is portable within rounding error;
    // coupling literals, no-effect state equality and RNG words remain exact.
    numerical_comparison::assert_values_close(
        actual,
        bits.into_iter().map(f64::from_bits),
        1e-14,
        1e-14,
        "initialized Slater coefficients",
    );
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
