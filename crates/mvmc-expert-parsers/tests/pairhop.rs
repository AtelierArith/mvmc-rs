//! Julia's directed expansion, strict section errors, and initialization contract.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use mvmc_expert_parsers::parsers::pairhop::parse_pairhop_def;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pairhop")
}

struct InputDirectory(PathBuf);
impl InputDirectory {
    fn new() -> Self {
        for attempt in 0.. {
            let directory = std::env::temp_dir().join(format!(
                "issue184-pairhop-component-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&directory) {
                Ok(()) => return Self(directory),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive PairHop input: {error}"),
            }
        }
        unreachable!()
    }
}
impl Drop for InputDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// This archived helper body has seven permissive rows but declares count1.
// Build the unchanged wavefunction through the existing historical orbital
// helper without handing this PairHop payload to the C-facing file reader,
// then explicitly compose the public component result. No fixture rewrite.
fn archived_component_model() -> mvmc_expert_parsers::ExpertModeData {
    let directory = InputDirectory::new();
    let namelist = directory.0.join("namelist.def");
    let original = std::fs::read_to_string(root().join("namelist.def")).unwrap();
    let mut wavefunction = String::new();
    let mut pairhop_count = 0;
    for line in original.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 2);
        if fields[0] == "PairHop" {
            assert_eq!(fields[1], "parser_cases.def");
            pairhop_count += 1;
        } else {
            wavefunction.push_str(&format!(
                "{} {}\n",
                fields[0],
                root().join(fields[1]).display()
            ));
        }
    }
    assert_eq!(pairhop_count, 1);
    std::fs::write(&namelist, wavefunction).unwrap();
    let mut data = parse_expert_mode_files(&namelist).unwrap();
    assert!(data.pair_hop_terms.is_empty());
    let component = parse_pairhop_def(root().join("parser_cases.def")).unwrap();
    assert!(component.is_success());
    data.pair_hop_terms = component.terms;
    data
}

#[test]
fn archived_component_replacement_and_single_entry_loader_errors_are_separate() {
    let data = archived_component_model();
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
    let invalid =
        mvmc_expert_parsers::parse_expert_mode_files(directory.join("namelist.def")).unwrap_err();
    std::fs::remove_dir_all(directory).unwrap();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = invalid else {
        panic!("invalid PairHop must be InvalidInput: {invalid:?}");
    };
    assert!(
        message.contains("Error parsing required PairHop"),
        "{message}"
    );
    let missing = mvmc_expert_parsers::parse_expert_mode_files(root().join("namelist_missing.def"))
        .unwrap_err();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = missing else {
        panic!("missing PairHop must be InvalidInput: {missing:?}");
    };
    assert!(
        message.contains("Required PairHop file not found"),
        "{message}"
    );
}

#[test]
fn public_loader_rejects_archived_count_one_seven_row_component_input() {
    let content = std::fs::read_to_string(root().join("parser_cases.def")).unwrap();
    assert_eq!(content.lines().nth(1), Some("NPairHopp 1"));
    assert_eq!(content.lines().skip(5).count(), 7);
    // Minimal valid filename list isolates PairHop rejection from archived
    // orbital definitions. The original helper source is used verbatim.
    let directory = InputDirectory::new();
    let namelist = directory.0.join("namelist.def");
    std::fs::write(
        &namelist,
        format!(
            "ModPara {}\nPairHop {}\n",
            root().join("../interall/modpara.def").display(),
            root().join("parser_cases.def").display()
        ),
    )
    .unwrap();
    let error = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap_err();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = error else {
        panic!("unsupported archived PairHop file must be InvalidInput: {error:?}");
    };
    assert!(
        message.contains("Error parsing required PairHop"),
        "{message}"
    );
    assert!(message.contains("excess records"), "{message}");
}
