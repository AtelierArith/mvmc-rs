use mvmc_expert_parsers::parsers::green::parse_green_two_ex_content;
use mvmc_expert_parsers::{GreenTwoExTerm, Spin};

fn header(count: usize) -> String {
    format!("# header\nTwoBodyGEx {count}\n# h3\n# h4\n# h5\n")
}

#[test]
fn parses_canonical_second_pair_reordering() {
    let content = format!("{}1 0 2 1 3 0 4 1\n", header(1));
    assert_eq!(
        parse_green_two_ex_content(&content).unwrap(),
        vec![GreenTwoExTerm {
            site1: 1,
            spin1: Spin::Up,
            site2: 2,
            spin2: Spin::Down,
            site3: 4,
            spin3: Spin::Down,
            site4: 3,
            spin4: Spin::Up,
        }]
    );
}

#[test]
fn rejects_count_mismatch_and_malformed_rows() {
    let mismatch = format!("{}1 0 2 1 3 0 4 1\n", header(2));
    assert!(parse_green_two_ex_content(&mismatch)
        .unwrap_err()
        .contains("header count 2"));

    let malformed = format!("{}1 0 2 1 3 0 4\n", header(1));
    assert!(parse_green_two_ex_content(&malformed)
        .unwrap_err()
        .contains("exactly 8 integer fields"));
}

#[test]
fn rejects_negative_sites_and_invalid_spins() {
    let negative = format!("{}-1 0 2 1 3 0 4 1\n", header(1));
    assert!(parse_green_two_ex_content(&negative)
        .unwrap_err()
        .contains("negative site"));
    let spin = format!("{}1 2 2 1 3 0 4 1\n", header(1));
    assert!(parse_green_two_ex_content(&spin)
        .unwrap_err()
        .contains("spin must be 0 or 1"));
}

#[test]
fn canonicalization_preserves_existing_duplicates_and_appends_missing_terms() {
    let first = mvmc_expert_parsers::GreenOneTerm {
        site1: 1,
        spin1: Spin::Up,
        site2: 2,
        spin2: Spin::Down,
    };
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.green_one_terms = vec![first, first];
    data.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 1,
        spin1: Spin::Up,
        site2: 2,
        spin2: Spin::Down,
        site3: 4,
        spin3: Spin::Down,
        site4: 3,
        spin4: Spin::Up,
    }];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms.len(), 3);
    assert_eq!(data.green_two_ex_indices, vec![(0, 2)]);
}

#[test]
fn repeated_constituents_keep_each_factored_reference_and_empty_sections_are_noops() {
    let term = GreenTwoExTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Down,
        site3: 0,
        spin3: Spin::Up,
        site4: 1,
        spin4: Spin::Down,
    };
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.green_two_ex_terms = vec![term, term];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms.len(), 1);
    assert_eq!(data.green_two_ex_indices, vec![(0, 0), (0, 0)]);

    let mut empty = mvmc_expert_parsers::ExpertModeData::new();
    empty.canonicalize_green_two_ex();
    assert!(empty.green_one_terms.is_empty());
    assert!(empty.green_two_ex_indices.is_empty());
}

#[test]
fn malformed_referenced_file_is_reported_by_expert_loader() {
    let dir = std::env::temp_dir().join(format!("mvmc-green-two-ex-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("modpara.def"), "Nsite 4\nNElec 1\nNMPTrans 1\n").unwrap();
    std::fs::write(
        dir.join("bad.def"),
        "# header\nTwoBodyGEx 1\n# h3\n# h4\n# h5\n0 0 1\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("namelist.def"),
        "ModPara modpara.def\nTwoBodyGEx bad.def\n",
    )
    .unwrap();
    let data = mvmc_expert_parsers::parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    assert!(data
        .input_errors
        .iter()
        .any(|error| error.contains("error parsing TwoBodyGEx file")));
    let _ = std::fs::remove_dir_all(dir);
}
