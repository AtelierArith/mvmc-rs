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
