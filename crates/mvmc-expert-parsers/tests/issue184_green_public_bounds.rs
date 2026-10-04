//! M0625: C-defined sites/counts at the full parser boundary, not a C runtime oracle.
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData, GreenOneTerm, Spin};
use std::fs;
use std::path::PathBuf;

struct InputDirectory(PathBuf);
impl InputDirectory {
    fn new() -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("mvmc-green-public-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        fs::write(
            path.join("modpara.def"),
            "# header\nModel_Parameters 0\n# h3\n# h4\n# h5\nNsite 2\nNcond 2\n",
        )
        .unwrap();
        Self(path)
    }
    fn definition(&self, family: &str, count: usize, rows: &str) {
        fs::write(
            self.0.join(format!("{family}.def")),
            format!("# header\n{family} {count}\n# h3\n# h4\n# h5\n{rows}"),
        )
        .unwrap();
        fs::write(
            self.0.join("namelist.def"),
            format!("{family} {family}.def\nModPara modpara.def\n"),
        )
        .unwrap();
    }
    fn parse(&self) -> Result<ExpertModeData, mvmc_expert_parsers::ParseError> {
        parse_expert_mode_files(self.0.join("namelist.def"))
    }
}
impl Drop for InputDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively created test inputs");
    }
}

#[test]
fn original_m0625_invalid_site_is_rejected_by_public_parser_with_valid_control() {
    let input = InputDirectory::new();
    input.definition("OneBodyG", 1, "0 0 5 0\n");
    let error = input.parse().unwrap_err().to_string();
    assert!(
        error.contains("OneBodyG") && error.contains("site2 (5) out of range [0, 2)"),
        "{error}"
    );
    input.definition("OneBodyG", 1, "0 0 1 0\n");
    let data = input.parse().unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(
        data.green_one_terms,
        vec![GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up
        }]
    );
}

#[test]
fn public_green_loaders_check_each_site_and_preserve_valid_spin_labels() {
    let input = InputDirectory::new();
    for (family, fields) in [("OneBodyG", 4), ("TwoBodyG", 8), ("TwoBodyGEx", 8)] {
        let valid: Vec<i64> = (0..fields).map(|i| if i % 4 < 2 { 0 } else { 1 }).collect();
        let row = |values: &[i64]| {
            values
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(" ")
                + "\n"
        };
        input.definition(family, 1, &row(&valid));
        let data = input.parse().unwrap();
        assert!(data.input_errors.is_empty());
        match family {
            "OneBodyG" => assert_eq!(
                (data.green_one_terms[0].spin1, data.green_one_terms[0].spin2),
                (Spin::Up, Spin::Down)
            ),
            "TwoBodyG" => assert_eq!(
                (data.green_two_terms[0].spin1, data.green_two_terms[0].spin2),
                (Spin::Up, Spin::Down)
            ),
            _ => assert_eq!(
                (
                    data.green_two_ex_terms[0].spin3,
                    data.green_two_ex_terms[0].spin4
                ),
                (Spin::Down, Spin::Up)
            ),
        }
        for position in (0..fields).step_by(2) {
            for invalid in [-1, 2, 5] {
                let mut values = valid.clone();
                values[position] = invalid;
                input.definition(family, 1, &row(&values));
                let error = input.parse().unwrap_err().to_string();
                assert!(
                    error.contains(family) && error.contains("out of range [0, 2)"),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn public_green_loader_failures_are_atomic_and_zero_count_uses_c_early_return() {
    let input = InputDirectory::new();
    for family in ["OneBodyG", "TwoBodyG", "TwoBodyGEx"] {
        let row = if family == "OneBodyG" {
            "0 0 1 1\n"
        } else {
            "0 0 1 1 1 0 0 1\n"
        };
        input.definition(family, 2, row);
        assert!(input
            .parse()
            .unwrap_err()
            .to_string()
            .contains("declared row count 2, got 1"));
        input.definition(family, 1, "0 0 1\n");
        assert!(input
            .parse()
            .unwrap_err()
            .to_string()
            .contains("integer fields"));
        input.definition(family, 1, row);
        fs::remove_file(input.0.join(format!("{family}.def"))).unwrap();
        assert!(input.parse().unwrap_err().to_string().contains("Required"));
        input.definition(family, 0, "5 9 -1 9\n");
        let data = input.parse().unwrap();
        assert!(
            data.green_one_terms.is_empty()
                && data.green_two_terms.is_empty()
                && data.green_two_ex_terms.is_empty()
        );
    }
    input.definition("OneBodyG", 1, "0 0 1 0\n");
    fs::write(
        input.0.join("TwoBodyGEx.def"),
        "# header\nTwoBodyGEx 1\n# h3\n# h4\n# h5\n0 0 1 0 1 0 5 0\n",
    )
    .unwrap();
    fs::write(
        input.0.join("namelist.def"),
        "TwoBodyGEx TwoBodyGEx.def\nOneBodyG OneBodyG.def\nModPara modpara.def\n",
    )
    .unwrap();
    assert!(input
        .parse()
        .unwrap_err()
        .to_string()
        .contains("TwoBodyGEx"));
    fs::write(
        input.0.join("TwoBodyGEx.def"),
        "# header\nTwoBodyGEx 1\n# h3\n# h4\n# h5\n0 0 1 0 1 0 0 0\n",
    )
    .unwrap();
    let data = input.parse().unwrap();
    assert_eq!(data.green_two_ex_terms.len(), 1);
    assert!(data.input_errors.is_empty());
}

#[test]
fn constructed_green_data_validator_checks_c_site_bounds_without_mutation() {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.green_one_terms = vec![GreenOneTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 5,
        spin2: Spin::Up,
    }];
    let original = data.green_one_terms.clone();
    let result = mvmc_expert_parsers::utils::validation::validate_green_terms(&data);
    assert!(!result.is_valid);
    assert!(result
        .errors
        .iter()
        .any(|e| e.contains("OneBodyG") && e.contains("site2 (5)")));
    assert_eq!(data.green_one_terms, original);
    data.green_one_terms[0].site2 = 1;
    assert!(mvmc_expert_parsers::utils::validation::validate_green_terms(&data).is_valid);
}

#[test]
fn each_constructed_green_coordinate_routes_through_combined_validation() {
    use mvmc_expert_parsers::{GreenTwoExTerm, GreenTwoTerm};
    let mut valid = ExpertModeData::new();
    valid.modpara.nsite = 2;
    valid.green_one_terms = vec![GreenOneTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Down,
    }];
    valid.green_two_terms = vec![GreenTwoTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Down,
        site3: 1,
        spin3: Spin::Up,
        site4: 0,
        spin4: Spin::Down,
    }];
    valid.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Down,
        site3: 1,
        spin3: Spin::Up,
        site4: 0,
        spin4: Spin::Down,
    }];
    let checked = mvmc_expert_parsers::utils::validation::validate_green_terms(&valid);
    assert!(checked.is_valid && checked.errors.is_empty());
    let combined = mvmc_expert_parsers::validate_expert_mode_data(&valid);
    assert!(!combined.errors.iter().any(|e| e.contains("BodyG")));
    for family in ["OneBodyG", "TwoBodyG", "TwoBodyGEx"] {
        for coordinate in 0..if family == "OneBodyG" { 2 } else { 4 } {
            for invalid in [-1, 2, 5] {
                let mut data = valid.clone();
                let sites = match family {
                    "OneBodyG" => {
                        let t = &mut data.green_one_terms[0];
                        vec![&mut t.site1, &mut t.site2]
                    }
                    "TwoBodyG" => {
                        let t = &mut data.green_two_terms[0];
                        vec![&mut t.site1, &mut t.site2, &mut t.site3, &mut t.site4]
                    }
                    _ => {
                        let t = &mut data.green_two_ex_terms[0];
                        vec![&mut t.site1, &mut t.site2, &mut t.site3, &mut t.site4]
                    }
                };
                *sites.into_iter().nth(coordinate).unwrap() = invalid;
                let before = data.clone();
                let checked = mvmc_expert_parsers::utils::validation::validate_green_terms(&data);
                assert!(!checked.is_valid && checked.errors.len() == 1);
                let expected = format!("{family} term 1: site{} ({invalid})", coordinate + 1);
                assert!(
                    checked.errors[0].contains(&expected),
                    "{:?}",
                    checked.errors
                );
                let combined = mvmc_expert_parsers::validate_expert_mode_data(&data);
                assert!(combined.errors.iter().any(|e| e.contains(&expected)));
                assert_eq!(data.green_one_terms, before.green_one_terms);
                assert_eq!(data.green_two_terms, before.green_two_terms);
                assert_eq!(data.green_two_ex_terms, before.green_two_ex_terms);
            }
        }
    }
}
