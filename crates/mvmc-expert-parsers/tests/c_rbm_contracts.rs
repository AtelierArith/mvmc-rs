//! Actual-C RBM reader fixtures; no toolbox or native compiler dependency.
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use std::{fs, path::PathBuf};

const CONTRACTS: &str = include_str!("../../../tests/fixtures/rbm/c_reader_contracts.txt");
const NAMES: [&str; 9] = mvmc_expert_parsers::parsers::rbm::SECTION_NAMES;

struct Input {
    dir: PathBuf,
}

impl Input {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("mvmc-c-rbm-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn parse(&self, record: &[&str]) -> ExpertModeData {
        let h: Vec<_> = record[0].split_whitespace().collect();
        let section: usize = h[1].parse().unwrap();
        let nsite: usize = h[2].parse().unwrap();
        fs::write(
            self.dir.join("namelist.def"),
            format!(
                "{} rbm.def\nGutzwiller g.def\nModPara modpara.def\n",
                NAMES[section]
            ),
        )
        .unwrap();
        fs::write(
            self.dir.join("modpara.def"),
            format!("Nsite {nsite}\nNElec 1\nNMPTrans -1\nNneuronCharge {}\nNneuronSpin {}\nNneuronGeneral {}\n", h[3], h[3], h[3]),
        )
        .unwrap();
        let maps: String = (0..nsite).map(|site| format!("{site} 0\n")).collect();
        fs::write(
            self.dir.join("g.def"),
            format!("===\nWidth 2\nComplex 0\n===\n===\n{maps}99 3\n99 -2\n"),
        )
        .unwrap();
        fs::write(self.dir.join("rbm.def"), record[1].replace('|', "\n")).unwrap();
        parse_expert_mode_files(self.dir.join("namelist.def")).unwrap()
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn records() -> Vec<&'static str> {
    CONTRACTS.lines().filter(|l| !l.starts_with('#')).collect()
}

#[test]
fn rbm_declarations_reserve_unused_slots_in_all_nine_sections() {
    let input = Input::new("widths");
    let records = records();
    let mut checked = 0;
    for record in records.as_chunks::<4>().0.iter() {
        let h: Vec<_> = record[0].split_whitespace().collect();
        if h[8] != "0" {
            continue;
        }
        let data = input.parse(record);
        assert!(
            data.input_errors.is_empty(),
            "{}: {:?}",
            h[0],
            data.input_errors
        );
        let section: usize = h[1].parse().unwrap();
        let mut widths = [0; 9];
        widths[section] = h[5].parse().unwrap();
        assert_eq!(data.rbm_section_sizes(), widths, "{}", h[0]);
        assert_eq!(data.rbm_parameters().len(), widths[section], "{}", h[0]);
        checked += 1;
    }
    assert!(checked > 700);
}

#[test]
fn rbm_raw_flags_follow_pair_order_with_nonzero_projection_prefix() {
    let input = Input::new("flags");
    let records = records();
    for record in records.as_chunks::<4>().0.iter() {
        let h: Vec<_> = record[0].split_whitespace().collect();
        if h[8] != "0" {
            continue;
        }
        let data = input.parse(record);
        let expected: Vec<i64> = record[3]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(data.optimization_flags, expected, "{}", h[0]);
    }
}

#[test]
fn rbm_requires_complete_c_mapping_and_declared_flag_counts() {
    let input = Input::new("acceptance");
    let records = records();
    let mut failures = Vec::new();
    for record in records.as_chunks::<4>().0.iter() {
        let h: Vec<_> = record[0].split_whitespace().collect();
        let expected = h[8] == "0";
        let data = input.parse(record);
        if data.input_errors.is_empty() != expected {
            failures.push(format!(
                "{}: Rust={}, C={expected}",
                h[0],
                data.input_errors.is_empty()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn final_rbm_coordinate_assignments_match_all_nine_native_c_arrays() {
    use mvmc_expert_parsers::parsers::rbm::parse_rbm_content;
    let records = records();
    let mut checked = 0;
    for record in records.as_chunks::<4>().0.iter() {
        let h: Vec<_> = record[0].split_whitespace().collect();
        if h[8] != "0" {
            continue;
        }
        let section: usize = h[1].parse().unwrap();
        let sites: usize = h[2].parse().unwrap();
        let hidden: usize = h[3].parse().unwrap();
        let parsed = parse_rbm_content(
            &record[1].replace('|', "\n"),
            section,
            sites as i64,
            hidden as i64,
        )
        .unwrap();
        let expected: Vec<i64> = record[2]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let mut actual = vec![-1; expected.len()];
        for row in parsed.mappings {
            let slot = match section {
                0 | 1 | 3..=5 => row[0] as usize,
                2 => row[0] as usize + row[1] as usize * sites,
                6 | 7 => row[0] as usize * hidden + row[1] as usize,
                8 => (row[0] as usize + row[1] as usize * sites) * hidden + row[2] as usize,
                _ => unreachable!(),
            };
            actual[slot] = *row.last().unwrap();
        }
        assert_eq!(actual, expected, "{} final coordinate assignments", h[0]);
        checked += 1;
    }
    assert_eq!(checked, 766);
}

#[test]
fn rbm_unsafe_dimensions_indices_and_scans_receive_bounded_rust_diagnostics() {
    use mvmc_expert_parsers::parsers::rbm::parse_rbm_content;
    for maps in [
        "-1 0 1 0",
        "2 0 1 0",
        "0 -1 1 0",
        "0 1 1 0",
        "0 x 1 0",
        "0 0 1 2147483648",
    ] {
        let payload = format!("===\nWidth 1\nComplexType 0\n===\n===\n{maps}\n0 1\n");
        assert!(parse_rbm_content(&payload, 0, 2, 1).is_err(), "{maps}");
    }
    let payload = "===\nWidth 1\nComplexType 0\n===\n===\n0 0\n0 1\n";
    for sites in [0, -1, i64::MAX] {
        assert!(parse_rbm_content(payload, 0, sites, 1).is_err());
    }
    for hidden in [0, -1, i64::MAX] {
        assert!(parse_rbm_content(payload, 3, 2, hidden).is_err());
    }
    assert!(parse_rbm_content(payload, 9, 2, 1).is_err());
    assert!(parse_rbm_content("0 0\n0 1\n", 0, 1, 1).is_err());
}
