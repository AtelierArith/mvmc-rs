//! F034/F041: original Coulomb helper conditions plus separate C file boundaries.
//! C readdef.c2017-2037/2064-2084 SHA256
//! 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
//! Independent literal values; no native/oracle/runtime dependency.
use mvmc_expert_parsers::parsers::coulomb::{
    parse_coulomb_inter_content, parse_coulomb_inter_definition, parse_coulomb_intra_content,
    parse_coulomb_intra_definition,
};
use mvmc_expert_parsers::{parse_expert_mode_files, ParseError};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Input(PathBuf);
impl Input {
    fn new() -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mvmc-coulomb-boundary-{}-{nonce}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn definition(&self, count: usize, body: &str) -> PathBuf {
        let path = self.0.join("coulomb.def");
        fs::write(
            &path,
            format!("===\nCount {count}\n===\nrecords\n===\n{body}"),
        )
        .unwrap();
        path
    }
    fn namelist(&self, family: &str) -> PathBuf {
        fs::write(self.0.join("modpara.def"), "NSite 4\nNElec 2\n").unwrap();
        let path = self.0.join("namelist.def");
        fs::write(
            &path,
            format!("ModPara modpara.def\n{family} coulomb.def\n"),
        )
        .unwrap();
        path
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn coulomb_original_valid_literals_preserve_coordinates_and_values() {
    let input = Input::new();
    let intra = parse_coulomb_intra_definition(input.definition(2, "0 4.0\n1 4.0\n"), 4).unwrap();
    assert_eq!(
        intra.iter().map(|t| (t.site, t.value)).collect::<Vec<_>>(),
        [(0, 4.0), (1, 4.0)]
    );
    let inter =
        parse_coulomb_inter_definition(input.definition(2, "0 1 1.0\n1 2 1.0\n"), 4).unwrap();
    assert_eq!(
        inter
            .iter()
            .map(|t| (t.site1, t.site2, t.value))
            .collect::<Vec<_>>(),
        [(0, 1, 1.0), (1, 2, 1.0)]
    );
}

#[test]
fn coulomb_native_hex_coefficients_do_not_use_zero_fallback() {
    let input = Input::new();
    assert_eq!(
        parse_coulomb_intra_definition(input.definition(1, "0 0x1.4p0\n"), 4).unwrap()[0].value,
        1.25
    );
    assert_eq!(
        parse_coulomb_inter_definition(input.definition(1, "0 1 -0x1p-1\n"), 4).unwrap()[0].value,
        -0.5
    );
}

#[test]
fn coulomb_omitted_scalar_retains_initialized_then_prior_value() {
    let input = Input::new();
    let intra = parse_coulomb_intra_definition(input.definition(3, "0\n1 -0.5\n2\n"), 4).unwrap();
    assert_eq!(
        intra.iter().map(|t| t.value).collect::<Vec<_>>(),
        [0.0, -0.5, -0.5]
    );
    let inter =
        parse_coulomb_inter_definition(input.definition(3, "0 1\n1 2 1.25\n2 3\n"), 4).unwrap();
    assert_eq!(
        inter.iter().map(|t| t.value).collect::<Vec<_>>(),
        [0.0, 1.25, 1.25]
    );
}

#[test]
fn coulomb_zero_count_ignores_body_and_diagonal_inter_is_not_rejected() {
    let input = Input::new();
    let path = input.definition(0, "not a row\n");
    assert!(parse_coulomb_intra_definition(&path, 4).unwrap().is_empty());
    assert!(parse_coulomb_inter_definition(&path, 4).unwrap().is_empty());
    let terms = parse_coulomb_inter_definition(input.definition(1, "2 2 -1.0\n"), 4).unwrap();
    assert_eq!(
        (terms[0].site1, terms[0].site2, terms[0].value),
        (2, 2, -1.0)
    );
}

#[test]
fn coulomb_short_zero_header_ignores_body_but_positive_header_is_incomplete() {
    // C ReadBuffInt reads line2; caller874 ignores EOF while skipping five
    // lines. Both native scalar readers return before any buffer use for0.
    let input = Input::new();
    let path = input.0.join("coulomb.def");
    fs::write(&path, "===\nCount 0\n").unwrap();
    assert!(parse_coulomb_intra_definition(&path, 4).unwrap().is_empty());
    assert!(parse_coulomb_inter_definition(&path, 4).unwrap().is_empty());
    fs::write(&path, "===\nCount 1\n").unwrap();
    assert_eq!(
        parse_coulomb_intra_definition(&path, 4).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        parse_coulomb_inter_definition(&path, 4).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn coulomb_file_reader_preserves_nonfinite_c_numeric_values_without_policy_rejection() {
    // These are parser values only, not acceptance of a runnable finite model.
    // Native %lf accepts them; runner numerical policy is a different layer.
    let input = Input::new();
    let intra =
        parse_coulomb_intra_definition(input.definition(3, "0 inf\n1 -inf\n2 nan\n"), 4).unwrap();
    assert_eq!(intra[0].value, f64::INFINITY);
    assert_eq!(intra[1].value, f64::NEG_INFINITY);
    assert!(intra[2].value.is_nan());
    let inter =
        parse_coulomb_inter_definition(input.definition(3, "0 1 inf\n1 2 -inf\n2 3 nan\n"), 4)
            .unwrap();
    assert_eq!(inter[0].value, f64::INFINITY);
    assert_eq!(inter[1].value, f64::NEG_INFINITY);
    assert!(inter[2].value.is_nan());
}

#[test]
fn coulomb_invalid_consumed_records_reject_without_partial_success() {
    let input = Input::new();
    for body in ["-1 4.0\n", "4 4.0\n", "0 invalid\n", "\n"] {
        assert_eq!(
            parse_coulomb_intra_definition(input.definition(2, &format!("0 1\n{body}")), 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
    for body in ["-1 0 1\n", "0 4 1\n", "0 1 invalid\n", "0\n"] {
        assert_eq!(
            parse_coulomb_inter_definition(input.definition(2, &format!("0 1 1\n{body}")), 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn public_coulomb_loader_rejects_missing_and_mismatched_declared_counts() {
    for family in ["CoulombIntra", "CoulombInter"] {
        let input = Input::new();
        let namelist = input.namelist(family);
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        let body = if family == "CoulombIntra" {
            "0 0x1p0\n"
        } else {
            "0 1 0x1p0\n"
        };
        input.definition(1, body);
        let data = parse_expert_mode_files(&namelist).unwrap();
        assert_eq!(
            if family == "CoulombIntra" {
                data.coulomb_intra_terms[0].value
            } else {
                data.coulomb_inter_terms[0].value
            },
            1.0
        );
        input.definition(2, body);
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        input.definition(1, &format!("{body}{body}"));
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
    }
}

#[test]
fn headerless_coulomb_helpers_are_separate_julia_architecture() {
    let mut context = mvmc_expert_parsers::ParsingContext::new("historical-payload");
    assert_eq!(
        parse_coulomb_intra_content("0 4.0\n", &mut context)[0].value,
        4.0
    );
    assert_eq!(parse_coulomb_inter_content("0 1 1.0\n")[0].value, 1.0);
}
