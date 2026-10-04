//! #264: native ReadPairHopValue input-count and ordered expansion boundaries.
//! C readdef.c SHA256 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
//! Ordinary independent literals; no oracle or toolbox dependency.
use mvmc_expert_parsers::{
    parse_expert_mode_files,
    parsers::pairhop::{parse_pairhop_content, parse_pairhop_definition},
    ParseError,
};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Input(PathBuf);
impl Input {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mvmc-pairhop-boundary-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn definition(&self, count: usize, body: &str) -> PathBuf {
        let path = self.0.join("pairhop.def");
        fs::write(
            &path,
            format!("===\nNPairHopp {count}\n===\npairs\n===\n{body}"),
        )
        .unwrap();
        path
    }
    fn rows(&self, count: usize, body: &str) -> Vec<(i64, i64, f64)> {
        parse_pairhop_definition(self.definition(count, body), 4)
            .unwrap()
            .into_iter()
            .map(|t| (t.site1, t.site2, t.value))
            .collect()
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn pairhop_native_expansion_preserves_forward_reverse_order_and_sign() {
    assert_eq!(
        Input::new().rows(2, "3 0 -0.5\n1 2 1.25\n"),
        [(3, 0, -0.5), (0, 3, -0.5), (1, 2, 1.25), (2, 1, 1.25)]
    );
}

#[test]
fn pairhop_diagonal_and_duplicate_rows_expand_without_deduplication() {
    assert_eq!(Input::new().rows(2, "2 2 1\n2 2 1\n"), [(2, 2, 1.0); 4]);
}

#[test]
fn pairhop_hex_and_signed_zero_are_preserved_in_both_directions() {
    let input = Input::new();
    assert_eq!(input.rows(1, "0 1 -0x1p-1\n"), [(0, 1, -0.5), (1, 0, -0.5)]);
    for row in input.rows(1, "0 1 -0.0\n") {
        assert_eq!(row.2.to_bits(), (-0.0_f64).to_bits());
    }
}

#[test]
fn pairhop_omitted_scalar_retains_initialized_then_prior_value() {
    assert_eq!(
        Input::new().rows(3, "0 1\n1 2 -0.5\n2 3\n"),
        [
            (0, 1, 0.0),
            (1, 0, 0.0),
            (1, 2, -0.5),
            (2, 1, -0.5),
            (2, 3, -0.5),
            (3, 2, -0.5)
        ]
    );
}

#[test]
fn pairhop_zero_and_short_zero_headers_ignore_body_positive_requires_header() {
    let input = Input::new();
    assert!(input.rows(0, "invalid body\n").is_empty());
    let path = input.0.join("pairhop.def");
    fs::write(&path, "===\nNPairHopp 0\n").unwrap();
    assert!(parse_pairhop_definition(&path, 4).unwrap().is_empty());
    fs::write(&path, "===\nNPairHopp 1\n").unwrap();
    assert_eq!(
        parse_pairhop_definition(&path, 4).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn pairhop_nonfinite_parser_values_are_not_runnable_model_approval() {
    let rows = Input::new().rows(3, "0 1 inf\n1 2 -inf\n2 3 nan\n");
    assert_eq!(rows[0].2, f64::INFINITY);
    assert_eq!(rows[1].2, f64::INFINITY);
    assert_eq!(rows[2].2, f64::NEG_INFINITY);
    assert_eq!(rows[3].2, f64::NEG_INFINITY);
    assert!(rows[4].2.is_nan() && rows[5].2.is_nan());
}

#[test]
fn pairhop_invalid_rows_and_input_count_mismatch_return_no_partial_payload() {
    let input = Input::new();
    for body in [
        "-1 0 1\n",
        "0 4 1\n",
        "4 0 1\n",
        "0 -1 1\n",
        "0 1 invalid\n",
        "0\n",
        "\n",
    ] {
        assert_eq!(
            parse_pairhop_definition(input.definition(2, &format!("0 1 1\n{body}")), 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
    for (count, body) in [(2, "0 1 1\n"), (1, "0 1 1\n1 2 2\n")] {
        assert_eq!(
            parse_pairhop_definition(input.definition(count, body), 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn public_pairhop_loader_rejects_missing_and_invalid_required_definition() {
    let input = Input::new();
    fs::write(input.0.join("modpara.def"), "NSite 4\nNElec 2\n").unwrap();
    let namelist = input.0.join("namelist.def");
    fs::write(&namelist, "ModPara modpara.def\nPairHop pairhop.def\n").unwrap();
    assert!(matches!(
        parse_expert_mode_files(&namelist),
        Err(ParseError::InvalidInput { .. })
    ));
    input.definition(1, "2 2 -0x1p-1\n");
    let data = parse_expert_mode_files(&namelist).unwrap();
    assert_eq!(
        data.pair_hop_terms
            .iter()
            .map(|t| (t.site1, t.site2, t.value))
            .collect::<Vec<_>>(),
        [(2, 2, -0.5), (2, 2, -0.5)]
    );
    for (count, body) in [(2, "0 1 1\n"), (1, "0 4 1\n")] {
        input.definition(count, body);
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
    }
}

#[test]
fn pairhop_headerless_helper_keeps_separate_partial_error_collection() {
    let section = parse_pairhop_content("0 1 0.5\ninvalid\n");
    assert!(!section.is_success());
    assert_eq!(
        section
            .terms
            .iter()
            .map(|t| (t.site1, t.site2, t.value))
            .collect::<Vec<_>>(),
        [(0, 1, 0.5), (1, 0, 0.5)]
    );
    assert_eq!(section.errors.len(), 1);
}
