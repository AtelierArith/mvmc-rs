//! #262: Hund/Exchange use native ReadPairDValue, not Julia's diagonal rejection.
//! C readdef.c SHA256 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
//! Independent literals only; no reference runtime or toolbox dependency.
use mvmc_expert_parsers::{
    parse_expert_mode_files,
    parsers::{exchange, hund},
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
            "mvmc-pair-boundary-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn definition(&self, count: usize, body: &str) -> PathBuf {
        let path = self.0.join("pair.def");
        fs::write(
            &path,
            format!("===\nCount {count}\n===\npairs\n===\n{body}"),
        )
        .unwrap();
        path
    }
    fn rows(&self, count: usize, body: &str) -> Vec<Vec<(i64, i64, f64)>> {
        let path = self.definition(count, body);
        vec![
            hund::parse_hund_definition(&path, 4)
                .unwrap()
                .into_iter()
                .map(|t| (t.site1, t.site2, t.value))
                .collect(),
            exchange::parse_exchange_definition(&path, 4)
                .unwrap()
                .into_iter()
                .map(|t| (t.site1, t.site2, t.value))
                .collect(),
        ]
    }
    fn reject(&self, count: usize, body: &str) {
        let path = self.definition(count, body);
        assert_eq!(
            hund::parse_hund_definition(&path, 4).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            exchange::parse_exchange_definition(&path, 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
    fn namelist(&self, family: &str) -> PathBuf {
        fs::write(self.0.join("modpara.def"), "NSite 4\nNElec 2\n").unwrap();
        let path = self.0.join("namelist.def");
        fs::write(&path, format!("ModPara modpara.def\n{family} pair.def\n")).unwrap();
        path
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn pair_literals_preserve_order_sign_and_diagonal_without_expansion() {
    for rows in Input::new().rows(3, "2 2 -1.5\n3 0 2.0\n0 3 -0.25\n") {
        assert_eq!(rows, [(2, 2, -1.5), (3, 0, 2.0), (0, 3, -0.25)]);
    }
}

#[test]
fn pair_hex_coefficients_are_not_zero_fallbacks() {
    for rows in Input::new().rows(2, "0 1 0x1.4p0\n1 2 -0x1p-1\n") {
        assert_eq!(rows, [(0, 1, 1.25), (1, 2, -0.5)]);
    }
}

#[test]
fn pair_omitted_scalar_retains_initialized_then_prior_value() {
    for rows in Input::new().rows(3, "0 1\n1 2 -0.5\n2 3\n") {
        assert_eq!(rows, [(0, 1, 0.0), (1, 2, -0.5), (2, 3, -0.5)]);
    }
}

#[test]
fn pair_zero_count_ignores_body() {
    for rows in Input::new().rows(0, "not a valid row\n") {
        assert!(rows.is_empty());
    }
}

#[test]
fn pair_short_zero_header_is_supported_but_positive_header_is_incomplete() {
    let input = Input::new();
    let path = input.0.join("pair.def");
    fs::write(&path, "===\nCount 0\n").unwrap();
    assert!(hund::parse_hund_definition(&path, 4).unwrap().is_empty());
    assert!(exchange::parse_exchange_definition(&path, 4)
        .unwrap()
        .is_empty());
    fs::write(&path, "===\nCount 1\n").unwrap();
    assert_eq!(
        hund::parse_hund_definition(&path, 4).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(
        exchange::parse_exchange_definition(&path, 4)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn pair_nonfinite_values_are_parser_only_not_runnable_model_policy() {
    for rows in Input::new().rows(3, "0 1 inf\n1 2 -inf\n2 3 nan\n") {
        assert_eq!(rows[0].2, f64::INFINITY);
        assert_eq!(rows[1].2, f64::NEG_INFINITY);
        assert!(rows[2].2.is_nan());
    }
}

#[test]
fn pair_invalid_consumed_rows_reject_partial_results_and_count_mismatch() {
    let input = Input::new();
    for bad in [
        "-1 0 1\n",
        "0 4 1\n",
        "4 0 1\n",
        "0 -1 1\n",
        "0 1 invalid\n",
        "0\n",
        "\n",
    ] {
        input.reject(2, &format!("0 1 1\n{bad}"));
    }
    input.reject(2, "0 1 1\n");
    input.reject(1, "0 1 1\n1 2 2\n");
}

#[test]
fn public_pair_loaders_reject_missing_and_invalid_required_files() {
    for family in ["Hund", "Exchange"] {
        let input = Input::new();
        let namelist = input.namelist(family);
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        input.definition(1, "2 2 -0x1p-1\n");
        let data = parse_expert_mode_files(&namelist).unwrap();
        let (count, value) = if family == "Hund" {
            (data.hund_terms.len(), data.hund_terms[0].value)
        } else {
            (data.exchange_terms.len(), data.exchange_terms[0].value)
        };
        assert_eq!((count, value), (1, -0.5));
        input.definition(2, "0 1 1\n");
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        input.definition(1, "0 4 1\n");
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
    }
}

#[test]
fn headerless_pair_helpers_remain_separate_convenience_apis() {
    let h = hund::parse_hund_content("0 1 0.5\n");
    let e = exchange::parse_exchange_content("0 1 0.5\n");
    assert_eq!((h[0].site1, h[0].site2, h[0].value), (0, 1, 0.5));
    assert_eq!((e[0].site1, e[0].site2, e[0].value), (0, 1, 0.5));
}
