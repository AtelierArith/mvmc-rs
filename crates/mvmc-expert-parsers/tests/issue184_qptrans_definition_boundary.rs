//! Literal C GetInfoTransSym boundary controls (readdef.c2232–2264).
//! No C/Julia program is compiled, invoked or read by these ordinary tests.
//! Holes/bounds/overflow rejection is safe Rust handling of undefined C input,
//! not a claim that the native unchecked scatter diagnoses those cases.
use mvmc_expert_parsers::{
    parse_expert_mode_files,
    parsers::qptrans::{parse_qptrans_content, parse_qptrans_def},
    ParseError,
};
use num_complex::Complex64;
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Bundle(PathBuf);
impl Bundle {
    fn new() -> Self {
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue269-qptrans-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("exclusive directory: {error}"),
            }
        }
    }
    fn definition(&self, count: &str, body: &str) -> PathBuf {
        let path = self.0.join("qptrans.def");
        fs::write(&path, format!("===\n{count}\n===\ncolumns\n===\n{body}")).unwrap();
        path
    }
    fn loader(&self, alias: &str, nmp: i64) -> PathBuf {
        fs::write(
            self.0.join("modpara.def"),
            format!("Nsite 2\nNElec 1\nNMPTrans {nmp}\n"),
        )
        .unwrap();
        let path = self.0.join("namelist.def");
        fs::write(&path, format!("ModPara modpara.def\n{alias} qptrans.def\n")).unwrap();
        path
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

const COMPLETE: &str = "1 -0x1.8p+1 0x1p-1\n0 0x1p-2 -0x1p-3\n1 1 0 -1\n0 0 1 -1\n1 0 1\n0 1 0 2\n";

#[test]
fn qptrans_hex_complex_weights_and_reordered_maps_are_indexed() {
    let bundle = Bundle::new();
    let section = parse_qptrans_def(bundle.definition("informational 2", COMPLETE), 2).unwrap();
    assert_eq!(section.n_qp_trans, 2);
    assert_eq!(section.entries[0].weight, Complex64::new(0.25, -0.125));
    assert_eq!(section.entries[1].weight, Complex64::new(-3.0, 0.5));
    for entry in &section.entries {
        assert_eq!(entry.site_map, [1, 0]);
        assert_eq!(entry.inverse_site_map().unwrap(), [1, 0]);
    }
    assert_eq!(section.entries[0].site_sign, [-1, 2]);
    assert_eq!(section.entries[1].site_sign, [-1, -1]);
}

#[test]
fn qptrans_weight_fields_reset_and_mapping_sign_retains_native_carry() {
    let bundle = Bundle::new();
    let body = "0 2 3\n1\n0 0 0\n0 1 1 -2\n1 0 0\n1 1 1\n";
    let section = parse_qptrans_def(bundle.definition("NQPTrans 2", body), 2).unwrap();
    assert_eq!(section.entries[0].weight, Complex64::new(2.0, 3.0));
    assert_eq!(section.entries[1].weight, Complex64::new(0.0, 0.0));
    assert_eq!(section.entries[0].site_sign, [0, -2]);
    assert_eq!(section.entries[1].site_sign, [-2, -2]);
}

#[test]
fn qptrans_defined_scan_failure_keeps_reset_weight_or_prior_mapping_fields() {
    let bundle = Bundle::new();
    // Initial blank weight row is defined: index/re/im all reset to zero.
    // Last mapping has two fields; itmp/sign retain 1/7 from its predecessor.
    let body = "\n0 1 1 7\n0 0\n";
    let section = parse_qptrans_def(bundle.definition("NQPTrans 1", body), 2);
    // This leaves inverse target zero unwritten: reject rather than invent it.
    assert_eq!(section.unwrap_err().kind(), io::ErrorKind::InvalidData);
    let body = "\n0 1 0 7\n0 0 1\n";
    let section = parse_qptrans_def(bundle.definition("NQPTrans 1", body), 2).unwrap();
    assert_eq!(section.entries[0].weight, Complex64::new(0.0, 0.0));
    assert_eq!(section.entries[0].site_map, [1, 0]);
    assert_eq!(section.entries[0].site_sign, [7, 7]);
}

#[test]
fn qptrans_physical_count_truncation_and_extra_records_fail() {
    let bundle = Bundle::new();
    for body in ["0 1\n", "0 1\n0 0 0 1\n", "0 1\n0 0 0 1\n0 1 1 1\n\n"] {
        assert_eq!(
            parse_qptrans_def(bundle.definition("NQPTrans 1", body), 2)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn qptrans_unwritten_weight_forward_and_inverse_slots_are_not_fabricated() {
    let bundle = Bundle::new();
    for body in [
        "0 1\n0 2\n0 0 0 1\n0 1 1 1\n1 0 0 1\n1 1 1 1\n",
        "0 1\n0 0 0 1\n0 0 1 1\n",
        "0 1\n0 0 0 1\n0 1 0 1\n",
    ] {
        let count = if body.starts_with("0 1\n0 2") { 2 } else { 1 };
        assert!(
            parse_qptrans_def(bundle.definition(&format!("NQPTrans {count}"), body), 2).is_err()
        );
    }
}

#[test]
fn qptrans_indices_integer_overflow_and_invalid_header_are_safe_errors() {
    let bundle = Bundle::new();
    for body in [
        "-1 1\n0 0 0 1\n0 1 1 1\n",
        "1 1\n0 0 0 1\n0 1 1 1\n",
        "0 1\n1 0 0 1\n0 1 1 1\n",
        "0 1\n0 -1 0 1\n0 1 1 1\n",
        "0 1\n0 0 2 1\n0 1 1 1\n",
        "0 1\n0 0 0 2147483648\n0 1 1 1\n",
    ] {
        assert!(parse_qptrans_def(bundle.definition("NQPTrans 1", body), 2).is_err());
    }
    for header in ["NQPTrans -1", "NQPTrans 2147483648", "NQPTrans nope"] {
        assert!(parse_qptrans_def(bundle.definition(header, ""), 2).is_err());
    }
}

#[test]
fn qptrans_zero_accepts_two_to_five_header_lines_and_ignores_unconsumed_body() {
    let bundle = Bundle::new();
    let section = parse_qptrans_def(bundle.definition("anything 0", "ignored\n"), 0).unwrap();
    assert_eq!(section.n_qp_trans, 0);
    assert!(section.entries.is_empty());
    let long_ignored = "x".repeat(300);
    assert!(parse_qptrans_def(bundle.definition("anything 0", &long_ignored), 0).is_ok());
    let path = bundle.0.join("short.def");
    for extra in ["", "===\n", "===\ncolumns\n", "===\ncolumns\n===\n"] {
        fs::write(&path, format!("===\nNQPTrans 0\n{extra}")).unwrap();
        let section = parse_qptrans_def(&path, 2).unwrap();
        assert_eq!(section.n_qp_trans, 0);
        assert!(section.entries.is_empty());
    }
    fs::write(&path, format!("===\nNQPTrans 0\n{long_ignored}\ninvalid\n")).unwrap();
    let section = parse_qptrans_def(&path, 2).unwrap();
    assert_eq!(section.n_qp_trans, 0);
    assert!(section.entries.is_empty());
}

#[test]
fn qptrans_count_header_errors_and_positive_short_headers_remain_rejected() {
    let bundle = Bundle::new();
    let path = bundle.0.join("short.def");
    for content in [
        "",
        "===\n",
        "===\n\n",
        "===\nNQPTrans\n",
        "===\nNQPTrans nope\n",
        "===\nNQPTrans -1\n",
        "===\nNQPTrans 2147483648\n",
        "===\nNQPTrans 1\n",
        "===\nNQPTrans 1\n===\n",
        "===\nNQPTrans 1\n===\ncolumns\n",
    ] {
        fs::write(&path, content).unwrap();
        assert!(parse_qptrans_def(&path, 2).is_err(), "{content:?}");
    }
}

#[test]
fn qptrans_consumed_header_length_limits_do_not_apply_to_zero_unconsumed_lines() {
    let bundle = Bundle::new();
    let path = bundle.0.join("length.def");
    for length in [255, 256] {
        let first = "x".repeat(length);
        let second = format!("NQPTrans 0{}", " ".repeat(length - "NQPTrans 0".len()));
        for content in [format!("{first}\nNQPTrans 0\n"), format!("===\n{second}\n")] {
            fs::write(&path, content).unwrap();
            assert!(parse_qptrans_def(&path, 2).is_err());
        }
    }
    let first = "x".repeat(254);
    let second = format!("NQPTrans 0{}", " ".repeat(254 - "NQPTrans 0".len()));
    fs::write(&path, format!("{first}\n{second}\n{}\n", "x".repeat(300))).unwrap();
    let section = parse_qptrans_def(&path, 2).unwrap();
    assert_eq!(section.n_qp_trans, 0);
    assert!(section.entries.is_empty());
    for header_index in 0..5 {
        let mut headers = ["===", "NQPTrans 2", "===", "columns", "==="].map(str::to_owned);
        headers[header_index] = if header_index == 1 {
            format!("NQPTrans 2{}", " ".repeat(255 - "NQPTrans 2".len()))
        } else {
            "x".repeat(255)
        };
        fs::write(&path, format!("{}\n{COMPLETE}", headers.join("\n"))).unwrap();
        assert!(parse_qptrans_def(&path, 2).is_err());
    }
}

#[test]
fn public_qptrans_aliases_accept_short_zero_as_empty_parsed_sections() {
    let bundle = Bundle::new();
    for alias in ["TransSym", "QPTrans"] {
        for nmp in [1, -1] {
            let namelist = bundle.loader(alias, nmp);
            for extra in ["", "===\n", "===\ncolumns\n", "===\ncolumns\n===\n"] {
                fs::write(
                    bundle.0.join("qptrans.def"),
                    format!("===\nNQPTrans 0\n{extra}"),
                )
                .unwrap();
                let data = parse_expert_mode_files(&namelist).unwrap();
                assert!(data.input_errors.is_empty());
                assert_eq!(data.n_qp_trans, 0);
                assert!(data.qp_trans_entries.is_empty());
                assert!(data.para_qp_trans.is_empty());
            }
        }
    }
}

#[test]
fn public_qptrans_aliases_reject_malformed_count_and_positive_short_header() {
    let bundle = Bundle::new();
    for alias in ["TransSym", "QPTrans"] {
        for nmp in [1, -1] {
            let namelist = bundle.loader(alias, nmp);
            for content in [
                "",
                "===\n",
                "===\n\n",
                "===\nNQPTrans\n",
                "===\nNQPTrans nope\n",
                "===\nNQPTrans -1\n",
                "===\nNQPTrans 2147483648\n",
                "===\nNQPTrans 1\n",
            ] {
                fs::write(bundle.0.join("qptrans.def"), content).unwrap();
                assert!(matches!(
                    parse_expert_mode_files(&namelist),
                    Err(ParseError::InvalidInput { .. })
                ));
            }
        }
    }
}

#[test]
fn public_qptrans_aliases_preserve_raw_signs_and_effective_ap_lifecycle() {
    let bundle = Bundle::new();
    bundle.definition("NQPTrans 2", COMPLETE);
    for alias in ["TransSym", "QPTrans"] {
        for nmp in [1, -1] {
            let data = parse_expert_mode_files(bundle.loader(alias, nmp)).unwrap();
            assert!(data.input_errors.is_empty());
            assert_eq!(
                data.para_qp_trans,
                [Complex64::new(0.25, -0.125), Complex64::new(-3.0, 0.5)]
            );
            assert_eq!(data.qp_trans_entries[0].site_sign, [-1, 2]);
            let effective: Vec<_> = (0..2)
                .map(|site| data.qp_trans_entries[0].boundary_sign(site, nmp < 0))
                .collect();
            assert_eq!(effective, if nmp < 0 { vec![-1, 2] } else { vec![1, 1] });
        }
    }
}

#[test]
fn public_qptrans_missing_or_invalid_section_never_returns_partial_model() {
    let bundle = Bundle::new();
    for alias in ["TransSym", "QPTrans"] {
        let namelist = bundle.loader(alias, 1);
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        bundle.definition("NQPTrans 1", "0 1\n0 0 0 1\n");
        assert!(matches!(
            parse_expert_mode_files(&namelist),
            Err(ParseError::InvalidInput { .. })
        ));
        fs::remove_file(bundle.0.join("qptrans.def")).unwrap();
    }
}

#[test]
fn qptrans_content_helper_remains_separate_and_permissive() {
    let section = parse_qptrans_content("===\nNQPTrans 1\n===\ncolumns\n===\n0 1\n0 0 1 -1\n", 2);
    assert_eq!(section.entries[0].site_map, [1, -1]);
}
