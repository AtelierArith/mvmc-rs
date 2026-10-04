//! Cross-language parity for `calc_m_all_{real,complex}`.
//!
//! Loads the deterministic text fixtures dumped by
//! `extern/Julia-mVMC/tools/dump_calc_m_all_reference.jl` and asserts
//! the Rust port reproduces Julia's `(Pfaffian, inv_m)` per QP plane.
//!
//! Regenerate after touching either the Julia kernel or the Rust port:
//!
//! ```sh
//! cd extern/Julia-mVMC
//! julia --startup-file=no --project=. tools/dump_calc_m_all_reference.jl
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use mvmc_core::{
    calc_m_all_complex, calc_m_all_real, InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace,
};
use num_complex::Complex64;
use pfapack::PivotIndex1Based;

const FIXTURE_DIR: &str = "../../tests/fixtures/calc_m_all";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

struct Fixture {
    n_site: usize,
    n_elec: usize, // per spin
    n_qp_full: usize,
    kind_value: String,
    payload: Vec<String>,
}

fn load_fixture(path: &Path) -> Fixture {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to open fixture {} (did you run dump_calc_m_all_reference.jl?): {}",
            path.display(),
            e
        )
    });
    let mut n_site = None;
    let mut n_elec = None;
    let mut n_qp_full = None;
    let mut kind_value = None;
    let mut payload = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_site ") {
            n_site = Some(rest.parse::<usize>().unwrap());
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_elec ") {
            n_elec = Some(rest.parse::<usize>().unwrap());
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_qp_full ") {
            n_qp_full = Some(rest.parse::<usize>().unwrap());
            continue;
        }
        if let Some(_rest) = t.strip_prefix("seed ") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("kind ") {
            let _ = rest;
            continue;
        }
        if let Some(rest) = t.strip_prefix("kind_value ") {
            kind_value = Some(rest.to_string());
            continue;
        }
        payload.push(t.to_string());
    }
    Fixture {
        n_site: n_site.expect("missing n_site"),
        n_elec: n_elec.expect("missing n_elec"),
        n_qp_full: n_qp_full.expect("missing n_qp_full"),
        kind_value: kind_value.expect("missing kind_value"),
        payload,
    }
}

fn parse_real(lines: &[String]) -> Vec<f64> {
    lines.iter().map(|s| s.parse::<f64>().unwrap()).collect()
}

fn parse_complex(lines: &[String]) -> Vec<Complex64> {
    lines
        .iter()
        .map(|s| {
            let mut it = s.split_ascii_whitespace();
            let re = it.next().unwrap().parse::<f64>().unwrap();
            let im = it.next().unwrap().parse::<f64>().unwrap();
            Complex64::new(re, im)
        })
        .collect()
}

fn parse_ele_idx(lines: &[String]) -> Vec<i64> {
    lines.iter().map(|s| s.parse::<i64>().unwrap()).collect()
}

fn check_real_case(n_site: usize, n_elec: usize, n_qp_full: usize, seed: u64) {
    let stem = format!(
        "real_ns{}_ne{}_nqp{}_seed{}",
        n_site, n_elec, n_qp_full, seed
    );
    let slater_fx = load_fixture(&fixture_path(&format!("{}.slater.txt", stem)));
    let ele_fx = load_fixture(&fixture_path(&format!("{}.ele_idx.txt", stem)));
    let pf_fx = load_fixture(&fixture_path(&format!("{}.pfaffian.txt", stem)));
    let inv_fx = load_fixture(&fixture_path(&format!("{}.inverse.txt", stem)));
    assert_eq!(slater_fx.kind_value, "slater");
    assert_eq!(ele_fx.kind_value, "ele_idx");
    assert_eq!(pf_fx.kind_value, "pfaffian");
    assert_eq!(inv_fx.kind_value, "inverse");
    assert_eq!(slater_fx.n_site, n_site);
    assert_eq!(slater_fx.n_elec, n_elec);
    assert_eq!(slater_fx.n_qp_full, n_qp_full);

    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;

    let slater_values = parse_real(&slater_fx.payload);
    assert_eq!(slater_values.len(), n_qp_full * n_site2 * n_site2);
    let ele_idx = parse_ele_idx(&ele_fx.payload);
    assert_eq!(ele_idx.len(), n_size);
    let pf_expected = parse_real(&pf_fx.payload);
    assert_eq!(pf_expected.len(), n_qp_full);
    let inv_expected = parse_real(&inv_fx.payload);
    assert_eq!(inv_expected.len(), n_qp_full * n_size * n_size);

    // Slater values are stored row-major within a QP plane (matching
    // the upstream `rsi*n_site2 + rsj` linearisation). Bring them into
    // a `SlaterElmFlat` by mirroring that index.
    let mut slater = SlaterElmFlat::<f64>::zeros(n_qp_full, n_site);
    for qp in 0..n_qp_full {
        for row in 0..n_site2 {
            for col in 0..n_site2 {
                let idx = qp * n_site2 * n_site2 + row * n_site2 + col;
                slater.set(qp, row, col, slater_values[idx]);
            }
        }
    }

    let mut inv_m = InvMColMajor::<f64>::zeros(n_qp_full, n_elec);
    let mut pf_m = vec![0.0; n_qp_full];
    let pool = ThreadedPfaPackWorkspace::new(n_size, 1);
    calc_m_all_real(
        &ele_idx, &slater, &mut inv_m, &mut pf_m, 0, n_qp_full, n_site, n_elec, &pool,
    )
    .expect("calc_m_all_real should succeed on the golden case");

    for qp in 0..n_qp_full {
        let want = pf_expected[qp];
        let got = pf_m[qp];
        assert!(
            close_f64(got, want, 1e-11, 1e-13),
            "real pf mismatch qp={qp} ns={n_site} ne={n_elec} seed={seed}: rust={got:.17e} julia={want:.17e}"
        );
        for col in 0..n_size {
            for row in 0..n_size {
                let want = inv_expected[qp * n_size * n_size + col * n_size + row];
                let got = inv_m.get(qp, row, col);
                assert!(
                    close_f64(got, want, 1e-11, 1e-13),
                    "real inv mismatch qp={qp} row={row} col={col}: rust={got:.17e} julia={want:.17e}"
                );
            }
        }
    }
    // Silence unused-import warning when the file compiles in isolation.
    let _ = PivotIndex1Based(0);
}

fn check_complex_case(n_site: usize, n_elec: usize, n_qp_full: usize, seed: u64) {
    let stem = format!(
        "complex_ns{}_ne{}_nqp{}_seed{}",
        n_site, n_elec, n_qp_full, seed
    );
    let slater_fx = load_fixture(&fixture_path(&format!("{}.slater.txt", stem)));
    let ele_fx = load_fixture(&fixture_path(&format!("{}.ele_idx.txt", stem)));
    let pf_fx = load_fixture(&fixture_path(&format!("{}.pfaffian.txt", stem)));
    let inv_fx = load_fixture(&fixture_path(&format!("{}.inverse.txt", stem)));

    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;

    let slater_values = parse_complex(&slater_fx.payload);
    assert_eq!(slater_values.len(), n_qp_full * n_site2 * n_site2);
    let ele_idx = parse_ele_idx(&ele_fx.payload);
    assert_eq!(ele_idx.len(), n_size);
    let pf_expected = parse_complex(&pf_fx.payload);
    assert_eq!(pf_expected.len(), n_qp_full);
    let inv_expected = parse_complex(&inv_fx.payload);
    assert_eq!(inv_expected.len(), n_qp_full * n_size * n_size);

    let mut slater = SlaterElmFlat::<Complex64>::zeros(n_qp_full, n_site);
    for qp in 0..n_qp_full {
        for row in 0..n_site2 {
            for col in 0..n_site2 {
                let idx = qp * n_site2 * n_site2 + row * n_site2 + col;
                slater.set(qp, row, col, slater_values[idx]);
            }
        }
    }

    let mut inv_m = InvMColMajor::<Complex64>::zeros(n_qp_full, n_elec);
    let mut pf_m = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let pool = ThreadedPfaPackWorkspace::new(n_size, 1);
    calc_m_all_complex(
        &ele_idx, &slater, &mut inv_m, &mut pf_m, 0, n_qp_full, n_site, n_elec, &pool,
    )
    .expect("calc_m_all_complex should succeed on the golden case");

    for qp in 0..n_qp_full {
        let want = pf_expected[qp];
        let got = pf_m[qp];
        assert!(
            close_complex(got, want, 1e-11, 1e-13),
            "complex pf mismatch qp={qp}: rust={got:?} julia={want:?}"
        );
        for col in 0..n_size {
            for row in 0..n_size {
                let want = inv_expected[qp * n_size * n_size + col * n_size + row];
                let got = inv_m.get(qp, row, col);
                assert!(
                    close_complex(got, want, 1e-11, 1e-13),
                    "complex inv mismatch qp={qp} row={row} col={col}: rust={got:?} julia={want:?}"
                );
            }
        }
    }
}

fn close_f64(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).abs();
    diff <= abs || diff <= rel * a.abs().max(b.abs())
}

fn close_complex(a: Complex64, b: Complex64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).norm();
    diff <= abs || diff <= rel * a.norm().max(b.norm())
}

// ---------- test cases ----------

#[test]
fn real_ns2_ne1_nqp1() {
    check_real_case(2, 1, 1, 42);
}
#[test]
fn real_ns4_ne2_nqp1() {
    check_real_case(4, 2, 1, 1234);
}
#[test]
fn real_ns4_ne2_nqp3() {
    check_real_case(4, 2, 3, 1234);
}
#[test]
fn real_ns8_ne4_nqp2() {
    check_real_case(8, 4, 2, 20240301);
}

#[test]
fn complex_ns2_ne1_nqp1() {
    check_complex_case(2, 1, 1, 42);
}
#[test]
fn complex_ns4_ne2_nqp1() {
    check_complex_case(4, 2, 1, 1234);
}
#[test]
fn complex_ns4_ne2_nqp3() {
    check_complex_case(4, 2, 3, 1234);
}
#[test]
fn complex_ns8_ne4_nqp2() {
    check_complex_case(8, 4, 2, 20240301);
}
