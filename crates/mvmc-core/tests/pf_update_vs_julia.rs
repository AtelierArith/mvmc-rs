//! Phase 4.3.4a parity for one-electron Pfaffian update helpers.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use mvmc_core::sampling::updates::{
    calculate_new_pf_m2_complex_flat, calculate_new_pf_m2_fsz_complex_flat,
    calculate_new_pf_m2_fsz_real_flat, calculate_new_pf_m2_real_flat,
    calculate_new_pf_m_two2_complex_flat, calculate_new_pf_m_two2_real_flat,
    calculate_new_pf_m_two_fsz_complex_flat, calculate_new_pf_m_two_fsz_real_flat,
    update_m_all_complex_flat, update_m_all_fsz_complex_flat, update_m_all_fsz_real_flat,
    update_m_all_real_flat, update_m_all_two_complex_flat, update_m_all_two_fsz_real_flat,
    update_m_all_two_real_flat,
};
use mvmc_core::SlaterElmFlat;
use num_complex::Complex64;

const FIXTURE: &str = "../../tests/fixtures/pf_update/pf_m2.txt";

#[derive(Debug)]
struct Fixture {
    n_site: usize,
    n_elec: usize,
    n_qp: usize,
    ma_spin: (usize, u8),
    mb_spin: (usize, u8),
    ma_spin_fsz: (usize, u8),
    mb_spin_fsz: (usize, u8),
    old_sites: (usize, usize),
    ele_idx: Vec<i64>,
    ele_spn: Vec<i64>,
    slater_real: Vec<f64>,
    inv_real: Vec<f64>,
    pf_real: Vec<f64>,
    out_real: Vec<f64>,
    out_real_fsz: Vec<f64>,
    out_real_two: Vec<f64>,
    out_real_two_fsz: Vec<f64>,
    slater_complex: Vec<Complex64>,
    inv_complex: Vec<Complex64>,
    pf_complex: Vec<Complex64>,
    out_complex: Vec<Complex64>,
    out_complex_fsz: Vec<Complex64>,
    out_complex_two: Vec<Complex64>,
    out_complex_two_fsz: Vec<Complex64>,
    update_real_pf: Vec<f64>,
    update_real_inv: Vec<f64>,
    update_real_fsz_pf: Vec<f64>,
    update_real_fsz_inv: Vec<f64>,
    update_complex_pf: Vec<Complex64>,
    update_complex_inv: Vec<Complex64>,
    update_complex_fsz_pf: Vec<Complex64>,
    update_complex_fsz_inv: Vec<Complex64>,
    update_two_real_pf: Vec<f64>,
    update_two_real_inv: Vec<f64>,
    update_two_complex_pf: Vec<Complex64>,
    update_two_complex_inv: Vec<Complex64>,
    update_two_fsz_real_pf: Vec<f64>,
    update_two_fsz_real_inv: Vec<f64>,
}

fn parse_i64s(rest: &str) -> Vec<i64> {
    rest.split_ascii_whitespace()
        .map(|s| s.parse::<i64>().unwrap())
        .collect()
}

fn parse_pair_usize_u8(rest: &str) -> (usize, u8) {
    let xs = parse_i64s(rest);
    (xs[0] as usize, xs[1] as u8)
}

fn parse_c64(s: &str) -> Complex64 {
    let mut it = s.split_ascii_whitespace();
    Complex64::new(
        it.next().unwrap().parse::<f64>().unwrap(),
        it.next().unwrap().parse::<f64>().unwrap(),
    )
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();

    let mut f = Fixture {
        n_site: 0,
        n_elec: 0,
        n_qp: 0,
        ma_spin: (0, 0),
        mb_spin: (0, 0),
        ma_spin_fsz: (0, 0),
        mb_spin_fsz: (0, 0),
        old_sites: (0, 0),
        ele_idx: Vec::new(),
        ele_spn: Vec::new(),
        slater_real: Vec::new(),
        inv_real: Vec::new(),
        pf_real: Vec::new(),
        out_real: Vec::new(),
        out_real_fsz: Vec::new(),
        out_real_two: Vec::new(),
        out_real_two_fsz: Vec::new(),
        slater_complex: Vec::new(),
        inv_complex: Vec::new(),
        pf_complex: Vec::new(),
        out_complex: Vec::new(),
        out_complex_fsz: Vec::new(),
        out_complex_two: Vec::new(),
        out_complex_two_fsz: Vec::new(),
        update_real_pf: Vec::new(),
        update_real_inv: Vec::new(),
        update_real_fsz_pf: Vec::new(),
        update_real_fsz_inv: Vec::new(),
        update_complex_pf: Vec::new(),
        update_complex_inv: Vec::new(),
        update_complex_fsz_pf: Vec::new(),
        update_complex_fsz_inv: Vec::new(),
        update_two_real_pf: Vec::new(),
        update_two_real_inv: Vec::new(),
        update_two_complex_pf: Vec::new(),
        update_two_complex_inv: Vec::new(),
        update_two_fsz_real_pf: Vec::new(),
        update_two_fsz_real_inv: Vec::new(),
    };

    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        macro_rules! read_real_vec {
            ($target:ident, $rest:expr) => {{
                let len = $rest.parse::<usize>().unwrap();
                f.$target = lines[i + 1..i + 1 + len]
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect();
                i += len;
            }};
        }
        macro_rules! read_complex_vec {
            ($target:ident, $rest:expr) => {{
                let len = $rest.parse::<usize>().unwrap();
                f.$target = lines[i + 1..i + 1 + len]
                    .iter()
                    .map(|s| parse_c64(s))
                    .collect();
                i += len;
            }};
        }
        if let Some(rest) = t.strip_prefix("n_site ") {
            f.n_site = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("n_elec ") {
            f.n_elec = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("n_qp ") {
            f.n_qp = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("ma_spin ") {
            f.ma_spin = parse_pair_usize_u8(rest);
        } else if let Some(rest) = t.strip_prefix("mb_spin ") {
            f.mb_spin = parse_pair_usize_u8(rest);
        } else if let Some(rest) = t.strip_prefix("ma_spin_fsz ") {
            f.ma_spin_fsz = parse_pair_usize_u8(rest);
        } else if let Some(rest) = t.strip_prefix("mb_spin_fsz ") {
            f.mb_spin_fsz = parse_pair_usize_u8(rest);
        } else if let Some(rest) = t.strip_prefix("old_sites ") {
            let xs = parse_i64s(rest);
            f.old_sites = (xs[0] as usize, xs[1] as usize);
        } else if let Some(rest) = t.strip_prefix("ele_idx ") {
            f.ele_idx = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_spn ") {
            f.ele_spn = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("slater_real_len ") {
            read_real_vec!(slater_real, rest);
        } else if let Some(rest) = t.strip_prefix("inv_real_len ") {
            read_real_vec!(inv_real, rest);
        } else if let Some(rest) = t.strip_prefix("pf_real_len ") {
            read_real_vec!(pf_real, rest);
        } else if let Some(rest) = t.strip_prefix("out_real_len ") {
            read_real_vec!(out_real, rest);
        } else if let Some(rest) = t.strip_prefix("out_real_fsz_len ") {
            read_real_vec!(out_real_fsz, rest);
        } else if let Some(rest) = t.strip_prefix("out_real_two_len ") {
            read_real_vec!(out_real_two, rest);
        } else if let Some(rest) = t.strip_prefix("out_real_two_fsz_len ") {
            read_real_vec!(out_real_two_fsz, rest);
        } else if let Some(rest) = t.strip_prefix("slater_complex_len ") {
            read_complex_vec!(slater_complex, rest);
        } else if let Some(rest) = t.strip_prefix("inv_complex_len ") {
            read_complex_vec!(inv_complex, rest);
        } else if let Some(rest) = t.strip_prefix("pf_complex_len ") {
            read_complex_vec!(pf_complex, rest);
        } else if let Some(rest) = t.strip_prefix("out_complex_len ") {
            read_complex_vec!(out_complex, rest);
        } else if let Some(rest) = t.strip_prefix("out_complex_fsz_len ") {
            read_complex_vec!(out_complex_fsz, rest);
        } else if let Some(rest) = t.strip_prefix("out_complex_two_len ") {
            read_complex_vec!(out_complex_two, rest);
        } else if let Some(rest) = t.strip_prefix("out_complex_two_fsz_len ") {
            read_complex_vec!(out_complex_two_fsz, rest);
        } else if let Some(rest) = t.strip_prefix("update_real_pf_len ") {
            read_real_vec!(update_real_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_real_inv_len ") {
            read_real_vec!(update_real_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_real_fsz_pf_len ") {
            read_real_vec!(update_real_fsz_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_real_fsz_inv_len ") {
            read_real_vec!(update_real_fsz_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_complex_pf_len ") {
            read_complex_vec!(update_complex_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_complex_inv_len ") {
            read_complex_vec!(update_complex_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_complex_fsz_pf_len ") {
            read_complex_vec!(update_complex_fsz_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_complex_fsz_inv_len ") {
            read_complex_vec!(update_complex_fsz_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_real_pf_len ") {
            read_real_vec!(update_two_real_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_real_inv_len ") {
            read_real_vec!(update_two_real_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_complex_pf_len ") {
            read_complex_vec!(update_two_complex_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_complex_inv_len ") {
            read_complex_vec!(update_two_complex_inv, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_fsz_real_pf_len ") {
            read_real_vec!(update_two_fsz_real_pf, rest);
        } else if let Some(rest) = t.strip_prefix("update_two_fsz_real_inv_len ") {
            read_real_vec!(update_two_fsz_real_inv, rest);
        }
        i += 1;
    }
    f
}

fn slater_real(f: &Fixture) -> SlaterElmFlat<f64> {
    let mut s = SlaterElmFlat::<f64>::zeros(f.n_qp, f.n_site);
    let n_site2 = 2 * f.n_site;
    for qp in 0..f.n_qp {
        for row in 0..n_site2 {
            for col in 0..n_site2 {
                let k = qp * n_site2 * n_site2 + row * n_site2 + col;
                s.set(qp, row, col, f.slater_real[k]);
            }
        }
    }
    s
}

fn slater_complex(f: &Fixture) -> SlaterElmFlat<Complex64> {
    let mut s = SlaterElmFlat::<Complex64>::zeros(f.n_qp, f.n_site);
    let n_site2 = 2 * f.n_site;
    for qp in 0..f.n_qp {
        for row in 0..n_site2 {
            for col in 0..n_site2 {
                let k = qp * n_site2 * n_site2 + row * n_site2 + col;
                s.set(qp, row, col, f.slater_complex[k]);
            }
        }
    }
    s
}

fn close_f64(a: f64, b: f64) -> bool {
    let d = (a - b).abs();
    d <= 1e-13 || d <= 1e-11 * a.abs().max(b.abs())
}

fn close_c64(a: Complex64, b: Complex64) -> bool {
    let d = (a - b).norm();
    d <= 1e-13 || d <= 1e-11 * a.norm().max(b.norm())
}

#[test]
fn pf_m2_helpers_match_julia() {
    let f = load_fixture();
    let sr = slater_real(&f);
    let sc = slater_complex(&f);
    let stride = (2 * f.n_elec) * (2 * f.n_elec);

    let mut out_r = vec![0.0; f.n_qp];
    calculate_new_pf_m2_real_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        &mut out_r,
        &f.ele_idx,
        &sr,
        &f.inv_real,
        stride,
        &f.pf_real,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_r.iter().zip(f.out_real.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "out_real[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut out_rf = vec![0.0; f.n_qp];
    calculate_new_pf_m2_fsz_real_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        &mut out_rf,
        &f.ele_idx,
        &f.ele_spn,
        &sr,
        &f.inv_real,
        stride,
        &f.pf_real,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_rf.iter().zip(f.out_real_fsz.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "out_real_fsz[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut out_c = vec![Complex64::new(0.0, 0.0); f.n_qp];
    calculate_new_pf_m2_complex_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        &mut out_c,
        &f.ele_idx,
        &sc,
        &f.inv_complex,
        stride,
        &f.pf_complex,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_c.iter().zip(f.out_complex.iter()).enumerate() {
        assert!(close_c64(*a, *b), "out_complex[{i}] rust={a:?} julia={b:?}");
    }

    let mut out_cf = vec![Complex64::new(0.0, 0.0); f.n_qp];
    calculate_new_pf_m2_fsz_complex_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        &mut out_cf,
        &f.ele_idx,
        &f.ele_spn,
        &sc,
        &f.inv_complex,
        stride,
        &f.pf_complex,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_cf.iter().zip(f.out_complex_fsz.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "out_complex_fsz[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut out_rt = vec![0.0; f.n_qp];
    calculate_new_pf_m_two2_real_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        f.mb_spin.0,
        f.mb_spin.1,
        &mut out_rt,
        &f.ele_idx,
        &sr,
        &f.inv_real,
        stride,
        &f.pf_real,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_rt.iter().zip(f.out_real_two.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "out_real_two[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut out_rtf = vec![0.0; f.n_qp];
    calculate_new_pf_m_two_fsz_real_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        f.mb_spin_fsz.0,
        f.mb_spin_fsz.1,
        &mut out_rtf,
        &f.ele_idx,
        &f.ele_spn,
        &sr,
        &f.inv_real,
        stride,
        &f.pf_real,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_rtf.iter().zip(f.out_real_two_fsz.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "out_real_two_fsz[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut out_ct = vec![Complex64::new(0.0, 0.0); f.n_qp];
    calculate_new_pf_m_two2_complex_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        f.mb_spin.0,
        f.mb_spin.1,
        &mut out_ct,
        &f.ele_idx,
        &sc,
        &f.inv_complex,
        stride,
        &f.pf_complex,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_ct.iter().zip(f.out_complex_two.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "out_complex_two[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut out_ctf = vec![Complex64::new(0.0, 0.0); f.n_qp];
    calculate_new_pf_m_two_fsz_complex_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        f.mb_spin_fsz.0,
        f.mb_spin_fsz.1,
        &mut out_ctf,
        &f.ele_idx,
        &f.ele_spn,
        &sc,
        &f.inv_complex,
        stride,
        &f.pf_complex,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in out_ctf.iter().zip(f.out_complex_two_fsz.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "out_complex_two_fsz[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut pf_r = f.pf_real.clone();
    let mut inv_r = f.inv_real.clone();
    update_m_all_real_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        &f.ele_idx,
        &sr,
        &mut inv_r,
        stride,
        &mut pf_r,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_r.iter().zip(f.update_real_pf.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_real_pf[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }
    for (i, (a, b)) in inv_r.iter().zip(f.update_real_inv.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_real_inv[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut pf_rf = f.pf_real.clone();
    let mut inv_rf = f.inv_real.clone();
    update_m_all_fsz_real_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        &f.ele_idx,
        &f.ele_spn,
        &sr,
        &mut inv_rf,
        stride,
        &mut pf_rf,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_rf.iter().zip(f.update_real_fsz_pf.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_real_fsz_pf[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }
    for (i, (a, b)) in inv_rf.iter().zip(f.update_real_fsz_inv.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_real_fsz_inv[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut pf_c = f.pf_complex.clone();
    let mut inv_c = f.inv_complex.clone();
    update_m_all_complex_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        &f.ele_idx,
        &sc,
        &mut inv_c,
        stride,
        &mut pf_c,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_c.iter().zip(f.update_complex_pf.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "update_complex_pf[{i}] rust={a:?} julia={b:?}"
        );
    }
    for (i, (a, b)) in inv_c.iter().zip(f.update_complex_inv.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "update_complex_inv[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut pf_cf = f.pf_complex.clone();
    let mut inv_cf = f.inv_complex.clone();
    update_m_all_fsz_complex_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        &f.ele_idx,
        &f.ele_spn,
        &sc,
        &mut inv_cf,
        stride,
        &mut pf_cf,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_cf.iter().zip(f.update_complex_fsz_pf.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "update_complex_fsz_pf[{i}] rust={a:?} julia={b:?}"
        );
    }
    for (i, (a, b)) in inv_cf
        .iter()
        .zip(f.update_complex_fsz_inv.iter())
        .enumerate()
    {
        assert!(
            close_c64(*a, *b),
            "update_complex_fsz_inv[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut pf_tr = f.pf_real.clone();
    let mut inv_tr = f.inv_real.clone();
    update_m_all_two_real_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        f.mb_spin.0,
        f.mb_spin.1,
        f.old_sites.0,
        f.old_sites.1,
        &f.ele_idx,
        &sr,
        &mut inv_tr,
        stride,
        &mut pf_tr,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_tr.iter().zip(f.update_two_real_pf.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_two_real_pf[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }
    for (i, (a, b)) in inv_tr.iter().zip(f.update_two_real_inv.iter()).enumerate() {
        assert!(
            close_f64(*a, *b),
            "update_two_real_inv[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }

    let mut pf_tc = f.pf_complex.clone();
    let mut inv_tc = f.inv_complex.clone();
    update_m_all_two_complex_flat(
        f.ma_spin.0,
        f.ma_spin.1,
        f.mb_spin.0,
        f.mb_spin.1,
        f.old_sites.0,
        f.old_sites.1,
        &f.ele_idx,
        &sc,
        &mut inv_tc,
        stride,
        &mut pf_tc,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_tc.iter().zip(f.update_two_complex_pf.iter()).enumerate() {
        assert!(
            close_c64(*a, *b),
            "update_two_complex_pf[{i}] rust={a:?} julia={b:?}"
        );
    }
    for (i, (a, b)) in inv_tc
        .iter()
        .zip(f.update_two_complex_inv.iter())
        .enumerate()
    {
        assert!(
            close_c64(*a, *b),
            "update_two_complex_inv[{i}] rust={a:?} julia={b:?}"
        );
    }

    let mut pf_tfr = f.pf_real.clone();
    let mut inv_tfr = f.inv_real.clone();
    update_m_all_two_fsz_real_flat(
        f.ma_spin_fsz.0,
        f.ma_spin_fsz.1,
        f.mb_spin_fsz.0,
        f.mb_spin_fsz.1,
        f.old_sites.0,
        f.old_sites.1,
        &f.ele_idx,
        &f.ele_spn,
        &sr,
        &mut inv_tfr,
        stride,
        &mut pf_tfr,
        0,
        f.n_qp,
        f.n_site,
        f.n_elec,
    );
    for (i, (a, b)) in pf_tfr
        .iter()
        .zip(f.update_two_fsz_real_pf.iter())
        .enumerate()
    {
        assert!(
            close_f64(*a, *b),
            "update_two_fsz_real_pf[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }
    for (i, (a, b)) in inv_tfr
        .iter()
        .zip(f.update_two_fsz_real_inv.iter())
        .enumerate()
    {
        assert!(
            close_f64(*a, *b),
            "update_two_fsz_real_inv[{i}] rust={a:.17e} julia={b:.17e}"
        );
    }
}
