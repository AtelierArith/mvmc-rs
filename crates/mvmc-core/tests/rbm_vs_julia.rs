//! Phase 4.3.2 parity for synthetic RBM helpers.
//!
//! The fixture is produced by `extern/Julia-mVMC/tools/dump_rbm_reference.jl`,
//! which builds a small RBM-bearing `ExpertModeData` by hand and calls
//! Julia's `make_rbm_cnt`, `update_rbm_cnt_hopping!`, `log_rbm_val`,
//! and `log_rbm_ratio`. Rust constructs the same synthetic [`RbmConfig`]
//! and compares all outputs.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use mvmc_core::sampling::rbm::{
    log_rbm_ratio, log_rbm_val, make_rbm_cnt, update_rbm_cnt_hopping, RbmConfig,
    RbmGeneralPhysHiddenTerm, RbmGeneralPhysLayerTerm, RbmHiddenLayerTerm, RbmPhysHiddenTerm,
    RbmPhysLayerTerm,
};
use num_complex::Complex64;

const FIXTURE_DIR: &str = "../../tests/fixtures/rbm";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

#[derive(Debug)]
struct Fixture {
    n_site: usize,
    nblock_size_rbm_ratio: usize,
    nneuron_charge: usize,
    nneuron_spin: usize,
    nneuron_general: usize,
    hop: (i64, i64, u8),
    ele_old: Vec<i64>,
    ele_new: Vec<i64>,
    cnt_old: Vec<Complex64>,
    cnt_new_update: Vec<Complex64>,
    cnt_new_fresh: Vec<Complex64>,
    log_val_old: Complex64,
    log_val_new: Complex64,
    log_ratio: Complex64,
}

fn parse_i64s(rest: &str) -> Vec<i64> {
    rest.split_ascii_whitespace()
        .map(|s| s.parse::<i64>().unwrap())
        .collect()
}

fn parse_complex_line(line: &str) -> Complex64 {
    let mut it = line.split_ascii_whitespace();
    let re = it.next().unwrap().parse::<f64>().unwrap();
    let im = it.next().unwrap().parse::<f64>().unwrap();
    Complex64::new(re, im)
}

fn load_fixture(path: &Path) -> Fixture {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to open fixture {} (did you run dump_rbm_reference.jl?): {}",
            path.display(),
            e
        )
    });
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();

    let mut n_site = None;
    let mut nblock = None;
    let mut nn_charge = None;
    let mut nn_spin = None;
    let mut nn_general = None;
    let mut hop = None;
    let mut ele_old = Vec::new();
    let mut ele_new = Vec::new();
    let mut cnt_old = Vec::new();
    let mut cnt_new_update = Vec::new();
    let mut cnt_new_fresh = Vec::new();
    let mut log_val_old = None;
    let mut log_val_new = None;
    let mut log_ratio = None;

    let mut idx = 0usize;
    while idx < lines.len() {
        let t = lines[idx].trim();
        if let Some(rest) = t.strip_prefix("n_site ") {
            n_site = Some(rest.parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("nblock_size_rbm_ratio ") {
            nblock = Some(rest.parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("nneuron_charge ") {
            nn_charge = Some(rest.parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("nneuron_spin ") {
            nn_spin = Some(rest.parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("nneuron_general ") {
            nn_general = Some(rest.parse().unwrap());
        } else if let Some(rest) = t.strip_prefix("hop ") {
            let ints = parse_i64s(rest);
            hop = Some((ints[0], ints[1], ints[2] as u8));
        } else if let Some(rest) = t.strip_prefix("ele_old ") {
            ele_old = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_new ") {
            ele_new = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("cnt_old_len ") {
            let len = rest.parse::<usize>().unwrap();
            idx += 2; // skip header line `cnt_old`
            cnt_old = lines[idx..idx + len]
                .iter()
                .map(|s| parse_complex_line(s))
                .collect();
            idx += len - 1;
        } else if let Some(rest) = t.strip_prefix("cnt_new_update_len ") {
            let len = rest.parse::<usize>().unwrap();
            idx += 2; // skip `cnt_new_update`
            cnt_new_update = lines[idx..idx + len]
                .iter()
                .map(|s| parse_complex_line(s))
                .collect();
            idx += len - 1;
        } else if let Some(rest) = t.strip_prefix("cnt_new_fresh_len ") {
            let len = rest.parse::<usize>().unwrap();
            idx += 2; // skip `cnt_new_fresh`
            cnt_new_fresh = lines[idx..idx + len]
                .iter()
                .map(|s| parse_complex_line(s))
                .collect();
            idx += len - 1;
        } else if let Some(rest) = t.strip_prefix("log_val_old ") {
            log_val_old = Some(parse_complex_line(rest));
        } else if let Some(rest) = t.strip_prefix("log_val_new ") {
            log_val_new = Some(parse_complex_line(rest));
        } else if let Some(rest) = t.strip_prefix("log_ratio ") {
            log_ratio = Some(parse_complex_line(rest));
        }
        idx += 1;
    }

    Fixture {
        n_site: n_site.unwrap(),
        nblock_size_rbm_ratio: nblock.unwrap(),
        nneuron_charge: nn_charge.unwrap(),
        nneuron_spin: nn_spin.unwrap(),
        nneuron_general: nn_general.unwrap(),
        hop: hop.unwrap(),
        ele_old,
        ele_new,
        cnt_old,
        cnt_new_update,
        cnt_new_fresh,
        log_val_old: log_val_old.unwrap(),
        log_val_new: log_val_new.unwrap(),
        log_ratio: log_ratio.unwrap(),
    }
}

fn synthetic_config<'a>(
    f: &Fixture,
    charge_phys: &'a [RbmPhysLayerTerm],
    spin_phys: &'a [RbmPhysLayerTerm],
    general_phys: &'a [RbmGeneralPhysLayerTerm],
    charge_hidden: &'a [RbmHiddenLayerTerm],
    spin_hidden: &'a [RbmHiddenLayerTerm],
    general_hidden: &'a [RbmHiddenLayerTerm],
    charge_ph: &'a [RbmPhysHiddenTerm],
    spin_ph: &'a [RbmPhysHiddenTerm],
    general_ph: &'a [RbmGeneralPhysHiddenTerm],
) -> RbmConfig<'a> {
    RbmConfig {
        n_site: f.n_site,
        nblock_size_rbm_ratio: f.nblock_size_rbm_ratio,
        nneuron_charge: f.nneuron_charge,
        nneuron_spin: f.nneuron_spin,
        nneuron_general: f.nneuron_general,
        charge_phys,
        spin_phys,
        general_phys,
        charge_hidden,
        spin_hidden,
        general_hidden,
        charge_phys_hidden: charge_ph,
        spin_phys_hidden: spin_ph,
        general_phys_hidden: general_ph,
    }
}

fn build_terms() -> (
    Vec<RbmPhysLayerTerm>,
    Vec<RbmPhysLayerTerm>,
    Vec<RbmGeneralPhysLayerTerm>,
    Vec<RbmHiddenLayerTerm>,
    Vec<RbmHiddenLayerTerm>,
    Vec<RbmHiddenLayerTerm>,
    Vec<RbmPhysHiddenTerm>,
    Vec<RbmPhysHiddenTerm>,
    Vec<RbmGeneralPhysHiddenTerm>,
) {
    let c = |re, im| Complex64::new(re, im);
    (
        vec![
            RbmPhysLayerTerm {
                site: 0,
                idx: 0,
                value: c(0.11, 0.02),
            },
            RbmPhysLayerTerm {
                site: 2,
                idx: 1,
                value: c(-0.03, 0.07),
            },
        ],
        vec![
            RbmPhysLayerTerm {
                site: 0,
                idx: 0,
                value: c(0.05, -0.01),
            },
            RbmPhysLayerTerm {
                site: 3,
                idx: 1,
                value: c(-0.08, 0.02),
            },
        ],
        vec![
            RbmGeneralPhysLayerTerm {
                site: 1,
                spin: 0,
                idx: 0,
                value: c(0.04, 0.03),
            },
            RbmGeneralPhysLayerTerm {
                site: 2,
                spin: 1,
                idx: 1,
                value: c(-0.06, -0.02),
            },
        ],
        vec![
            RbmHiddenLayerTerm {
                site: 0,
                value: c(0.20, -0.04),
            },
            RbmHiddenLayerTerm {
                site: 1,
                value: c(-0.15, 0.05),
            },
            RbmHiddenLayerTerm {
                site: 2,
                value: c(0.07, 0.01),
            },
        ],
        vec![
            RbmHiddenLayerTerm {
                site: 0,
                value: c(-0.12, 0.08),
            },
            RbmHiddenLayerTerm {
                site: 1,
                value: c(0.09, -0.03),
            },
        ],
        vec![
            RbmHiddenLayerTerm {
                site: 0,
                value: c(0.05, 0.02),
            },
            RbmHiddenLayerTerm {
                site: 1,
                value: c(-0.04, 0.06),
            },
        ],
        vec![
            RbmPhysHiddenTerm {
                site1: 0,
                site2: 0,
                value: c(0.13, 0.02),
            },
            RbmPhysHiddenTerm {
                site1: 1,
                site2: 1,
                value: c(-0.05, 0.04),
            },
            RbmPhysHiddenTerm {
                site1: 3,
                site2: 2,
                value: c(0.03, -0.07),
            },
        ],
        vec![
            RbmPhysHiddenTerm {
                site1: 0,
                site2: 0,
                value: c(-0.09, 0.02),
            },
            RbmPhysHiddenTerm {
                site1: 2,
                site2: 1,
                value: c(0.06, 0.05),
            },
        ],
        vec![
            RbmGeneralPhysHiddenTerm {
                site1: 1,
                spin: 0,
                site2: 0,
                value: c(0.10, -0.03),
            },
            RbmGeneralPhysHiddenTerm {
                site1: 2,
                spin: 1,
                site2: 1,
                value: c(-0.07, 0.09),
            },
        ],
    )
}

fn close(a: Complex64, b: Complex64) -> bool {
    let diff = (a - b).norm();
    diff <= 1e-13 || diff <= 1e-11 * a.norm().max(b.norm())
}

#[test]
fn synthetic_rbm_matches_julia() {
    let fx = load_fixture(&fixture_path("synthetic.txt"));
    let (
        charge_phys,
        spin_phys,
        general_phys,
        charge_hidden,
        spin_hidden,
        general_hidden,
        charge_ph,
        spin_ph,
        general_ph,
    ) = build_terms();
    let cfg = synthetic_config(
        &fx,
        &charge_phys,
        &spin_phys,
        &general_phys,
        &charge_hidden,
        &spin_hidden,
        &general_hidden,
        &charge_ph,
        &spin_ph,
        &general_ph,
    );

    let cnt_old = make_rbm_cnt(&fx.ele_old, &cfg);
    assert_eq!(cnt_old.len(), fx.cnt_old.len());
    for (i, (got, want)) in cnt_old.iter().zip(fx.cnt_old.iter()).enumerate() {
        assert!(
            close(*got, *want),
            "cnt_old[{i}] rust={got:?} julia={want:?}"
        );
    }

    let (ri, rj, spin) = fx.hop;
    let mut cnt_new = cnt_old.clone();
    update_rbm_cnt_hopping(&mut cnt_new, &cnt_old, ri, rj, spin, &cfg);
    for (i, (got, want)) in cnt_new.iter().zip(fx.cnt_new_update.iter()).enumerate() {
        assert!(
            close(*got, *want),
            "cnt_new_update[{i}] rust={got:?} julia={want:?}"
        );
    }

    let cnt_new_fresh = make_rbm_cnt(&fx.ele_new, &cfg);
    for (i, (got, want)) in cnt_new_fresh
        .iter()
        .zip(fx.cnt_new_fresh.iter())
        .enumerate()
    {
        assert!(
            close(*got, *want),
            "cnt_new_fresh[{i}] rust={got:?} julia={want:?}"
        );
    }

    let log_old = log_rbm_val(&fx.ele_old, &cfg);
    let log_new = log_rbm_val(&fx.ele_new, &cfg);
    let log_ratio = log_rbm_ratio(&cnt_new, &cnt_old, &cfg);
    assert!(
        close(log_old, fx.log_val_old),
        "log_val_old rust={log_old:?} julia={:?}",
        fx.log_val_old
    );
    assert!(
        close(log_new, fx.log_val_new),
        "log_val_new rust={log_new:?} julia={:?}",
        fx.log_val_new
    );
    assert!(
        close(log_ratio, fx.log_ratio),
        "log_ratio rust={log_ratio:?} julia={:?}",
        fx.log_ratio
    );
}
