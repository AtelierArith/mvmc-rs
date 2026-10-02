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
};
use mvmc_expert_parsers::{
    ChargeRBMHiddenLayerTerm, ChargeRBMPhysHiddenTerm, ChargeRBMPhysLayerTerm, ExpertModeData,
    GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm, GeneralRBMPhysLayerTerm,
    SpinRBMHiddenLayerTerm, SpinRBMPhysHiddenTerm, SpinRBMPhysLayerTerm,
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

fn build_data(f: &Fixture) -> ExpertModeData {
    let c = |re, im| Complex64::new(re, im);
    let mut data = ExpertModeData::new();
    data.rbm_section_widths = [2, 2, 2, 3, 2, 2, 1, 1, 1];
    data.rbm_params = vec![c(0.0, 0.0); 16];
    data.modpara.nsite = f.n_site as i64;
    data.modpara.nblock_size_rbm_ratio = f.nblock_size_rbm_ratio as i64;
    data.modpara.nneuron_charge = f.nneuron_charge as i64;
    data.modpara.nneuron_spin = f.nneuron_spin as i64;
    data.modpara.nneuron_general = f.nneuron_general as i64;
    data.charge_rbm_phys_layer_terms = vec![
        ChargeRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: c(0.11, 0.02),

            is_complex: true,
        },
        ChargeRBMPhysLayerTerm {
            site: 2,
            idx: 1,
            value: c(-0.03, 0.07),

            is_complex: true,
        },
    ];
    data.spin_rbm_phys_layer_terms = vec![
        SpinRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: c(0.05, -0.01),

            is_complex: true,
        },
        SpinRBMPhysLayerTerm {
            site: 3,
            idx: 1,
            value: c(-0.08, 0.02),

            is_complex: true,
        },
    ];
    data.general_rbm_phys_layer_terms = vec![
        GeneralRBMPhysLayerTerm {
            site: 1,
            spin: 0,
            idx: 0,
            value: c(0.04, 0.03),

            is_complex: true,
        },
        GeneralRBMPhysLayerTerm {
            site: 2,
            spin: 1,
            idx: 1,
            value: c(-0.06, -0.02),

            is_complex: true,
        },
    ];
    data.charge_rbm_hidden_layer_terms = vec![
        ChargeRBMHiddenLayerTerm {
            site: 0,
            value: c(0.20, -0.04),

            idx: 0,
            is_complex: true,
        },
        ChargeRBMHiddenLayerTerm {
            site: 1,
            value: c(-0.15, 0.05),

            idx: 1,
            is_complex: true,
        },
        ChargeRBMHiddenLayerTerm {
            site: 2,
            value: c(0.07, 0.01),

            idx: 2,
            is_complex: true,
        },
    ];
    data.spin_rbm_hidden_layer_terms = vec![
        SpinRBMHiddenLayerTerm {
            site: 0,
            value: c(-0.12, 0.08),

            idx: 0,
            is_complex: true,
        },
        SpinRBMHiddenLayerTerm {
            site: 1,
            value: c(0.09, -0.03),

            idx: 1,
            is_complex: true,
        },
    ];
    data.general_rbm_hidden_layer_terms = vec![
        GeneralRBMHiddenLayerTerm {
            site: 0,
            value: c(0.05, 0.02),

            idx: 0,
            is_complex: true,
        },
        GeneralRBMHiddenLayerTerm {
            site: 1,
            value: c(-0.04, 0.06),

            idx: 1,
            is_complex: true,
        },
    ];
    data.charge_rbm_phys_hidden_terms = vec![
        ChargeRBMPhysHiddenTerm {
            site1: 0,
            site2: 0,
            value: c(0.13, 0.02),

            idx: 0,
            is_complex: true,
        },
        ChargeRBMPhysHiddenTerm {
            site1: 1,
            site2: 1,
            value: c(-0.05, 0.04),

            idx: 0,
            is_complex: true,
        },
        ChargeRBMPhysHiddenTerm {
            site1: 3,
            site2: 2,
            value: c(0.03, -0.07),

            idx: 0,
            is_complex: true,
        },
    ];
    data.spin_rbm_phys_hidden_terms = vec![
        SpinRBMPhysHiddenTerm {
            site1: 0,
            site2: 0,
            value: c(-0.09, 0.02),

            idx: 0,
            is_complex: true,
        },
        SpinRBMPhysHiddenTerm {
            site1: 2,
            site2: 1,
            value: c(0.06, 0.05),

            idx: 0,
            is_complex: true,
        },
    ];
    data.general_rbm_phys_hidden_terms = vec![
        GeneralRBMPhysHiddenTerm {
            site1: 1,
            spin: 0,
            site2: 0,
            value: c(0.10, -0.03),

            idx: 0,
            is_complex: true,
        },
        GeneralRBMPhysHiddenTerm {
            site1: 2,
            spin: 1,
            site2: 1,
            value: c(-0.07, 0.09),

            idx: 0,
            is_complex: true,
        },
    ];
    data
}

fn close(a: Complex64, b: Complex64) -> bool {
    let diff = (a - b).norm();
    diff <= 1e-13 || diff <= 1e-11 * a.norm().max(b.norm())
}

#[test]
fn synthetic_rbm_matches_julia() {
    let fx = load_fixture(&fixture_path("synthetic.txt"));
    let data = build_data(&fx);
    let cfg = RbmConfig::from(&data);

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
