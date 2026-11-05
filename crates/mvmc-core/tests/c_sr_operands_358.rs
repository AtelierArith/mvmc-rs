//! Step-1 sampled SR operands against the instrumented native C 1.4.0 `vmc.out`
//! (issue #358).
//!
//! `tests/fixtures/c_order_sr_operands/` holds, per case, the post-`WeightAverage`
//! `SROptOO`/`SROptHO`/`SROptO` (or `SROptO_Store`) arrays and the step-1 energy of
//! one optimization step of C (`c_toolbox/sr_operand_dump/`, provenance in
//! `PROVENANCE.md`), and the inputs C ran under `inputs/<case>/`. This test runs the
//! production entry point on the same inputs with the same modpara overrides
//! (`NSROptItrStep = NSROptItrSmp = 1`, `RndSeed 1`, `NSRCG`/`NStore` per file,
//! `NSRCGAbortOnFail 0`). The operands are formed before the SR solve, so they
//! do not pass through the amplified CG recurrence. Rust tests never run C.

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::{run_para_opt_from_namelist_observed, InitialDef, RunConfig, SingleProcessReducer};

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/c_order_sr_operands")
}

/// The modpara overrides of `c_toolbox/sr_operand_dump/capture.py`.
fn prepare(case: &str, cg: bool, store: i64, dest: &Path) -> PathBuf {
    let source = fixture_root().join("inputs").join(case);
    fs::create_dir_all(dest).unwrap();
    for entry in fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), dest.join(entry.file_name())).unwrap();
    }
    let modpara = dest.join("modpara.def");
    let mut lines: Vec<String> = fs::read_to_string(&modpara)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    for (key, value) in [
        ("NSROptItrStep", "1".to_owned()),
        ("NSROptItrSmp", "1".to_owned()),
        ("NSRCG", i64::from(cg).to_string()),
        ("NStore", store.to_string()),
        ("RndSeed", "1".to_owned()),
        ("NSRCGAbortOnFail", "0".to_owned()),
    ] {
        match lines
            .iter_mut()
            .find(|line| line.split_whitespace().next() == Some(key))
        {
            Some(line) => *line = format!("{key} {value}"),
            None => lines.push(format!("{key} {value}")),
        }
    }
    fs::write(&modpara, lines.join("\n") + "\n").unwrap();
    dest.join("namelist.def")
}

fn record(text: &str, path: &Path, name: &str) -> Vec<f64> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| line.strip_prefix(&format!("{name} ")))
        .map(|values| {
            values
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect()
        })
        .unwrap_or_else(|| panic!("{path:?} lacks {name}"))
}

fn check(case: &str, cg: bool, store: i64) {
    let path = fixture_root().join(format!(
        "{case}-{}-store{store}.txt",
        if cg { "cg" } else { "direct" }
    ));
    let text = fs::read_to_string(&path).unwrap();
    let work = std::env::temp_dir().join(format!(
        "mvmc-c-sr-operands-{case}-{cg}-{store}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&work);
    let namelist = prepare(case, cg, store, &work);
    let mut config = RunConfig::new(1, "real");
    config.initial_def = InitialDef::None;
    config.output_dir = Some(work.join("output"));
    config.enable_opt_trans = Some(case == "opt_real");
    let (_, state, _) =
        run_para_opt_from_namelist_observed(&namelist, config, &SingleProcessReducer).unwrap();
    fs::remove_dir_all(&work).unwrap();

    // First divergence from C is the last-bit summation order of the sampled sums
    // (observed <= ~2 ulp, docs/NUMERICAL_COMPARISONS.md); ABS covers exact-zero C
    // entries that hold roundoff (~1e-18).
    const ABS: f64 = 1e-14;
    const REL: f64 = 1e-13;
    let size = state.sr_opt.sr_opt_size;
    let mut checks = vec![
        ("ho", state.sr_opt.sr_opt_ho_real.clone()),
        ("o", state.sr_opt.sr_opt_o_real.clone()),
    ];
    if store == 0 {
        checks.push(("oo", state.sr_opt.sr_opt_oo_real.clone()));
    } else {
        checks.push(("o_store", state.sr_opt.sr_opt_o_store_real.clone()));
    }
    for (name, actual) in checks {
        let expected = record(&text, &path, name);
        // Rust's buffers are larger; the leading elements use C's layout.
        assert!(actual.len() >= expected.len(), "{case} {name} length");
        // C writes real-mode OptTrans derivatives through a complex pointer offset
        // (vmccal.c calculateOptTransDiff: slot 2*k+2 holds parameter k), so with
        // NQPOptTrans > 1 the last slots are wrong in C (#370, C defect not
        // reproduced); those entries are excluded.
        let skip = |i: usize| case == "opt_real" && i % size >= size - 2;
        let pairs: Vec<(f64, f64)> = (0..expected.len())
            .filter(|&i| !skip(i))
            .map(|i| (actual[i], expected[i]))
            .collect();
        let worst = pairs
            .iter()
            .map(|(a, e)| (a - e).abs() / (ABS / REL + e.abs()))
            .fold(0.0, f64::max);
        // Observed worst scaled error |d| / (ABS/REL + |c|) on Linux x86_64 is
        // 8.6e-16 (fixture PROVENANCE.md); a much larger value means the
        // operands no longer follow C's sampling, not roundoff.
        assert!(worst < 1e-2, "{case} {name}: scaled error {worst:e}");
        numerical_comparison::assert_values_close(
            pairs.iter().map(|p| p.0),
            pairs.iter().map(|p| p.1),
            ABS,
            REL,
            format!("{case} step-1 native C {name}"),
        );
    }
    numerical_comparison::assert_close(
        state.energy.etot.re,
        record(&text, &path, "energy")[0],
        1e-12,
        1e-12,
        format!("{case} step-1 native C energy"),
    );
}

#[test]
fn real_cg_step_one_operands_match_native_c() {
    check("real", true, 0);
}

#[test]
fn real_direct_step_one_operands_match_native_c() {
    check("real", false, 0);
}

#[test]
fn real_direct_stored_step_one_operands_match_native_c() {
    check("real", false, 1);
}

#[test]
fn hubbard_cg_step_one_operands_match_native_c() {
    check("hubbard", true, 0);
}

#[test]
fn dh2_real_cg_step_one_operands_match_native_c() {
    check("dh2_real", true, 0);
}

#[test]
fn dh4_real_cg_step_one_operands_match_native_c() {
    check("dh4_real", true, 0);
}

#[test]
fn dh24_real_cg_step_one_operands_match_native_c() {
    check("dh24_real", true, 0);
}

#[test]
fn pairhop_real_cg_step_one_operands_match_native_c() {
    check("pairhop_real", true, 0);
}

#[test]
fn opt_real_cg_step_one_operands_match_native_c() {
    check("opt_real", true, 0);
}
