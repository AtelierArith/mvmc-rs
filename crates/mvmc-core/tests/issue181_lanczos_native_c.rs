//! Issue #181: the Lanczos formula fed with NATIVE C moments reproduces native C.
//!
//! Complements `mvmc-cli/tests/issue181_native_c_physcal.rs`: there Rust samples
//! and its moments differ from C's by ~1e-16 relative, which the ill-conditioned
//! alpha amplifies by up to ~1e9. Here the C moments (`zvo_ls_qqqq_*`) are the
//! input, so Rust must reproduce C's `zvo_ls_out_*` (energy, relative variance,
//! alpha) to roundoff, and must fail exactly where C returns -1 (zero-byte files).
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use num_complex::Complex64;
use std::{
    fs,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/native_c_physcal_181")
}

fn scenario_class(name: &str) -> String {
    let table = fs::read_to_string(root().join("scenarios.tsv")).unwrap();
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] == name {
            return f[6].to_owned();
        }
    }
    panic!("scenario {name} is not listed in scenarios.tsv");
}

/// The Lanczos formula, fed with the native C moments, reproduces C's outputs
/// (no sampling roundoff, so no alpha amplification).
fn lanczos_formula(name: &str) {
    let class = scenario_class(name);
    let dir = root().join(name).join("expected");
    let mut checked = 0;
    for entry in fs::read_dir(&dir).unwrap().flatten() {
        let file = entry.file_name().into_string().unwrap();
        let Some(tag) = file.strip_prefix("zvo_ls_qqqq_") else {
            continue;
        };
        let moments: Vec<Complex64> = fs::read_to_string(entry.path())
            .unwrap()
            .split_whitespace()
            .map(|v| Complex64::new(v.parse().unwrap(), 0.0))
            .collect();
        let out = fs::read_to_string(dir.join(format!("zvo_ls_out_{tag}"))).unwrap();
        if moments.is_empty() {
            assert!(out.is_empty());
            continue;
        }
        let result = mvmc_core::lanczos::lanczos_energy(&moments);
        if out.is_empty() {
            let error = result.expect_err("C returned -1 for these moments");
            assert!(error.is_c_early_return(), "{name}/{file}: {error:?}");
            checked += 1;
            continue;
        }
        if class == "c_singular" {
            continue; // C output is uninitialised memory or a near-singular alpha
        }
        let c: Vec<f64> = out.split_whitespace().map(|v| v.parse().unwrap()).collect();
        let r = result.unwrap();
        // Linux/glibc pow(x, 3) is the reference platform; another libm may differ
        // by one ulp, which alpha amplifies, so other platforms use the end-to-end bound.
        let relative = if cfg!(all(target_os = "linux", target_env = "gnu")) {
            1e-13
        } else {
            1e-6
        };
        for (label, rust, native) in [
            ("energy", r.energy, c[0]),
            ("variance", r.variance, c[1]),
            ("alpha", r.alpha, c[2]),
        ] {
            numerical_comparison::assert_close(
                rust,
                native,
                0.0,
                relative,
                format!("{name}/{file} {label}"),
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "{name}: no Lanczos moments checked");
}

macro_rules! native_lanczos {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            lanczos_formula($name);
        }
    )*};
}

native_lanczos! {
    lanczos_formula_reproduces_native_c_hubbard_lanczos1 => "hubbard_lanczos1",
    lanczos_formula_reproduces_native_c_hubbard_lanczos2 => "hubbard_chain_real",
    lanczos_formula_reproduces_native_c_heisenberg_real => "heisenberg_real_lanczos2",
    lanczos_formula_reproduces_native_c_heisenberg_complex => "heisenberg_cmp_lanczos2",
    lanczos_formula_reproduces_native_c_kondo => "kondo_real_lanczos2",
    lanczos_formula_reproduces_native_c_all_terms => "all_terms_lanczos2_real",
    lanczos_formula_reproduces_native_c_spin_chain => "spin_chain_lanczos1",
    lanczos_formula_reproduces_native_c_hubbard_dh => "hubbard_dh_real_lanczos2",
}

#[test]
fn lanczos_failure_leaves_every_ls_file_empty_like_c() {
    // C PhysCalLanczos_* returns -1 when CalculateEne fails (negative discriminant)
    // before any fprintf; InitFilePhysCal had already created all zvo_ls_* files.
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.lanczos_mode = 2;
    data.modpara.n_data_idx_start = 7;
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    let mut phys = mvmc_core::state::PhysicalQuantities::zeros(0, 0, 0);
    // Moments of the lanczos.rs negative-discriminant unit test.
    for (slot, value) in [
        (2, -2.152457446309748),
        (3, -2.9568890514902515),
        (10, 0.6960797422130414),
        (11, 3.1910420776032336),
        (15, 2.2685266679515284),
    ] {
        phys.phys_lanczos_qqqq[slot] = Complex64::new(value, 0.0);
    }
    state.phys_quantities = Some(phys);
    let out = std::env::temp_dir().join(format!("issue181-lanczos-fail-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).unwrap();
    mvmc_core::io::output_phys_data(&data, &state, 0, Some(&out)).unwrap();
    for file in [
        "zvo_ls_out_007.dat",
        "zvo_ls_qqqq_007.dat",
        "zvo_ls_cisajs_007.dat",
        "zvo_ls_cisajscktalt_007.dat",
        "zvo_ls_cisajscktaltex_007.dat",
    ] {
        let meta = fs::metadata(out.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(meta.len(), 0, "{file} must be an empty file");
    }
    // The ordinary moment/OneBody/direct files are still written.
    assert!(out.join("zvo_out_007.dat").exists());
    let _ = fs::remove_dir_all(out);
}
