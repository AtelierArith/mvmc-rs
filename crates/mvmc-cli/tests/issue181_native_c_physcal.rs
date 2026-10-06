//! Issue #181: serial PhysCal and Lanczos modes 1/2 against the NATIVE C `vmc.out`.
//!
//! References in `tests/fixtures/native_c_physcal_181/` come from the unmodified
//! vendored mVMC 1.3.0 executable (`c_toolbox/physcal_native/`); this test never
//! builds or runs C. Per scenario it checks
//!
//! * exact: file inventory, `zvo_time` deterministic columns (acceptance counters),
//!   saved configurations, projection counters, C `Counter[0..6]`, RNG draw count
//!   and the next 624 SFMT words after every sample (probe build, see provenance);
//! * class `c_defect_fsz_rbm` (issue #403): C's FSZ code has no RBM factor; Rust samples and
//!   measures with it, so only the file inventory is compared (the RBM-zeroed scenarios
//!   carry the numerical and exact-state comparison);
//! * toleranced: every `zvo_*` numerical output, with bounds justified by the
//!   first-divergence analysis in
//!   `docs/reference/c-to-julia/verification/issue-181-physcal-lanczos-matrix.md`
//!   (roundoff-order differences; FSZ is ill conditioned, Lanczos alpha amplifies
//!   moment roundoff by up to ~1e9).
//!
//! The Lanczos formula itself is checked separately and tightly by feeding the
//! native C moments to the Rust implementation.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Work(PathBuf);
impl Work {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "issue181-native-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/native_c_physcal_181")
}

struct Scenario {
    mode: String,
    opt_trans: bool,
    stages: Vec<String>,
    class: String,
}

fn scenario(name: &str) -> Scenario {
    let table = fs::read_to_string(root().join("scenarios.tsv")).unwrap();
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert!(f.len() >= 7, "{line}");
        if f[0] == name {
            return Scenario {
                mode: f[2].to_owned(),
                opt_trans: f[3] == "1",
                stages: f[4].split(';').map(str::to_owned).collect(),
                class: f[6].to_owned(),
            };
        }
    }
    panic!("scenario {name} is not listed in scenarios.tsv");
}

fn words(path: impl AsRef<Path>) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// (absolute, relative) bound for one output file; see the matrix document.
fn tolerance(name: &str, file: &str) -> (f64, f64) {
    if file.starts_with("zvo_ls_qqqq") {
        // Moments: accumulation order only, relative ~1e-15.
        (1e-13, 1e-12)
    } else if file.starts_with("zvo_ls_") {
        // Lanczos alpha-dependent values: moment roundoff (1 ulp) amplified by the
        // condition number of alpha, measured up to 8e8 (lanczos_sensitivity.py).
        (1e-7, 1e-6)
    } else if name.starts_with("fsz_warm0_") {
        // One configuration, FSZ: a 1e-16 relative input perturbation moves the
        // energy by up to 1.1e-11; Rust and C differ by 1.1e-10 (matrix document).
        (3e-10, 1e-9)
    } else if name.contains("fsz") {
        (1e-11, 1e-9)
    } else {
        (1e-13, 1e-10)
    }
}

/// Largest absolute and relative difference over the numeric tokens of two outputs
/// (measurement aid for choosing tolerances; enabled by `MVMC_RS_REPORT_MAXDIFF`).
fn max_differences(produced: &str, reference: &str) -> (f64, f64) {
    let (mut abs, mut rel) = (0.0_f64, 0.0_f64);
    for (a, b) in produced
        .split_whitespace()
        .zip(reference.split_whitespace())
    {
        if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
            if a.is_finite() && b.is_finite() {
                let d = (a - b).abs();
                abs = abs.max(d);
                let scale = a.abs().max(b.abs());
                if scale > 0.0 {
                    rel = rel.max(d / scale);
                }
            }
        }
    }
    (abs, rel)
}

fn run(name: &str, check_numbers: bool) {
    run_with(name, check_numbers, 1, 1);
}

/// `threads` sets `MVMC_RS_INNER_THREADS` (with threshold 1 so the worker pool is used
/// even for these small ranges); `ranks > 1` launches the binary through `mpiexec`
/// (set `MVMC_RS_MPIEXEC` to select the launcher). The native-C reference is the same
/// for every thread count; for ranks it is the C run with the same rank count.
fn run_with(name: &str, check_numbers: bool, threads: usize, ranks: usize) {
    let sc = scenario(name);
    assert!(
        matches!(sc.mode.as_str(), "real" | "cmp" | "fsz"),
        "{}",
        sc.mode
    );
    let dir = root().join(name);
    let work = Work::new(name);
    let out = work.0.join("out");
    let trace = work.0.join("trace");
    let last = sc.stages.len() - 1;
    for index in 0..=last {
        let inputs = work.0.join(format!("in{index}"));
        fs::create_dir(&inputs).unwrap();
        for entry in fs::read_dir(dir.join("inputs")).unwrap().flatten() {
            fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
        }
        if index > 0 {
            fs::copy(
                dir.join(format!("modpara-stage{index}.def")),
                inputs.join("modpara.def"),
            )
            .unwrap();
        }
        // `--mode` is a checked label (#347): pass the mode the inputs declare. The
        // scenario table's label never selected an arithmetic path, so a `cmp` label on
        // real-declared inputs would now be rejected.
        let parsed =
            mvmc_expert_parsers::parse_expert_mode_files(inputs.join("namelist.def")).unwrap();
        let declared = if parsed.i_flg_orbital_general != 0 {
            "fsz"
        } else if mvmc_core::get_all_complex_flag(&parsed).unwrap() {
            "cmp"
        } else {
            "real"
        };
        let mut command = if ranks > 1 {
            let mut launcher = Command::new(
                std::env::var("MVMC_RS_MPIEXEC").unwrap_or_else(|_| "mpiexec".to_owned()),
            );
            launcher.args(["-n", &ranks.to_string(), env!("CARGO_BIN_EXE_mvmc")]);
            launcher
        } else {
            Command::new(env!("CARGO_BIN_EXE_mvmc"))
        };
        if threads > 1 {
            command
                .env("MVMC_RS_INNER_THREADS", threads.to_string())
                .env("MVMC_RS_INNER_THRESHOLD", "1");
        }
        command
            .arg(inputs.join("namelist.def"))
            .arg("--physcal")
            .arg(dir.join("zqp_opt.dat"))
            .args(["--seed", "1", "--mode", declared, "--out-dir"])
            .arg(&out)
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1");
        if sc.opt_trans {
            command.arg("--opt-trans");
        }
        if index == last && ranks == 1 {
            command.arg("--physcal-trace").arg(&trace);
        }
        let result = command.output().unwrap();
        if sc.class == "c_rejected" || sc.class == "c_defect_not_reproduced" {
            // c_rejected: C aborts in the definition reader (nonzero exit, no zvo_ls_*).
            // c_defect_not_reproduced: C accepts the input through its `else if` in
            // readdef.c:610-617 and writes NaN moments (fixture shows it); Rust rejects.
            assert!(!result.status.success(), "{name}: Rust rejects this input");
            let leaked = out.exists()
                && fs::read_dir(&out)
                    .unwrap()
                    .flatten()
                    .any(|e| e.file_name().to_string_lossy().starts_with("zvo_ls_"));
            assert!(!leaked, "{name}: rejected run produced Lanczos files");
            // Version control drops an empty directory, so a missing one also means
            // "no C output".
            let c_empty =
                fs::read_dir(dir.join("expected")).map_or(true, |mut e| e.next().is_none());
            assert_eq!(
                c_empty,
                sc.class == "c_rejected",
                "{name}: native C evidence"
            );
            return;
        }
        assert!(
            result.status.success(),
            "{name} stage {index}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    // File inventory: C's indexed outputs plus its _time_ files (wall-clock body).
    let mut expected: Vec<String> = fs::read_dir(dir.join("expected"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    let time_files: Vec<String> = fs::read_to_string(dir.join("time-files.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("zvo_time_"))
        .map(str::to_owned)
        .collect();
    expected.extend(time_files.iter().cloned());
    expected.sort();
    let mut actual: Vec<String> = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    actual.sort();
    assert_eq!(actual, expected, "{name}: exact C file inventory");

    // zvo_time: everything but the trailing ctime() stamp.
    for file in time_files.iter().filter(|_| sc.class != "c_defect_fsz_rbm") {
        let rows = |text: String| -> Vec<String> {
            text.lines()
                .map(|l| l.rsplit_once(": ").map_or(l, |(head, _)| head).to_owned())
                .collect()
        };
        assert_eq!(
            rows(fs::read_to_string(out.join(file)).unwrap()),
            rows(fs::read_to_string(dir.join("time-rows").join(file)).unwrap())
                .into_iter()
                .collect::<Vec<_>>(),
            "{name}/{file}: acceptance columns (C Counter[0..6])"
        );
    }

    if ranks == 1 && sc.class != "c_defect_fsz_rbm" {
        // Saved state after every sample of the final stage: exact.
        // (--physcal-trace is serial-only; MPI cells are checked through their outputs.)
        let mut frames: Vec<_> = fs::read_dir(dir.join("native-state"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        frames.sort();
        assert!(!frames.is_empty());
        for frame in &frames {
            let k: usize = frame["state_dump_".len()..frame.len() - 4].parse().unwrap();
            let mut native = std::collections::HashMap::new();
            for line in fs::read_to_string(dir.join("native-state").join(frame))
                .unwrap()
                .lines()
            {
                let mut parts = line.split_whitespace();
                let key = parts.next().unwrap().to_owned();
                native.insert(key, parts.map(str::to_owned).collect::<Vec<_>>());
            }
            let stage = trace.join(format!("sample-{k}"));
            for key in ["ele_idx", "ele_cfg", "ele_num", "ele_proj_cnt", "ele_spn"] {
                assert_eq!(
                    words(stage.join(format!("{key}.txt"))),
                    native[key],
                    "{name} {frame} {key}"
                );
            }
            assert_eq!(
                words(stage.join("counter.txt"))[..6],
                native["counter"][..],
                "{name} {frame}: C Counter[0..6]"
            );
            assert_eq!(
                words(stage.join("draw-count.txt")),
                native["draws"],
                "{name} {frame}: primitive RNG draw count"
            );
            assert_eq!(
                words(stage.join("next624.txt")),
                native["next624"],
                "{name} {frame}: next 624 SFMT words"
            );
        }
    }

    // Numerical outputs.
    for file in fs::read_dir(dir.join("expected")).unwrap() {
        let file = file.unwrap().file_name().into_string().unwrap();
        let reference = fs::read_to_string(dir.join("expected").join(&file)).unwrap();
        let produced = fs::read_to_string(out.join(&file)).unwrap();
        if file.starts_with("zvo_ls_") && reference.is_empty() {
            // C PhysCalLanczos returned -1 (illegal alpha) and left zero-byte files.
            // Whether Rust's moments land on the same side of the negative-discriminant
            // test is roundoff-dependent, so this is asserted only for ordinary
            // scenarios; the writer contract has its own synthetic test.
            if sc.class != "c_singular" {
                assert!(
                    produced.is_empty(),
                    "{name}/{file}: C leaves this file empty"
                );
            }
            continue;
        }
        if !check_numbers
            || sc.class == "c_defect_fsz_rbm"
            || (file.starts_with("zvo_ls_") && sc.class == "c_singular")
        {
            continue;
        }
        let (absolute, relative) = tolerance(name, &file);
        if std::env::var_os("MVMC_RS_REPORT_MAXDIFF").is_some() {
            let (abs, rel) = max_differences(produced.trim_end(), reference.trim_end());
            eprintln!("NATIVE181 {name}/{file} max_abs {abs:.3e} max_rel {rel:.3e}");
        }
        numerical_comparison::assert_numeric_text(
            produced.trim_end(),
            reference.trim_end(),
            absolute,
            relative,
            &[],
            format!("{name}/{file}"),
        );
    }
}

macro_rules! native_scenarios {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            run($name, true);
        }
    )*};
}

native_scenarios! {
    hubbard_lanczos1_complex_path_matches_native_c => "hubbard_lanczos1_cmp",
    fsz_dh2_matches_native_c => "fsz_dh2_physcal",
    fsz_dh2_fixed_two_sz_matches_native_c => "fsz_dh2_csz_physcal",
    fsz_dh2_dh4_matches_native_c => "fsz_dh24_physcal",
    fsz_dh_opttrans_matches_native_c => "fsz_dh24_opttrans_physcal",
    fsz_rbm_inventory_matches_native_c_but_rust_applies_the_rbm_weight => "fsz_rbm_physcal",
    fsz_dh_rbm_opttrans_inventory_matches_native_c_but_rust_applies_the_rbm_weight => "fsz_dh24_rbm_opttrans_physcal",
    fsz_zero_rbm_matches_native_c => "fsz_rbm_zero_physcal",
    fsz_dh_zero_rbm_opttrans_matches_native_c => "fsz_dh24_rbm_opttrans_zero_physcal",
    heisenberg_real_matches_native_c => "heisenberg_chain_real",
    heisenberg_complex_matches_native_c => "heisenberg_chain_cmp",
    heisenberg_fsz_matches_native_c => "heisenberg_chain_fsz",
    hubbard_real_lanczos2_matches_native_c => "hubbard_chain_real",
    hubbard_dh_real_matches_native_c => "hubbard_chain_dh_real",
    kondo_real_matches_native_c => "kondo_chain_real",
    hubbard_dh_overlays_match_native_c => "hubbard_chain_dh_overlays",
    hubbard_dh_opttrans_matches_native_c => "hubbard_chain_dh_opttrans",
    hubbard_dh_rbm_opttrans_matches_native_c => "hubbard_chain_dh_rbm_opttrans",
    hubbard_lanczos1_matches_native_c => "hubbard_lanczos1",
    hubbard_lanczos2_complex_path_matches_native_c => "hubbard_lanczos2_cmp",
    heisenberg_real_lanczos1_matches_native_c => "heisenberg_real_lanczos1",
    heisenberg_real_lanczos2_matches_native_c => "heisenberg_real_lanczos2",
    heisenberg_complex_lanczos1_matches_native_c => "heisenberg_cmp_lanczos1",
    heisenberg_complex_lanczos2_matches_native_c => "heisenberg_cmp_lanczos2",
    heisenberg_fsz_lanczos1_is_rejected_like_c => "heisenberg_fsz_lanczos1",
    heisenberg_fsz_lanczos2_is_rejected_like_c => "heisenberg_fsz_lanczos2",
    heisenberg_fsz_lanczos_with_gauss_leg_is_a_c_defect_rust_rejects => "heisenberg_fsz_lanczos1_gauss8",
    kondo_real_lanczos1_matches_native_c => "kondo_real_lanczos1",
    kondo_real_lanczos2_matches_native_c => "kondo_real_lanczos2",
    hubbard_dh_real_lanczos2_matches_native_c => "hubbard_dh_real_lanczos2",
    hubbard_dh_real_lanczos1_matches_native_c => "hubbard_dh_real_lanczos1",
    all_terms_lanczos1_real_matches_native_c => "all_terms_lanczos1_real",
    all_terms_lanczos2_real_matches_native_c => "all_terms_lanczos2_real",
    all_terms_lanczos1_complex_matches_native_c => "all_terms_lanczos1_cmp",
    all_terms_lanczos2_complex_matches_native_c => "all_terms_lanczos2_cmp",
    hubbard_chain_lanczos1_matches_native_c => "hubbard_chain_lanczos1",
    hubbard_chain_lanczos2_matches_native_c => "hubbard_chain_lanczos2",
    spin_chain_lanczos1_matches_native_c => "spin_chain_lanczos1",
    spin_chain_lanczos2_matches_native_c => "spin_chain_lanczos2",
    indexed_start3_qty3_matches_native_c => "idx_start3_qty3",
    indexed_start1_qty3_lanczos2_matches_native_c => "idx_start1_qty3_lanczos2",
    rerun_truncation_matches_native_c => "rerun_truncate",
    rerun_with_lanczos_switched_off_keeps_stale_files_like_c => "rerun_lanczos_off",
    exact_eigenstate_real_lanczos_has_c_singular_alpha => "heisenberg_real_lanczos1_exact_state",
    exact_eigenstate_complex_lanczos_has_c_singular_alpha => "heisenberg_cmp_lanczos1_exact_state",
    heisenberg_real_gutzwiller_jastrow_orbital_overlays_match_native_c => "heisenberg_overlays_gjo",
    hubbard_dh2_only_overlay_matches_native_c => "hubbard_dh2_only_overlay",
    hubbard_dh4_only_overlay_matches_native_c => "hubbard_dh4_only_overlay",
    negative_index_start_matches_native_c => "idx_negative_start",
    fsz_single_configuration_matches_native_c => "fsz_warm0_sample1",
    fsz_two_configurations_match_native_c => "fsz_warm0_sample2",
    fsz_ten_configurations_match_native_c => "fsz_warm0_sample10",
    fsz_fifty_configurations_match_native_c => "fsz_warm10_sample50",
}

macro_rules! native_threaded {
    ($($test:ident => ($name:literal, $threads:literal)),* $(,)?) => {$(
        #[test]
        fn $test() {
            run_with($name, true, $threads, 1);
        }
    )*};
}

native_threaded! {
    heisenberg_real_workers2_match_native_c => ("heisenberg_chain_real", 2),
    heisenberg_real_workers4_match_native_c => ("heisenberg_chain_real", 4),
    heisenberg_complex_workers2_match_native_c => ("heisenberg_chain_cmp", 2),
    heisenberg_complex_workers4_match_native_c => ("heisenberg_chain_cmp", 4),
    heisenberg_fsz_workers2_match_native_c => ("heisenberg_chain_fsz", 2),
    heisenberg_fsz_workers4_match_native_c => ("heisenberg_chain_fsz", 4),
    hubbard_lanczos2_workers2_match_native_c => ("hubbard_chain_real", 2),
    hubbard_lanczos2_workers4_match_native_c => ("hubbard_chain_real", 4),
    hubbard_dh_workers2_match_native_c => ("hubbard_chain_dh_real", 2),
    hubbard_dh_workers4_match_native_c => ("hubbard_chain_dh_real", 4),
    kondo_workers2_match_native_c => ("kondo_chain_real", 2),
    kondo_workers4_match_native_c => ("kondo_chain_real", 4),
    hubbard_dh_rbm_opttrans_workers2_match_native_c => ("hubbard_chain_dh_rbm_opttrans", 2),
    hubbard_dh_rbm_opttrans_workers4_match_native_c => ("hubbard_chain_dh_rbm_opttrans", 4),
    fsz_dh_opttrans_workers2_match_native_c => ("fsz_dh24_opttrans_physcal", 2),
    fsz_dh_opttrans_workers4_match_native_c => ("fsz_dh24_opttrans_physcal", 4),
}

/// Optional MPI gate: needs `--features mpi`, an MPI launcher (`MVMC_RS_MPIEXEC`, default
/// `mpiexec`) and `MVMC_RS_NATIVE_C_MPI=1`. Selecting the ignored tests without the gate
/// fails instead of passing silently.
fn require_mpi_gate() {
    if !cfg!(feature = "mpi") {
        panic!("build mvmc-cli with --features mpi to run the native-C MPI gate");
    }
    assert_eq!(
        std::env::var("MVMC_RS_NATIVE_C_MPI").as_deref(),
        Ok("1"),
        "set MVMC_RS_NATIVE_C_MPI=1 to run the native-C MPI gate"
    );
}

macro_rules! native_mpi {
    ($($test:ident => ($name:literal, $ranks:literal)),* $(,)?) => {$(
        #[test]
        #[ignore = "optional MPI gate: --features mpi, MVMC_RS_NATIVE_C_MPI=1, MPI launcher"]
        fn $test() {
            require_mpi_gate();
            run_with($name, true, 1, $ranks);
        }
    )*};
}

native_mpi! {
    heisenberg_real_two_ranks_match_native_c => ("heisenberg_chain_real_mpi2", 2),
    heisenberg_complex_two_ranks_match_native_c => ("heisenberg_chain_cmp_mpi2", 2),
    heisenberg_fsz_two_ranks_match_native_c => ("heisenberg_chain_fsz_mpi2", 2),
    hubbard_lanczos2_two_ranks_match_native_c => ("hubbard_chain_real_mpi2", 2),
    hubbard_dh_two_ranks_match_native_c => ("hubbard_chain_dh_real_mpi2", 2),
    kondo_two_ranks_match_native_c => ("kondo_chain_real_mpi2", 2),
    hubbard_dh_rbm_opttrans_two_ranks_match_native_c => ("hubbard_chain_dh_rbm_opttrans_mpi2", 2),
    fsz_dh_opttrans_two_ranks_match_native_c => ("fsz_dh24_opttrans_physcal_mpi2", 2),
    heisenberg_real_four_ranks_match_native_c => ("heisenberg_chain_real_mpi4", 4),
    heisenberg_fsz_four_ranks_match_native_c => ("heisenberg_chain_fsz_mpi4", 4),
    hubbard_lanczos2_four_ranks_match_native_c => ("hubbard_chain_real_mpi4", 4),
    hubbard_dh_opttrans_four_ranks_match_native_c => ("hubbard_chain_dh_opttrans_mpi4", 4),
    kondo_four_ranks_match_native_c => ("kondo_chain_real_mpi4", 4),
}
