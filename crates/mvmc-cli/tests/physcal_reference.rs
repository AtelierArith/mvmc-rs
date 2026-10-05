//! Process-level output parity only: the CLI does not export RNG/configuration.
//! Normal fixtures contain independent Julia values with C output layout;
//! Lanczos outputs use historical native-C references. OptTrans is a labelled
//! phase-order harness, not original Julia/full-C runner parity.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct OutputDir(PathBuf);
impl OutputDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mvmc-cli-reference-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for OutputDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn check(model: &str, mode: &str, opt_trans: bool) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/two-samples")
        .join(model);
    let out = OutputDir::new();
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command
        .arg(root.join("inputs/namelist.def"))
        .arg("--physcal")
        .arg(root.join("zqp_opt.dat"))
        .args(["--seed", "1", "--mode", mode, "--out-dir"])
        .arg(&out.0)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1");
    if opt_trans {
        command.arg("--opt-trans");
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{model}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("Completed 2 PhysCal samples"));
    let mut expected_names = Vec::new();
    for file in fs::read_dir(root.join("expected")).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().into_string().unwrap();
        let expected = fs::read_to_string(file.path()).unwrap();
        let actual = fs::read_to_string(out.0.join(&name)).unwrap();
        let indices: &[usize] = if name.contains("cisajscktaltex") {
            &[]
        } else if name.contains("cisajscktalt") {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        } else if name.contains("cisajs") {
            &[0, 1, 2, 3]
        } else {
            &[]
        };
        // C outputData appends a terminal blank line to indexed OneBody/direct
        // Green files. This test checks numerical data-row shape, not terminal
        // whitespace or byte-format parity; never flatten/drop indexed rows.
        numerical_comparison::assert_numeric_text(
            actual.trim_end(),
            expected.trim_end(),
            1e-12,
            1e-10,
            indices,
            &name,
        );
        expected_names.push(name);
    }
    let mut actual_names = fs::read_dir(&out.0)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    // C InitFile always creates one `_time_` file (NDataIdxStart=7); its content
    // ends with a wall-clock ctime string and is checked in run_log_files.rs.
    expected_names.push("zvo_time_007.dat".to_string());
    expected_names.sort();
    actual_names.sort();
    assert_eq!(
        actual_names, expected_names,
        "exact C indexed-file lifecycle"
    );
}

#[test]
fn real_cli_matches_independent_indexed_outputs() {
    check("heisenberg_chain_real", "real", false);
}
#[test]
fn complex_cli_matches_independent_indexed_outputs() {
    check("heisenberg_chain_cmp", "cmp", false);
}
#[test]
fn fsz_cli_matches_independent_indexed_outputs() {
    check("heisenberg_chain_fsz", "fsz", false);
}
#[test]
fn opttrans_cli_matches_labelled_phase_order_outputs() {
    check("hubbard_chain_dh_opttrans", "real", true);
}

fn is_c_empty_lanczos_gex(bytes: &[u8]) -> bool {
    // Original C PhysCalLanczos_real/fcmp always writes one LF after the
    // zero-length GEx loop; this is not a generic whitespace-empty contract.
    bytes == b"\n"
}

#[test]
fn empty_lanczos_gex_requires_exact_c_lf() {
    assert!(is_c_empty_lanczos_gex(b"\n"));
    for invalid in [b"".as_slice(), b" ", b"\r\n", b"\n\n", b"0\n"] {
        assert!(
            !is_c_empty_lanczos_gex(invalid),
            "invalid bytes: {invalid:?}"
        );
    }
}

fn check_lanczos(model: &str, mode: i32, arithmetic: &str) {
    // Historical checked-in native C reference, revision and known numerical
    // divergence documented in its metadata.txt. No Julia/C runtime invocation.
    let reference = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/test/integration/reference")
        .join(model)
        .join("physcal_ref");
    let inputs = OutputDir::new();
    for entry in fs::read_dir(reference.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), inputs.0.join(entry.file_name())).unwrap();
    }
    let modpara = inputs.0.join("modpara.def");
    let mut replacements = 0;
    let edited = fs::read_to_string(&modpara)
        .unwrap()
        .lines()
        .map(|line| {
            if line.split_whitespace().next() == Some("NLanczosMode") {
                replacements += 1;
                format!("NLanczosMode {mode}")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(replacements, 1);
    fs::write(&modpara, edited).unwrap();
    let namelist = inputs.0.join("namelist.def");
    let parsed = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
    assert!(parsed.inter_all_terms.is_empty());
    if model.starts_with("hubbard") {
        assert!(!parsed.transfer_terms.is_empty() && !parsed.coulomb_intra_terms.is_empty());
    } else {
        assert!(!parsed.exchange_terms.is_empty());
        assert!(!parsed.coulomb_inter_terms.is_empty() || !parsed.hund_terms.is_empty());
    }
    let output = OutputDir::new();
    let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .arg("--physcal")
        .arg(reference.join("zqp_opt.dat"))
        .args(["--seed", "1", "--mode", arithmetic, "--out-dir"])
        .arg(&output.0)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{model} mode {mode}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut expected_names = Vec::new();
    for entry in fs::read_dir(reference.join("expected")).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if !name.starts_with("zvo_ls_") || (mode == 1 && name.contains("cisajs")) {
            continue;
        }
        if name == "zvo_ls_cisajscktaltex_001.dat"
            && parsed.green_two_ex_terms.is_empty()
            && parsed.green_two_ex_indices.is_empty()
        {
            assert!(entry.file_type().unwrap().is_file());
            assert!(is_c_empty_lanczos_gex(&fs::read(entry.path()).unwrap()));
            // Runtime bytes are checked below independently of reference presence.
            expected_names.push(name);
            continue;
        }
        let actual = fs::read_to_string(output.0.join(&name)).unwrap();
        let expected = fs::read_to_string(entry.path()).unwrap();
        let indices: &[usize] = if name.contains("cisajscktalt") {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        } else if name.contains("cisajs") {
            &[0, 1, 2, 3]
        } else {
            &[]
        };
        // Preserve existing independent Lanczos absolute budgets: moment
        // accumulation 1e-10; ill-conditioned alpha/corrected results 1e-8
        // (SpinChain C/Julia energy first divergence documented as 3.4e-9).
        let absolute = if name.contains("qqqq") { 1e-10 } else { 1e-8 };
        numerical_comparison::assert_numeric_text(
            actual.trim_end(),
            expected.trim_end(),
            absolute,
            0.0,
            indices,
            &name,
        );
        expected_names.push(name);
    }
    if mode == 2 {
        assert!(parsed.green_two_ex_terms.is_empty());
        let name = "zvo_ls_cisajscktaltex_001.dat";
        assert!(fs::symlink_metadata(output.0.join(name))
            .unwrap()
            .file_type()
            .is_file());
        assert!(parsed.green_two_ex_indices.is_empty());
        assert!(is_c_empty_lanczos_gex(
            &fs::read(output.0.join(name)).unwrap()
        ));
        if !expected_names.iter().any(|expected| expected == name) {
            expected_names.push(name.to_owned());
        }
    }
    let mut actual_names = fs::read_dir(&output.0)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.starts_with("zvo_ls_"))
        .collect::<Vec<_>>();
    expected_names.sort();
    actual_names.sort();
    assert_eq!(actual_names, expected_names);
}

#[test]
fn hopping_intra_cli_lanczos_modes_match_native_c_reference() {
    for mode in [1, 2] {
        check_lanczos("hubbard_chain_lanczos", mode, "real");
    }
}
#[test]
fn exchange_spin_cli_lanczos_modes_match_native_c_reference() {
    for mode in [1, 2] {
        check_lanczos("spin_chain_lanczos", mode, "real");
    }
}

#[test]
fn all_hamiltonian_terms_cli_lanczos_matches_independent_base_and_corrected_outputs() {
    // `--mode` is a checked label (#347): a `cmp` label on these real-declared inputs
    // is rejected, and it never selected a different arithmetic path before.
    for (mode, arithmetic) in [(1, "real"), (2, "real")] {
        // Existing Julia 1.13.1 values with C indexed layout; provenance.txt
        // records the source, seed and BLAS. Not a full native-C trajectory.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181")
            .join(format!("hubbard_all_terms_lanczos{mode}"));
        let parsed =
            mvmc_expert_parsers::parse_expert_mode_files(root.join("inputs/namelist.def")).unwrap();
        assert_eq!(parsed.modpara.lanczos_mode, mode);
        assert!(parsed.inter_all_terms.is_empty());
        assert!(!parsed.transfer_terms.is_empty());
        assert!(!parsed.coulomb_intra_terms.is_empty());
        assert!(!parsed.coulomb_inter_terms.is_empty());
        assert!(!parsed.hund_terms.is_empty());
        assert!(!parsed.exchange_terms.is_empty());
        assert!(!parsed.pair_hop_terms.is_empty());
        let out = OutputDir::new();
        let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(root.join("inputs/namelist.def"))
            .arg("--physcal")
            .arg(root.join("zqp_opt.dat"))
            .args(["--seed", "1", "--mode", arithmetic, "--out-dir"])
            .arg(&out.0)
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("Completed 1 PhysCal samples"));
        let mut expected_names = Vec::new();
        for entry in fs::read_dir(root.join("expected")).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            let indices: &[usize] = if name.contains("cisajscktaltex") {
                &[]
            } else if name.contains("cisajscktalt") {
                &[0, 1, 2, 3, 4, 5, 6, 7]
            } else if name.contains("cisajs") {
                &[0, 1, 2, 3]
            } else {
                &[]
            };
            // Same per-output budgets as physcal_issue181::assert_reference.
            let (atol, rtol) = if name.starts_with("zvo_ls_qqqq_") {
                (1e-10, 0.0)
            } else if name.starts_with("zvo_ls_") {
                (1e-8, 0.0)
            } else if name.contains("cisajscktalt") {
                (1e-12, 1e-9)
            } else {
                (1e-12, 1e-10)
            };
            let actual = fs::read_to_string(out.0.join(&name)).unwrap();
            let expected = fs::read_to_string(entry.path()).unwrap();
            numerical_comparison::assert_numeric_text(
                actual.trim_end(),
                expected.trim_end(),
                atol,
                rtol,
                indices,
                &name,
            );
            // The shared text helper checks ordered shape and exact discrete
            // columns. Also retain #181's stricter max, not sum, float budget.
            for (row, (a, e)) in actual
                .lines()
                .filter(|s| !s.trim().is_empty())
                .zip(expected.lines().filter(|s| !s.trim().is_empty()))
                .enumerate()
            {
                for (column, (a, e)) in a.split_whitespace().zip(e.split_whitespace()).enumerate() {
                    if indices.contains(&column) || e.parse::<i64>().is_ok() {
                        continue;
                    }
                    let a: f64 = a.parse().unwrap();
                    let e: f64 = e.parse().unwrap();
                    let matches = if a.is_nan() || e.is_nan() {
                        a.is_nan() && e.is_nan()
                    } else if !a.is_finite() || !e.is_finite() {
                        a == e
                    } else {
                        (a - e).abs() <= atol.max(rtol * a.abs().max(e.abs()))
                    };
                    assert!(
                        matches,
                        "{name} row {row} column {column}: {a:.17e} vs {e:.17e}"
                    );
                }
            }
            expected_names.push(name);
        }
        let mut actual_names = fs::read_dir(&out.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        // C InitFile always creates one `_time_` file (NDataIdxStart=1).
        expected_names.push("zvo_time_001.dat".to_string());
        expected_names.sort();
        actual_names.sort();
        assert_eq!(
            actual_names, expected_names,
            "mode {mode}: complete indexed inventory"
        );
    }
}
