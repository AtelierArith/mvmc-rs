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
    // Inputs and fixed parameters from the integration reference; expected outputs
    // regenerated from C 1.4.0 (tests/fixtures/c_integration_reference/provenance.txt,
    // c_toolbox/integration_reference). No Julia/C runtime invocation.
    let reference = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/test/integration/reference")
        .join(model)
        .join("physcal_ref");
    let expected = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/c_integration_reference")
        .join(model)
        .join("physcal_ref/expected");
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
    for entry in fs::read_dir(&expected).unwrap() {
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
