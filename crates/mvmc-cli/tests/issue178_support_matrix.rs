//! Actual subprocess counterparts to the public runtime support matrix.
//! Grouped restrictions are Julia feature limits, not blanket C-invalid input.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Owned(PathBuf);
impl Owned {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue178-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn input(parent: &Path, model: &str, phys: bool, changes: &[(&str, i64)]) -> (PathBuf, PathBuf) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(model);
    let dir = parent.join("inputs");
    fs::create_dir(&dir).unwrap();
    for entry in fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
    }
    let namelist = dir.join("namelist.def");
    if !phys {
        let text = fs::read_to_string(&namelist).unwrap();
        fs::write(
            &namelist,
            text.lines()
                .filter(|line| line.split_whitespace().next() != Some("TwoBodyGEx"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
    }
    let settings = [
        ("NVMCCalMode", i64::from(phys)),
        ("NSROptItrStep", 1),
        ("NSROptItrSmp", 1),
        ("NVMCWarmUp", 1),
        ("NVMCInterval", 1),
        ("NDataQtySmp", 1),
    ];
    let modpara = dir.join("modpara.def");
    let mut text = fs::read_to_string(&modpara).unwrap();
    for &(name, value) in settings.iter().chain(changes) {
        text = text
            .lines()
            .filter(|line| line.split_whitespace().next() != Some(name))
            .collect::<Vec<_>>()
            .join("\n");
        text.push_str(&format!("\n{name} {value}\n"));
    }
    fs::write(modpara, text).unwrap();
    let fixed = parent.join("fixed.dat");
    fs::copy(source.join("zqp_opt.dat"), &fixed).unwrap();
    (namelist, fixed)
}
fn command(input: &Path, fixed: &Path, out: &Path, phys: bool, opt_trans: bool) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command
        .arg(input)
        .args([
            "--seed",
            "11272",
            "--nsteps",
            "1",
            "--nsmp",
            "1",
            "--initial-def",
            "none",
            "--out-dir",
        ])
        .arg(out);
    if phys {
        command.arg("--physcal").arg(fixed);
    }
    if opt_trans {
        command.arg("--opt-trans");
    }
    command
}
fn failure(output: Output, diagnostic: &str) {
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains(diagnostic),
        "wrong actual subprocess error: {stderr}"
    );
}
#[test]
fn grouped_matrix_rejections_reach_actual_cli_before_output_creation() {
    for (model, phys, opt, changes, diagnostic) in [
        (
            "heisenberg_chain_real",
            false,
            false,
            vec![("NSplitSize", 2), ("NSRCG", 1)],
            "undefined in mVMC C",
        ),
        (
            "heisenberg_chain_fsz",
            true,
            false,
            vec![("NSplitSize", 2)],
            "MPI group communicator",
        ),
        (
            "heisenberg_chain_fsz",
            false,
            false,
            vec![("NSplitSize", 2), ("NSPGaussLeg", 2)],
            "MPI group communicator",
        ),
        (
            "heisenberg_chain_fsz",
            false,
            false,
            vec![("NSplitSize", 2), ("NMPTrans", 2)],
            "MPI group communicator",
        ),
        (
            "heisenberg_chain_fsz",
            false,
            false,
            vec![("NSplitSize", 2), ("NMPTrans", -2)],
            "MPI group communicator",
        ),
        (
            "hubbard_chain_dh_opttrans",
            false,
            true,
            vec![("NSplitSize", 2)],
            "MPI group communicator",
        ),
        (
            "hubbard_chain_dh_opttrans",
            true,
            true,
            vec![("NSplitSize", 2)],
            "MPI group communicator",
        ),
        (
            "heisenberg_chain_real",
            false,
            false,
            vec![("NLanczosMode", 1)],
            "NLanczosMode",
        ),
        (
            "heisenberg_chain_real",
            false,
            false,
            vec![("NLanczosMode", 2)],
            "NLanczosMode",
        ),
        (
            "heisenberg_chain_real",
            true,
            false,
            vec![("NSplitSize", 2), ("NLanczosMode", 1)],
            "MPI group communicator",
        ),
        (
            "heisenberg_chain_real",
            true,
            false,
            vec![("NSplitSize", 2), ("NLanczosMode", 2)],
            "MPI group communicator",
        ),
    ] {
        let root = Owned::new();
        let (input, fixed) = input(&root.0, model, phys, &changes);
        let before = fs::read(&input).unwrap();
        let params = fs::read(&fixed).unwrap();
        let out = root.0.join("must-not-exist");
        failure(
            command(&input, &fixed, &out, phys, opt).output().unwrap(),
            diagnostic,
        );
        assert!(!out.exists());
        assert_eq!(fs::read(&input).unwrap(), before);
        assert_eq!(fs::read(&fixed).unwrap(), params);
    }
}
#[test]
fn signed_projection_physcal_with_cg_metadata_is_not_overrejected_by_cli() {
    for nmp in [1, -1] {
        let root = Owned::new();
        let (input, fixed) = input(
            &root.0,
            "heisenberg_chain_real",
            true,
            &[
                ("NSplitSize", 1),
                ("NMPTrans", nmp),
                ("NSRCG", 1),
                ("NVMCSample", 3),
            ],
        );
        let out = root.0.join("output");
        let actual = command(&input, &fixed, &out, true, false).output().unwrap();
        assert!(
            actual.status.success(),
            "{}",
            String::from_utf8_lossy(&actual.stderr)
        );
        assert!(out.join("zvo_out_001.dat").is_file());
        // Process RNG is not exported by the ordinary CLI; this is a genuine
        // success/output assertion, not a subprocess raw-state claim.
    }
}

#[test]
fn parse_fixed_initial_and_overlay_failures_reach_cli_status_without_output() {
    for (phys, case, diagnostic) in [
        (true, "parse", "failed to read namelist.def"),
        (false, "parse", "failed to read namelist.def"),
        (true, "fixed-missing", "fixed parameter file not found"),
        (true, "fixed-short", "too short"),
        (true, "fixed-token", "non-numeric token"),
        (false, "initial", "explicitly requested path"),
        (true, "overlay", "UTF-8"),
        (false, "overlay", "UTF-8"),
    ] {
        let root = Owned::new();
        let (input, fixed) = input(&root.0, "heisenberg_chain_real", phys, &[]);
        let missing = root.0.join("missing.def");
        let selected = if case == "parse" { &missing } else { &input };
        let params = if case == "fixed-missing" {
            &missing
        } else {
            &fixed
        };
        if case == "fixed-short" {
            fs::write(&fixed, "0 0 0 0 0 0").unwrap();
        }
        if case == "fixed-token" {
            fs::write(&fixed, "0 0 invalid 0 0 0").unwrap();
        }
        if case == "overlay" {
            let mut text = fs::read_to_string(&input).unwrap();
            text.push_str("\nInGutzwiller invalid-utf8.def\n");
            fs::write(&input, text).unwrap();
            fs::write(input.parent().unwrap().join("invalid-utf8.def"), [0xff]).unwrap();
        }
        let before = fs::read(&input).unwrap();
        let fixed_before = fs::read(&fixed).unwrap();
        let output = root.0.join("must-not-exist");
        let mut actual = command(selected, params, &output, phys, false);
        if case == "initial" {
            actual.arg("--initial-def").arg(&missing);
        }
        failure(actual.output().unwrap(), diagnostic);
        assert!(!output.exists());
        assert_eq!(fs::read(&input).unwrap(), before);
        assert_eq!(fs::read(&fixed).unwrap(), fixed_before);
    }
}
