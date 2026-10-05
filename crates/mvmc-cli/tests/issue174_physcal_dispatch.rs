//! Issue #174: the CLI selects optimization or fixed-parameter PhysCal from
//! `NVMCCalMode` (C `vmcmain.c`) and rejects disagreeing options before any
//! initialization or output. Numerical parity of the PhysCal CLI outputs with
//! independent Julia/C references lives in `physcal_reference.rs`; the test
//! here checks that the CLI path equals a direct library call on the same input.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "issue174-dispatch-{}-{}",
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

fn model_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/two-samples/heisenberg_chain_real")
}

/// Copy the fixture inputs, optionally replacing the NVMCCalMode value.
fn inputs(work: &Work, mode: Option<i64>) -> (PathBuf, PathBuf) {
    let dir = work.0.join("inputs");
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(model_root().join("inputs")).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    if let Some(mode) = mode {
        let path = dir.join("modpara.def");
        let text = fs::read_to_string(&path).unwrap();
        let mut replaced = false;
        let text: Vec<String> = text
            .lines()
            .map(|line| {
                if line.trim_start().starts_with("NVMCCalMode") {
                    replaced = true;
                    format!("NVMCCalMode    {mode}")
                } else {
                    line.to_owned()
                }
            })
            .collect();
        assert!(replaced, "fixture modpara.def has no NVMCCalMode line");
        fs::write(&path, text.join("\n") + "\n").unwrap();
    }
    (dir.join("namelist.def"), model_root().join("zqp_opt.dat"))
}

fn run(namelist: &Path, fixed: Option<&Path>, out: &Path) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command.arg(namelist);
    if let Some(fixed) = fixed {
        command.arg("--physcal").arg(fixed);
    }
    command
        .args(["--seed", "1", "--mode", "real", "--out-dir"])
        .arg(out)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap()
}

fn rejected(output: Output, diagnostic: &str, out: &Path) {
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(diagnostic), "{stderr}");
    assert!(!out.exists(), "output directory created before rejection");
}

#[test]
fn physcal_mode_one_with_fixed_file_matches_direct_library_call() {
    let work = Work::new();
    let (namelist, fixed) = inputs(&work, Some(1));
    let cli_out = work.0.join("cli");
    let output = run(&namelist, Some(&fixed), &cli_out);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Completed 2 PhysCal samples"));

    let lib_out = work.0.join("lib");
    let preparation =
        mvmc_core::prepare_phys_cal_from_namelist(&namelist, &fixed, "real", Some(1)).unwrap();
    let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &lib_out).unwrap();
    assert_eq!(result.iterations, 2);

    // Same implementation, same input and seed: outputs are reproducible text.
    let names = |dir: &Path| {
        let mut names: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        names
    };
    let lib_names = names(&lib_out);
    assert!(lib_names.iter().any(|name| name == "zvo_out_007.dat"));
    assert!(lib_names.iter().any(|name| name == "zvo_cisajs_008.dat"));
    assert_eq!(names(&cli_out), lib_names);
    for name in &lib_names {
        // `_time_` rows end in a wall-clock ctime string.
        if name.to_string_lossy().contains("_time_") {
            continue;
        }
        assert_eq!(
            fs::read(cli_out.join(name)).unwrap(),
            fs::read(lib_out.join(name)).unwrap(),
            "{name:?} differs between CLI and library PhysCal"
        );
    }
}

#[test]
fn physcal_flag_with_optimization_mode_is_rejected_before_output() {
    let work = Work::new();
    let (namelist, fixed) = inputs(&work, Some(0));
    let out = work.0.join("out");
    rejected(
        run(&namelist, Some(&fixed), &out),
        "--physcal requires NVMCCalMode=1",
        &out,
    );
}

#[test]
fn physcal_mode_without_parameter_file_runs_like_c_vmc_out() {
    // C treats the positional initpara as optional for NVMCCalMode=1 (#347); the
    // C InitParameter draws then supply the parameters.
    let work = Work::new();
    let (namelist, _) = inputs(&work, Some(1));
    let out = work.0.join("out");
    let output = run(&namelist, None, &out);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.join("zvo_out_007.dat").is_file());
}

#[test]
fn unsupported_calculation_modes_are_rejected_with_and_without_fixed_file() {
    for mode in [2, 3, -1] {
        let work = Work::new();
        let (namelist, fixed) = inputs(&work, Some(mode));
        for fixed in [Some(fixed.as_path()), None] {
            let out = work.0.join("out");
            rejected(
                run(&namelist, fixed, &out),
                &format!("unsupported NVMCCalMode={mode}"),
                &out,
            );
        }
    }
}

#[test]
fn physcal_missing_fixed_file_with_mode_one_is_rejected_before_output() {
    let work = Work::new();
    let (namelist, _) = inputs(&work, Some(1));
    let out = work.0.join("out");
    let missing = work.0.join("missing.dat");
    rejected(
        run(&namelist, Some(&missing), &out),
        "fixed parameter file not found",
        &out,
    );
}
