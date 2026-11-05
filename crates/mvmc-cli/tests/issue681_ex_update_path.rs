//! Issue #681: `NExUpdatePath` values outside 0-6 have no update in mVMC
//! (`getUpdateType` returns `NONE`), so the sampler would never move and every
//! sample would be the initial configuration; 4-6 are 1.4.0 updates that are
//! not ported. The CLI must reject them, for
//! optimization and PhysCal, before any output is created.
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
            "issue681-ex-update-path-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Copy `source` into the work directory with `NExUpdatePath` set to `path`.
fn inputs(work: &Work, source: &Path, path: &str) -> PathBuf {
    let dir = work.0.join("inputs");
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(source).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    let modpara = dir.join("modpara.def");
    let text = fs::read_to_string(&modpara).unwrap();
    let mut replaced = 0;
    let text: String = text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("NExUpdatePath") {
                replaced += 1;
                format!("NExUpdatePath  {path}")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(replaced, 1, "fixture must declare NExUpdatePath once");
    fs::write(&modpara, text + "\n").unwrap();
    dir.join("namelist.def")
}

fn run(namelist: &Path, out: &Path, physcal: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command.arg(namelist);
    match physcal {
        Some(parameters) => {
            command
                .arg("--physcal")
                .arg(parameters)
                .args(["--mode", "real"]);
        }
        None => {
            command.args(["--nsteps", "2", "--nsmp", "2"]);
        }
    }
    command
        .args(["--seed", "1", "--out-dir"])
        .arg(out)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap()
}

fn assert_rejected(output: &Output, out: &Path, path: &str, context: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{context}: run must fail");
    assert!(
        stderr.contains(&format!("NExUpdatePath = {path} is not supported")),
        "{context}: stderr was {stderr}"
    );
    if matches!(path, "4" | "5" | "6") {
        assert!(stderr.contains("mVMC 1.4.0"), "{context}: {stderr}");
    }
    assert!(!out.exists(), "{context}: output directory was created");
}

#[test]
fn optimization_rejects_unported_update_paths_before_output() {
    let source = repo().join("extern/Julia-mVMC/examples/inputs/hubbard_chain_real");
    for path in ["4", "5", "6", "-1", "7"] {
        let work = Work::new();
        let namelist = inputs(&work, &source, path);
        let out = work.0.join("out");
        let output = run(&namelist, &out, None);
        assert_rejected(&output, &out, path, &format!("opt path={path}"));
    }
}

#[test]
fn physcal_rejects_unported_update_paths_before_output() {
    let model = repo().join("tests/fixtures/physcal_181/two-samples/heisenberg_chain_real");
    for path in ["4", "5", "6", "-1"] {
        let work = Work::new();
        let namelist = inputs(&work, &model.join("inputs"), path);
        let out = work.0.join("out");
        let output = run(&namelist, &out, Some(&model.join("zqp_opt.dat")));
        assert_rejected(&output, &out, path, &format!("physcal path={path}"));
    }
}

#[test]
fn supported_update_path_still_runs() {
    let source = repo().join("extern/Julia-mVMC/examples/inputs/hubbard_chain_real");
    let work = Work::new();
    let namelist = inputs(&work, &source, "0");
    let out = work.0.join("out");
    let output = run(&namelist, &out, None);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.join("zqp_opt.dat").exists());
}
