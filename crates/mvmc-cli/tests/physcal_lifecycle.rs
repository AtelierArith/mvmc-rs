//! Actual CLI-process observation against immutable, independently generated
//! C parameter stages and Julia sampling records. No Rust expected generator,
//! C/Julia invocation or toolbox access. C stage replay is not full C sampling.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct OwnedDir(PathBuf);
impl OwnedDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "mvmc-cli-lifecycle-{}-{}",
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
impl Drop for OwnedDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn fixture(model: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(model)
}

fn command(root: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command
        .arg(root.join("inputs/namelist.def"))
        .arg("--physcal")
        .arg(root.join("zqp_opt.dat"))
        .args(["--seed", "1", "--mode", "real", "--opt-trans", "--out-dir"])
        .arg(output)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1");
    command
}

fn success(result: Output) {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn integers(path: &Path) -> Vec<u128> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

fn configuration(path: &Path) -> Vec<i64> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

fn parameters(path: &Path) -> Vec<f64> {
    let values: Vec<f64> = fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    assert!(!values.is_empty() && values.len().is_multiple_of(2));
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

fn assert_parameters(actual: &Path, expected: &Path, exact: bool) {
    let a = parameters(actual);
    let e = parameters(expected);
    assert_eq!(a.len(), e.len(), "complete declared parameter pack");
    for (index, (&a, &e)) in a.iter().zip(&e).enumerate() {
        if exact {
            assert_eq!(a, e, "stored coefficient {index}");
        } else {
            // Existing physcal_issue181 C-stage policy: cabs/hypot rescaling
            // rounds differently. No larger or downstream solver budget.
            assert!(
                (a - e).abs() <= 1e-14 * e.abs().max(1.0),
                "coefficient {index}: {a} vs {e}"
            );
        }
    }
}

fn assert_rng(actual: &Path, next: &Path, count: &Path) {
    let a = integers(&actual.join("next624.txt"));
    let e = integers(next);
    assert_eq!(e.len(), 624);
    assert_eq!(a, e, "independent nonconsuming future outputs");
    let e = integers(count);
    assert_eq!(e.len(), 1);
    assert_eq!(
        integers(&actual.join("draw-count.txt")),
        e,
        "actual primitive draw count"
    );
}

fn same_outputs(a: &Path, b: &Path) {
    let names = |p: &Path| -> std::collections::BTreeSet<_> {
        fs::read_dir(p)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect()
    };
    assert_eq!(names(a), names(b));
    for name in names(a) {
        // `_time_` rows end in a wall-clock ctime string.
        if name.to_string_lossy().contains("Timer") || name.to_string_lossy().contains("_time_") {
            continue;
        }
        assert_eq!(
            fs::read(a.join(&name)).unwrap(),
            fs::read(b.join(&name)).unwrap(),
            "trace must not change output {name:?}"
        );
    }
}

fn check(model: &str) {
    let root = fixture(model);
    let native = root.join("native-c-stages");
    let provenance = fs::read_to_string(native.join("provenance.txt")).unwrap();
    assert!(provenance.contains("seed=1\n"));
    assert!(provenance.contains(
        "source_parameter_sha256=46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0"
    ));
    assert!(provenance.contains(
        "source_readdef_sha256=6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9"
    ));
    let julia = fs::read_to_string(root.join("provenance.txt")).unwrap();
    assert!(julia.contains("julia=1.13.1\n") && julia.contains("seed=1\n"));
    // Direct fixture paths: unchanged input files, seed and sampling settings.
    let directory = OwnedDir::new();
    let off = directory.0.join("off");
    let on = directory.0.join("on");
    let trace = directory.0.join("trace");
    success(command(&root, &off).output().unwrap());
    success(
        command(&root, &on)
            .arg("--physcal-trace")
            .arg(&trace)
            .output()
            .unwrap(),
    );
    same_outputs(&off, &on);
    assert_eq!(
        fs::read_to_string(trace.join("terminal.txt")).unwrap(),
        "status=0\n"
    );
    assert!(fs::read_to_string(trace.join("request.txt"))
        .unwrap()
        .contains("requested_opt_trans=true\n"));
    for stage in [
        "fixed-loaded",
        "overlaid",
        "synchronized",
        "seeded",
        "initialized-clone",
        "sample-0",
    ] {
        let resolved = fs::read_to_string(trace.join(stage).join("resolved.txt")).unwrap();
        assert!(resolved.contains("actual_all_complex=false\n"));
        assert!(resolved.contains("actual_orbital_general=0\n"));
        assert!(resolved.contains("actual_n_qp_opt_trans="));
        assert!(resolved.contains("actual_c_opt_trans_flags="));
        if stage == "sample-0" {
            assert!(resolved.contains("actual_qp_total=Some("));
        }
    }
    assert_eq!(
        fs::read_to_string(trace.join("stages.txt")).unwrap(),
        "fixed-loaded\noverlaid\nsynchronized\nseeded\ninitialized-clone\nsample-0\n"
    );
    for (actual, expected, exact) in [
        ("fixed-loaded", "fixed", true),
        ("overlaid", "overlaid", true),
        ("synchronized", "synchronized", false),
        ("initialized-clone", "initialized", false),
    ] {
        assert_parameters(
            &trace.join(actual).join("parameters.txt"),
            &native.join(format!("{expected}-parameters.txt")),
            exact,
        );
    }
    assert_rng(
        &trace.join("seeded"),
        &root.join("seeded/next624.txt"),
        &root.join("seeded/draw-count.txt"),
    );
    assert_rng(
        &trace.join("initialized-clone"),
        &native.join("initialized-next624.txt"),
        &native.join("initialized-draw-count.txt"),
    );
    assert_rng(
        &trace.join("sample-0"),
        &root.join("sample-0/next624.txt"),
        &root.join("sample-0/draw-count.txt"),
    );
    assert_parameters(
        &trace.join("sample-0/parameters.txt"),
        &native.join("synchronized-parameters.txt"),
        false,
    );
    for name in [
        "ele_idx",
        "ele_cfg",
        "ele_num",
        "ele_proj_cnt",
        "ele_spn",
        "counter",
    ] {
        assert_eq!(
            configuration(&trace.join("sample-0").join(format!("{name}.txt"))),
            configuration(&root.join("sample-0").join(format!("{name}.txt"))),
            "{model} independent {name}"
        );
    }
    for stage in ["fixed-loaded", "overlaid", "synchronized"] {
        assert!(
            !trace.join(stage).join("next624.txt").exists(),
            "Rust has not seeded yet; don't fabricate C chronology"
        );
    }
    let count = parameters(&native.join("fixed-parameters.txt")).len() / 2;
    assert!(fs::read_to_string(trace.join("fixed-loaded/settings.txt"))
        .unwrap()
        .contains(&format!("consumed=Some({count})\n")));
    // The explicit record already includes all families; caller cannot silently
    // reload a neighboring optimization initial.def during PhysCal.
    let copy = directory.0.join("neighbor");
    fs::create_dir(&copy).unwrap();
    fs::create_dir(copy.join("inputs")).unwrap();
    for entry in fs::read_dir(root.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        let destination = copy.join("inputs").join(entry.file_name());
        fs::copy(entry.path(), &destination).unwrap();
        assert_eq!(
            fs::read(entry.path()).unwrap(),
            fs::read(destination).unwrap()
        );
    }
    fs::copy(root.join("zqp_opt.dat"), copy.join("zqp_opt.dat")).unwrap();
    fs::write(
        copy.join("inputs/initial.def"),
        "must not be read by explicit PhysCal\n",
    )
    .unwrap();
    let neighbor_out = directory.0.join("neighbor-out");
    success(command(&copy, &neighbor_out).output().unwrap());
    same_outputs(&off, &neighbor_out);
}

#[test]
fn cli_dh_overlays_match_independent_c_stages_and_julia_rng_config() {
    check("hubbard_chain_dh_overlays");
}

#[test]
fn cli_dh_opttrans_match_independent_c_stages_and_julia_rng_config() {
    check("hubbard_chain_dh_opttrans");
}

#[test]
fn trace_existing_directory_is_rejected_without_overwrite_or_output() {
    let directory = OwnedDir::new();
    let trace = directory.0.join("existing");
    fs::create_dir(&trace).unwrap();
    fs::write(trace.join("sentinel"), "preserve\n").unwrap();
    let out = directory.0.join("out");
    let result = command(&fixture("hubbard_chain_dh_overlays"), &out)
        .arg("--physcal-trace")
        .arg(&trace)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("NEW directory"));
    assert_eq!(
        fs::read_to_string(trace.join("sentinel")).unwrap(),
        "preserve\n"
    );
    assert!(!out.exists());
    assert!(!trace.join("terminal.txt").exists());
}

#[test]
fn trace_requires_physcal_and_rejects_mpi_before_initialization() {
    let directory = OwnedDir::new();
    let root = fixture("hubbard_chain_dh_overlays");
    let trace = directory.0.join("trace");
    let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(root.join("inputs/namelist.def"))
        .arg("--physcal-trace")
        .arg(&trace)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(!trace.exists());
    let result = command(&root, &directory.0.join("out"))
        .arg("--physcal-trace")
        .arg(&trace)
        .env("OMPI_COMM_WORLD_SIZE", "2")
        .env("OMPI_COMM_WORLD_RANK", "0")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(!trace.exists());
    assert!(!directory.0.join("out").exists());
}
