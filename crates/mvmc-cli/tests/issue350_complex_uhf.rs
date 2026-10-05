//! #350: `mvmc uhf namelist.def [OptParaFile]` is the port of the C ComplexUHF
//! executable. The generated orbital file feeds the Rust optimizer through
//! `InOrbital`, exactly like the C tutorial (`samples/tutorial_1.3/run_uhf.sh`).
//! Expected observables come from the C fixtures in `tests/fixtures/complex_uhf`.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str, case: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue350-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/complex_uhf")
            .join(case)
            .join("input");
        for entry in fs::read_dir(input).unwrap().flatten() {
            fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn mvmc(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

fn expected(case: &str, name: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/complex_uhf")
            .join(case)
            .join("expected")
            .join(name),
    )
    .unwrap()
}

/// `energy`/`num` values of a `_result.dat`.
fn result_values(text: &str) -> (f64, f64) {
    let mut values = text.lines().map(|line| {
        line.split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<f64>()
            .unwrap()
    });
    (values.next().unwrap(), values.next().unwrap())
}

#[test]
fn uhf_orbitals_feed_the_optimizer() {
    let dir = TestDir::new("e2e", "hubbard_chain_real");
    let uhf = mvmc(&dir.0, &["uhf", "namelist.def"]);
    assert!(
        uhf.status.success(),
        "{}",
        String::from_utf8_lossy(&uhf.stderr)
    );
    let (energy, num) = result_values(&fs::read_to_string(dir.0.join("zvo_result.dat")).unwrap());
    let (c_energy, c_num) = result_values(&expected("hubbard_chain_real", "zvo_result.dat"));
    // Eight printed decimals of an O(1) energy; see tests/c_parity.rs in mvmc-uhf.
    assert!((energy - c_energy).abs() <= 2e-10, "{energy} vs {c_energy}");
    assert!((num - c_num).abs() <= 2e-10);
    let orbitals = fs::read_to_string(dir.0.join("zqp_APOrbital_opt.dat")).unwrap();
    assert_eq!(orbitals.lines().nth(1), Some("NOrbitalAP  12"));
    assert_eq!(orbitals.lines().count(), 5 + 12);

    // Baseline optimizer start without the UHF orbitals.
    let namelist = fs::read_to_string(dir.0.join("namelist.def")).unwrap();
    let common = ["--nsteps", "2", "--nsmp", "2", "--seed", "1"];
    let run_vmc = |label: &str, text: &str| {
        let list = format!("namelist_{label}.def");
        fs::write(dir.0.join(&list), text).unwrap();
        let out = format!("out_{label}");
        let output = mvmc(
            &dir.0,
            &[
                &list,
                common[0],
                common[1],
                common[2],
                common[3],
                common[4],
                common[5],
                "--out-dir",
                &out,
            ],
        );
        assert!(
            output.status.success(),
            "{label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        fs::read_to_string(dir.0.join(&out).join("zvo_out.dat")).unwrap()
    };
    let random_start = run_vmc("random", &namelist);
    let uhf_start = run_vmc(
        "uhf",
        &format!("{namelist} InOrbital zqp_APOrbital_opt.dat\n"),
    );
    let first = |text: &str| {
        text.lines()
            .next()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .to_string()
    };
    assert_ne!(
        first(&random_start),
        first(&uhf_start),
        "InOrbital must change the optimizer's starting orbitals"
    );
    for text in [&random_start, &uhf_start] {
        assert_eq!(text.lines().count(), 2, "two SR steps");
        assert!(text.lines().all(|line| line
            .split_whitespace()
            .all(|word| word.parse::<f64>().is_ok_and(f64::is_finite))));
    }
}

#[test]
fn unconverged_run_exits_255_after_writing_files_like_c() {
    let dir = TestDir::new("unconverged", "uhf_hubbard_square_unconverged");
    // The optional OptParaFile argument is accepted and unused, as in C.
    let output = mvmc(&dir.0, &["uhf", "namelist.def", "unused_opt_file.dat"]);
    assert_eq!(output.status.code(), Some(255));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Hartree-Fock calculation is not finished at 5  step!!"));
    for file in [
        "zvo_check.dat",
        "zvo_result.dat",
        "zvo_eigen.dat",
        "zvo_gap.dat",
        "zvo_UHF_cisajs.dat",
        "zqp_AP_Fij.dat",
        "zqp_APOrbital_opt.dat",
    ] {
        assert!(dir.0.join(file).exists(), "{file}");
    }
}

#[test]
fn usage_and_missing_files_are_errors_not_crashes() {
    let dir = TestDir::new("errors", "uhf_hubbard_square");
    let none = mvmc(&dir.0, &["uhf"]);
    assert_eq!(none.status.code(), Some(1));
    let three = mvmc(&dir.0, &["uhf", "a.def", "b.dat", "c.dat"]);
    assert_eq!(three.status.code(), Some(1));
    // C: fclose(NULL) segfault; Rust: a diagnostic and exit status 1.
    fs::remove_file(dir.0.join("qptransidx.def")).unwrap();
    let missing = mvmc(&dir.0, &["uhf", "namelist.def"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("qptransidx.def"));
    assert!(!dir.0.join("zvo_result.dat").exists());
}
