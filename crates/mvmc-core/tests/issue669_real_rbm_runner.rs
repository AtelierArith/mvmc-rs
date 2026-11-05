//! Issue #669: real-orbital RBM optimization is consistent with the complex normal path.
//!
//! C's real path (`vmcmake_real.c`, `calham_real.c`) has no RBM code (a C defect,
//! tmisawa/Julia-mVMC#59); Rust applies the RBM factor in real-mode sampling and in every
//! local-energy ratio, as C's complex normal path (`vmcmake.c`, `calham.c`) does.
//!
//! The input is `tests/fixtures/issue669_real_rbm` (a copy of the PR #668 `hubbard_rbm_L16`
//! input, all nine RBM sections with real coefficients), with three test-only edits so the
//! real and the complex runner consume the same RNG stream:
//!
//! * Slater and RBM optimization flags are set to 0, so `InitParameter` draws nothing for them
//!   (its real and complex branches draw different numbers of words); the coefficients come
//!   from `InOrbital` and the `In*RBM` overlays. Projection parameters remain optimized.
//! * An `InterAll` section adds four-operator terms, covering the real InterAll Green kernel.
//! * `Orbital ComplexType` selects the real or the complex path.
//!
//! Both paths then sample the same configurations (`issue669_real_rbm_sampler` checks the
//! sampler itself), so the step energies differ only by the rounding of the real and complex
//! Pfaffian-ratio kernels.
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::{run_para_opt_from_namelist, InitialDef, RunConfig};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rbm {
    Absent,
    Zero,
    Real,
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/issue669_real_rbm")
}

fn scratch(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "mvmc-issue669-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

/// Set the last `count` lines (the optimization flags) of an index file to 0.
fn fix_flags(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let count: usize = lines[1].split_whitespace().nth(1).unwrap().parse().unwrap();
    let first = lines.len() - count;
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i >= first {
            let index = line.split_whitespace().next().unwrap();
            out.push_str(&format!("{index} 0\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn zero_values(text: &str) -> String {
    let mut out = String::new();
    for (i, line) in text.lines().enumerate() {
        if i >= 5 {
            let index = line.split_whitespace().next().unwrap();
            out.push_str(&format!("{index} 0.0 0.0\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

const BAR: &str = "=============================================";

fn prepare(complex: bool, rbm: Rbm) -> PathBuf {
    let dir = scratch(&format!("{complex}-{rbm:?}"));
    for entry in fs::read_dir(fixture()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let mut text = fs::read_to_string(&path).unwrap();
        if name == "orbitalidx.def" {
            text = fix_flags(&text);
            if complex {
                text = text.replacen("ComplexType          0", "ComplexType          1", 1);
            }
        } else if name.starts_with("rbm_") {
            text = fix_flags(&text);
        } else if name.starts_with("in_rbm_") && rbm == Rbm::Zero {
            text = zero_values(&text);
        }
        fs::write(dir.join(&name), text).unwrap();
    }
    // Fixed, nonsingular real orbitals for the 16 translational indices.
    let mut orbital = format!("{BAR}\nNOrbitalIdx 16\n{BAR}\n{BAR}\n{BAR}\n");
    for k in 0..16 {
        let value = [0.875, -0.5, 0.375, 0.25, -0.625, 0.125, -0.25, 0.5][k % 8]
            * if k < 8 { 1.0 } else { 0.75 };
        orbital.push_str(&format!("{k} {value:.15} 0.000000000000000\n"));
    }
    fs::write(dir.join("inorbital.def"), orbital).unwrap();
    // Hermitian, spin-conserving four-operator terms (fixed TwoSz): density-assisted hops
    // of both spins.
    let mut terms = Vec::new();
    for i in [0, 5, 10] {
        let j = (i + 1) % 16;
        let k = (i + 3) % 16;
        terms.push(format!("{i} 1 {j} 1 {k} 0 {k} 0 0.125 0.0"));
        terms.push(format!("{j} 1 {i} 1 {k} 0 {k} 0 0.125 0.0"));
        terms.push(format!("{i} 0 {j} 0 {k} 1 {k} 1 -0.25 0.0"));
        terms.push(format!("{j} 0 {i} 0 {k} 1 {k} 1 -0.25 0.0"));
    }
    fs::write(
        dir.join("interall.def"),
        format!(
            "{BAR}\nNInterAll {}\n{BAR}\n{BAR}\n{BAR}\n{}\n",
            terms.len(),
            terms.join("\n")
        ),
    )
    .unwrap();
    let namelist = fs::read_to_string(dir.join("namelist.def")).unwrap();
    let mut out = String::new();
    for line in namelist.lines() {
        if rbm == Rbm::Absent && line.contains("RBM_") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("InOrbital inorbital.def\nInterAll interall.def\n");
    fs::write(dir.join("namelist.def"), out).unwrap();
    dir
}

struct Outcome {
    energies: Vec<f64>,
    output: PathBuf,
}

fn run(complex: bool, rbm: Rbm) -> Outcome {
    let input = prepare(complex, rbm);
    let output = input.join("output");
    let mut config = RunConfig::new(2, if complex { "cmp" } else { "real" });
    config.nsmp = Some(2);
    config.initial_def = InitialDef::None;
    config.output_dir = Some(output.clone());
    let summary = run_para_opt_from_namelist(input.join("namelist.def"), config).unwrap();
    let energies = summary
        .zvo_first_n
        .iter()
        .map(|row| row.split_whitespace().next().unwrap().parse().unwrap())
        .collect();
    Outcome { energies, output }
}

fn assert_close(a: &[f64], b: &[f64], tolerance: f64, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() <= tolerance * x.abs().max(1.0),
            "{label}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

// Per-step energies average 64 local energies of O(10). On the same configurations
// the real and complex Pfaffian-ratio kernels round differently: observed difference
// 1 ulp (1.8e-15 at 10.35), and zero vs absent RBM agree bit for bit. The 1e-10
// relative bound leaves room for the conditioning of the rank-1/rank-2 ratio updates
// and is far below the defects it guards: without the fix the real sampler moves the
// step-2 energy by 1.8, and the RBM-free real InterAll kernel alone by 1.3e-2.
const SAME_PATH_RELATIVE: f64 = 1e-10;

#[test]
fn real_rbm_energies_match_complex_orbital_path() {
    let real = run(false, Rbm::Real);
    let complex = run(true, Rbm::Real);
    assert_close(
        &real.energies,
        &complex.energies,
        SAME_PATH_RELATIVE,
        "real vs complex orbitals, RBM",
    );
    // The RBM factor is not negligible here.
    let absent = run(false, Rbm::Absent);
    assert!(
        (real.energies[0] - absent.energies[0]).abs() > 1e-3,
        "{:?} vs {:?}",
        real.energies,
        absent.energies
    );
}

#[test]
fn zero_rbm_coefficients_leave_both_paths_unchanged() {
    for complex in [false, true] {
        let zero = run(complex, Rbm::Zero);
        let absent = run(complex, Rbm::Absent);
        // A zero RBM factor is exactly one; the RBM-enabled kernels group some
        // sums differently from the RBM-free fast paths (last-bit differences).
        assert_close(
            &zero.energies,
            &absent.energies,
            SAME_PATH_RELATIVE,
            &format!("complex={complex}: zero vs absent RBM"),
        );
    }
}

#[test]
fn real_rbm_optimized_parameter_files_include_rbm_rows() {
    let real = run(false, Rbm::Real);
    let opt: Vec<f64> = fs::read_to_string(real.output.join("zqp_opt.dat"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    // Energy and energy squared, then projection (1 Gutzwiller + Jastrow), 76 RBM and 16
    // Slater slots, each as mean real/imaginary/deviation.
    let jastrow: usize = fs::read_to_string(fixture().join("jastrowidx.def"))
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(opt.len(), 3 * (2 + 1 + jastrow + 76 + 16));
    for (file, label, width) in [
        (
            "zqp_chargeRBM_physlayer_opt.dat",
            "NChargeRBM_PhysLayerIdx",
            1,
        ),
        (
            "zqp_generalRBM_physhidden_opt.dat",
            "NGeneralRBM_PhysHiddenIdx",
            32,
        ),
    ] {
        let text = fs::read_to_string(real.output.join(file)).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.nth(1).unwrap().split_whitespace().collect::<Vec<_>>(),
            [label, &width.to_string()]
        );
        assert_eq!(text.lines().count(), 5 + width, "{file}");
    }
    // The fixed RBM coefficients are written back unchanged (flags are 0).
    let charge = fs::read_to_string(real.output.join("zqp_chargeRBM_physlayer_opt.dat")).unwrap();
    let row: Vec<f64> = charge
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let input = fs::read_to_string(fixture().join("in_rbm_charge_phys.def")).unwrap();
    let expected: f64 = input
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(row[1], expected);
}
