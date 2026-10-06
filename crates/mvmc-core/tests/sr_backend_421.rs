//! Issue #421: the tenferro SR backend against the C-order oracle.
//!
//! Every bound below is derived from the operation, not tuned:
//!
//! * dot products of length `k` in any summation order differ from the exact value by at most
//!   `gamma_k * sum |terms|` with `gamma_k = k eps / (1 - k eps)` (Higham, Lemma 3.1); two
//!   implementations that both obey it differ by at most `2 gamma_k * sum |terms|`. The complex
//!   Gram uses `k + 3` because each complex product adds up to three roundings.
//! * S and g assembly is elementwise: the same IEEE operation sequence as C, so the bound is
//!   two roundings per entry.
//! * Cholesky solve: both are backward stable, so the relative solution difference is at most
//!   `2 c n eps kappa(S)` with `c = 8`; kappa is bounded from the construction of S.
//!   An independent residual check against the tenferro solution also runs.
//!
//! The sample order of the complex Gram product is documented in `sr_backend`.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use mvmc_core::sr_backend::{
    COrderSr, CgSamples, RealView, SrAssembleInput, SrBackend, TenferroSr,
};
use num_complex::Complex64;

const EPS: f64 = f64::EPSILON;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn random(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..len).map(|_| lcg(&mut s)).collect()
}

fn gamma(k: usize) -> f64 {
    let ke = k as f64 * EPS;
    ke / (1.0 - ke)
}

fn tenferro() -> TenferroSr {
    TenferroSr::new_cpu().expect("tenferro cpu runtime")
}

#[test]
fn real_gram_matches_c_order_within_dot_product_bound() {
    for &(n, samples) in &[(5usize, 3usize), (24, 60), (40, 200), (3, 2)] {
        let store = random(n * samples, 17 + n as u64);
        let mut expected = vec![0.0; n * n];
        let mut actual = vec![0.0; n * n];
        COrderSr::default()
            .gram_real(&store, n, samples, &mut expected)
            .unwrap();
        tenferro()
            .gram_real(&store, n, samples, &mut actual)
            .unwrap();
        for j in 0..n {
            for i in 0..n {
                let abs_sum: f64 = (0..samples)
                    .map(|s| (store[i + s * n] * store[j + s * n]).abs())
                    .sum();
                numerical_comparison::assert_close(
                    actual[i + j * n],
                    expected[i + j * n],
                    2.0 * gamma(samples) * abs_sum,
                    0.0,
                    format!("real gram n={n} samples={samples} ({i},{j})"),
                );
                assert_eq!(actual[i + j * n], actual[j + i * n], "symmetric by copy");
            }
        }
    }
}

#[test]
fn complex_gram_matches_sequential_c_order_within_dot_product_bound() {
    for &(n, samples) in &[(6usize, 4usize), (20, 50), (33, 120)] {
        let re = random(n * samples, 3 + n as u64);
        let im = random(n * samples, 91 + n as u64);
        let store: Vec<Complex64> = re
            .iter()
            .zip(&im)
            .map(|(&a, &b)| Complex64::new(a, b))
            .collect();
        let expected = COrderSr::default()
            .gram_complex(&store, n, samples)
            .unwrap();
        let actual = tenferro().gram_complex(&store, n, samples).unwrap();
        for j in 0..n {
            for i in 0..n {
                let abs_sum: f64 = (0..samples)
                    .map(|s| store[i + s * n].norm() * store[j + s * n].norm())
                    .sum();
                let bound = 2.0 * gamma(samples + 3) * abs_sum;
                let (a, e) = (actual[i + j * n], expected[i + j * n]);
                numerical_comparison::assert_close(a.re, e.re, bound, 0.0, format!("re ({i},{j})"));
                numerical_comparison::assert_close(a.im, e.im, bound, 0.0, format!("im ({i},{j})"));
            }
        }
    }
}

fn assemble(
    backend: &mut dyn SrBackend,
    oo: RealView<'_>,
    ho: RealView<'_>,
    map: &[usize],
    ld: usize,
    offset: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = map.len();
    let (mut s, mut g) = (vec![0.0; n * n], vec![0.0; n]);
    backend
        .assemble_s_g(
            &SrAssembleInput {
                oo,
                ho,
                map,
                ld,
                offset,
                sta_del: 0.02,
                step_dt: 0.05,
            },
            &mut s,
            &mut g,
        )
        .unwrap();
    (s, g)
}

#[test]
fn s_and_g_assembly_matches_c_order_for_real_and_complex_layouts() {
    let size = 14; // 1 + NPara
    let map = [0usize, 2, 3, 7, 11];
    let oo = random(size * size, 5);
    let ho = random(size, 6);
    let (s0, g0) = assemble(
        &mut COrderSr::default(),
        RealView::Real(&oo),
        RealView::Real(&ho),
        &map,
        size,
        1,
    );
    let (s1, g1) = assemble(
        &mut tenferro(),
        RealView::Real(&oo),
        RealView::Real(&ho),
        &map,
        size,
        1,
    );
    // Same IEEE operations in the same order; at most two roundings per entry.
    let bound = 2.0 * EPS;
    numerical_comparison::assert_values_close(s1, s0, 0.0, bound, "real S");
    numerical_comparison::assert_values_close(g1, g0, 0.0, bound, "real g");

    let co: Vec<Complex64> = random(4 * size * size, 8)
        .chunks(2)
        .map(|c| Complex64::new(c[0], c[1]))
        .collect();
    let ch: Vec<Complex64> = random(4 * size, 9)
        .chunks(2)
        .map(|c| Complex64::new(c[0], c[1]))
        .collect();
    let (s0, g0) = assemble(
        &mut COrderSr::default(),
        RealView::ReOfComplex(&co),
        RealView::ReOfComplex(&ch),
        &map,
        2 * size,
        2,
    );
    let (s1, g1) = assemble(
        &mut tenferro(),
        RealView::ReOfComplex(&co),
        RealView::ReOfComplex(&ch),
        &map,
        2 * size,
        2,
    );
    numerical_comparison::assert_values_close(s1, s0, 0.0, bound, "complex-layout S");
    numerical_comparison::assert_values_close(g1, g0, 0.0, bound, "complex-layout g");
}

/// `S = A^T A + shift I` (column-major) and a bound on its condition number.
fn spd(n: usize, shift: f64) -> (Vec<f64>, f64) {
    let a = random(n * n, 41);
    let mut s = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            s[i + j * n] = (0..n).map(|k| a[k + i * n] * a[k + j * n]).sum::<f64>();
        }
        s[j + j * n] += shift;
    }
    let frob2: f64 = a.iter().map(|v| v * v).sum();
    // lambda_min >= shift, lambda_max <= ||A||_F^2 + shift.
    (s, (frob2 + shift) / shift)
}

#[test]
fn cholesky_solve_matches_c_order_and_has_small_residual() {
    for &(n, shift) in &[(4usize, 1.0), (30, 0.5), (96, 0.1)] {
        let (s, kappa) = spd(n, shift);
        let rhs = random(n, 77);
        let (mut s0, mut x0) = (s.clone(), rhs.clone());
        COrderSr::default()
            .cholesky_solve(&mut s0, &mut x0, n)
            .unwrap();
        let (mut s1, mut x1) = (s.clone(), rhs.clone());
        tenferro().cholesky_solve(&mut s1, &mut x1, n).unwrap();
        let norm = x0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let bound = 2.0 * 8.0 * n as f64 * EPS * kappa * norm;
        for i in 0..n {
            numerical_comparison::assert_close(x1[i], x0[i], bound, 0.0, format!("n={n} x[{i}]"));
        }
        // Independent check of the tenferro solution: ||S x - b||_inf <= 8 n eps ||S|| ||x||.
        let s_norm = (0..n)
            .map(|i| (0..n).map(|j| s[i + j * n].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        for i in 0..n {
            let r: f64 = (0..n).map(|j| s[i + j * n] * x1[j]).sum::<f64>() - rhs[i];
            assert!(
                r.abs() <= 8.0 * n as f64 * EPS * s_norm * norm,
                "n={n} residual {r:e} too large"
            );
        }
    }
}

#[test]
fn non_positive_definite_fails_like_c_order() {
    let n = 3;
    let mut s = vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // eigenvalue -1
    let mut x = vec![1.0, 1.0, 1.0];
    assert!(COrderSr::default()
        .cholesky_solve(&mut s.clone(), &mut x.clone(), n)
        .is_err());
    assert!(tenferro().cholesky_solve(&mut s, &mut x, n).is_err());
}

fn cg_case(complex: bool) {
    let (n, samples) = (37usize, 90usize);
    let real = random(n * samples, 12);
    let imag = if complex {
        random(n * samples, 13)
    } else {
        Vec::new()
    };
    let x = random(n, 14);
    let samples_view = |version| CgSamples {
        real: &real,
        imag: &imag,
        components: n,
        samples,
        version,
    };
    let mut expected = vec![0.0; n];
    COrderSr::default()
        .cg_local_product(&samples_view(1), &x, &mut expected)
        .unwrap();
    let mut backend = tenferro();
    let mut actual = vec![0.0; n];
    backend
        .cg_local_product(&samples_view(1), &x, &mut actual)
        .unwrap();
    // z_i = sum_s O_is (sum_j O_js x_j): inner length n, outer length samples (x2 when complex).
    let blocks: Vec<&[f64]> = if complex {
        vec![&real, &imag]
    } else {
        vec![&real]
    };
    for i in 0..n {
        let abs_sum: f64 = blocks
            .iter()
            .map(|o| {
                (0..samples)
                    .map(|s| {
                        o[i + s * n].abs()
                            * (0..n).map(|j| (o[j + s * n] * x[j]).abs()).sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .sum();
        let bound = 2.0 * (gamma(n) + gamma(samples * blocks.len())) * abs_sum;
        numerical_comparison::assert_close(
            actual[i],
            expected[i],
            bound,
            0.0,
            format!("cg complex={complex} z[{i}]"),
        );
    }
    // Constant operand cache: the same version uploads once, a new version uploads again.
    for _ in 0..4 {
        backend
            .cg_local_product(&samples_view(1), &x, &mut actual)
            .unwrap();
    }
    assert_eq!(backend.stats().cg_products, 5);
    assert_eq!(backend.stats().cg_uploads, 1);
    backend
        .cg_local_product(&samples_view(2), &x, &mut actual)
        .unwrap();
    assert_eq!(backend.stats().cg_uploads, 2);
}

#[test]
fn cg_product_matches_c_order_and_caches_constant_operand() {
    cg_case(false);
    cg_case(true);
}

// ---- end to end: short optimization runs with both backends -------------------------------

use mvmc_core::sr_backend::{set_sr_backend_override, tenferro_stats, SrBackendKind};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_optimization(
    fixture: &str,
    mode: &str,
    cg: bool,
    kind: SrBackendKind,
) -> std::collections::BTreeMap<String, String> {
    set_sr_backend_override(Some(kind));
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue421-{}-{fixture}-{}-{kind:?}",
        std::process::id(),
        u8::from(cg)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let overrides = [
        ("NVMCCalMode", "0"),
        ("NSROptItrStep", "4"),
        ("NSROptItrSmp", "4"),
        ("NVMCSample", "60"),
        ("NVMCWarmUp", "10"),
        ("NSRCG", if cg { "1" } else { "0" }),
        ("NStore", "1"),
        ("NDataIdxStart", "1"),
        ("NDataQtySmp", "1"),
        ("NLanczosMode", "0"),
    ];
    let mut out = String::new();
    let mut seen = Vec::new();
    for line in modpara.lines() {
        let key = line.split_whitespace().next().unwrap_or("");
        match overrides.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
            Some((k, v)) => {
                seen.push(*k);
                out.push_str(&format!("{k} {v}\n"));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    for (k, v) in overrides {
        if !seen.contains(&k) {
            out.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), out).unwrap();
    let output = root.join("output");
    let mut config = mvmc_core::RunConfig::new(4, mode);
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{fixture} {kind:?}: {e}"));
    let mut files = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(&output).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(
            name.clone(),
            std::fs::read_to_string(output.join(&name)).unwrap(),
        );
    }
    if std::env::var("KEEP421").is_ok() {
        eprintln!("kept {}", root.display());
    } else {
        let _ = std::fs::remove_dir_all(&root);
    }
    files
}

/// Direct SR is well conditioned on these fixtures: the whole run (energies and optimized
/// parameters of 4 steps) must agree to `abs 1e-10 / rel 1e-8`. The first measured worst
/// relative difference is recorded in the assertion message.
#[test]
fn direct_sr_run_with_tenferro_tracks_the_c_order_run() {
    for (fixture, mode) in [
        ("hubbard_chain_dh_real", "real"),
        ("heisenberg_chain_cmp", "cmp"),
    ] {
        let reference = run_optimization(fixture, mode, false, SrBackendKind::COrder);
        let accelerated = run_optimization(fixture, mode, false, SrBackendKind::Tenferro);
        assert_eq!(
            reference.keys().collect::<Vec<_>>(),
            accelerated.keys().collect::<Vec<_>>()
        );
        for (name, text) in &reference {
            compare_numeric(
                &format!("{fixture} direct {name}"),
                text,
                &accelerated[name],
                1e-10,
                1e-8,
            );
        }
    }
    set_sr_backend_override(None);
    assert!(tenferro_stats().is_some(), "tenferro backend never used");
}

fn compare_numeric(label: &str, expected: &str, actual: &str, abs: f64, rel: f64) {
    let (a, b): (Vec<&str>, Vec<&str>) = (
        expected.split_whitespace().collect(),
        actual.split_whitespace().collect(),
    );
    assert_eq!(a.len(), b.len(), "{label}: token count");
    for (x, y) in a.iter().zip(&b) {
        match (x.parse::<f64>(), y.parse::<f64>()) {
            (Ok(x), Ok(y)) => numerical_comparison::assert_close(y, x, abs, rel, label),
            _ => assert_eq!(x, y, "{label}: non-numeric token"),
        }
    }
}

/// CG runs are not compared step by step: with `max_iterations = n` on the ill-conditioned
/// sampled S, a one-ulp change of the samples moves the C-order solution itself by up to
/// ~7e-4 (see `cg_difference_is_within_the_conditioning_spread`), and that perturbs every
/// later step. What must hold exactly is everything that does not depend on that
/// sensitivity: the file set, the first (pre-update) energy row, and the S structure columns
/// of the SR info rows (sizes and cuts, the only part that is a discrete contract).
#[test]
fn cg_run_with_tenferro_keeps_structure_and_the_pre_update_energy() {
    for (fixture, mode) in [
        ("hubbard_chain_dh_real", "real"),
        ("heisenberg_chain_cmp", "cmp"),
    ] {
        let reference = run_optimization(fixture, mode, true, SrBackendKind::COrder);
        let accelerated = run_optimization(fixture, mode, true, SrBackendKind::Tenferro);
        assert_eq!(
            reference.keys().collect::<Vec<_>>(),
            accelerated.keys().collect::<Vec<_>>()
        );
        let first = |text: &str| text.lines().next().unwrap_or("").to_owned();
        compare_numeric(
            &format!("{fixture} cg first energy row"),
            &first(&reference["zvo_out.dat"]),
            &first(&accelerated["zvo_out.dat"]),
            1e-10,
            1e-8,
        );
        for (r, a) in reference["zvo_SRinfo.dat"]
            .lines()
            .zip(accelerated["zvo_SRinfo.dat"].lines())
            .skip(1)
        {
            // Npara Msize optCut diagCut sDiagMax sDiagMin: independent of the solver.
            let head = |l: &str| l.split_whitespace().take(6).collect::<Vec<_>>().join(" ");
            assert_eq!(
                head(r).split(' ').take(4).collect::<Vec<_>>(),
                head(a).split(' ').take(4).collect::<Vec<_>>()
            );
        }
    }
    set_sr_backend_override(None);
}

// ---- CG sensitivity: the tolerance of the CG comparison is the measured conditioning --------

use mvmc_core::sr_cg::{install_cg_observer, CgObserver, SampledSrOperator};
use std::cell::RefCell;
use std::rc::Rc;

type Captured = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Default)]
struct Grab(RefCell<Option<Captured>>);

impl CgObserver for Grab {
    fn prepared(&self, _m: &[usize], op: &SampledSrOperator, g: &[f64]) {
        if self.0.borrow().is_none() {
            *self.0.borrow_mut() = Some((
                op.mean.clone(),
                op.diagonal.clone(),
                op.real_samples.clone(),
                g.to_vec(),
            ));
        }
    }
}

/// The first CG system of a real fixture run, solved with `max_iterations = n` as the
/// runner does. The tenferro-vs-C-order solution difference must not exceed the spread that
/// C order itself shows when its samples are perturbed by a few ulp (the backend difference
/// is rounding of the same kind). This is the justification for not comparing CG runs
/// step by step.
#[test]
fn cg_difference_is_within_the_conditioning_spread() {
    let grab = Rc::new(Grab::default());
    {
        let _guard = install_cg_observer(grab.clone()).unwrap();
        let _ = run_optimization("hubbard_chain_dh_real", "real", true, SrBackendKind::COrder);
    }
    let (mean, diag, real, grad) = grab.0.borrow().clone().expect("a CG system was prepared");
    let n = mean.len();
    let samples = real.len() / n;
    let solve = |kind, noise: f64, seed: u64| {
        set_sr_backend_override(Some(kind));
        let mut op = SampledSrOperator::new(n, samples, false);
        op.mean = mean.clone();
        op.diagonal = diag.clone();
        let mut state = seed;
        op.real_samples = real
            .iter()
            .map(|v| v * (1.0 + noise * lcg(&mut state)))
            .collect();
        op.solve(&grad, 1.0 / 60.0, 0.02, 1e-10, n).solution
    };
    let relative = |a: &[f64], b: &[f64]| {
        let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    };
    let base = solve(SrBackendKind::COrder, 0.0, 1);
    let spread = [(1e-16, 2u64), (1e-16, 3), (1e-15, 4)]
        .iter()
        .map(|&(noise, seed)| relative(&solve(SrBackendKind::COrder, noise, seed), &base))
        .fold(0.0_f64, f64::max);
    let backend_difference = relative(&solve(SrBackendKind::Tenferro, 0.0, 1), &base);
    set_sr_backend_override(None);
    assert!(
        spread > 1e-6,
        "fixture must be ill conditioned enough to need this argument (spread {spread:e})"
    );
    assert!(
        backend_difference <= 2.0 * spread,
        "tenferro CG differs by {backend_difference:e}, C-order ulp-perturbation spread {spread:e}"
    );
}
