//! Optional CUDA gate for the production routing of SR to the resident device steps (issue #452).
//! Ignored by default; `MVMC_RS_CUDA_GATE=1` requires a device (hard failure without one),
//! anything else skips explicitly.
//!
//! Real sampled optimizer runs (`run_para_opt_from_namelist`, not synthetic operands), once with
//! the C-order backend and once with `MVMC_RS_SR_BACKEND=cuda` semantics (the override selects
//! the provider's resident backend):
//!
//! * **Routing**: the resident counters show one store upload per SR step, of exactly
//!   `n x samples` values (the O store, not an `n x n` host `OO`), and one device CG solve per
//!   CG step.
//! * **Step-1 operands** (sampling does not depend on the SR backend until the first parameter
//!   update, so both runs see identical samples): the SR observer captures `S`, `g` and the
//!   solution of the first solve. With `k` samples, `a_i = sqrt(OO_ii)` and means `m_i`
//!   (`|m_i| <= a_i` by Cauchy-Schwarz, `OO` normalized by `1 / wc`), the Gram entries of two
//!   summation orders differ by at most `2 (k + 2) eps a_i a_j`, hence
//!   `|dS_ij| <= 6 (k + 3) eps (1 + DSROptStaDel [i = j]) a_i a_j` and
//!   `|dg_i| <= 2 dt (2 (k + 2) eps |HO_0| a_i + 8 eps (|HO_i| + |HO_0 m_i|))` (`HO` is a host
//!   accumulation and identical in both runs). The solution obeys the first-order perturbation
//!   bound `|dx| / |x| <= 2 kappa (|B|_F / |S|_2 + |B_g| / |g| + 4 n eps)` with `kappa`
//!   estimated on the host (power and inverse iteration through the C-order Cholesky).
//! * **Trajectories (#358)**: the resident direct run is bitwise repeatable and finite; its
//!   difference from C order over several steps is reported (the accumulated effect of the
//!   step-1 differences) and bounded with the measured amplification stated in the assertion.
//!   CG runs are compared as in the tenferro test: structure (file set, S size and cuts of the
//!   SR info rows) and the pre-update energy row exactly, repeatability bitwise; the CG
//!   solution itself is validated at stage level by `sr_device_gate`.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_core::sr::observer::{
    capture, capture_with_normalized, DirectSolveObservation, NormalizedObservation,
};
use mvmc_core::sr_backend::{COrderSr, SrStages};
use mvmc_core::stage_backend::{acquire_kind, set_stage_backend_override, StageBackendKind};

const EPS: f64 = f64::EPSILON;

fn gate() -> bool {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("sr-routing-gate: ExplicitSkip: skipped, no device ({why})");
            false
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("sr-routing-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => true,
    }
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Case {
    name: &'static str,
    inputs: PathBuf,
    mode: &'static str,
    samples: usize,
}

fn case(name: &'static str, inputs: PathBuf, samples: usize) -> Case {
    Case {
        name,
        inputs,
        mode: "real",
        samples,
    }
}

fn cases() -> Vec<Case> {
    vec![
        case(
            "hubbard_chain_dh_real",
            repo().join("tests/fixtures/physcal_181/hubbard_chain_dh_real/inputs"),
            60,
        ),
        case(
            "hubbard_chain_L16",
            repo().join("benchmark/hubbard_chain/inputs/hubbard_chain_L16"),
            120,
        ),
        case(
            "hubbard_chain_L32",
            repo().join("benchmark/hubbard_chain/inputs/hubbard_chain_L32"),
            160,
        ),
    ]
}

/// Complex-parameter case: the resident CG solve supports imaginary samples; the resident direct
/// step is real only, so complex direct SR stays on the per-stage path.
fn complex_case() -> Case {
    Case {
        name: "heisenberg_chain_cmp",
        inputs: repo().join("tests/fixtures/physcal_181/heisenberg_chain_cmp/inputs"),
        mode: "cmp",
        samples: 60,
    }
}

#[derive(Default)]
struct Outcome {
    files: BTreeMap<String, String>,
    observations: Vec<DirectSolveObservation>,
    normalized: Vec<NormalizedObservation>,
}

/// One optimization run of `case` with `steps` SR steps; `capture_ops` records the direct solves
/// (and, for the reference, the normalized `OO`/`HO`).
fn run(
    c: &Case,
    tag: &str,
    steps: i64,
    cg: bool,
    kind: StageBackendKind,
    capture_ops: bool,
) -> Outcome {
    set_stage_backend_override(Some(kind));
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue452-{}-{}-{tag}-{}",
        std::process::id(),
        c.name,
        u8::from(cg)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(&c.inputs).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
        }
    }
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let samples = c.samples.to_string();
    let steps_text = steps.to_string();
    let overrides = [
        ("NVMCCalMode", "0"),
        ("NSROptItrStep", steps_text.as_str()),
        ("NSROptItrSmp", "1"),
        ("NVMCSample", samples.as_str()),
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
    let mut config = mvmc_core::RunConfig::new(steps, c.mode);
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    let mut guard = capture_ops.then(|| {
        if kind == StageBackendKind::COrder {
            capture_with_normalized().expect("capture")
        } else {
            capture().expect("capture")
        }
    });
    let result = mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config);
    let mut outcome = Outcome::default();
    if let Some(g) = guard.as_mut() {
        outcome.normalized = g.take_normalized();
    }
    if let Some(g) = guard {
        outcome.observations = g.finish();
    }
    set_stage_backend_override(None);
    result.unwrap_or_else(|e| panic!("{} {kind:?}: {e}", c.name));
    for entry in std::fs::read_dir(&output).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        outcome.files.insert(
            name.clone(),
            std::fs::read_to_string(output.join(&name)).unwrap(),
        );
    }
    let _ = std::fs::remove_dir_all(&root);
    outcome
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for j in 0..n {
        for i in 0..n {
            y[i] += a[i + j * n] * x[j];
        }
    }
    y
}

/// `(lambda_max, kappa)` of the SPD matrix `s`.
fn spectrum(s: &[f64], n: usize) -> (f64, f64) {
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut lmax = 0.0;
    for _ in 0..60 {
        let y = matvec(s, &x, n);
        lmax = norm(&y);
        x = y.iter().map(|v| v / lmax).collect();
    }
    let mut host = COrderSr::default();
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mut inv_norm = 0.0;
    for _ in 0..30 {
        let nv = norm(&v);
        let mut w: Vec<f64> = v.iter().map(|a| a / nv).collect();
        let mut sc = s.to_vec();
        host.cholesky_solve(&mut sc, &mut w, n).expect("SPD");
        inv_norm = norm(&w);
        v = w;
    }
    (lmax, lmax * inv_norm)
}

fn counters() -> mvmc_core::sr_backend::ResidentCounters {
    acquire_kind(StageBackendKind::Cuda(0))
        .sr()
        .resident_counters()
        .expect("the CUDA stage backend must be resident")
}

/// Step-1 operands and solution of the real direct step against C order.
fn check_direct(c: &Case) {
    let reference = run(c, "ref1", 1, false, StageBackendKind::COrder, true);
    let before = counters();
    let device = run(c, "dev1", 1, false, StageBackendKind::Cuda(0), true);
    let after = counters();

    let (r, d) = (&reference.observations[0], &device.observations[0]);
    let norm_ref = &reference.normalized[0];
    // n = 1 + NPara from the normalized OO (length n * (n + 2))
    let len = norm_ref.oo_real.len();
    let n = (((len + 1) as f64).sqrt() as usize).saturating_sub(1);
    assert_eq!(n * (n + 2), len, "{}: OO layout", c.name);
    assert!(
        device.normalized.is_empty() || device.normalized.iter().all(|x| x.oo_real.is_empty()),
        "{}: the resident run must not materialize the host OO",
        c.name
    );
    assert_eq!(
        after.direct_steps - before.direct_steps,
        1,
        "{}: one resident direct step",
        c.name
    );
    assert_eq!(after.store_uploads - before.store_uploads, 1);
    assert_eq!(
        after.store_values - before.store_values,
        (n * c.samples) as u64,
        "{}: the O store (n x samples) is the only large upload",
        c.name
    );

    assert_eq!(r.active_indices, d.active_indices, "{}: active set", c.name);
    assert_eq!(r.dimension, d.dimension);
    let nm = r.dimension;
    let (k, sta_del, dt) = (
        c.samples as f64,
        r.settings.diagonal_shift,
        r.settings.step_dt,
    );
    let oo = &norm_ref.oo_real;
    let ho = &norm_ref.ho_real;
    let a_of = |a: usize| oo[a + a * n].sqrt();
    let mean = |a: usize| oo[a];
    let comp: Vec<usize> = r.active_indices.iter().map(|&i| i / 2 + 1).collect();

    // S: elementwise derived bound
    let mut worst_s = 0.0f64;
    let mut bound_fro = 0.0f64;
    for j in 0..nm {
        for i in 0..nm {
            let (ai, aj) = (a_of(comp[i]), a_of(comp[j]));
            let diag = if i == j { 1.0 + sta_del } else { 1.0 };
            let bound = 6.0 * (k + 3.0) * EPS * diag * ai * aj;
            let diff = (r.matrix[i + j * nm] - d.matrix[i + j * nm]).abs();
            worst_s = worst_s.max(diff / bound);
            bound_fro += bound * bound;
        }
    }
    let bound_fro = bound_fro.sqrt();
    assert!(worst_s <= 1.0, "{}: S error / bound = {worst_s}", c.name);

    // g: the same sampled HO enters both; only the mean differs
    let mut worst_g = 0.0f64;
    let mut bound_g2 = 0.0f64;
    for i in 0..nm {
        let a = comp[i];
        let bound = 2.0
            * dt.abs()
            * (2.0 * (k + 2.0) * EPS * ho[0].abs() * a_of(a)
                + 8.0 * EPS * (ho[a].abs() + (ho[0] * mean(a)).abs()));
        let diff = (r.rhs[i] - d.rhs[i]).abs();
        worst_g = worst_g.max(diff / bound.max(f64::MIN_POSITIVE));
        bound_g2 += bound * bound;
    }
    let bound_g = bound_g2.sqrt();
    assert!(worst_g <= 1.0, "{}: g error / bound = {worst_g}", c.name);

    // solution: first-order perturbation bound with the C-order S
    let (lmax, kappa) = spectrum(&r.matrix, nm);
    let dx: Vec<f64> = r
        .increment
        .iter()
        .zip(&d.increment)
        .map(|(a, b)| a - b)
        .collect();
    let rel = norm(&dx) / norm(&r.increment);
    let bound = 2.0 * kappa * (bound_fro / lmax + bound_g / norm(&r.rhs) + 4.0 * nm as f64 * EPS);
    eprintln!(
        "  {:22} n={n:4} nm={nm:4} k={:4}: S err/bound {worst_s:.2e}, g err/bound {worst_g:.2e}, kappa~{kappa:.2e}, |dx|/|x| {rel:.2e} (bound {bound:.2e})",
        c.name, c.samples
    );
    assert!(rel <= bound, "{}: solution error {rel} > {bound}", c.name);
}

fn compare_files(label: &str, a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> f64 {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "{label}: file set"
    );
    let mut worst = 0.0f64;
    for (name, text) in a {
        let (x, y): (Vec<&str>, Vec<&str>) = (
            text.split_whitespace().collect(),
            b[name].split_whitespace().collect(),
        );
        assert_eq!(x.len(), y.len(), "{label}: {name} token count");
        for (p, q) in x.iter().zip(&y) {
            match (p.parse::<f64>(), q.parse::<f64>()) {
                (Ok(p), Ok(q)) => {
                    assert!(q.is_finite(), "{label}: {name} not finite");
                    worst = worst.max((p - q).abs() / p.abs().max(1.0));
                }
                _ => assert_eq!(p, q, "{label}: {name}"),
            }
        }
    }
    worst
}

/// Several-step direct runs: bitwise repeatable on the device, finite, and close to C order.
fn check_direct_trajectory(c: &Case, steps: i64) {
    let reference = run(c, "ref", steps, false, StageBackendKind::COrder, false);
    let before = counters();
    let dev_a = run(c, "deva", steps, false, StageBackendKind::Cuda(0), false);
    let after = counters();
    let dev_b = run(c, "devb", steps, false, StageBackendKind::Cuda(0), false);
    assert_eq!(
        after.direct_steps - before.direct_steps,
        steps as u64,
        "{}: one resident direct step per SR step",
        c.name
    );
    assert_eq!(
        dev_a.files, dev_b.files,
        "{}: resident run not repeatable",
        c.name
    );
    let worst = compare_files(c.name, &reference.files, &dev_a.files);
    eprintln!(
        "  {:22} {steps} steps: worst relative difference of any output number vs C order {worst:.2e}",
        c.name
    );
    // 1e-6 is of the order of the derived step-1 solution bounds of `check_direct` (2e-8 to
    // 5e-7 for these cases); the observed differences are about 1e-14 (docs/manual 12.4)
    assert!(worst <= 1e-6, "{}: trajectory differs by {worst}", c.name);
}

/// CG runs: device CG solve per step, structure equal to C order, repeatable on the device.
fn check_cg(c: &Case, steps: i64) {
    let reference = run(c, "cgref", steps, true, StageBackendKind::COrder, false);
    let before = counters();
    let dev_a = run(c, "cga", steps, true, StageBackendKind::Cuda(0), false);
    let after = counters();
    let dev_b = run(c, "cgb", steps, true, StageBackendKind::Cuda(0), false);
    assert_eq!(
        after.cg_solves - before.cg_solves,
        steps as u64,
        "{}: one device CG solve per SR step",
        c.name
    );
    assert_eq!(
        after.direct_steps, before.direct_steps,
        "{}: CG must not use the direct step",
        c.name
    );
    assert_eq!(
        dev_a.files, dev_b.files,
        "{}: resident CG not repeatable",
        c.name
    );
    assert_eq!(
        reference.files.keys().collect::<Vec<_>>(),
        dev_a.files.keys().collect::<Vec<_>>()
    );
    let first = |files: &BTreeMap<String, String>| {
        files["zvo_out.dat"].lines().next().unwrap_or("").to_owned()
    };
    // the pre-update energy row does not depend on the SR backend
    assert_eq!(first(&reference.files), first(&dev_a.files), "{}", c.name);
    // S size and cuts of the first SR info row (the discrete contract)
    let info_cols = |files: &BTreeMap<String, String>| {
        files["zvo_SRinfo.dat"]
            .lines()
            .next()
            .map(|l| {
                l.split_whitespace()
                    .take(4)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    assert_eq!(
        info_cols(&reference.files),
        info_cols(&dev_a.files),
        "{}",
        c.name
    );
    for (name, text) in &dev_a.files {
        for token in text.split_whitespace() {
            if let Ok(v) = token.parse::<f64>() {
                assert!(v.is_finite(), "{}: {name} not finite", c.name);
            }
        }
    }
    eprintln!(
        "  {:22} cg {steps} steps: routed, repeatable, structure equal",
        c.name
    );
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn resident_direct_sr_run_matches_c_order() {
    if !gate() {
        return;
    }
    for c in cases() {
        check_direct(&c);
        check_direct_trajectory(&c, 4);
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn resident_cg_sr_run_is_routed_and_repeatable() {
    if !gate() {
        return;
    }
    for c in cases().into_iter().chain([complex_case()]) {
        check_cg(&c, 3);
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn complex_direct_sr_stays_on_the_per_stage_path() {
    if !gate() {
        return;
    }
    let c = complex_case();
    let reference = run(&c, "cref", 2, false, StageBackendKind::COrder, false);
    let before = counters();
    let device = run(&c, "cdev", 2, false, StageBackendKind::Cuda(0), false);
    assert_eq!(
        counters().direct_steps,
        before.direct_steps,
        "complex direct SR must not use the real resident step"
    );
    let worst = compare_files(c.name, &reference.files, &device.files);
    eprintln!(
        "  {:22} complex direct 2 steps (per-stage path): worst relative difference {worst:.2e}",
        c.name
    );
    assert!(
        worst <= 1e-6,
        "{}: complex direct differs by {worst}",
        c.name
    );
}
