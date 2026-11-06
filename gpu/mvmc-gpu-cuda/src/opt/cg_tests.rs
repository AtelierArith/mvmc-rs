//! Resident SR-CG (`NSRCG != 0`) against the Rust CPU C-order CG (#670, #694, #572).
//!
//! Julia GPU Opt rejects CG SR, so the reference is C `StochasticOptCG_real`
//! (`stcopt_cg_impl.c`) as ported by `mvmc_core::sr_cg`: the plain and
//! DiagScale solves of `SampledSrOperator::solve_variant`, the `NSRCGFallback`
//! rerun and `RescaleSmat` of `stochastic_opt_cg`, applied to the derivatives
//! and local energies the GPU produced. Integers (active set, rescale decision,
//! iteration counts, statuses, fallback decision) are exact; FP64 bounds are
//! stated at each assertion and derived in `docs/design/gpu-resident-opt.md`
//! (SR-CG section).
#![allow(clippy::needless_range_loop)] // index-parallel host references
use super::dh_tests::worst;
use super::tests::close;
use super::*;
use mvmc_core::sr_cg::{
    install_cg_observer, CgIterationView, CgObserver, CgSolution, CgVariant, SampledSrOperator,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use tenferro_ext::raw_exec::raw_exec_counters;

const EPS: f64 = f64::EPSILON;

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn source(name: &str) -> PathBuf {
    match name {
        "L16" => repo("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16"),
        "L128" => repo("../../benchmark/hubbard_chain/inputs/hubbard_chain_L128"),
        "dh_L16" => repo("tests/fixtures/dh/hubbard_dh_L16"),
        "rbm_L16" => repo("tests/fixtures/rbm/hubbard_rbm_L16"),
        "opttrans_L16" => repo("tests/fixtures/opttrans/hubbard_opttrans_L16"),
        other => panic!("unknown input {other}"),
    }
}

/// Copy an input with `NSRCG 1` (or the `NSRCG` line of `extra`) and extra
/// modpara.def lines (C keywords).
pub(crate) fn cg_input(name: &str, tag: &str, extra: &[&str]) -> PathBuf {
    let from = source(name);
    let dir = std::env::temp_dir().join(format!("mvmc-cg-670-{name}-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&from).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let modpara = dir.join("modpara.def");
    let text = std::fs::read_to_string(&modpara).unwrap();
    let key = |l: &str| l.split_whitespace().next().unwrap_or("").to_string();
    let mut lines: Vec<String> = text
        .lines()
        .filter(|l| key(l) != "NSRCG" && !extra.iter().any(|e| key(e) == key(l)))
        .map(str::to_string)
        .collect();
    if !extra.iter().any(|e| key(e) == "NSRCG") {
        lines.push("NSRCG          1".into());
    }
    lines.extend(extra.iter().map(|s| s.to_string()));
    std::fs::write(&modpara, lines.join("\n") + "\n").unwrap();
    dir.join("namelist.def")
}

fn download<T: Copy + Default>(
    exec: &RawExec<'_, '_>,
    buffer: &DeviceBuffer,
    len: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let mut out = vec![T::default(); len];
    // SAFETY: every caller passes a length within the buffer's allocation.
    unsafe {
        exec.download(&mut out, exec.ptr(buffer)?)?;
    }
    Ok(out)
}

/// Device inputs and outputs of one CG solve.
struct Solve {
    /// Derivatives before the solve (RescaleSmat scales the stored ones).
    raw: Vec<f64>,
    /// Derivatives after the solve.
    derivatives: Vec<f64>,
    energy: Vec<f64>,
    mean: Vec<f64>,
    variance: Vec<f64>,
    ho: Vec<f64>,
    oodiag: Vec<f64>,
    x: Vec<f64>,
    gradient: Vec<f64>,
    active: Vec<i32>,
    /// `[delta, threshold, alpha, coef, 1/Wc, rescaleOO, rescaleO, -]`.
    st: Vec<f64>,
    /// `[stop, iterations, max iterations, nSmat, C status, first status,
    /// fallback ran, rescaled]`.
    ist: Vec<i32>,
    /// `delta` of the first solve (`limit + 1`), then of the fallback solve.
    trace: Vec<f64>,
    info: Vec<i32>,
    /// zqp_SRinfo.dat info and fallback status of step 0.
    srinfo: i32,
    fallback: i32,
    /// Parameters before and after the solve (RescaleSmat divides Slater).
    before: Vec<f64>,
    after: Vec<f64>,
    seconds: f64,
}

type Edit<'a> = &'a dyn Fn(&RawExec<'_, '_>, &State) -> tenferro_tensor::Result<()>;

/// Observe the current walkers, apply `edit` to the device state, and run one
/// device CG solve (no update).
fn solve_once(backend: &mut CudaEngine, state: &State, edit: Option<Edit<'_>>) -> Solve {
    let m = &state.model;
    let (np, w) = (m.np, state.resident.sampler.geom.walkers);
    with_raw_exec(backend, "cg.solve", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        if let Some(edit) = edit {
            edit(exec, state)?;
        }
        let b = &state.buffers;
        let raw = download(exec, &b.derivatives, w * np)?;
        let before = download(exec, &b.parameters, np)?;
        exec.synchronize()?;
        let start = std::time::Instant::now();
        state.solve_cg(exec, &module, 0)?;
        exec.synchronize()?;
        let seconds = start.elapsed().as_secs_f64();
        let cg = b.cg.as_ref().unwrap();
        let fallback = usize::from(m.data.modpara.nsrcg_fallback);
        Ok(Solve {
            raw,
            derivatives: download(exec, &b.derivatives, w * np)?,
            energy: download(exec, &b.local_energy, w)?,
            mean: download(exec, &b.mean, np + 1)?,
            variance: download(exec, &b.variance, np)?,
            ho: download(exec, &b.ho, np)?,
            oodiag: download(exec, &cg.oodiag, np)?,
            x: download(exec, &b.rhs, np)?,
            gradient: download(exec, &cg.vec, np)?,
            active: download(exec, &cg.active, np)?,
            st: download(exec, &cg.st, 8)?,
            ist: download(exec, &cg.ist, 8)?,
            trace: download(exec, &cg.trace, (cg.limit + 1) * (1 + fallback))?,
            info: download(exec, &b.info, 2)?,
            srinfo: download::<i32>(exec, &cg.iterations, 1)?[0],
            fallback: download::<i32>(exec, &cg.fallback, 1)?[0],
            before,
            after: download(exec, &b.parameters, np)?,
            seconds,
        })
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

#[derive(Default)]
struct Trace {
    delta: RefCell<Vec<f64>>,
}

impl CgObserver for Trace {
    fn iteration(&self, state: CgIterationView<'_>) {
        self.delta.borrow_mut().push(state.delta);
    }
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// C `get_absmax` (stcopt_cg.c:35-46).
fn absmax(p: &[f64]) -> f64 {
    let Some(first) = p.first() else { return 0.0 };
    let mut max = first.abs();
    for v in &p[1..] {
        if max < v.abs() {
            max = v.abs();
        }
    }
    max
}

/// Host `Rescale4SRCG_real` (stcopt_cg_impl.c:772-875) decision on the
/// unscaled variance: `(start, end, rescaleOO, rescaleO)`, or `None` where C
/// returns without rescaling.
fn host_rescale(model: &Model, variance: &[f64]) -> Option<(usize, usize, f64, f64)> {
    let (start, end) = cg::rescale_block(model.nproj, model.nrbm, model.nopt, model.np)?;
    let proj = absmax(&variance[..model.nproj]);
    let mut slater = absmax(&variance[start..end]);
    if proj < 1e-10 {
        return None;
    }
    if slater < 1e-10 {
        slater = 1e-10;
    }
    let oo = proj / slater;
    Some((start, end, oo, oo.sqrt()))
}

/// C DiagScale preconditioner entry (stcopt_cg_impl.c:383-389).
fn precon(sdiag: f64, sta_del: f64) -> f64 {
    let mut dq = (1.0 + sta_del) * sdiag;
    if !dq.is_finite() || dq.abs() < 1e-12 {
        dq = if dq < 0.0 { -1e-12 } else { 1e-12 };
    }
    dq
}

fn variant_of(diag: bool) -> CgVariant {
    if diag {
        CgVariant::DiagScale
    } else {
        CgVariant::Plain
    }
}

/// Iterations of a solve from its C status: the count on convergence, the
/// failing iteration after a breakdown, `max_iter` at the limit.
fn iterations_of(status: i64) -> usize {
    if status >= 0 {
        status as usize
    } else {
        (-status - 1) as usize
    }
}

/// Operator spectrum of one variant: `(lambda_min(S), lambda_max(S), kappa,
/// max M_ii)`, where kappa is that of S (plain) or of M^-1/2 S M^-1/2
/// (DiagScale). The lower bounds are rigorous: S = Cov + DSROptStaDel
/// diag(S_ii) with Cov positive semidefinite, so lambda_min(S) >= DSROptStaDel
/// min S_ii and lambda_min(M^-1/2 S M^-1/2) >= DSROptStaDel min S_ii/M_ii.
/// The upper eigenvalues are power-iteration estimates.
fn spectrum(
    operator: &mut SampledSrOperator,
    variant: CgVariant,
    inv_w: f64,
    sta_del: f64,
) -> (f64, f64, f64, f64) {
    let n = operator.diagonal.len();
    let diag = operator.diagonal.clone();
    let m: Vec<f64> = diag.iter().map(|&s| precon(s, sta_del)).collect();
    let power = |operator: &mut SampledSrOperator, scale: &dyn Fn(usize) -> f64| {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut z = vec![0.; n];
        let mut lambda = 0.;
        for _ in 0..60 {
            let u: Vec<f64> = (0..n).map(|i| v[i] * scale(i)).collect();
            operator.apply(&mut z, &u, inv_w, sta_del);
            z.iter_mut().enumerate().for_each(|(i, x)| *x *= scale(i));
            lambda = norm(&z);
            v.iter_mut().zip(&z).for_each(|(a, b)| *a = b / lambda);
        }
        lambda
    };
    let lambda_min = sta_del * diag.iter().cloned().fold(f64::INFINITY, f64::min);
    let lambda_max = power(operator, &|_| 1.0);
    let kappa = match variant {
        CgVariant::Plain => 2.0 * lambda_max / lambda_min,
        CgVariant::DiagScale => {
            assert!(m.iter().all(|&x| x > 0.0), "positive preconditioner");
            let low = sta_del * (0..n).map(|i| diag[i] / m[i]).fold(f64::INFINITY, f64::min);
            2.0 * power(operator, &|i| 1.0 / m[i].sqrt()) / low
        }
    };
    let mmax = m.iter().cloned().fold(0f64, f64::max);
    (lambda_min, lambda_max, kappa, mmax)
}

/// One CPU solve of `variant` (optionally with the samples in reverse order)
/// and its `delta` trace.
struct CpuSolve {
    result: CgSolution,
    delta: Vec<f64>,
    seconds: f64,
}

/// Device trajectory of one solve against the CPU solve and its reordered
/// twin. Returns whether the iteration count (and status) is determined.
#[allow(clippy::too_many_arguments)]
fn compare_trajectory(
    name: &str,
    trace: &[f64],
    gpu_status: i64,
    cpu: &CpuSolve,
    reordered: &CpuSolve,
    kappa: f64,
    w: usize,
    n: usize,
    thresh: f64,
    max_iter: usize,
) -> bool {
    let gpu_k = iterations_of(gpu_status);
    let breakdown = gpu_status < 0 && gpu_k < max_iter;
    // C stopping rule on the device's own values (stcopt_cg_impl.c:528-535,
    // :397-402): every completed iteration started with delta >= threshold
    // (or NaN), and the loop ended by max_iter, by delta < threshold, or by a
    // breakdown guard.
    assert!(gpu_k <= max_iter, "{name}: iterations beyond max_iter");
    assert!(
        trace[..gpu_k].iter().all(|&d| d >= thresh || d.is_nan()),
        "{name}: device iterated past C's stop test"
    );
    assert!(
        breakdown || gpu_k == max_iter || trace[gpu_k] < thresh,
        "{name}: device stopped before C's stop test ({gpu_k} of {max_iter})"
    );
    // Deterministic regime. Both runs apply C's control flow and differ only in
    // the reduction order of the sampled products and dots, a relative
    // perturbation <= (W + nSmat) eps per operation. While CG is stable, the
    // divergence of two such runs grows at most like k kappa times that (the
    // finite-precision CG perturbation bounds of Greenbaum 1989), with kappa
    // of the operator CG actually iterates on (M^-1/2 S M^-1/2 for DiagScale).
    // Once the residual has fallen far below ||g||, CG amplifies rounding by
    // orders of magnitude per iteration and different summation orders give
    // different (equally valid) iterates. That regime starts where the CPU's
    // own reordered run reaches 1/64 of the budget; iterates are compared
    // before it.
    let budget = |i: usize| 8.0 * (i.max(1) as f64) * kappa * (w + n) as f64 * EPS;
    let rel = |a: f64, b: f64| ((a - b) / b.abs().max(f64::MIN_POSITIVE)).abs();
    let k = cpu.result.iterations;
    let last = k
        .min(reordered.result.iterations)
        .min(cpu.delta.len() - 1)
        .min(reordered.delta.len() - 1);
    let chaotic = (0..=last)
        .find(|&i| rel(reordered.delta[i], cpu.delta[i]) > budget(i) / 64.0)
        .unwrap_or(last + 1);
    let compared = chaotic.min(gpu_k + 1).min(cpu.delta.len());
    for i in 0..compared {
        let r = rel(trace[i], cpu.delta[i]);
        assert!(
            r <= budget(i),
            "{name}: delta at iteration {i} differs by {r:e} > {:e} (deterministic until {chaotic})",
            budget(i)
        );
    }
    // Same iteration count and status as C whenever the stop (by max_iter, by
    // a breakdown at iteration 0, or by the tolerance with a margin beyond
    // rounding) falls in the deterministic regime.
    let stop_margin = rel(cpu.delta[k.min(cpu.delta.len() - 1)], thresh);
    let count_determined = k < chaotic
        && (k == max_iter
            || (cpu.result.status < 0 && k == 0 && cpu.delta[0].is_nan())
            || stop_margin > budget(k));
    // A failure at the limit far above the threshold is also decided past the
    // deterministic regime when the CPU's own reordered run fails there with
    // the same margin (> 1e3): rounding-order sensitivity then cannot move
    // delta below the threshold, so the status (and the fallback decision)
    // is compared exactly.
    let robust_limit = k == max_iter
        && reordered.result.iterations == max_iter
        && cpu.result.status < 0
        && reordered.result.status == cpu.result.status
        && stop_margin > 1e3
        && rel(reordered.delta[max_iter], thresh) > 1e3;
    let count_determined = count_determined || robust_limit;
    if count_determined {
        assert_eq!(gpu_k, k, "{name}: C CG iteration count");
        assert_eq!(gpu_status, cpu.result.status, "{name}: C 1.4.0 CG status");
    }
    println!(
        "{name}: iterations GPU/CPU/CPU-reordered={gpu_k}/{k}/{} status GPU/CPU={gpu_status}/{} \
         (max {max_iter}, determined: {count_determined}), stop margin={stop_margin:.2e}; \
         kappa<={kappa:.2e}; delta agrees to {:.1e} (budget {:.1e}) through iteration {}; \
         reordered run reaches budget/64 at {chaotic}; CPU solve={:.2}ms",
        reordered.result.iterations,
        cpu.result.status,
        (0..compared)
            .map(|i| rel(trace[i], cpu.delta[i]))
            .fold(0f64, f64::max),
        budget(compared.saturating_sub(1)),
        compared.saturating_sub(1),
        cpu.seconds * 1e3,
    );
    count_determined
}

/// Per-entry check `|a - b| <= bound(i)`; returns the worst ratio.
fn within(label: &str, a: &[f64], b: &[f64], bound: impl Fn(usize) -> f64) -> f64 {
    assert_eq!(a.len(), b.len(), "{label} length");
    let mut worst = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let limit = bound(i);
        assert!(
            a[i].is_finite() && b[i].is_finite() && d <= limit,
            "{label}[{i}]: {} vs {} (bound {limit:e})",
            a[i],
            b[i]
        );
        if limit > 0.0 {
            worst = worst.max(d / limit);
        }
    }
    worst
}

/// Exact integer outcome of one checked step.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    /// zqp_SRinfo.dat info (the first solve's status).
    info: i32,
    /// Fallback solve status, when it ran.
    fallback: Option<i32>,
    /// Iterations of the final solve.
    iterations: usize,
    /// RescaleSmat applied.
    rescaled: bool,
}

/// Check one device step against the CPU C-order SR-CG on the same samples.
fn assert_solve(name: &str, model: &Model, g: &Solve) -> Outcome {
    let mp = &model.data.modpara;
    let np = model.np;
    let w = g.energy.len();
    let wc = w as f64;
    let raw = |s: usize, p: usize| g.raw[s * np + p];

    // C real-path moments (vmccal.c with w=1, weightAverageSROpt): samples
    // summed in order, then divided by Wc.
    let mut mean = vec![0.; np + 1];
    let mut oo = vec![0.; np];
    let mut ho = vec![0.; np];
    for s in 0..w {
        for p in 0..np {
            mean[p] += raw(s, p);
            oo[p] += raw(s, p) * raw(s, p);
            ho[p] += g.energy[s] * raw(s, p);
        }
        mean[np] += g.energy[s];
    }
    mean.iter_mut().for_each(|v| *v /= wc);
    oo.iter_mut().for_each(|v| *v /= wc);
    ho.iter_mut().for_each(|v| *v /= wc);
    let raw_mean = mean.clone();
    let raw_oo = oo.clone();
    let raw_ho = ho.clone();
    let mut variance: Vec<f64> = (0..np).map(|p| oo[p] - mean[p] * mean[p]).collect();
    // The device performs the same IEEE operations in the same order without
    // FMA, so the unscaled moments agree within a few ulps: 4 eps of the
    // vector scale plus 4 eps relative. a(p) is that variance bound.
    let vscale = variance.iter().fold(0f64, |m, v| m.max(v.abs()));
    let raw_variance = variance.clone();
    let a = |p: usize| 4.0 * EPS * vscale + 4.0 * EPS * raw_variance[p].abs();

    // RescaleSmat (stcopt_cg.c:58-61). The decision is exact: the device flag
    // equals the host decision (proj_max is never within rounding of 1e-10 in
    // these inputs, checked below). The ratios follow from the two maxima;
    // each maximum is 1-Lipschitz in the sup norm, so its relative error is at
    // most max a(p) / maximum, and rescaleOO = proj/slater adds one rounding.
    let block = cg::rescale_block(model.nproj, model.nrbm, model.nopt, np);
    let rescale = if mp.rescale_smat {
        host_rescale(model, &variance)
    } else {
        None
    };
    let device_rescaled = mp.rescale_smat && block.is_some() && g.ist[7] != 0;
    assert_eq!(
        device_rescaled,
        rescale.is_some(),
        "{name}: RescaleSmat decision"
    );
    let mut delta_ratio = 0.;
    let (start, end) = rescale.map_or((0, 0), |r| (r.0, r.1));
    let scaled = |p: usize| (start..end).contains(&p);
    let o_ratio = rescale.map_or(1.0, |r| r.3);
    if let Some((start, end, oo_ratio, o_ratio)) = rescale {
        let proj = absmax(&variance[..model.nproj]);
        let slater = absmax(&variance[start..end]);
        assert!(
            (proj - 1e-10).abs() > (0..model.nproj).map(a).fold(0., f64::max),
            "{name}: proj_max {proj:e} within rounding of C's 1e-10 early return"
        );
        let e_proj = (0..model.nproj).map(a).fold(0., f64::max) / proj;
        let e_slater = if slater < 1e-10 {
            0.0
        } else {
            (start..end).map(a).fold(0., f64::max) / slater
        };
        delta_ratio = e_proj + e_slater + 4.0 * EPS;
        let r_oo = ((g.st[5] - oo_ratio) / oo_ratio).abs();
        let r_o = ((g.st[6] - o_ratio) / o_ratio).abs();
        assert!(
            r_oo <= delta_ratio && r_o <= delta_ratio,
            "{name}: rescale ratios {:e}/{:e} vs {oo_ratio:e}/{o_ratio:e} (bound {delta_ratio:e})",
            g.st[5],
            g.st[6]
        );
        // Host Rescale4SRCG on the averaged operands.
        for p in start..end {
            oo[p] *= oo_ratio;
            mean[p] *= o_ratio;
            ho[p] *= o_ratio;
            variance[p] = oo[p] - mean[p] * mean[p];
        }
        println!(
            "{name}: RescaleSmat block {start}..{end} proj_max={proj:.3e} slater_max={slater:.3e} \
             rescaleOO={oo_ratio:.6e} (device rel {r_oo:.1e}, bound {delta_ratio:.1e})"
        );
    }
    let o = |s: usize, p: usize| {
        if scaled(p) {
            raw(s, p) * o_ratio
        } else {
            raw(s, p)
        }
    };
    // Rescaled entries carry the ratio error delta_ratio (+1 rounding); the
    // variance of a rescaled entry, OO*rho - (O*sqrt(rho))^2, in addition its
    // cancellation: (delta_ratio + 8 eps)(|OO| + O^2).
    let d = |p: usize| if scaled(p) { delta_ratio + EPS } else { 0.0 };
    let mscale = mean.iter().fold(0f64, |m, v| m.max(v.abs()));
    let hscale = ho.iter().fold(0f64, |m, v| m.max(v.abs()));
    let ratios = [
        within(&format!("{name} mean"), &g.mean, &mean, |p| {
            4.0 * EPS * mscale + (4.0 * EPS + d(p)) * mean[p].abs()
        }),
        within(&format!("{name} ho"), &g.ho, &ho, |p| {
            4.0 * EPS * hscale + (4.0 * EPS + d(p)) * ho[p].abs()
        }),
        within(&format!("{name} variance"), &g.variance, &variance, |p| {
            a(p) + if scaled(p) {
                (delta_ratio + 8.0 * EPS) * (oo[p].abs() + mean[p] * mean[p])
            } else {
                0.0
            }
        }),
        within(&format!("{name} oodiag"), &g.oodiag, &oo, |p| {
            (4.0 * EPS + d(p)) * oo[p].abs()
        }),
    ];
    // Stored samples and parameters: scaled entries by one rounding plus the
    // ratio error; the rest untouched.
    let expected: Vec<f64> = (0..w * np).map(|i| o(i / np, i % np)).collect();
    within(
        &format!("{name} derivatives"),
        &g.derivatives,
        &expected,
        |i| d(i % np) * expected[i].abs(),
    );
    let divided: Vec<f64> = (0..np)
        .map(|p| {
            if scaled(p) {
                g.before[p] / o_ratio
            } else {
                g.before[p]
            }
        })
        .collect();
    within(
        &format!("{name} Slater division"),
        &g.after,
        &divided,
        |p| (d(p) + EPS) * divided[p].abs(),
    );

    // stcopt_cg_impl.c:127-176: non-absolute maximum from sDiagElm[0], cut on `<`.
    let mut max = variance[0];
    for &v in &variance {
        if v > max {
            max = v;
        }
    }
    let threshold = max * mp.dsr_opt_red_cut;
    let mut mapping = vec![];
    let mut margin = f64::INFINITY;
    for p in 0..np {
        if model.flags[p] == 0 {
            continue;
        }
        margin =
            margin.min((variance[p] - threshold).abs() / threshold.abs().max(f64::MIN_POSITIVE));
        if variance[p] >= threshold {
            mapping.push(p);
        }
    }
    let n = mapping.len();
    let active: Vec<usize> = (0..np).filter(|&p| g.active[p] != 0).collect();
    assert_eq!(active, mapping, "{name}: C diagonal-cut active set");
    assert_eq!(g.ist[3] as usize, n, "{name}: nSmat");
    // The cut decision depends on rounding only within a few ulps of the threshold.
    assert!(margin > 1e-12, "{name}: diagonal-cut margin {margin:e}");

    let dt = 2.0 * mp.dsr_opt_step_dt;
    let gradient: Vec<f64> = mapping
        .iter()
        .map(|&p| -dt * (ho[p] - mean[np] * mean[p]))
        .collect();
    let device_gradient: Vec<f64> = mapping.iter().map(|&p| g.gradient[p]).collect();
    let gscale = gradient.iter().fold(0f64, |m, v| m.max(v.abs()));
    let (ratio, _) = worst(
        &device_gradient,
        &gradient,
        (8.0 * EPS + 2.0 * delta_ratio) * gscale,
        8.0 * EPS,
    );
    assert!(ratio <= 1.0, "{name}: gradient ratio {ratio:e}");
    for p in 0..np {
        if g.active[p] == 0 {
            assert_eq!(g.x[p], 0.0, "{name}: inactive parameter {p} changed");
        }
    }

    let max_iter = if mp.nsr_opt_cg_max_iter > 0 {
        mp.nsr_opt_cg_max_iter as usize
    } else {
        n
    };
    let inv_w = 1.0 / wc;
    let operator = || {
        let mut operator = SampledSrOperator::new(n, w, false);
        for (si, &p) in mapping.iter().enumerate() {
            operator.mean[si] = mean[p];
            operator.diagonal[si] = variance[p];
        }
        operator
    };
    let cpu_solve = |variant: CgVariant, reverse: bool| {
        let mut operator = operator();
        for (si, &p) in mapping.iter().enumerate() {
            for s in 0..w {
                let t = if reverse { w - 1 - s } else { s };
                operator.real_samples[t * n + si] = o(s, p);
            }
        }
        let trace = Rc::new(Trace::default());
        let guard = install_cg_observer(trace.clone()).unwrap();
        let start = std::time::Instant::now();
        let result = operator.solve_variant(
            variant,
            &gradient,
            inv_w,
            mp.dsr_opt_sta_del,
            mp.dsr_opt_cg_tol,
            max_iter,
        );
        let seconds = start.elapsed().as_secs_f64();
        drop(guard);
        let delta = trace.delta.borrow().clone();
        (
            operator,
            CpuSolve {
                result,
                delta,
                seconds,
            },
        )
    };
    let thresh = mp.dsr_opt_cg_tol * mp.dsr_opt_cg_tol * n as f64 * n as f64;
    let first = variant_of(mp.use_diag_scale);
    let limit = g.trace.len() / (1 + usize::from(mp.nsrcg_fallback));

    let (mut op, cpu1) = cpu_solve(first, false);
    let (_, re1) = cpu_solve(first, true);
    let (lambda_min, lambda_max, kappa1, mmax1) =
        spectrum(&mut op, first, inv_w, mp.dsr_opt_sta_del);
    let ran = g.ist[6] != 0;
    let gpu_first = i64::from(if ran { g.ist[5] } else { g.ist[4] });
    let determined1 = compare_trajectory(
        &format!("{name} {first:?}"),
        &g.trace[..limit],
        gpu_first,
        &cpu1,
        &re1,
        kappa1,
        w,
        n,
        thresh,
        max_iter,
    );
    // NSRCGFallback (stcopt_cg_impl.c:225-251): exact decision on the device.
    assert_eq!(
        ran,
        mp.nsrcg_fallback && gpu_first < 0,
        "{name}: device fallback decision"
    );
    if mp.nsrcg_fallback {
        assert!(
            determined1,
            "{name}: the first solve must stop in the deterministic regime so that the \
             fallback decision is comparable"
        );
    }
    assert_eq!(g.srinfo as i64, gpu_first, "{name}: zqp_SRinfo.dat info");
    let (final_variant, cpu, mmax, kappa, gpu_status) = if ran {
        let other = variant_of(!mp.use_diag_scale);
        let (mut op2, cpu2) = cpu_solve(other, false);
        let (_, re2) = cpu_solve(other, true);
        let (_, _, kappa2, mmax2) = spectrum(&mut op2, other, inv_w, mp.dsr_opt_sta_del);
        let gpu2 = i64::from(g.ist[4]);
        assert_eq!(g.fallback, g.ist[4], "{name}: fallback status of the step");
        compare_trajectory(
            &format!("{name} fallback {other:?}"),
            &g.trace[limit..],
            gpu2,
            &cpu2,
            &re2,
            kappa2,
            w,
            n,
            thresh,
            max_iter,
        );
        (other, cpu2, mmax2, kappa2, gpu2)
    } else {
        assert_eq!(g.fallback, i32::MIN, "{name}: no fallback status");
        (first, cpu1, mmax1, kappa1, gpu_first)
    };
    let gpu_k = iterations_of(gpu_status);
    assert_eq!(g.ist[1] as usize, gpu_k, "{name}: final iteration count");

    // A posteriori solution bound, independent of the trajectory: for the true
    // residuals r = g - S x (evaluated on the host), ||x - x*|| <= ||r|| /
    // lambda_min, so ||x_gpu - x_cpu|| <= (||r_gpu|| + ||r_cpu||) / lambda_min.
    let mut operator = op;
    let gpu_x: Vec<f64> = mapping.iter().map(|&p| g.x[p]).collect();
    let true_residual = |operator: &mut SampledSrOperator, x: &[f64]| {
        let mut sx = vec![0.; n];
        operator.apply(&mut sx, x, inv_w, mp.dsr_opt_sta_del);
        let r: Vec<f64> = gradient.iter().zip(&sx).map(|(a, b)| a - b).collect();
        norm(&r)
    };
    let gpu_residual = true_residual(&mut operator, &gpu_x);
    let cpu_residual = true_residual(&mut operator, &cpu.result.solution);
    let diff: Vec<f64> = gpu_x
        .iter()
        .zip(&cpu.result.solution)
        .map(|(a, b)| a - b)
        .collect();
    let x_diff = norm(&diff);
    let x_norm = norm(&cpu.result.solution).max(f64::MIN_POSITIVE);
    // Host residual evaluation error: (W + nSmat) eps (lambda_max ||x|| + ||g||),
    // plus the device-host operand difference of a rescaled Slater block
    // (relative 2 delta_ratio in S and g).
    let evaluation =
        ((w + n) as f64 * EPS + 2.0 * delta_ratio) * (2.0 * lambda_max * x_norm + norm(&gradient));
    let posterior = (gpu_residual + cpu_residual + 2.0 * evaluation) / lambda_min;
    assert!(
        x_diff <= posterior,
        "{name}: ||x_gpu - x_cpu|| = {x_diff:e} > {posterior:e}"
    );

    // Recurrence vs true residual of the device solution: their gap (the
    // residual drift) is <= k (W+nSmat) eps (||S|| ||x|| + ||g||); the
    // 20-iteration refresh only reduces it. DiagScale's delta = r^T M^-1 r, so
    // ||r|| <= sqrt(max M_ii |delta|).
    let drift = 8.0
        * (gpu_k.max(1) as f64)
        * (w + n) as f64
        * EPS
        * (2.0 * lambda_max * norm(&gpu_x) + norm(&gradient));
    let last = if ran { limit + gpu_k } else { gpu_k };
    let recurrence = match final_variant {
        CgVariant::Plain => g.trace[last].abs().sqrt(),
        CgVariant::DiagScale => (g.trace[last].abs() * mmax).sqrt(),
    };
    if gpu_status >= 0 || gpu_k == max_iter {
        assert!(
            gpu_residual <= recurrence + drift,
            "{name}: true residual {gpu_residual:e} > {recurrence:e} + {drift:e}"
        );
    }
    assert_eq!(g.info, [0, 0], "{name}: info");

    // The complete CPU step, mvmc_core::sr_cg::stochastic_opt_cg (RescaleSmat,
    // variant, fallback and update), on the unscaled host moments and samples.
    // Its operands are the oracle's above, so its update equals the oracle's
    // within 2 eps per parameter; the device parameters plus increment agree
    // within ||x_gpu - x_cpu|| plus the parameter rounding.
    let mut data = model.data.clone();
    data.modpara.nvmc_sample = w as i64;
    let packed = |data: &mvmc_core::ExpertModeData| -> Vec<f64> {
        mvmc_core::pack_parameters(data)
            .unwrap()
            .iter()
            .map(|z| z.re)
            .collect()
    };
    close("host parameters", &packed(&data), &g.before, 0.0, 0.0);
    let size = np + 1;
    let mut cpu_state = mvmc_core::VmcOptimizationState::zeros(1, 1, 0, np, 1, w, false, false);
    cpu_state.energy.wc = num_complex::Complex64::new(wc, 0.0);
    let sr = &mut cpu_state.sr_opt;
    sr.sr_opt_oo_real[0] = 1.0;
    sr.sr_opt_ho_real[0] = raw_mean[np];
    for p in 0..np {
        sr.sr_opt_oo_real[p + 1] = raw_mean[p];
        sr.sr_opt_oo_real[size + p + 1] = raw_oo[p];
        sr.sr_opt_ho_real[p + 1] = raw_ho[p];
        for s in 0..w {
            sr.sr_opt_o_store_real[s * size + p + 1] = raw(s, p);
        }
    }
    let step = mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &cpu_state, None);
    assert_eq!(step.unwrap(), 0, "{name}: CPU StochasticOptCG return value");
    let cpu_after = packed(&data);
    let mut oracle: Vec<f64> = (0..np)
        .map(|p| {
            if scaled(p) {
                g.before[p] / rescale.unwrap().3
            } else {
                g.before[p]
            }
        })
        .collect();
    for (si, &p) in mapping.iter().enumerate() {
        oracle[p] += cpu.result.solution[si];
    }
    close(
        "CPU StochasticOptCG step",
        &cpu_after,
        &oracle,
        0.0,
        2.0 * EPS,
    );
    let gpu_after: Vec<f64> = (0..np).map(|p| g.after[p] + g.x[p]).collect();
    within(
        &format!("{name} updated parameters"),
        &gpu_after,
        &cpu_after,
        |p| posterior + (d(p) + 2.0 * EPS) * (g.after[p].abs() + cpu_after[p].abs()),
    );

    println!(
        "{name}: W={w} np={np} nSmat={n} variant={first:?} fallback ran={ran} final={final_variant:?} \
         info={} fallback status={} kappa(final)<={kappa:.2e} cut margin={margin:.2e}; moments \
         worst/bound={:.2}; x rel diff={:.2e} (a posteriori bound {:.2e}); true residual \
         GPU/CPU={gpu_residual:.2e}/{cpu_residual:.2e} (recurrence {recurrence:.2e}, drift bound \
         {drift:.2e}); device solve={:.2}ms",
        g.srinfo,
        g.fallback,
        ratios.iter().cloned().fold(0., f64::max),
        x_diff / x_norm,
        posterior / x_norm,
        g.seconds * 1e3,
    );
    Outcome {
        info: g.srinfo,
        fallback: ran.then_some(g.ist[4]),
        iterations: gpu_k,
        rescaled: device_rescaled,
    }
}

fn config(walkers: usize, interval: usize) -> OptConfig {
    OptConfig {
        nsteps: 1,
        walkers: Some(walkers),
        warmup: Some(4 * interval),
        interval: Some(interval),
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

/// Sample, then solve on the device and on the CPU, with an optional edit of
/// the observed device state. `NSRCGAbortOnFail 0` keeps a solve that fails
/// (C 1.4.0 status `-(max_iter+1)` or a breakdown) inspectable instead of
/// aborting.
fn check_edited(
    name: &str,
    tag: &str,
    extra: &[&str],
    walkers: usize,
    edit: Option<Edit<'_>>,
) -> Outcome {
    let mut extra = extra.to_vec();
    extra.push("NSRCGAbortOnFail 0");
    let input = cg_input(name, tag, &extra);
    let probe = Model::prepare(&input, &config(walkers, 1)).unwrap();
    let config = config(walkers, 8 * probe.ns);
    let model = Model::prepare(&input, &config).unwrap();
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    with_raw_exec(backend, "cg.sample", |exec| {
        state
            .resident
            .enqueue_random(exec, config.warmup.unwrap())?;
        state
            .resident
            .enqueue_random(exec, config.interval.unwrap())
    })
    .unwrap();
    let solve = solve_once(backend, &state, edit);
    state.resident.check_status(backend).unwrap();
    let outcome = assert_solve(&format!("{name}/{tag}"), &state.model, &solve);
    std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    outcome
}

/// [`check_edited`] without an edit; returns the final iteration count.
fn check(name: &str, tag: &str, extra: &[&str], walkers: usize) -> usize {
    check_edited(name, tag, extra, walkers, None).iterations
}

#[test]
#[ignore = "requires CUDA; SR-CG on the L16 chain against the CPU C-order CG"]
fn resident_cg_l16_matches_cpu_c_order() {
    // Default C tolerance (1e-10) and max_iter = nSmat: crosses the
    // 20-iteration residual refresh.
    let k = check("L16", "default", &[], 64);
    assert!(
        k >= 20,
        "the solve must cover a residual refresh ({k} iterations)"
    );
    // NSROptCGMaxIter caps the count exactly; 25 also crosses a refresh.
    assert_eq!(check("L16", "max7", &["NSROptCGMaxIter 7"], 64), 7);
    assert_eq!(check("L16", "max25", &["NSROptCGMaxIter 25"], 64), 25);
    // Looser tolerances stop on C's delta < tol^2 nSmat^2 test inside the
    // deterministic regime, where the count must equal C's.
    for (tag, tol) in [
        ("tol1e-4", "DSROptCGTol 1e-4"),
        ("tol2e-5", "DSROptCGTol 2e-5"),
    ] {
        let loose = check("L16", tag, &[tol], 64);
        assert!(
            0 < loose && loose < k,
            "{tag}: must stop on the tolerance ({loose} vs {k})"
        );
    }
    // delta_0 < tol^2 nSmat^2 already: zero iterations and a zero increment.
    assert_eq!(check("L16", "tol1e-3", &["DSROptCGTol 1e-3"], 64), 0);
}

/// DiagScale CG (`NSRCG 2`, or `useDiagScale 1` with `NSRCG 1`): C
/// `StochasticOptPreconCG_DiagScale_Main` (stcopt_cg_impl.c:352-480).
#[test]
#[ignore = "requires CUDA; DiagScale SR-CG on the L16 chain against the CPU C-order CG"]
fn resident_cg_diag_scale_l16_matches_cpu_c_order() {
    let k = check("L16", "diag", &["NSRCG 2"], 64);
    assert!(
        k >= 20,
        "the solve must cover a residual refresh ({k} iterations)"
    );
    let diag = "useDiagScale 1";
    assert_eq!(
        check("L16", "diag-max7", &[diag, "NSROptCGMaxIter 7"], 64),
        7
    );
    assert_eq!(
        check("L16", "diag-max25", &[diag, "NSROptCGMaxIter 25"], 64),
        25
    );
    for (tag, tol) in [
        ("diag-tol1e-4", "DSROptCGTol 1e-4"),
        ("diag-tol2e-5", "DSROptCGTol 2e-5"),
    ] {
        let loose = check("L16", tag, &[diag, tol], 64);
        assert!(
            0 < loose && loose < k,
            "{tag}: must stop on the tolerance ({loose} vs {k})"
        );
    }
    // DiagScale's delta_0 = r.M^-1 r is still above tol^2 nSmat^2 at 1e-3:
    // one iteration, as in C.
    assert_eq!(
        check("L16", "diag-tol1e-3", &[diag, "DSROptCGTol 1e-3"], 64),
        1
    );
}

#[test]
#[ignore = "requires CUDA; SR-CG with DH2/DH4 and RBM factors against the CPU C-order CG"]
fn resident_cg_dh_rbm_match_cpu_c_order() {
    check("dh_L16", "default", &[], 64);
    check("rbm_L16", "default", &[], 64);
    check("dh_L16", "diag", &["NSRCG 2"], 64);
    check("rbm_L16", "diag", &["NSRCG 2"], 64);
}

#[test]
#[ignore = "requires CUDA; SR-CG on the L128 chain against the CPU C-order CG"]
fn resident_cg_l128_matches_cpu_c_order() {
    check("L128", "default", &[], 64);
    check("L128", "max40", &["NSROptCGMaxIter 40"], 320);
    check("L128", "diag", &["NSRCG 2"], 64);
    check(
        "L128",
        "diag-max40",
        &["NSRCG 2", "NSROptCGMaxIter 40"],
        320,
    );
}

/// NSRCGFallback (stcopt_cg_impl.c:225-251): a failed first solve is rerun
/// with the other variant from x = 0; zqp_SRinfo.dat keeps the first status.
#[test]
#[ignore = "requires CUDA; NSRCGFallback against the CPU C-order fallback"]
fn resident_cg_fallback_reruns_the_other_variant() {
    let fallback = "NSRCGFallback 1";
    // A converging first solve needs no fallback.
    let ok = check_edited("L16", "fb-none", &[fallback, "DSROptCGTol 1e-4"], 64, None);
    assert_eq!((ok.info, ok.fallback), (3, None));
    // Plain CG reaches max_iter = nSmat; DiagScale converges from x = 0.
    let default = check_edited("L16", "fb-default", &[fallback], 64, None);
    assert_eq!(default.info, -101, "plain status at max_iter = 100");
    assert!(default.fallback.is_some_and(|s| s > 0), "{default:?}");
    // DiagScale fails at max_iter 4; plain converges in 3 (deterministic).
    let diag = check_edited(
        "L16",
        "fb-diag",
        &[fallback, "NSRCG 2", "NSROptCGMaxIter 4", "DSROptCGTol 1e-4"],
        64,
        None,
    );
    assert_eq!((diag.info, diag.fallback), (-5, Some(3)));
    // Both variants fail at max_iter 6: the first status is reported.
    let both = check_edited(
        "L16",
        "fb-both",
        &[fallback, "NSROptCGMaxIter 6", "DSROptCGTol 2e-5"],
        64,
        None,
    );
    assert_eq!((both.info, both.fallback), (-7, Some(-7)));
}

/// RescaleSmat (stcopt_cg_impl.c:772-875) with both variants, on inputs whose
/// Slater block starts after RBM or DH parameters or ends before OptTrans.
#[test]
#[ignore = "requires CUDA; RescaleSmat against the CPU C-order StochasticOptCG"]
fn resident_cg_rescale_smat_matches_cpu_c_order() {
    let rescale = "RescaleSmat 1";
    for name in ["L16", "dh_L16", "rbm_L16", "opttrans_L16"] {
        let plain = check_edited(name, "rescale", &[rescale], 64, None);
        assert!(plain.rescaled, "{name}: rescaled");
        let diag = check_edited(name, "rescale-diag", &[rescale, "NSRCG 2"], 64, None);
        assert!(diag.rescaled, "{name}: rescaled with DiagScale");
    }
    check_edited(
        "L128",
        "rescale",
        &[rescale, "NSROptCGMaxIter 40"],
        320,
        None,
    );
}

/// Replace the observed derivative columns `columns` of every walker by zero.
fn zero_columns(
    columns: std::ops::Range<usize>,
) -> impl Fn(&RawExec<'_, '_>, &State) -> tenferro_tensor::Result<()> {
    move |exec, state| {
        let np = state.model.np;
        let w = state.resident.sampler.geom.walkers;
        let b = &state.buffers;
        let mut o = download::<f64>(exec, &b.derivatives, w * np)?;
        for s in 0..w {
            o[s * np + columns.start..s * np + columns.end].fill(0.0);
        }
        // SAFETY: the same w*np FP64 derivatives just downloaded.
        unsafe { exec.upload(exec.ptr(&b.derivatives)?, &o) }
    }
}

/// C's data-dependent RescaleSmat branches: max |S_ii| below 1e-10 over the
/// projections returns without rescaling; below 1e-10 over the Slater block
/// it is clamped to 1e-10 (stcopt_cg_impl.c:841-846).
#[test]
#[ignore = "requires CUDA; RescaleSmat early return and Slater clamp"]
fn resident_cg_rescale_smat_follows_c_early_returns() {
    let rescale = "RescaleSmat 1";
    let probe = Model::prepare(&cg_input("L16", "probe", &[]), &config(8, 16)).unwrap();
    let (nproj, np) = (probe.nproj, probe.np);
    let none = check_edited(
        "L16",
        "rescale-proj0",
        &[rescale],
        64,
        Some(&zero_columns(0..nproj)),
    );
    assert!(!none.rescaled, "zero projection variance: no rescale");
    let clamp = check_edited(
        "L16",
        "rescale-slater0",
        &[rescale],
        64,
        Some(&zero_columns(nproj..np)),
    );
    assert!(clamp.rescaled, "zero Slater variance: clamped rescale");
}

/// A NaN gradient makes C 1.4.0 return -(0+1) before the loop
/// (`stcopt_cg_impl.c:526`); with the default `NSRCGAbortOnFail = 1` C aborts
/// (`:252-262`). The device latches RESIDENT_CG_FAILED and publishes nothing.
/// With `NSRCGFallback` the other variant fails the same way (`:397`) and C
/// aborts after it; with `NSRCGAbortOnFail 0` it continues with x = 0.
#[test]
#[ignore = "requires CUDA; a failed SR-CG solve must abort without publishing parameters"]
fn resident_cg_failed_solve_aborts_by_default() {
    for (tag, extra, abort) in [
        ("nan", &[][..], true),
        ("nan-fb", &["NSRCGFallback 1"][..], true),
        ("nan-fb-diag", &["NSRCGFallback 1", "NSRCG 2"][..], true),
        (
            "nan-fb-continue",
            &["NSRCGFallback 1", "NSRCGAbortOnFail 0"][..],
            false,
        ),
    ] {
        let mut extra = extra.to_vec();
        extra.push("NSROptCGMaxIter 5");
        let input = cg_input("L16", tag, &extra);
        let config = config(8, 16);
        let model = Model::prepare(&input, &config).unwrap();
        let fallback = model.data.modpara.nsrcg_fallback;
        let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
        let backend: &mut CudaEngine = &mut backend;
        let mut state = State::prepare(backend, model, &config).unwrap();
        state.initialize(backend).unwrap();
        let np = state.model.np;
        let (before, after, ist, srinfo) = with_raw_exec(backend, "cg.nan", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            state.observe(exec, &module)?;
            let before = download::<f64>(exec, &state.buffers.parameters, np)?;
            // A NaN local energy gives a NaN gradient and a nonfinite initial delta.
            // SAFETY: walker 0's local energy.
            unsafe {
                exec.upload(exec.ptr(&state.buffers.local_energy)?, &[f64::NAN])?;
            }
            state.update(exec, &module, 0)?;
            let cg = state.buffers.cg.as_ref().unwrap();
            Ok((
                before,
                download::<f64>(exec, &state.buffers.parameters, np)?,
                download::<i32>(exec, &cg.ist, 8)?,
                [
                    download::<i32>(exec, &cg.iterations, 1)?[0],
                    download::<i32>(exec, &cg.fallback, 1)?[0],
                ],
            ))
        })
        .unwrap();
        assert_eq!((ist[1], ist[4]), (0, -1), "{tag}: C 1.4.0 status -(0+1)");
        let expected = if fallback { -1 } else { i32::MIN };
        assert_eq!(srinfo, [-1, expected], "{tag}: info and fallback status");
        if abort {
            close("unpublished NaN update", &after, &before, 0.0, 0.0);
            let error = state.result(backend).unwrap_err();
            assert!(error.contains("device status 8"), "{tag}: {error}");
        } else {
            // x = 0 is finite: C continues and the update adds zeros; the
            // gauge then renormalizes as after any step.
            let result = state.result(backend).unwrap();
            assert_eq!(result.cg_iterations, [-1]);
            assert_eq!(result.cg_fallback, [Some(-1)]);
        }
        std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
}

/// Complete 20-step CG Opt runs repeat exactly within one implementation,
/// after an intervening differently seeded run, with zero explicit transfers
/// and one stream synchronization for the whole series.
#[test]
#[ignore = "requires CUDA; 20-step SR-CG repeatability, transfers and timing"]
fn resident_cg_opt_repeats_twenty_steps() {
    for (name, walkers, extra) in [
        ("L16", 64, &[][..]),
        ("L128", 64, &[][..]),
        (
            "L16",
            64,
            &["NSRCG 2", "NSRCGFallback 1", "RescaleSmat 1"][..],
        ),
        (
            "L128",
            64,
            &["NSRCG 2", "NSRCGFallback 1", "RescaleSmat 1"][..],
        ),
        ("L16", 64, &["NSRCGFallback 1", "RescaleSmat 1"][..]),
    ] {
        // Some of the 20 solves reach max_iter = nSmat: C 1.4.0 aborts there by
        // default, so the series continues them like the 1.3.0 reference.
        let mut extra = extra.to_vec();
        extra.push("NSRCGAbortOnFail 0");
        let label = format!("{name} {extra:?}");
        let input = cg_input(name, "repeat", &extra);
        let config = OptConfig {
            nsteps: 20,
            walkers: Some(walkers),
            warmup: Some(128),
            interval: Some(128),
            seed: Some(79),
            initial_def: InitialDef::None,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let first = run_gpu_para_opt_from_namelist(&input, config.clone()).unwrap();
        let first_seconds = start.elapsed().as_secs_f64();
        run_gpu_para_opt_from_namelist(
            &input,
            OptConfig {
                nsteps: 1,
                walkers: Some(8),
                warmup: Some(0),
                interval: Some(16),
                seed: Some(23),
                ..config.clone()
            },
        )
        .unwrap();
        let start = std::time::Instant::now();
        let second = run_gpu_para_opt_from_namelist(&input, config.clone()).unwrap();
        let second_seconds = start.elapsed().as_secs_f64();
        // Same implementation, inputs and seed: bitwise repeatability is the
        // contract, so the bounds are only formatting slack.
        close("energies", &first.energies, &second.energies, 1e-12, 1e-12);
        close(
            "parameters",
            &first.parameters,
            &second.parameters,
            1e-12,
            1e-12,
        );
        assert_eq!(first.cg_iterations, second.cg_iterations);
        assert_eq!(first.cg_fallback, second.cg_fallback);
        assert_eq!(first.cg_iterations.len(), 20);
        assert_eq!(first.cg_fallback.len(), 20);
        // C 1.4.0 status: the count, or -(max_iter+1) for a continued limit.
        assert!(first.cg_iterations.iter().all(|&k| k != 0 && k != -1));
        let fallback = extra.contains(&"NSRCGFallback 1");
        for (&info, status) in first.cg_iterations.iter().zip(&first.cg_fallback) {
            assert_eq!(
                status.is_some(),
                fallback && info < 0,
                "{label}: fallback runs"
            );
        }
        let (traffic, run_seconds) = with_runtime(0, |backend| {
            let model = Model::prepare(&input, &config)?;
            let mut state = State::prepare(backend, model, &config)?;
            state.initialize(backend)?;
            let before = raw_exec_counters();
            let start = std::time::Instant::now();
            state.run(backend)?;
            let seconds = start.elapsed().as_secs_f64();
            let traffic = raw_exec_counters() - before;
            let result = state.result(backend)?;
            close(
                "resident run",
                &result.energies,
                &first.energies,
                1e-12,
                1e-12,
            );
            Ok((traffic, seconds))
        })
        .unwrap();
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.download_calls,
                traffic.synchronizations
            ),
            (0, 0, 1),
            "{label}: CG optimization must stay resident"
        );
        // Direct SR on the unedited input, same workload (warm), for the timing report.
        let direct_input = source(name).join("namelist.def");
        let direct = run_gpu_para_opt_from_namelist(&direct_input, config.clone()).unwrap();
        assert!(direct.cg_iterations.is_empty());
        let start = std::time::Instant::now();
        run_gpu_para_opt_from_namelist(&direct_input, config.clone()).unwrap();
        let direct_seconds = start.elapsed().as_secs_f64();
        println!(
            "{label} CG: walkers={walkers} e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s (direct SR warm={direct_seconds:.3}s) traffic={traffic:?} \
             iterations={:?} fallback={:?} energies[0,19]=({:.6},{:.6}) direct=({:.6},{:.6})",
            first.cg_iterations,
            first.cg_fallback,
            first.energies[0],
            first.energies[19],
            direct.energies[0],
            direct.energies[19]
        );
        std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
}
