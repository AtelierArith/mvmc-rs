//! Resident SR-CG (`NSRCG != 0`) against the Rust CPU C-order CG (#670, #572).
//!
//! Julia GPU Opt rejects CG SR, so the reference is C `StochasticOptCG_real`
//! (`stcopt_cg_impl.c`) as ported by `mvmc_core::sr_cg::SampledSrOperator`,
//! applied to the derivatives and local energies the GPU produced. Integers
//! (active set, iteration count) are exact; FP64 bounds are stated at each
//! assertion and derived in `docs/design/gpu-resident-opt.md` (SR-CG section).
#![allow(clippy::needless_range_loop)] // index-parallel host references
use super::dh_tests::worst;
use super::tests::close;
use super::*;
use mvmc_core::sr_cg::{install_cg_observer, CgIterationView, CgObserver, SampledSrOperator};
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
        other => panic!("unknown input {other}"),
    }
}

/// Copy an input with `NSRCG 1` and extra modpara.def lines (C keywords).
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
    let mut lines: Vec<String> = text
        .lines()
        .filter(|l| {
            let key = l.split_whitespace().next().unwrap_or("");
            key != "NSRCG"
                && !extra
                    .iter()
                    .any(|e| e.split_whitespace().next() == Some(key))
        })
        .map(str::to_string)
        .collect();
    lines.push("NSRCG          1".into());
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
    derivatives: Vec<f64>,
    energy: Vec<f64>,
    mean: Vec<f64>,
    variance: Vec<f64>,
    ho: Vec<f64>,
    x: Vec<f64>,
    gradient: Vec<f64>,
    active: Vec<i32>,
    /// `[stop, iterations, max iterations, nSmat]`.
    ist: Vec<i32>,
    trace: Vec<f64>,
    info: Vec<i32>,
    seconds: f64,
}

/// Observe the current walkers and run one device CG solve (no update).
fn solve_once(backend: &mut CudaEngine, state: &State) -> Solve {
    let m = &state.model;
    let (np, w) = (m.np, state.resident.sampler.geom.walkers);
    with_raw_exec(backend, "cg.solve", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        exec.synchronize()?;
        let start = std::time::Instant::now();
        state.solve_cg(exec, &module, 0)?;
        exec.synchronize()?;
        let seconds = start.elapsed().as_secs_f64();
        let b = &state.buffers;
        let cg = b.cg.as_ref().unwrap();
        Ok(Solve {
            derivatives: download(exec, &b.derivatives, w * np)?,
            energy: download(exec, &b.local_energy, w)?,
            mean: download(exec, &b.mean, np + 1)?,
            variance: download(exec, &b.variance, np)?,
            ho: download(exec, &b.ho, np)?,
            x: download(exec, &b.rhs, np)?,
            gradient: download(exec, &cg.vec, np)?,
            active: download(exec, &cg.active, np)?,
            ist: download(exec, &cg.ist, 4)?,
            trace: download(exec, &cg.trace, cg.limit + 1)?,
            info: download(exec, &b.info, 2)?,
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

/// Check one device solve against the CPU C-order CG on the same samples.
/// Returns the iteration count.
fn assert_solve(name: &str, model: &Model, g: &Solve) -> usize {
    let mp = &model.data.modpara;
    let np = model.np;
    let w = g.energy.len();
    let wc = w as f64;
    let o = |s: usize, p: usize| g.derivatives[s * np + p];

    // C real-path moments (vmccal.c with w=1, weightAverageSROpt): samples
    // summed in order, then divided by Wc. The device performs the same IEEE
    // operations in the same order without FMA, so a few ulps of headroom only.
    let mut mean = vec![0.; np + 1];
    let mut oo = vec![0.; np];
    let mut ho = vec![0.; np];
    for s in 0..w {
        for p in 0..np {
            mean[p] += o(s, p);
            oo[p] += o(s, p) * o(s, p);
            ho[p] += g.energy[s] * o(s, p);
        }
        mean[np] += g.energy[s];
    }
    mean.iter_mut().for_each(|v| *v /= wc);
    oo.iter_mut().for_each(|v| *v /= wc);
    ho.iter_mut().for_each(|v| *v /= wc);
    let variance: Vec<f64> = (0..np).map(|p| oo[p] - mean[p] * mean[p]).collect();
    for (label, a, b) in [
        ("mean", &g.mean, &mean),
        ("variance", &g.variance, &variance),
        ("ho", &g.ho, &ho),
    ] {
        let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
        let (ratio, diff) = worst(a, b, 4.0 * EPS * scale, 4.0 * EPS);
        assert!(
            ratio <= 1.0,
            "{name} {label}: ratio {ratio:e} diff {diff:e}"
        );
    }

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
    let (ratio, _) = worst(&device_gradient, &gradient, 8.0 * EPS * gscale, 8.0 * EPS);
    assert!(ratio <= 1.0, "{name}: gradient ratio {ratio:e}");
    for p in 0..np {
        if g.active[p] == 0 {
            assert_eq!(g.x[p], 0.0, "{name}: inactive parameter {p} changed");
        }
    }

    // CPU C-order CG on the same samples and prepared vectors, plus a second
    // CPU run with the samples in reverse order: the same operator with a
    // different product summation order, measuring CG's own rounding sensitivity.
    let max_iter = if mp.nsr_opt_cg_max_iter > 0 {
        mp.nsr_opt_cg_max_iter as usize
    } else {
        n
    };
    let cpu_solve = |reverse: bool| {
        let mut operator = SampledSrOperator::new(n, w, false);
        for (si, &p) in mapping.iter().enumerate() {
            operator.mean[si] = mean[p];
            operator.diagonal[si] = variance[p];
            for s in 0..w {
                let t = if reverse { w - 1 - s } else { s };
                operator.real_samples[t * n + si] = o(s, p);
            }
        }
        let trace = Rc::new(Trace::default());
        let guard = install_cg_observer(trace.clone()).unwrap();
        let start = std::time::Instant::now();
        let result = operator.solve(
            &gradient,
            1.0 / wc,
            mp.dsr_opt_sta_del,
            mp.dsr_opt_cg_tol,
            max_iter,
        );
        let seconds = start.elapsed().as_secs_f64();
        drop(guard);
        let delta = trace.delta.borrow().clone();
        (operator, result, delta, seconds)
    };
    let (mut operator, cpu, cpu_delta, cpu_seconds) = cpu_solve(false);
    let (_, reordered, reordered_delta, _) = cpu_solve(true);
    let k = cpu.iterations;
    let gpu_k = g.ist[1] as usize;
    let thresh = mp.dsr_opt_cg_tol * mp.dsr_opt_cg_tol * n as f64 * n as f64;

    // Conditioning: S = Cov + DSROptStaDel diag(S_ii) with Cov positive
    // semidefinite, so lambda_min >= DSROptStaDel * min S_ii (rigorous).
    // lambda_max is a power-iteration estimate.
    let lambda_min = mp.dsr_opt_sta_del
        * operator
            .diagonal
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut z = vec![0.; n];
    let mut lambda_max = 0.;
    for _ in 0..60 {
        operator.apply(&mut z, &v, 1.0 / wc, mp.dsr_opt_sta_del);
        lambda_max = norm(&z);
        v.iter_mut().zip(&z).for_each(|(a, b)| *a = b / lambda_max);
    }
    let kappa = 2.0 * lambda_max / lambda_min;

    // C stopping rule on the device's own values (stcopt_cg_impl.c:495-497):
    // every completed iteration started with delta >= threshold (or NaN), and
    // the loop ended by max_iter or by delta < threshold.
    assert!(gpu_k <= max_iter, "{name}: iterations beyond max_iter");
    assert!(
        g.trace[..gpu_k].iter().all(|&d| d >= thresh || d.is_nan()),
        "{name}: device iterated past C's stop test"
    );
    assert!(
        gpu_k == max_iter || g.trace[gpu_k] < thresh,
        "{name}: device stopped before C's stop test ({gpu_k} of {max_iter})"
    );

    // Deterministic regime. Both runs apply C's control flow and differ only in
    // the reduction order of the sampled products and dots, a relative
    // perturbation <= (W + nSmat) eps per operation. While CG is stable, the
    // divergence of two such runs grows at most like k kappa times that (the
    // finite-precision CG perturbation bounds of Greenbaum 1989). Once the
    // residual has fallen far below ||g||, CG amplifies rounding by orders of
    // magnitude per iteration and different summation orders give different
    // (equally valid) iterates. That regime starts where the CPU's own
    // reordered run reaches 1/64 of the budget (the growth there is ~10x per
    // iteration, so the device's differently ordered trees may reach the
    // budget one or two iterations earlier); iterates are compared before it.
    let budget = |i: usize| 8.0 * (i.max(1) as f64) * kappa * (w + n) as f64 * EPS;
    let rel = |a: f64, b: f64| ((a - b) / b.abs().max(f64::MIN_POSITIVE)).abs();
    let last = k.min(reordered.iterations);
    let chaotic = (0..=last)
        .find(|&i| rel(reordered_delta[i], cpu_delta[i]) > budget(i) / 64.0)
        .unwrap_or(last + 1);
    for i in 0..chaotic.min(gpu_k + 1) {
        let r = rel(g.trace[i], cpu_delta[i]);
        assert!(
            r <= budget(i),
            "{name}: delta at iteration {i} differs by {r:e} > {:e} (deterministic until {chaotic})",
            budget(i)
        );
    }
    let gpu_divergence = (0..=gpu_k.min(k))
        .find(|&i| rel(g.trace[i], cpu_delta[i]) > budget(i))
        .unwrap_or(gpu_k.min(k) + 1);
    // Same iteration count as C whenever the stop (by max_iter, or by the
    // tolerance with a margin beyond rounding) falls in the deterministic
    // regime. Later, each run's count follows from its own trace, checked above.
    let stop_margin = rel(cpu_delta[k.min(cpu_delta.len() - 1)], thresh);
    let count_determined = k < chaotic && (k == max_iter || stop_margin > budget(k));
    if count_determined {
        assert_eq!(gpu_k, k, "{name}: C CG iteration count");
    }

    // A posteriori solution bound, independent of the trajectory: for the true
    // residuals r = g - S x (evaluated on the host), ||x - x*|| <= ||r|| /
    // lambda_min, so ||x_gpu - x_cpu|| <= (||r_gpu|| + ||r_cpu||) / lambda_min.
    let gpu_x: Vec<f64> = mapping.iter().map(|&p| g.x[p]).collect();
    let true_residual = |operator: &mut SampledSrOperator, x: &[f64]| {
        let mut sx = vec![0.; n];
        operator.apply(&mut sx, x, 1.0 / wc, mp.dsr_opt_sta_del);
        let r: Vec<f64> = gradient.iter().zip(&sx).map(|(a, b)| a - b).collect();
        norm(&r)
    };
    let gpu_residual = true_residual(&mut operator, &gpu_x);
    let cpu_residual = true_residual(&mut operator, &cpu.solution);
    let diff: Vec<f64> = gpu_x
        .iter()
        .zip(&cpu.solution)
        .map(|(a, b)| a - b)
        .collect();
    let x_diff = norm(&diff);
    let x_norm = norm(&cpu.solution).max(f64::MIN_POSITIVE);
    // Host residual evaluation error: (W + nSmat) eps (lambda_max ||x|| + ||g||).
    let evaluation = (w + n) as f64 * EPS * (2.0 * lambda_max * x_norm + norm(&gradient));
    let posterior = (gpu_residual + cpu_residual + 2.0 * evaluation) / lambda_min;
    assert!(
        x_diff <= posterior,
        "{name}: ||x_gpu - x_cpu|| = {x_diff:e} > {posterior:e}"
    );

    // Recurrence vs true residual of the device solution: their gap (the
    // residual drift) is <= k (W+nSmat) eps (||S|| ||x|| + ||g||); the
    // 20-iteration refresh only reduces it.
    let drift = 8.0
        * (gpu_k.max(1) as f64)
        * (w + n) as f64
        * EPS
        * (2.0 * lambda_max * norm(&gpu_x) + norm(&gradient));
    let recurrence = g.trace[gpu_k].abs().sqrt();
    assert!(
        gpu_residual <= recurrence + drift,
        "{name}: true residual {gpu_residual:e} > {recurrence:e} + {drift:e}"
    );
    assert_eq!(g.info, [0, 0], "{name}: info");
    println!(
        "{name}: W={w} np={np} nSmat={n} iterations GPU/CPU/CPU-reordered={gpu_k}/{k}/{} \
         (max {max_iter}, count determined: {count_determined}), \
         stop margin={stop_margin:.2e}, cut margin={margin:.2e}; kappa<={kappa:.2e}; \
         delta agrees to {:.1e} (budget {:.1e}) through iteration {}; CPU reordered run \
         reaches budget/64 at {chaotic}, GPU exceeds the budget at {gpu_divergence}; \
         x rel diff={:.2e} (a posteriori bound {:.2e}); true residual GPU/CPU={gpu_residual:.2e}/{cpu_residual:.2e} \
         (recurrence {recurrence:.2e}, drift bound {drift:.2e}); device solve={:.2}ms, CPU solve={:.2}ms",
        reordered.iterations,
        (0..chaotic.min(gpu_k + 1))
            .map(|i| rel(g.trace[i], cpu_delta[i]))
            .fold(0f64, f64::max),
        budget(chaotic.min(gpu_k + 1).saturating_sub(1)),
        chaotic.min(gpu_k + 1).saturating_sub(1),
        x_diff / x_norm,
        posterior / x_norm,
        g.seconds * 1e3,
        cpu_seconds * 1e3
    );
    gpu_k
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

/// Sample, then solve on the device and on the CPU. Returns the C iteration count.
fn check(name: &str, tag: &str, extra: &[&str], walkers: usize) -> usize {
    let input = cg_input(name, tag, extra);
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
    let solve = solve_once(backend, &state);
    state.resident.check_status(backend).unwrap();
    let k = assert_solve(&format!("{name}/{tag}"), &state.model, &solve);
    std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    k
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

#[test]
#[ignore = "requires CUDA; SR-CG with DH2/DH4 and RBM factors against the CPU C-order CG"]
fn resident_cg_dh_rbm_match_cpu_c_order() {
    check("dh_L16", "default", &[], 64);
    check("rbm_L16", "default", &[], 64);
}

#[test]
#[ignore = "requires CUDA; SR-CG on the L128 chain against the CPU C-order CG"]
fn resident_cg_l128_matches_cpu_c_order() {
    check("L128", "default", &[], 64);
    check("L128", "max40", &["NSROptCGMaxIter 40"], 320);
}

/// A nonfinite increment is C `StochasticOptCG` info=1 (`stcopt_cg_impl.c:315-327`):
/// C stops the optimization (`vmcmain.c:503-506`); the device latches
/// RESIDENT_CG and publishes nothing.
#[test]
#[ignore = "requires CUDA; nonfinite SR-CG increment must never publish parameters"]
fn resident_cg_rejects_nonfinite_increment() {
    let input = cg_input("L16", "nan", &["NSROptCGMaxIter 5"]);
    let config = config(8, 16);
    let model = Model::prepare(&input, &config).unwrap();
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let np = state.model.np;
    let (before, after, iterations) = with_raw_exec(backend, "cg.nan", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        let before = download::<f64>(exec, &state.buffers.parameters, np)?;
        // A NaN local energy gives a NaN gradient: C's `delta < cg_thresh` is
        // false for NaN, so CG runs to max_iter and returns a NaN increment.
        // SAFETY: walker 0's local energy.
        unsafe {
            exec.upload(exec.ptr(&state.buffers.local_energy)?, &[f64::NAN])?;
        }
        state.update(exec, &module, 0)?;
        let cg = state.buffers.cg.as_ref().unwrap();
        Ok((
            before,
            download::<f64>(exec, &state.buffers.parameters, np)?,
            download::<i32>(exec, &cg.ist, 4)?,
        ))
    })
    .unwrap();
    assert_eq!(iterations[1], 5, "NaN delta never stops before max_iter");
    close("unpublished NaN update", &after, &before, 0.0, 0.0);
    let error = state.result(backend).unwrap_err();
    assert!(error.contains("device status 7"), "{error}");
    std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
}

/// Complete 20-step CG Opt runs repeat exactly within one implementation,
/// after an intervening differently seeded run, with zero explicit transfers
/// and one stream synchronization for the whole series.
#[test]
#[ignore = "requires CUDA; 20-step SR-CG repeatability, transfers and timing"]
fn resident_cg_opt_repeats_twenty_steps() {
    for (name, walkers) in [("L16", 64), ("L128", 64)] {
        let input = cg_input(name, "repeat", &[]);
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
        assert_eq!(first.cg_iterations.len(), 20);
        assert!(first.cg_iterations.iter().all(|&k| k > 0));
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
            "{name}: CG optimization must stay resident"
        );
        // Direct SR on the unedited input, same workload (warm), for the timing report.
        let direct_input = source(name).join("namelist.def");
        let direct = run_gpu_para_opt_from_namelist(&direct_input, config.clone()).unwrap();
        assert!(direct.cg_iterations.is_empty());
        let start = std::time::Instant::now();
        run_gpu_para_opt_from_namelist(&direct_input, config.clone()).unwrap();
        let direct_seconds = start.elapsed().as_secs_f64();
        println!(
            "{name} CG: walkers={walkers} e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s (direct SR warm={direct_seconds:.3}s) traffic={traffic:?} \
             iterations={:?} energies[0,19]=({:.6},{:.6}) direct=({:.6},{:.6})",
            first.cg_iterations,
            first.energies[0],
            first.energies[19],
            direct.energies[0],
            direct.energies[19]
        );
        std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
}
