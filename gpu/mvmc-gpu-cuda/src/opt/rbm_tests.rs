//! RBM correlation factors against the Rust CPU C-order path (#667, #572).
//!
//! Julia GPU has no RBM support, so the references are mvmc-core's C-order
//! RBM kernels (C `rbm.c`) evaluated on the configurations the GPU sampled:
//! `make_rbm_cnt` (MakeRBMCnt), `update_rbm_cnt_hopping` (UpdateRBMCnt),
//! `log_rbm_ratio` (LogRBMRatio), `set_rbm_diff` (RBMDiff) and the RBM-aware
//! `calculate_local_energy`. The supplied-tape trajectory is replayed on the
//! host with the GPU tape contract, C UpdateRBMCnt/LogRBMRatio and full CPU
//! Pfaffians. Integers (configurations, acceptance counts, counters) and the
//! hidden fields, which both sides form with the same IEEE operations in the
//! same order, are compared exactly; FP64 bounds are stated at each assertion.
//!
//! The physics is C's complex normal path (`vmcmake.c`, `calham.c`,
//! `vmccal.c`), which keeps the RBM factor in the Metropolis ratio and the
//! local energy. C's real path omits both (see `resident_rbm.cuh`).
use super::dh_tests::{arrays, tape, worst, Cpu};
use super::tests::{close, sr_backward_error};
use super::*;
use mvmc_core::sampling::projection::{log_proj_ratio, make_proj_cnt, update_proj_cnt};
use mvmc_core::sampling::rbm::{log_rbm_ratio, make_rbm_cnt, update_rbm_cnt_hopping, RbmConfig};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

pub(crate) fn input(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rbm")
        .join(name)
        .join("namelist.def")
}

/// Number of physical-layer counters (C NRBM_PhysLayerIdx).
pub(crate) fn nphys(model: &Model) -> usize {
    model.data.rbm_section_sizes()[..3].iter().sum()
}

/// Host replay of the resident tape contract (`interval_hop_impl`), as
/// `dh_tests::replay`, with C UpdateRBMCnt on the proposal (an exchange is C's
/// two hops, `vmcmake.c:250-260`) and C LogRBMRatio added to LogProjRatio.
pub(crate) struct Replay {
    pub cfg: Vec<i32>,
    pub accepted: Vec<i32>,
    /// Smallest |log u - log ratio| over all decisions.
    pub margin: f64,
    /// Hidden fields after the tape, `[walker][neuron]`.
    pub theta: Vec<f64>,
    /// Decisions that the RBM factor changed.
    pub flips: usize,
}

pub(crate) fn replay(
    cpu: &mut Cpu,
    model: &Model,
    initial: &[i32],
    tape: &[u32],
    walkers: usize,
    steps: usize,
) -> Replay {
    let (ns, ne) = (model.ns, model.ne);
    let n = 2 * ne;
    let np = nphys(model);
    let mut cfg = initial.to_vec();
    let mut accepted = vec![0i32; walkers];
    let mut theta = vec![];
    let mut margin = f64::INFINITY;
    let mut flips = 0;
    let scale = 4294967296.0;
    let data = cpu.data.clone();
    let config = RbmConfig::from(&data);
    for w in 0..walkers {
        let e = &mut cfg[w * n..(w + 1) * n];
        let (_, _, mut num) = arrays(e, ns, ne);
        let mut counts = cpu.counts(&num);
        let mut rbm = make_rbm_cnt(&num, &config);
        let ip = cpu.factor(&e.iter().map(|&r| r as i64).collect::<Vec<_>>());
        let mut log_old = (ip.abs() + 1e-100).ln();
        for step in 0..steps {
            let t = &tape[5 * (w + walkers * step)..][..5];
            let mi = (t[0] % ne as u32) as usize;
            let spin = (t[1] >> 31) as usize;
            let ri = e[mi + spin * ne] as usize;
            let exchange = (t[3] as f64) / scale < model.exchange;
            let (rj, partner, valid) = if exchange {
                let partner = (t[2] % ne as u32) as usize + (1 - spin) * ne;
                let rj = e[partner] as usize;
                let valid = ri != rj && num[rj + spin * ns] == 0 && num[ri + (1 - spin) * ns] == 0;
                (rj, partner, valid)
            } else {
                let empty: Vec<usize> = (0..ns)
                    .filter(|&r| num[r + spin * ns] == 0 && model.localspins[r] == 0)
                    .collect();
                let valid = !empty.is_empty() && model.localspins[ri] == 0;
                let rj = if empty.is_empty() {
                    0
                } else {
                    empty[(t[2] % empty.len() as u32) as usize]
                };
                (rj, usize::MAX, valid)
            };
            if !valid {
                continue;
            }
            let previous = (e.to_vec(), num.clone());
            e[mi + spin * ne] = rj as i32;
            num[ri + spin * ns] = 0;
            num[rj + spin * ns] = 1;
            let mut proposed = vec![0i64; counts.len()];
            let mut rbm_new = vec![Complex64::new(0., 0.); rbm.len()];
            update_rbm_cnt_hopping(
                &mut rbm_new,
                &rbm,
                ri as i64,
                rj as i64,
                spin as u8,
                &config,
            );
            if exchange {
                e[partner] = ri as i32;
                num[rj + (1 - spin) * ns] = 0;
                num[ri + (1 - spin) * ns] = 1;
                make_proj_cnt(&mut proposed, &num, &cpu.data);
                let mid = rbm_new.clone();
                update_rbm_cnt_hopping(
                    &mut rbm_new,
                    &mid,
                    rj as i64,
                    ri as i64,
                    (1 - spin) as u8,
                    &config,
                );
            } else {
                update_proj_cnt(
                    ri as i64,
                    rj as i64,
                    spin as u8,
                    &mut proposed,
                    &counts,
                    &num,
                    &cpu.data,
                );
            }
            // Physical-layer counters are integers: the incremental update
            // equals a full recount exactly.
            let fresh = make_rbm_cnt(&num, &config);
            for (a, b) in rbm_new[..np].iter().zip(&fresh[..np]) {
                assert_eq!(a, b, "UpdateRBMCnt physical counters");
            }
            let proj = if exchange {
                0.0
            } else {
                log_proj_ratio(&proposed, &counts, &cpu.data)
            };
            let x = log_rbm_ratio(&rbm_new, &rbm, &config);
            assert!(x.im == 0.0, "real RBM coefficients give a real ratio");
            let ip_new = cpu.factor(&e.iter().map(|&r| r as i64).collect::<Vec<_>>());
            let log_new = (ip_new.abs() + 1e-100).ln();
            let ratio = 2.0 * ((proj + x.re) + log_new - log_old);
            let without = 2.0 * (proj + log_new - log_old);
            let u = (t[4] as f64 / scale).ln();
            if u.is_finite() {
                margin = margin.min((u - ratio).abs());
            }
            let ok = ip_new.is_finite() && ip_new != 0.0;
            flips += usize::from(ok && (u < ratio) != (u < without));
            if ok && u < ratio {
                accepted[w] += 1;
                log_old = log_new;
                counts = proposed;
                rbm = rbm_new;
            } else {
                e.copy_from_slice(&previous.0);
                num = previous.1;
            }
        }
        theta.extend(rbm[np..].iter().map(|z| z.re));
    }
    Replay {
        cfg,
        accepted,
        margin,
        theta,
        flips,
    }
}

pub(crate) fn download<T: Copy + Default>(
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

/// Resident hidden fields `[walker][neuron]`.
pub(crate) fn device_theta(backend: &mut CudaEngine, state: &State) -> Vec<f64> {
    let r = state.resident.rbm.as_ref().expect("RBM input");
    with_raw_exec(backend, "rbm.theta", |exec| {
        download(
            exec,
            &r.theta,
            state.resident.sampler.geom.walkers * r.nneuron,
        )
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Hidden fields equal bit for bit: the same IEEE operations in the same order.
pub(crate) fn assert_theta_exact(what: &str, gpu: &[f64], cpu: &[f64]) {
    assert_eq!(gpu.len(), cpu.len());
    for (k, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: hidden field {k}: {a} vs {b}"
        );
    }
}

/// Contributions to each RBM derivative slot (C RBMDiff), from the tables.
fn rbm_multiplicity(model: &Model) -> Vec<usize> {
    let (ns, h) = (model.ns, model.nneuron);
    let t = &model.rbm;
    let mut count = vec![0usize; model.nrbm];
    for &p in &t[..4 * ns] {
        if p >= 0 {
            count[p as usize] += 1;
        }
    }
    for k in 0..h {
        let b = t[4 * ns + h + k];
        if b >= 0 {
            count[b as usize] += 1;
        }
        let width = if t[4 * ns + k] == 2 { 2 * ns } else { ns };
        for rs in 0..width {
            let c = t[4 * ns + 2 * h + k * 2 * ns + rs];
            if c >= 0 {
                count[c as usize] += 1;
            }
        }
    }
    count
}

/// Per-walker RBM error budget of an operator log-ratio, relative to the
/// ratio: the fields differ from a fresh MakeRBMCnt by at most `dtheta`
/// (incremental C UpdateRBMCnt, measured), |d log cosh / d theta| <= 1, and
/// each neuron adds a few roundings of exp/log/division near max(1, |theta|).
pub(crate) fn rbm_ratio_budget(nneuron: usize, dtheta: f64, theta_max: f64) -> f64 {
    nneuron as f64 * (2.0 * dtheta + 16.0 * f64::EPSILON * (1.0 + theta_max))
}

struct Measured {
    counts: Vec<i32>,
    energy: Vec<f64>,
    derivatives: Vec<f64>,
    overlaps: Vec<f64>,
    terms: Vec<f64>,
    aw: Vec<i32>,
}

fn measure(backend: &mut CudaEngine, state: &State) -> Measured {
    with_raw_exec(backend, "rbm.measure", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        let w = state.resident.sampler.geom.walkers;
        let m = &state.model;
        Ok(Measured {
            counts: download(exec, &state.buffers.counts, w * m.nproj)?,
            energy: download(exec, &state.buffers.local_energy, w)?,
            derivatives: download(exec, &state.buffers.derivatives, w * m.np)?,
            overlaps: download(exec, &state.buffers.overlaps, w)?,
            terms: download(exec, &state.buffers.values, w * m.energies.len())?,
            aw: download(exec, &state.resident.sampler.aw, w)?,
        })
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Compare one stage with the CPU C-order evaluation of the same walkers.
/// `expected_theta` is the exact C-order hidden-field state (a fresh
/// MakeRBMCnt initially, the replayed UpdateRBMCnt chain after the tape).
fn assert_stage(
    name: &str,
    stage: &str,
    cpu: &mut Cpu,
    plain: &mut Cpu,
    state: &State,
    backend: &mut CudaEngine,
    expected_theta: Option<&[f64]>,
) {
    let cfg = with_raw_exec(backend, "rbm.cfg", |exec| {
        let g = state.resident.sampler.geom;
        download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let theta = device_theta(backend, state);
    let g = measure(backend, state);
    let m = &state.model;
    let n = 2 * m.ne;
    let h = m.nneuron;
    let walkers = cfg.len() / n;
    let multiplicity = rbm_multiplicity(m);
    let mut cpu_overlaps = vec![];
    let (mut energy_error, mut derivative_error, mut rbm_error) = (0f64, 0f64, 0f64);
    let (mut energy_abs, mut rbm_abs, mut dtheta_max, mut kappa_max) = (0f64, 0f64, 0f64, 0f64);
    let mut rbm_effect = 0f64;
    let mut bound_max = 0f64;
    let mut fresh_theta = vec![];
    for w in 0..walkers {
        let walker = &cfg[w * n..(w + 1) * n];
        let gpu_theta = &theta[w * h..(w + 1) * h];
        let e = cpu.evaluate_with(walker, Some(gpu_theta));
        fresh_theta.extend_from_slice(&e.theta);
        let gpu: Vec<i64> = g.counts[w * m.nproj..(w + 1) * m.nproj]
            .iter()
            .map(|&c| c as i64)
            .collect();
        assert_eq!(
            gpu, e.counts,
            "{name} {stage} walker {w}: projection counters"
        );
        let row = &g.derivatives[w * m.np..(w + 1) * m.np];
        for (&a, &c) in row.iter().zip(&e.counts).take(m.nproj) {
            assert_eq!(
                a, c as f64,
                "{name} {stage} walker {w}: projection derivative"
            );
        }
        // Distance of the incremental fields from a fresh MakeRBMCnt.
        let dtheta = gpu_theta
            .iter()
            .zip(&e.theta)
            .fold(0f64, |a, (x, y)| a.max((x - y).abs()));
        let theta_max = e.theta.iter().fold(0f64, |a, x| a.max(x.abs()));
        dtheta_max = dtheta_max.max(dtheta);
        // RBM derivatives (C RBMDiff real part) of the GPU fields. Physical
        // counters are integers and match exactly. A hidden or coupling slot
        // sums m terms x tanh(theta), |x tanh| <= 1: each tanh is within
        // 3 eps (CUDA 2 ulp, host 1 ulp) and the sum order adds <= m(m-1) eps.
        let rbm = m.nproj..m.nproj + m.nrbm;
        let np_phys = nphys(m);
        for (k, p) in rbm.enumerate() {
            let (a, b) = (row[p], e.derivatives[p]);
            if k < np_phys {
                assert_eq!(a, b, "{name} {stage} walker {w}: RBM physical counter {k}");
                continue;
            }
            let c = multiplicity[k] as f64;
            let bound = (c * c + 3.0 * c) * f64::EPSILON;
            let d = (a - b).abs();
            assert!(
                d <= bound,
                "{name} {stage} walker {w} RBM derivative {k}: {a} vs {b} (bound {bound:e})"
            );
            rbm_abs = rbm_abs.max(d);
            if d > 0.0 {
                rbm_error = rbm_error.max(d / bound);
            }
        }
        cpu_overlaps.push(e.ip);
        // #656 interval bound (dh_tests::assert_stage) plus the RBM log-ratio
        // budget: every Hamiltonian ratio carries both relative errors.
        let budget = 16.0 * (g.aw[w] + 1) as f64 * n as f64 * f64::EPSILON * e.kappa;
        let rbm_budget = rbm_ratio_budget(h, dtheta, theta_max);
        kappa_max = kappa_max.max(e.kappa);
        let terms = m.energies.len();
        let scale: f64 = (0..terms)
            .map(|t| (m.energies[t] * g.terms[w * terms + t]).abs())
            .sum();
        let d = (g.energy[w] - e.energy).abs();
        let bound = 1e-13 + (budget + rbm_budget) * scale;
        bound_max = bound_max.max(bound);
        assert!(
            d <= bound,
            "{name} {stage} walker {w}: energy {} vs {} (kappa {:e})",
            g.energy[w],
            e.energy,
            e.kappa
        );
        energy_abs = energy_abs.max(d);
        energy_error = energy_error.max(d / (budget * scale / 16.0));
        // The reference is RBM-aware: zeroing the RBM coefficients moves it.
        let without = plain.evaluate(walker);
        rbm_effect = rbm_effect.max((e.energy - without.energy).abs());
        let largest = e.derivatives[m.nproj + m.nrbm..]
            .iter()
            .fold(0f64, |a, x| a.max(x.abs()));
        let slater = m.nproj + m.nrbm;
        for (p, (&a, &b)) in row.iter().zip(&e.derivatives).enumerate().skip(slater) {
            let d = (a - b).abs();
            assert!(
                d <= 1e-13 + budget * largest,
                "{name} {stage} walker {w} Slater derivative {p}: {a} vs {b}"
            );
            derivative_error = derivative_error.max(d / (budget * largest / 16.0));
        }
    }
    match expected_theta {
        Some(expected) => assert_theta_exact(&format!("{name} {stage}"), &theta, expected),
        None => assert_theta_exact(&format!("{name} {stage}"), &theta, &fresh_theta),
    }
    assert!(
        rbm_effect > 1e2 * bound_max,
        "{name} {stage}: the RBM factor must change the local energy ({rbm_effect:e})"
    );
    let (ratio, diff) = worst(&g.overlaps, &cpu_overlaps, 1e-12, 5e-11);
    assert!(ratio <= 1.0, "{name} {stage} overlaps: ratio {ratio:e}");
    println!(
        "{name} {stage}: max kappa_F={kappa_max:.3e}; |theta - MakeRBMCnt|<={dtheta_max:e}; \
         RBM energy effect={rbm_effect:.3e}; max abs diff overlap={diff:e} energy={energy_abs:e} \
         RBM derivative={rbm_abs:e} (normalized {rbm_error:.3}, bound 1); normalized by \
         (aw+1) n eps kappa_F scale: energy={energy_error:.3} Slater derivative={derivative_error:.3} \
         (bound 16)"
    );
}

fn stage_config(steps: usize) -> OptConfig {
    OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(steps),
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

/// CPU evaluator with the RBM coefficients set to zero (factor exactly 1).
pub(crate) fn without_rbm(model: &Model) -> Cpu {
    let mut parameters = model.parameters.clone();
    parameters[model.nproj..model.nproj + model.nrbm].fill(0.0);
    Cpu::new(model, &parameters)
}

fn assert_opt_stages(name: &str, steps_per_site: usize) {
    let probe = Model::prepare(&input(name), &stage_config(1)).unwrap();
    let steps = steps_per_site * probe.ns;
    let config = stage_config(steps);
    let model = Model::prepare(&input(name), &config).unwrap();
    assert!(model.nrbm > 0 && model.nneuron > 0);
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut plain = without_rbm(&model);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "initial", &mut cpu, &mut plain, &state, backend, None);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let host = replay(&mut cpu, &state.model, &initial, &tape, walkers, steps);
    assert_eq!(last.ele, host.cfg, "{name}: supplied-tape configurations");
    assert_eq!(accepted, host.accepted, "{name}: acceptance counts");
    assert!(accepted.iter().any(|&a| a > 0));
    assert!(host.flips > 0, "{name}: the RBM factor changed no decision");
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, \
         {} decisions changed by RBM, smallest decision margin {:e}",
        host.flips, host.margin
    );
    assert_stage(
        name,
        "final",
        &mut cpu,
        &mut plain,
        &state,
        backend,
        Some(&host.theta),
    );
    // One SR update plus the gauge, against the C-order sync; then the
    // rebuilt hidden fields against MakeRBMCnt with the new coefficients.
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "rbm.update", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            let m = &state.model;
            let w = state.resident.sampler.geom.walkers;
            state.observe(exec, &module)?;
            let before = download::<f64>(exec, &state.buffers.parameters, m.np)?;
            let energy = download::<f64>(exec, &state.buffers.local_energy, w)?;
            let derivatives = download::<f64>(exec, &state.buffers.derivatives, w * m.np)?;
            state.update(exec, &module, 0)?;
            Ok((
                before,
                download(exec, &state.buffers.rhs, m.np)?,
                download(exec, &state.buffers.parameters, m.np)?,
                energy,
                derivatives,
            ))
        })
        .unwrap();
    state.resident.check_status(backend).unwrap();
    let relative = sr_backward_error(&state.model, &derivatives, &energy, &update);
    assert!(relative < 1e-11, "{name}: SR backward error {relative:e}");
    let m = &state.model;
    assert!(
        update[m.nproj..m.nproj + m.nrbm].iter().any(|&x| x != 0.0),
        "{name}: SR moved no RBM coefficient"
    );
    let pre: Vec<Complex64> = before
        .iter()
        .zip(&update)
        .map(|(p, u)| Complex64::new(p + u, 0.))
        .collect();
    let mut data = m.data.clone();
    mvmc_core::unpack_parameters(&mut data, &pre).unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    let expected: Vec<f64> = mvmc_core::pack_parameters(&data)
        .unwrap()
        .iter()
        .map(|p| p.re)
        .collect();
    // Same IEEE operations in the same order as the C-order sync (no FMA);
    // C SyncModifiedParameter leaves RBM unchanged and rescales Slater only.
    let (ratio, diff) = worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    let theta = device_theta(backend, &state);
    let cfg = state.resident.snapshot(backend).unwrap().ele;
    let config = RbmConfig::from(&data);
    let n = 2 * m.ne;
    let rebuilt: Vec<f64> = (0..walkers)
        .flat_map(|w| {
            let (_, _, num) = arrays(&cfg[w * n..(w + 1) * n], m.ns, m.ne);
            make_rbm_cnt(&num, &config)[nphys(m)..]
                .iter()
                .map(|z| z.re)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_theta_exact(&format!("{name} after SR"), &theta, &rebuilt);
    println!("{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}");
}

#[test]
#[ignore = "requires CUDA; RBM L16 doped chain with exchanges against the CPU C-order path"]
fn resident_opt_rbm_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_rbm_L16", 8);
}

#[test]
#[ignore = "requires CUDA; RBM L128 chain (delayed queue, paired inverse) against the CPU C-order path"]
fn resident_opt_rbm_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_rbm_L128", 1);
}

/// Complete 20-step Opt runs repeat exactly within one implementation, after
/// an intervening differently seeded run, with zero explicit transfers and one
/// stream synchronization for the whole warmup/optimization series.
#[test]
#[ignore = "requires CUDA; 20-step RBM repeatability, transfers and timing"]
fn resident_opt_rbm_repeats_twenty_steps() {
    for (name, walkers) in [("hubbard_rbm_L16", 64), ("hubbard_rbm_L128", 64)] {
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
        let first = run_gpu_para_opt_from_namelist(input(name), config.clone()).unwrap();
        let first_seconds = start.elapsed().as_secs_f64();
        run_gpu_para_opt_from_namelist(
            input(name),
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
        let second = run_gpu_para_opt_from_namelist(input(name), config.clone()).unwrap();
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
        let (traffic, run_seconds) = with_runtime(0, |backend| {
            let model = Model::prepare(&input(name), &config)?;
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
            "{name}: sampling/optimization must stay resident"
        );
        println!(
            "{name}: walkers={walkers} e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s traffic={traffic:?} energies[0,19]=({:.6},{:.6})",
            first.energies[0], first.energies[19]
        );
    }
}

/// #701: the minimal hidden-layer-only Charge/Spin/General inputs, whose family
/// tags are not below the coefficient count, set up on CUDA for resident Opt
/// and PhysCal. Each hidden field is its bias coefficient (C MakeRBMCnt with
/// no coupling section: `0 + w[h]`, one exact addition), checked against the
/// CPU MakeRBMCnt; both complete runners then finish with finite results.
#[test]
#[ignore = "requires CUDA; minimal Charge/Spin/General RBM tables for Opt and PhysCal"]
fn resident_rbm_minimal_family_tables_run_on_cuda() {
    use super::physcal::{run_gpu_phys_cal_from_namelist, PhysCalConfig};
    use super::scope_tests::minimal_rbm_inputs;
    for (label, input, nrbm, nneuron, _) in minimal_rbm_inputs("rbm-cuda") {
        let config = OptConfig {
            nsteps: 2,
            walkers: Some(8),
            warmup: Some(4),
            interval: Some(12),
            seed: Some(79),
            initial_def: InitialDef::None,
            ..Default::default()
        };
        for physcal in [false, true] {
            let model = if physcal {
                Model::prepare_physcal(&input, &config, None).unwrap()
            } else {
                Model::prepare(&input, &config).unwrap()
            };
            assert_eq!((model.nrbm, model.nneuron), (nrbm, nneuron), "{label}");
            let cfg = with_runtime(0, |backend| {
                let mut state = State::prepare_with_derivatives(backend, model, &config, !physcal)?;
                state.initialize(backend)?;
                let theta = device_theta(backend, &state);
                let m = &state.model;
                let ele = state.resident.snapshot(backend)?.ele;
                let rbm = RbmConfig::from(&m.data);
                let n = 2 * m.ne;
                let expected: Vec<f64> = (0..8)
                    .flat_map(|w| {
                        let (_, _, num) = arrays(&ele[w * n..(w + 1) * n], m.ns, m.ne);
                        make_rbm_cnt(&num, &rbm)[nphys(m)..]
                            .iter()
                            .map(|z| z.re)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok((theta, expected))
            })
            .unwrap();
            // One exact IEEE addition on both sides: a 1-ulp relative bound.
            let (ratio, diff) = worst(&cfg.0, &cfg.1, 0.0, f64::EPSILON);
            assert!(
                ratio <= 1.0,
                "{label} physcal={physcal}: theta diff {diff:e}"
            );
        }
        let opt = run_gpu_para_opt_from_namelist(&input, config.clone()).unwrap();
        assert_eq!(opt.energies.len(), 2, "{label}");
        assert!(opt.energies.iter().all(|e| e.is_finite()), "{label}");
        assert!(opt.parameters.iter().all(|p| p.is_finite()), "{label}");
        let phys = run_gpu_phys_cal_from_namelist(
            &input,
            PhysCalConfig {
                walkers: Some(8),
                nsamples: Some(2),
                warmup: Some(4),
                interval: Some(12),
                seed: Some(79),
                initial_def: InitialDef::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(phys.energy.is_finite(), "{label}");
        println!(
            "{label}: Opt energies {:?}, PhysCal energy {}",
            opt.energies, phys.energy
        );
        std::fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
}
