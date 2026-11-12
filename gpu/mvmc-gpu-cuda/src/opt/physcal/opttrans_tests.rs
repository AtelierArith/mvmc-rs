//! Resident PhysCal with OptTrans weights against the Rust CPU C-order path (#674).
//!
//! Per-walker Hamiltonian and Green operator values of GPU-sampled
//! configurations are compared with mvmc-core's `calculate_local_energy`,
//! `green_func1` and `green_func2` over every `NQPFix * NQPOptTrans` plane
//! with C's full weights (`QPFullWeight = OptTrans * QPFixWeight`). The
//! supplied-tape trajectory uses PhysCal's immediate inverse updates.
use super::*;
use crate::opt::dh_tests::{arrays, tape, Cpu};
use crate::opt::opttrans_tests::{identity_only, input, replay};
use mvmc_core::observables::{green_func1, green_func2};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

fn download<T: Copy + Default>(
    backend: &mut CudaEngine,
    buffer: &DeviceBuffer,
    len: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); len];
    with_raw_exec(backend, "opttrans.physcal.download", |exec| {
        // SAFETY: every caller passes a length within the buffer's allocation.
        unsafe { exec.download(&mut out, exec.ptr(buffer)?) }
    })
    .map_err(|e| e.to_string())
    .unwrap();
    out
}

/// Per-walker operator values `[walker][term]` after one observation.
fn values(backend: &mut CudaEngine, state: &State, measurement: &Measurement) -> Vec<f64> {
    with_raw_exec(backend, "opttrans.physcal.observe", |exec| {
        measurement.observe(exec, state, 0)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let error: Vec<i32> = download(backend, &state.resident.error, 1);
    assert_eq!(error, [0], "observation latched a failure");
    let w = state.resident.sampler.geom.walkers;
    download(
        backend,
        &measurement.values,
        w * measurement.operators.terms(),
    )
}

/// CPU operator values of one walker: energy, OneBodyG, TwoBodyG.
fn cpu_rows(cpu: &mut Cpu, model: &Model, walker: &[i32]) -> (Vec<f64>, f64) {
    let e = cpu.evaluate(walker);
    let (idx, ele_cfg, num) = arrays(walker, model.ns, model.ne);
    let z = Complex64::new(e.ip, 0.);
    let mut rows = vec![e.energy];
    let data = &model.data;
    for t in &data.green_one_terms {
        rows.push(
            green_func1(
                t.site1 as usize,
                t.site2 as usize,
                t.spin1.as_code(),
                t.spin2.as_code(),
                z,
                &cpu.data,
                &cpu.state,
                &idx,
                &ele_cfg,
                &num,
                &e.counts,
            )
            .re,
        );
    }
    for t in &data.green_two_terms {
        rows.push(
            green_func2(
                t.site1 as usize,
                t.site2 as usize,
                t.site3 as usize,
                t.site4 as usize,
                t.spin1.as_code(),
                t.spin3.as_code(),
                z,
                &cpu.data,
                &cpu.state,
                &idx,
                &ele_cfg,
                &num,
                &e.counts,
            )
            .re,
        );
    }
    (rows, e.kappa)
}

fn assert_values(
    name: &str,
    stage: &str,
    backend: &mut CudaEngine,
    state: &State,
    measurement: &Measurement,
    cpu: &mut Cpu,
    plain: &mut Cpu,
) {
    let m = &state.model;
    let g = state.resident.sampler.geom;
    let cfg: Vec<i32> = download(backend, &state.resident.sampler.ele, g.walkers * g.n);
    let aw: Vec<i32> = download(backend, &state.resident.sampler.aw, g.walkers);
    let values = values(backend, state, measurement);
    let ops = &measurement.operators;
    let terms = ops.terms();
    let (mut worst, mut normalized, mut effect, mut bound_max) = (0f64, 0f64, 0f64, 0f64);
    for w in 0..g.walkers {
        let walker = &cfg[w * g.n..(w + 1) * g.n];
        let (expected, kappa) = cpu_rows(cpu, m, walker);
        let (identity, _) = cpu_rows(plain, m, walker);
        let v = &values[w * terms..(w + 1) * terms];
        let mut actual = vec![(0..ops.n_energy).map(|t| m.energies[t] * v[t]).sum::<f64>()];
        let mut scale = vec![(0..ops.n_energy)
            .map(|t| (m.energies[t] * v[t]).abs())
            .sum::<f64>()];
        for k in 0..ops.n_one + ops.n_two {
            actual.push(v[ops.n_energy + k]);
            scale.push(expected[1 + k].abs());
        }
        // The Opt stage bound (#656 interval bound, see opt::opttrans_tests).
        let budget = 16.0 * (aw[w] + 1) as f64 * g.n as f64 * f64::EPSILON * kappa;
        for (row, ((&a, &b), &s)) in actual.iter().zip(&expected).zip(&scale).enumerate() {
            let d = (a - b).abs();
            let bound = 1e-13 + budget * s;
            bound_max = bound_max.max(bound);
            assert!(
                d <= bound,
                "{name} {stage} walker {w} row {row}: {a} vs {b}"
            );
            worst = worst.max(d);
            if s > 0. {
                normalized = normalized.max(d / (budget * s / 16.0));
            }
        }
        for (a, b) in expected.iter().zip(&identity) {
            effect = effect.max((a - b).abs());
        }
    }
    assert!(
        effect > 1e2 * bound_max,
        "{name} {stage}: the OptTrans planes must change the operator values ({effect:e})"
    );
    println!(
        "{name} {stage}: max abs diff={worst:e}; OptTrans effect={effect:.3e}; normalized by \
         (aw+1) n eps kappa_F scale={normalized:.2e} (bound 16)"
    );
}

fn assert_physcal(name: &str) {
    let walkers = 8;
    with_runtime(0, |backend| {
        let probe = Model::prepare_physcal(&input(name), &OptConfig::default(), None)?;
        let steps = 8 * probe.ns.min(16);
        let sampling = OptConfig {
            walkers: Some(walkers),
            nsteps: 1,
            warmup: Some(0),
            interval: Some(steps),
            seed: Some(79),
            initial_def: InitialDef::None,
            ..Default::default()
        };
        let model = Model::prepare_physcal(&input(name), &sampling, None)?;
        assert!(model.nopt > 1);
        let mut cpu = Cpu::new(&model, &model.parameters);
        let mut plain = identity_only(&model, &model.parameters);
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state, false)?;
        let initial = state.resident.snapshot(backend)?.ele;
        assert_values(
            name,
            "initial",
            backend,
            &state,
            &measurement,
            &mut cpu,
            &mut plain,
        );
        let tape = tape(walkers, steps);
        state.resident.advance_with_tape(backend, &tape)?;
        let last = state.resident.snapshot(backend)?.ele;
        let accepted = state.resident.accepted(backend)?;
        let (cfg, host_accepted, margin, flips) = replay(
            &mut cpu,
            &mut plain,
            &state.model,
            &initial,
            &tape,
            walkers,
            steps,
        );
        assert_eq!(last, cfg, "{name}: supplied-tape configurations");
        assert_eq!(accepted, host_accepted, "{name}: acceptance counts");
        assert!(
            flips > 0,
            "{name}: the OptTrans weights changed no decision"
        );
        println!(
            "{name}: PhysCal accepted {accepted:?}, {flips} decisions changed by OptTrans, \
             smallest decision margin {margin:e}"
        );
        assert_values(
            name,
            "final",
            backend,
            &state,
            &measurement,
            &mut cpu,
            &mut plain,
        );
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; OptTrans L16 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_opttrans_l16_matches_cpu_c_order() {
    assert_physcal("hubbard_opttrans_L16");
}

#[test]
#[ignore = "requires CUDA; OptTrans L128 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_opttrans_l128_matches_cpu_c_order() {
    assert_physcal("hubbard_opttrans_L128");
}

#[test]
#[ignore = "requires CUDA; 20-window OptTrans PhysCal repeatability and residency"]
fn resident_physcal_opttrans_repeats_twenty_windows() {
    for name in ["hubbard_opttrans_L16", "hubbard_opttrans_L128"] {
        let config = PhysCalConfig {
            walkers: Some(64),
            nsamples: Some(20),
            warmup: Some(128),
            interval: Some(128),
            seed: Some(79),
            initial_def: InitialDef::None,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let first = run_gpu_phys_cal_from_namelist(input(name), config.clone()).unwrap();
        let first_seconds = start.elapsed().as_secs_f64();
        run_gpu_phys_cal_from_namelist(
            input(name),
            PhysCalConfig {
                seed: Some(23),
                nsamples: Some(1),
                ..config.clone()
            },
        )
        .unwrap();
        let start = std::time::Instant::now();
        let second = run_gpu_phys_cal_from_namelist(input(name), config.clone()).unwrap();
        let second_seconds = start.elapsed().as_secs_f64();
        let all = |r: &PhysCalResult| -> Vec<f64> {
            std::iter::once(r.energy)
                .chain(r.green_one.iter().copied())
                .chain(r.green_two.iter().copied())
                .collect()
        };
        for (a, b) in all(&first).iter().zip(all(&second)) {
            // Same implementation, input and seed: repeatability is exact.
            assert!(a.is_finite() && (a - b).abs() <= 1e-12 * a.abs().max(1.0));
        }
        let sampling = OptConfig {
            walkers: config.walkers,
            nsteps: 20,
            warmup: config.warmup,
            interval: config.interval,
            seed: config.seed,
            initial_def: InitialDef::None,
            ..Default::default()
        };
        let (traffic, seconds) = with_runtime(0, |backend| {
            let model = Model::prepare_physcal(&input(name), &sampling, None)?;
            let operators = Operators::prepare(&model)?;
            let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
            state.resident.delayed_updates = false;
            state.initialize(backend)?;
            let mut measurement = Measurement::prepare(backend, operators, &state, false)?;
            let before = raw_exec_counters();
            let start = std::time::Instant::now();
            measurement.run(backend, &mut state)?;
            let seconds = start.elapsed().as_secs_f64();
            let traffic = raw_exec_counters() - before;
            let result = measurement.result(backend, &state)?;
            assert!((result.energy - first.energy).abs() <= 1e-12 * first.energy.abs());
            Ok((traffic, seconds))
        })
        .unwrap();
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.download_calls,
                traffic.synchronizations
            ),
            (0, 0, 1)
        );
        println!(
            "{name}: PhysCal e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={seconds:.3}s traffic={traffic:?} energy={:.9}",
            first.energy
        );
    }
}
