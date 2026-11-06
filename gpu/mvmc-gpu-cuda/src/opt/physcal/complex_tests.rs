//! Resident PhysCal with complex normal orbitals against the Rust CPU C-order
//! complex path (#678, #572 milestone 6).
//!
//! Per-walker Hamiltonian and Green operator values of GPU-sampled
//! configurations are compared with mvmc-core's complex
//! `calculate_local_energy`, `green_func1` and `green_func2` (C calham.c,
//! locgrn.c, which return `conj(<x|O|psi>/<x|psi>)`) and C calgrn.c's
//! factored TwoBodyGEx product `G1[idx0] * conj(G1[idx1])`. The supplied-tape
//! trajectory is the complex Opt replay; complex PhysCal and Opt both use
//! immediate C rank-two updates.
use super::*;
use crate::opt::complex_tests::{budget, complex, input, replay, Cpu};
use crate::opt::dh_tests::{arrays, tape};
use mvmc_core::observables::{green_func1, green_func2};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

fn download<T: Copy + Default>(
    backend: &mut CudaEngine,
    buffer: &DeviceBuffer,
    len: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); len];
    with_raw_exec(backend, "complex.physcal.download", |exec| {
        // SAFETY: every caller passes a length within the buffer's allocation.
        unsafe { exec.download(&mut out, exec.ptr(buffer)?) }
    })
    .map_err(|e| e.to_string())
    .unwrap();
    out
}

/// CPU rows of one walker: energy, OneBodyG, TwoBodyG, TwoBodyGEx, plus the
/// magnitude scale of each row and the Frobenius condition number.
fn cpu_rows(cpu: &mut Cpu, model: &Model, walker: &[i32]) -> (Vec<Complex64>, f64) {
    let e = cpu.evaluate(walker);
    let (idx, ele_cfg, num) = arrays(walker, model.ns, model.ne);
    let data = &model.data;
    let one = |cpu: &Cpu, i: i64, j: i64, s: u8| {
        green_func1(
            i as usize, j as usize, s, s, e.ip, &cpu.data, &cpu.state, &idx, &ele_cfg, &num,
            &e.counts,
        )
    };
    let mut rows = vec![e.energy];
    for t in &data.green_one_terms {
        rows.push(one(cpu, t.site1, t.site2, t.spin1.as_code()));
    }
    for t in &data.green_two_terms {
        rows.push(green_func2(
            t.site1 as usize,
            t.site2 as usize,
            t.site3 as usize,
            t.site4 as usize,
            t.spin1.as_code(),
            t.spin3.as_code(),
            e.ip,
            &cpu.data,
            &cpu.state,
            &idx,
            &ele_cfg,
            &num,
            &e.counts,
        ));
    }
    for t in &data.green_two_ex_terms {
        let a = one(cpu, t.site1, t.site2, t.spin1.as_code());
        let b = one(cpu, t.site3, t.site4, t.spin3.as_code());
        rows.push(a * b.conj());
    }
    (rows, e.kappa)
}

/// Device rows of one walker from its operator values, with the magnitude
/// scale of each row (sum |c_t v_t| for the energy; |a||b| for GEx).
fn gpu_rows(model: &Model, ops: &Operators, v: &[Complex64]) -> (Vec<Complex64>, Vec<f64>) {
    let coefficients = complex(&model.energies);
    let mut rows = vec![(0..ops.n_energy)
        .map(|t| coefficients[t] * v[t])
        .sum::<Complex64>()];
    let mut scale = vec![(0..ops.n_energy)
        .map(|t| (coefficients[t] * v[t]).norm())
        .sum::<f64>()];
    for k in 0..ops.n_one + ops.n_two {
        rows.push(v[ops.n_energy + k]);
        scale.push(v[ops.n_energy + k].norm());
    }
    for k in 0..ops.n_ex {
        let first = ops.n_energy + ops.n_one + ops.n_two + 2 * k;
        rows.push(v[first] * v[first + 1].conj());
        // Each factor carries the budget, so the product carries twice it.
        scale.push(2.0 * v[first].norm() * v[first + 1].norm());
    }
    (rows, scale)
}

/// CPU C-order complex PhysCal rows of one walker and its condition number:
/// the normal complex path ([`Cpu`]) or complex FSZ (`fsz_complex_tests`).
pub(crate) trait ComplexRows {
    fn rows(&mut self, model: &Model, walker: &[i32]) -> (Vec<Complex64>, f64);
}

impl ComplexRows for Cpu {
    fn rows(&mut self, model: &Model, walker: &[i32]) -> (Vec<Complex64>, f64) {
        cpu_rows(self, model, walker)
    }
}

pub(crate) fn assert_values(
    name: &str,
    stage: &str,
    backend: &mut CudaEngine,
    state: &State,
    measurement: &Measurement,
    cpu: &mut impl ComplexRows,
) {
    let m = &state.model;
    let g = state.resident.sampler.geom;
    let ops = &measurement.operators;
    let cfg: Vec<i32> = download(backend, &state.resident.sampler.ele, g.walkers * g.n);
    let aw: Vec<i32> = download(backend, &state.resident.sampler.aw, g.walkers);
    let rows = ops.rows();
    let before = complex(&download::<f64>(backend, &measurement.means, 2 * rows));
    with_raw_exec(backend, "complex.physcal.observe", |exec| {
        measurement.observe(exec, state)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let error: Vec<i32> = download(backend, &state.resident.error, 1);
    assert_eq!(error, [0], "observation latched a failure");
    let terms = ops.terms();
    let values = complex(&download::<f64>(
        backend,
        &measurement.values,
        2 * g.walkers * terms,
    ));
    let after = complex(&download::<f64>(backend, &measurement.means, 2 * rows));
    let (mut worst, mut normalized, mut imag) = (0f64, 0f64, 0f64);
    let mut window = vec![Complex64::new(0., 0.); rows];
    let mut window_scale = vec![0f64; rows];
    for w in 0..g.walkers {
        let walker = &cfg[w * g.n..(w + 1) * g.n];
        let (expected, kappa) = cpu.rows(m, walker);
        let (actual, scale) = gpu_rows(m, ops, &values[w * terms..(w + 1) * terms]);
        // The Opt stage bound (#656 interval bound, see opt::complex_tests).
        let budget = budget(aw[w], g.n, kappa);
        for (row, ((&a, &b), &s)) in actual.iter().zip(&expected).zip(&scale).enumerate() {
            let d = (a - b).norm();
            assert!(
                d <= 1e-13 + budget * s,
                "{name} {stage} walker {w} row {row}: {a} vs {b}"
            );
            worst = worst.max(d);
            if s > 0. {
                normalized = normalized.max(d / (budget * s / 16.0));
            }
            if row > 0 {
                imag = imag.max(b.im.abs());
            }
            window[row] += a;
            window_scale[row] += a.norm();
        }
    }
    // physcal_reduce_c: the window mean of the same device rows, summed in a
    // warp tree instead of the host's order (<= W eps of the magnitude sum).
    for row in 0..rows {
        let mean = window[row] / g.walkers as f64;
        let d = (after[row] - before[row] - mean).norm();
        let bound = 1e-15 + 4.0 * g.walkers as f64 * f64::EPSILON * window_scale[row];
        assert!(
            d <= bound,
            "{name} {stage}: reduced row {row} {} vs {mean}",
            after[row] - before[row]
        );
    }
    assert!(
        imag > 1e-3,
        "{name} {stage}: Green functions must be complex ({imag:e})"
    );
    println!(
        "{name} {stage}: max abs diff={worst:e}; max |Im G|={imag:.3e}; normalized by \
         (aw+1) n eps kappa_F scale={normalized:.3} (bound 16)"
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
        assert!(model.complex && !model.data.green_two_ex_terms.is_empty());
        let mut cpu = Cpu::new(&model, &model.parameters);
        let mut plain = Cpu::real_weights(&model, &model.parameters);
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state)?;
        let initial = state.resident.snapshot(backend)?.ele;
        assert_values(name, "initial", backend, &state, &measurement, &mut cpu);
        let tape = tape(walkers, steps);
        state.resident.advance_with_tape(backend, &tape)?;
        let last = state.resident.snapshot(backend)?.ele;
        let accepted = state.resident.accepted(backend)?;
        let host = replay(
            &mut cpu,
            &mut plain,
            &state.model,
            &initial,
            &tape,
            walkers,
            steps,
        );
        assert_eq!(last, host.cfg, "{name}: supplied-tape configurations");
        assert_eq!(accepted, host.accepted, "{name}: acceptance counts");
        assert!(
            host.flips > 0,
            "{name}: Im QPFullWeight changed no decision"
        );
        println!(
            "{name}: PhysCal accepted {accepted:?}, {} decisions changed by Im QPFullWeight, \
             smallest decision margin {:e}",
            host.flips, host.margin
        );
        assert_values(name, "final", backend, &state, &measurement, &mut cpu);
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; complex L16 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_complex_l16_matches_cpu_c_order() {
    assert_physcal("hubbard_complex_L16");
}

#[test]
#[ignore = "requires CUDA; complex L128 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_complex_l128_matches_cpu_c_order() {
    assert_physcal("hubbard_complex_L128");
}

#[test]
#[ignore = "requires CUDA; 20-window complex PhysCal repeatability and residency"]
fn resident_physcal_complex_repeats_twenty_windows() {
    for name in ["hubbard_complex_L16", "hubbard_complex_L128"] {
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
            [r.energy, r.energy_imag]
                .into_iter()
                .chain(r.green_one.iter().copied())
                .chain(r.green_two.iter().copied())
                .chain(r.green_two_ex.iter().copied())
                .chain(r.green_one_imag.iter().copied())
                .chain(r.green_two_imag.iter().copied())
                .chain(r.green_two_ex_imag.iter().copied())
                .collect()
        };
        for (a, b) in all(&first).iter().zip(all(&second)) {
            // Same implementation, input and seed: repeatability is exact.
            assert!(a.is_finite() && (a - b).abs() <= 1e-12 * a.abs().max(1.0));
        }
        assert!(first.green_one_imag.iter().any(|&x| x != 0.0));
        assert_eq!(first.green_two_ex.len(), first.green_two_ex_imag.len());
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
            let mut measurement = Measurement::prepare(backend, operators, &state)?;
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
             run={seconds:.3}s traffic={traffic:?} energy={:.9}{:+.9}i",
            first.energy, first.energy_imag
        );
    }
}
