//! Resident PhysCal with general orbitals / FSZ against the Rust CPU C-order
//! FSZ path (#714, #572 milestone 7).
//!
//! Per-walker Hamiltonian and Green operator values of GPU-sampled
//! configurations are compared with mvmc-core's `calculate_local_energy_fsz`
//! (C CalculateHamiltonian_fsz_real) and the C FSZ Green kernels that C
//! PhysCal uses in both modes (calgrn_fsz.c: GreenFunc1_fsz/_fsz2,
//! GreenFunc2_fsz/_fsz2 on the complex shadow), including spin-flip pairs,
//! and C's factored TwoBodyGEx product `G1[idx0] * conj(G1[idx1])`. The
//! supplied-tape trajectory is the FSZ Opt replay; PhysCal uses immediate
//! rank-two updates, so the same moves are accepted.
use super::*;
use crate::opt::complex_tests::budget;
use crate::opt::dh_tests::tape;
use crate::opt::fsz_tests::{arrays, assert_initial, replay, Cpu};
use crate::opt::scope_tests::fsz_input as input;
use mvmc_core::observables::{green_func1_fsz_complex, green_func2_fsz_complex};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

fn download<T: Copy + Default>(
    backend: &mut CudaEngine,
    buffer: &DeviceBuffer,
    len: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); len];
    with_raw_exec(backend, "fsz.physcal.download", |exec| {
        // SAFETY: every caller passes a length within the buffer's allocation.
        unsafe { exec.download(&mut out, exec.ptr(buffer)?) }
    })
    .map_err(|e| e.to_string())
    .unwrap();
    out
}

/// CPU rows of one walker: energy, OneBodyG, TwoBodyG, TwoBodyGEx, and the
/// Frobenius condition number.
fn cpu_rows(cpu: &mut Cpu, model: &Model, walker: &[i32]) -> (Vec<f64>, f64) {
    let e = cpu.evaluate(walker);
    let (idx, spn, ele_cfg, num) = arrays(walker, model.ns);
    let ip = Complex64::new(e.ip, 0.);
    let data = &model.data;
    let code = |s: mvmc_expert_parsers::types::Spin| s.as_code();
    let one = |cpu: &Cpu, i: i64, s: u8, j: i64, t: u8| {
        let z = green_func1_fsz_complex(
            i as usize, j as usize, s, t, ip, &cpu.data, &cpu.state, &idx, &ele_cfg, &num,
            &e.counts, &spn,
        );
        z.re
    };
    let mut rows = vec![e.energy];
    for t in &data.green_one_terms {
        rows.push(one(cpu, t.site1, code(t.spin1), t.site2, code(t.spin2)));
    }
    for t in &data.green_two_terms {
        let z = green_func2_fsz_complex(
            t.site1 as usize,
            t.site2 as usize,
            t.site3 as usize,
            t.site4 as usize,
            code(t.spin1),
            code(t.spin2),
            code(t.spin3),
            code(t.spin4),
            ip,
            &cpu.data,
            &cpu.state,
            &idx,
            &ele_cfg,
            &num,
            &e.counts,
            &spn,
        );
        rows.push(z.re);
    }
    for t in &data.green_two_ex_terms {
        let a = one(cpu, t.site1, code(t.spin1), t.site2, code(t.spin2));
        let b = one(cpu, t.site3, code(t.spin3), t.site4, code(t.spin4));
        rows.push(a * b);
    }
    (rows, e.kappa)
}

/// Device rows of one walker from its operator values, with the magnitude
/// scale of each row (sum |c_t v_t| for the energy; 2 |a||b| for GEx).
fn gpu_rows(model: &Model, ops: &Operators, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let c = &model.energies;
    let mut rows = vec![(0..ops.n_energy).map(|t| c[t] * v[t]).sum::<f64>()];
    let mut scale = vec![(0..ops.n_energy).map(|t| (c[t] * v[t]).abs()).sum::<f64>()];
    for k in 0..ops.n_one + ops.n_two {
        rows.push(v[ops.n_energy + k]);
        scale.push(v[ops.n_energy + k].abs());
    }
    for k in 0..ops.n_ex {
        let first = ops.n_energy + ops.n_one + ops.n_two + 2 * k;
        rows.push(v[first] * v[first + 1]);
        scale.push(2.0 * (v[first] * v[first + 1]).abs());
    }
    (rows, scale)
}

fn assert_values(
    name: &str,
    stage: &str,
    backend: &mut CudaEngine,
    state: &State,
    measurement: &Measurement,
    cpu: &mut Cpu,
) {
    let m = &state.model;
    let g = state.resident.sampler.geom;
    let ops = &measurement.operators;
    let cfg: Vec<i32> = download(backend, &state.resident.sampler.ele, g.walkers * g.n);
    let aw: Vec<i32> = download(backend, &state.resident.sampler.aw, g.walkers);
    let rows = ops.rows();
    let before: Vec<f64> = download(backend, &measurement.means, rows);
    with_raw_exec(backend, "fsz.physcal.observe", |exec| {
        measurement.observe(exec, state)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let error: Vec<i32> = download(backend, &state.resident.error, 1);
    assert_eq!(error, [0], "observation latched a failure");
    let terms = ops.terms();
    let values: Vec<f64> = download(backend, &measurement.values, g.walkers * terms);
    let after: Vec<f64> = download(backend, &measurement.means, rows);
    let (mut worst, mut normalized) = (0f64, 0f64);
    let mut window = vec![0f64; rows];
    let mut window_scale = vec![0f64; rows];
    // Spin-flip Green rows (s != t in a pair) that are nonzero somewhere.
    let mut flips = 0usize;
    let flip_rows: Vec<bool> = std::iter::once(false)
        .chain(m.data.green_one_terms.iter().map(|t| t.spin1 != t.spin2))
        .chain(
            m.data
                .green_two_terms
                .iter()
                .map(|t| t.spin1 != t.spin2 || t.spin3 != t.spin4),
        )
        .chain(
            m.data
                .green_two_ex_terms
                .iter()
                .map(|t| t.spin1 != t.spin2 || t.spin3 != t.spin4),
        )
        .collect();
    for w in 0..g.walkers {
        let walker = &cfg[w * g.n..(w + 1) * g.n];
        let (expected, kappa) = cpu_rows(cpu, m, walker);
        let (actual, scale) = gpu_rows(m, ops, &values[w * terms..(w + 1) * terms]);
        // The Opt stage bound (#656 interval bound, see opt::fsz_tests).
        let budget = budget(aw[w], g.n, kappa);
        for (row, ((&a, &b), &s)) in actual.iter().zip(&expected).zip(&scale).enumerate() {
            let d = (a - b).abs();
            assert!(
                d <= 1e-13 + budget * s,
                "{name} {stage} walker {w} row {row}: {a} vs {b}"
            );
            worst = worst.max(d);
            if s > 0. {
                normalized = normalized.max(d / (budget * s / 16.0));
            }
            flips += usize::from(flip_rows[row] && b != 0.0);
            window[row] += a;
            window_scale[row] += a.abs();
        }
    }
    // physcal_reduce: the window mean of the same device rows, summed in a
    // warp tree instead of the host's order (<= W eps of the magnitude sum).
    for row in 0..rows {
        let mean = window[row] / g.walkers as f64;
        let d = (after[row] - before[row] - mean).abs();
        let bound = 1e-15 + 4.0 * g.walkers as f64 * f64::EPSILON * window_scale[row];
        assert!(
            d <= bound,
            "{name} {stage}: reduced row {row} {} vs {mean}",
            after[row] - before[row]
        );
    }
    if m.spin_free {
        assert!(
            flips > 0,
            "{name} {stage}: every spin-flip Green row vanished"
        );
    }
    println!(
        "{name} {stage}: max abs diff={worst:e}; nonzero spin-flip rows={flips}; normalized by \
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
        assert!(model.fsz);
        let mut cpu = Cpu::new(&model, &model.parameters);
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state)?;
        let initial = state.resident.snapshot(backend)?.ele;
        assert_initial(&state.model, &initial);
        assert_values(name, "initial", backend, &state, &measurement, &mut cpu);
        let tape = tape(walkers, steps);
        state.resident.advance_with_tape(backend, &tape)?;
        let last = state.resident.snapshot(backend)?.ele;
        let accepted = state.resident.accepted(backend)?;
        let host = replay(&mut cpu, &state.model, &initial, &tape, walkers, steps);
        assert_eq!(last, host.cfg, "{name}: supplied-tape configurations");
        assert_eq!(accepted, host.accepted, "{name}: acceptance counts");
        println!(
            "{name}: PhysCal accepted {accepted:?}, {} spin-changing moves ({} local spin \
             flips), smallest decision margin {:e}",
            host.spin_moves, host.local_flips, host.margin
        );
        assert_values(name, "final", backend, &state, &measurement, &mut cpu);
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; FSZ L16 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_fsz_l16_matches_cpu_c_order() {
    assert_physcal("hubbard_fsz_L16");
}

#[test]
#[ignore = "requires CUDA; FSZ Kondo 2Sz=2 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_fsz_kondo_2sz2_matches_cpu_c_order() {
    assert_physcal("kondo_fsz_2sz2");
}

#[test]
#[ignore = "requires CUDA; FSZ L128 PhysCal operator values against the CPU C-order path"]
fn resident_physcal_fsz_l128_matches_cpu_c_order() {
    assert_physcal("hubbard_fsz_L128");
}

#[test]
#[ignore = "requires CUDA; FSZ KondoGC PhysCal (NExUpdatePath=3) against the CPU C-order path"]
fn resident_physcal_fsz_kondo_kgc_matches_cpu_c_order() {
    assert_physcal("kondo_fsz_kgc");
}

#[test]
#[ignore = "requires CUDA; FSZ KondoGC 64+64 PhysCal against the CPU C-order path"]
fn resident_physcal_fsz_kondo_kgc_l128_matches_cpu_c_order() {
    assert_physcal("kondo_fsz_kgc_L128");
}

#[test]
#[ignore = "requires CUDA; FSZ Heisenberg PhysCal (local spin flips) against the CPU C-order path"]
fn resident_physcal_fsz_heisenberg_matches_cpu_c_order() {
    assert_physcal("heisenberg_fsz_L16");
}

#[test]
#[ignore = "requires CUDA; 20-window FSZ PhysCal repeatability and residency"]
fn resident_physcal_fsz_repeats_twenty_windows() {
    for name in [
        "hubbard_fsz_L16",
        "kondo_fsz_2sz2",
        "hubbard_fsz_L128",
        "kondo_fsz_kgc",
        "heisenberg_fsz_L16",
        "kondo_fsz_kgc_L128",
    ] {
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
            [r.energy]
                .into_iter()
                .chain(r.green_one.iter().copied())
                .chain(r.green_two.iter().copied())
                .chain(r.green_two_ex.iter().copied())
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
             run={seconds:.3}s traffic={traffic:?} energy={:.9}",
            first.energy
        );
    }
}
