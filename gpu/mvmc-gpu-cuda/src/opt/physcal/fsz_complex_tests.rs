//! Resident PhysCal with complex general orbitals / FSZ against the Rust CPU
//! C-order complex FSZ path (#714 (c), #572 milestone 7).
//!
//! Per-walker operator values are compared with `calculate_local_energy_fsz`
//! (C CalculateHamiltonian_fsz) and the C FSZ Green kernels
//! (`green_func1_fsz_complex`, `green_func2_fsz_complex`; locgrn_fsz.c returns
//! `conj(z/ip)`), and TwoBodyGEx with C calgrn_fsz.c's
//! `G1[idx0] conj(G1[idx1])`, through the complex PhysCal checks of
//! `complex_tests`.
use super::complex_tests::{assert_values, ComplexRows};
use super::*;
use crate::opt::complex_tests::ComplexReference;
use crate::opt::dh_tests::tape;
use crate::opt::fsz_complex_tests::ComplexFszCpu;
use crate::opt::fsz_tests::{assert_initial, replay};
use crate::opt::scope_tests::fsz_input as input;
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

impl ComplexRows for ComplexFszCpu {
    fn rows(&mut self, model: &Model, walker: &[i32]) -> (Vec<Complex64>, f64) {
        let e = self.evaluate(walker);
        let data = &model.data;
        let code = |s: mvmc_expert_parsers::types::Spin| s.as_code();
        let site = |r: i64| r as usize;
        let mut rows = vec![e.energy];
        let one = |cpu: &Self, i: i64, s, j: i64, t| {
            cpu.green(
                walker,
                e.ip,
                &e.counts,
                [site(i), site(j), 0, 0],
                [s, t, 0, 0],
                1,
            )
        };
        for t in &data.green_one_terms {
            rows.push(one(self, t.site1, code(t.spin1), t.site2, code(t.spin2)));
        }
        for t in &data.green_two_terms {
            rows.push(self.green(
                walker,
                e.ip,
                &e.counts,
                [site(t.site1), site(t.site2), site(t.site3), site(t.site4)],
                [code(t.spin1), code(t.spin2), code(t.spin3), code(t.spin4)],
                2,
            ));
        }
        for t in &data.green_two_ex_terms {
            let a = one(self, t.site1, code(t.spin1), t.site2, code(t.spin2));
            let b = one(self, t.site3, code(t.spin3), t.site4, code(t.spin4));
            rows.push(a * b.conj());
        }
        (rows, e.kappa)
    }
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
        assert!(model.fsz && model.complex && !model.data.green_two_ex_terms.is_empty());
        let mut cpu = ComplexFszCpu::new(&model, &model.parameters);
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state, false)?;
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
            "{name}: PhysCal accepted {accepted:?}, {} spin-changing moves, smallest decision \
             margin {:e}",
            host.spin_moves, host.margin
        );
        assert_values(name, "final", backend, &state, &measurement, &mut cpu);
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; complex FSZ L16 PhysCal against the CPU C-order complex FSZ path"]
fn resident_physcal_fsz_complex_l16_matches_cpu_c_order() {
    assert_physcal("hubbard_fsz_complex_L16");
}

#[test]
#[ignore = "requires CUDA; complex FSZ L128 PhysCal against the CPU C-order complex FSZ path"]
fn resident_physcal_fsz_complex_l128_matches_cpu_c_order() {
    assert_physcal("hubbard_fsz_complex_L128");
}

#[test]
#[ignore = "requires CUDA; 20-window complex FSZ PhysCal repeatability and residency"]
fn resident_physcal_fsz_complex_repeats_twenty_windows() {
    for name in ["hubbard_fsz_complex_L16", "hubbard_fsz_complex_L128"] {
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
             run={seconds:.3}s traffic={traffic:?} energy={:.9}{:+.9}i",
            first.energy, first.energy_imag
        );
    }
}
