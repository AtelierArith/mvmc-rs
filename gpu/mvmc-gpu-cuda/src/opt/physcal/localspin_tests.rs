//! Resident PhysCal with local spins and exchanges against Julia GPU (#660).
use super::*;
use crate::opt::localspin_tests::{assert_local_spins, input, integers, parse_fixture, tape};
use tenferro_ext::raw_exec::raw_exec_counters;

fn means(
    backend: &mut CudaEngine,
    state: &State,
    measurement: &Measurement,
) -> Result<Vec<f64>, String> {
    with_raw_exec(backend, "localspin.physcal.observe", |exec| {
        // SAFETY: means owns exactly rows FP64 values.
        unsafe {
            exec.zero_bytes(
                exec.ptr(&measurement.means)?,
                measurement.operators.rows() * 8,
            )?;
        }
        measurement.observe(exec, state, 0)
    })
    .map_err(|e| e.to_string())?;
    let result = measurement.result(backend, state)?;
    Ok(std::iter::once(result.energy)
        .chain(result.green_one)
        .chain(result.green_two)
        .chain(result.green_two_ex)
        .collect())
}

fn assert_physcal_flavor(name: &str, fixture_text: &str, exchange: f64) {
    let fixture = parse_fixture(fixture_text);
    let walkers = 8;
    with_runtime(0, |backend| {
        let probe = Model::prepare_physcal(&input(name), &OptConfig::default(), None)?;
        let steps = 3 * probe.ns;
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
        assert_eq!(model.exchange, exchange);
        for (a, b) in model.parameters.iter().zip(&fixture["parameters"]) {
            assert!((a - b).abs() <= 1e-14 + 1e-14 * a.abs().max(b.abs()));
        }
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        // Julia PhysCal: immediate inverse updates (cuda_runners.jl:122).
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state, false)?;
        let initial = state.resident.snapshot(backend)?;
        assert_eq!(initial.ele, integers(&fixture["initial_cfg"]));
        assert_local_spins(&state.model, &initial.ele);
        for (a, b) in initial.pf.iter().zip(&fixture["initial_pf"]) {
            assert!((a - b).abs() <= 1e-12 + 5e-11 * a.abs().max(b.abs()));
        }
        let rows = measurement.operators.rows();
        for stage in ["initial_values", "final_values"] {
            if stage == "final_values" {
                state
                    .resident
                    .advance_with_tape(backend, &tape(&fixture, walkers, steps))?;
                let last = state.resident.snapshot(backend)?;
                assert_eq!(
                    last.ele,
                    integers(&fixture["final_cfg"]),
                    "{name} trajectory"
                );
                assert_eq!(
                    state.resident.accepted(backend)?,
                    integers(&fixture["accepted"])
                );
                assert_local_spins(&state.model, &last.ele);
                for (a, b) in last.pf.iter().zip(&fixture["final_pf"]) {
                    assert!((a - b).abs() <= 1e-12 + 5e-11 * a.abs().max(b.abs()));
                }
            }
            let actual = means(backend, &state, &measurement)?;
            let expected = &fixture[stage];
            assert_eq!(expected.len(), rows * walkers);
            let mut worst = 0f64;
            for (row, a) in actual.iter().enumerate() {
                let b =
                    (0..walkers).map(|w| expected[row + rows * w]).sum::<f64>() / walkers as f64;
                worst = worst.max((a - b).abs());
                // The PhysCal L128 fixture bound (physcal_julia.md): roundoff
                // starts in independent factorization and QP/dot reductions.
                assert!(
                    (a - b).abs() <= 1e-10 + 1e-10 * a.abs().max(b.abs()),
                    "{name} {stage}[{row}] {a} vs {b}"
                );
            }
            println!("{name} {stage}: max abs observable difference={worst:e}");
        }
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; Julia GPU Heisenberg L64 PhysCal observables and tape"]
fn resident_physcal_matches_julia_heisenberg_exchange() {
    assert_physcal_flavor(
        "heisenberg_chain_L64",
        include_str!("../../../tests/fixtures/localspin/heisenberg_physcal.txt"),
        1.0,
    );
}

#[test]
#[ignore = "requires CUDA; Julia GPU Kondo PhysCal observables and tape"]
fn resident_physcal_matches_julia_kondo_mixed_moves() {
    assert_physcal_flavor(
        "kondo_chain_L64",
        include_str!("../../../tests/fixtures/localspin/kondo_physcal.txt"),
        0.5,
    );
}

#[test]
#[ignore = "requires CUDA; 20-window local-spin PhysCal repeatability and residency"]
fn resident_physcal_localspin_repeats_twenty_windows() {
    for name in ["heisenberg_chain_L64", "kondo_chain_L64"] {
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
