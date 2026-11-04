use super::*;
use tenferro_ext::raw_exec::raw_exec_counters;

fn input() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/cpu_scaling_l128/inputs/phys/namelist.def")
}

fn config() -> PhysCalConfig {
    PhysCalConfig {
        walkers: Some(8),
        nsamples: Some(20),
        warmup: Some(128),
        interval: Some(128),
        seed: Some(79),
        opt_para: Some(input().with_file_name("zqp_opt.dat")),
        ..Default::default()
    }
}

fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 + 1e-12 * a.abs().max(b.abs()),
        "{a} vs {b}"
    );
}

#[test]
fn physcal_operator_validation_precedes_cuda_allocation() {
    let cfg = OptConfig {
        seed: Some(79),
        ..Default::default()
    };
    let model = Model::prepare_physcal(&input(), &cfg, config().opt_para.as_deref()).unwrap();
    let operators = Operators::prepare(&model).unwrap();
    assert_eq!(
        (operators.n_one, operators.n_two, operators.n_ex),
        (256, 768, 0)
    );
    let mut model = model;
    model.data.green_one_terms[0].spin2 = mvmc_expert_parsers::types::Spin::Down;
    model.data.green_one_terms[0].spin1 = mvmc_expert_parsers::types::Spin::Up;
    assert!(Operators::prepare(&model)
        .unwrap_err()
        .contains("spin-conserving"));
}

#[test]
fn physcal_explicit_input_parameters_override_fixed_parameters() {
    use std::fs;
    let baseline = Model::prepare_physcal(
        &input(),
        &OptConfig::default(),
        config().opt_para.as_deref(),
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("mvmc-physcal-overlay-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(input().parent().unwrap()).unwrap() {
        let source = entry.unwrap().path();
        if source.is_file() {
            fs::copy(&source, dir.join(source.file_name().unwrap())).unwrap();
        }
    }
    let namelist = dir.join("namelist.def");
    let mut content = fs::read_to_string(&namelist).unwrap();
    content.push_str("\nInGutzwiller gutz-input.def\nInOrbital orbital-input.def\n");
    fs::write(&namelist, content).unwrap();
    fs::write(
        dir.join("gutz-input.def"),
        "===\nNGutzwillerIdx 4\nComplexType 0\n===\n===\n0 0.625 0\n1 0.75 0\n2 0.875 0\n3 1 0\n",
    )
    .unwrap();
    fs::write(
        dir.join("orbital-input.def"),
        "===\nNOrbitalIdx 1\nComplexType 0\n===\n===\n0 0.875 0\n",
    )
    .unwrap();
    let overlaid = Model::prepare_physcal(
        &namelist,
        &OptConfig::default(),
        Some(&dir.join("zqp_opt.dat")),
    )
    .unwrap();
    // Explicit constants verify C's fixed-file -> In*.def precedence. The
    // other coefficients must retain the supplied fixed-file values.
    for (&actual, expected) in overlaid.parameters[..4]
        .iter()
        .zip([0.625, 0.75, 0.875, 1.0])
    {
        close(actual, expected);
    }
    close(overlaid.parameters[overlaid.ng + overlaid.nj], 0.875);
    for (index, (&actual, &fixed)) in overlaid
        .parameters
        .iter()
        .zip(&baseline.parameters)
        .enumerate()
    {
        if index >= overlaid.ng && index != overlaid.ng + overlaid.nj {
            close(actual, fixed);
        }
    }
    // Removing the overlays recovers every no-In coefficient used by the
    // recorded L128 benchmark; no parameter reinitialization is introduced.
    fs::write(&namelist, fs::read_to_string(input()).unwrap()).unwrap();
    let restored = Model::prepare_physcal(
        &namelist,
        &OptConfig::default(),
        Some(&dir.join("zqp_opt.dat")),
    )
    .unwrap();
    for (&actual, &fixed) in restored.parameters.iter().zip(&baseline.parameters) {
        close(actual, fixed);
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn physcal_one_body_output_retains_c_indirect_reader_contract() {
    use mvmc_expert_parsers::types::{GreenOneTerm, GreenTwoExTerm, Spin::Up};
    let mut model = Model::prepare_physcal(
        &input(),
        &OptConfig::default(),
        config().opt_para.as_deref(),
    )
    .unwrap();
    let term = GreenOneTerm {
        site1: 0,
        site2: 1,
        spin1: Up,
        spin2: Up,
    };
    model.data.green_one_terms = vec![term, term];
    model.data.green_two_terms.clear();
    model.data.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 0,
        site2: 1,
        spin1: Up,
        spin2: Up,
        site3: 0,
        site4: 1,
        spin3: Up,
        spin4: Up,
    }];
    // C's independent duplicate-reader fixture expects one canonical row,
    // rather than Julia GPU's two raw OneBodyG rows. Keep shared input semantics.
    model.data.canonicalize_green_two_ex();
    assert_eq!(model.data.green_one_terms, vec![term]);
    let operators = Operators::prepare(&model).unwrap();
    assert_eq!(
        (operators.n_one, operators.n_two, operators.n_ex),
        (1, 0, 1)
    );
}

#[test]
#[ignore = "requires CUDA; 20-window repeatability and transfer/synchronization contract"]
fn resident_physcal_repeats_and_keeps_sampling_resident() {
    let cfg = config();
    let first = run_gpu_phys_cal_from_namelist(input(), cfg.clone()).unwrap();
    run_gpu_phys_cal_from_namelist(
        input(),
        PhysCalConfig {
            seed: Some(23),
            nsamples: Some(1),
            ..cfg.clone()
        },
    )
    .unwrap();
    let second = run_gpu_phys_cal_from_namelist(input(), cfg.clone()).unwrap();
    close(first.energy, second.energy);
    for (a, b) in first
        .green_one
        .iter()
        .chain(&first.green_two)
        .chain(&first.green_two_ex)
        .zip(
            second
                .green_one
                .iter()
                .chain(&second.green_two)
                .chain(&second.green_two_ex),
        )
    {
        close(*a, *b);
    }
    let sampling = OptConfig {
        walkers: cfg.walkers,
        nsteps: 20,
        warmup: cfg.warmup,
        interval: cfg.interval,
        seed: cfg.seed,
        ..Default::default()
    };
    with_runtime(0, |backend| {
        let model = Model::prepare_physcal(&input(), &sampling, cfg.opt_para.as_deref())?;
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let mut measurement = Measurement::prepare(backend, operators, &state)?;
        let before = raw_exec_counters();
        measurement.run(backend, &mut state)?;
        let traffic = raw_exec_counters() - before;
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.download_calls,
                traffic.synchronizations
            ),
            (0, 0, 1)
        );
        let result = measurement.result(backend, &state)?;
        close(first.energy, result.energy);
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CUDA; independent Julia fixed-state and supplied-tape oracle"]
fn resident_physcal_matches_julia_observables_and_tape() {
    use mvmc_expert_parsers::types::{GreenTwoExTerm, Spin::*};
    let mut tokens =
        include_str!("../../../tests/fixtures/physcal_julia_l128.txt").split_whitespace();
    let mut fixture = HashMap::new();
    while let Some(name) = tokens.next() {
        let len: usize = tokens.next().unwrap().parse().unwrap();
        fixture.insert(
            name,
            (0..len)
                .map(|_| tokens.next().unwrap().parse::<f64>().unwrap())
                .collect::<Vec<_>>(),
        );
    }
    let sampling = OptConfig {
        walkers: Some(2),
        nsteps: 1,
        warmup: Some(0),
        interval: Some(20),
        seed: Some(79),
        ..Default::default()
    };
    let parameters = config().opt_para;
    with_runtime(0, |backend| {
        let mut model = Model::prepare_physcal(&input(), &sampling, parameters.as_deref())?;
        for (a, b) in model.parameters.iter().zip(&fixture["parameters"]) {
            assert!((a - b).abs() <= 1e-14 + 1e-14 * a.abs().max(b.abs()));
        }
        for (a, b, s, c, d, t) in [
            (0, 1, Up, 0, 2, Up),
            (0, 1, Down, 0, 1, Down),
            (0, 0, Up, 0, 0, Down),
        ] {
            model.data.green_two_ex_terms.push(GreenTwoExTerm {
                site1: a,
                site2: b,
                spin1: s,
                spin2: s,
                site3: c,
                site4: d,
                spin3: t,
                spin4: t,
            });
        }
        let operators = Operators::prepare(&model)?;
        let mut state = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        state.resident.delayed_updates = false;
        state.initialize(backend)?;
        let measurement = Measurement::prepare(backend, operators, &state)?;
        let expected_cfg = |name| fixture[name].iter().map(|x| *x as i32).collect::<Vec<_>>();
        let initial = state.resident.snapshot(backend)?;
        assert_eq!(
            initial.ele,
            expected_cfg("initial_cfg"),
            "exact Julia XORWOW initialization"
        );
        let mut worst_pf_relative = 0.0f64;
        for (a, b) in initial.pf.iter().zip(&fixture["initial_pf"]) {
            worst_pf_relative = worst_pf_relative.max((a - b).abs() / a.abs().max(b.abs()));
            assert!((a - b).abs() <= 1e-12 + 5e-11 * a.abs().max(b.abs()));
        }
        println!("initial Pfaffian maximum relative difference={worst_pf_relative:e}");
        for stage in ["initial_values", "final_values"] {
            if stage == "final_values" {
                let tape: Vec<_> = fixture["tape"].iter().map(|x| *x as u32).collect();
                state.resident.advance_with_tape(backend, &tape)?;
                assert_eq!(
                    state.resident.snapshot(backend)?.ele,
                    expected_cfg("final_cfg")
                );
                assert_eq!(state.resident.accepted(backend)?, expected_cfg("accepted"));
            }
            with_raw_exec(backend, "physcal.julia.observe", |exec| {
                unsafe {
                    exec.zero_bytes(
                        exec.ptr(&measurement.means)?,
                        measurement.operators.rows() * 8,
                    )?;
                }
                measurement.observe(exec, &state)
            })
            .map_err(|e| e.to_string())?;
            let result = measurement.result(backend, &state)?;
            let actual: Vec<_> = std::iter::once(result.energy)
                .chain(result.green_one)
                .chain(result.green_two)
                .chain(result.green_two_ex)
                .collect();
            let expected = &fixture[stage];
            assert_eq!(expected.len(), 2 * actual.len());
            let mut worst = 0.0f64;
            for (row, a) in actual.iter().enumerate() {
                let b = (expected[row] + expected[row + actual.len()]) / 2.0;
                worst = worst.max((a - b).abs());
                // Same fixed operators; first roundoff is in independent GPU
                // factorization and QP/dot reduction, as in the Opt stage gate.
                assert!(
                    a.is_finite()
                        && b.is_finite()
                        && (a - b).abs() <= 1e-10 + 1e-10 * a.abs().max(b.abs()),
                    "{stage}[{row}] {a} vs {b}"
                );
            }
            println!("{stage}: max abs observable difference={worst:e}");
        }
        // Supplying a tape must leave the ensemble's own XORWOW generator
        // untouched. Compare its next words to a freshly initialized run.
        let next_words = |backend: &mut CudaEngine, state: &mut State| {
            with_raw_exec(backend, "physcal.julia.next_rng", |exec| {
                let resident = &mut state.resident;
                resident
                    .rng
                    .fill_u32(exec, &mut resident.sampler.tape, 200)?;
                let mut words = vec![0u32; 200];
                unsafe {
                    exec.download(&mut words, exec.ptr(&resident.sampler.tape)?)?;
                }
                Ok(words)
            })
            .map_err(|e| e.to_string())
        };
        let actual_words = next_words(backend, &mut state)?;
        let model = Model::prepare_physcal(&input(), &sampling, parameters.as_deref())?;
        let mut fresh = State::prepare_with_derivatives(backend, model, &sampling, false)?;
        fresh.initialize(backend)?;
        assert_eq!(
            actual_words,
            next_words(backend, &mut fresh)?,
            "supplied tape must not consume XORWOW words"
        );
        Ok(())
    })
    .unwrap();
}
