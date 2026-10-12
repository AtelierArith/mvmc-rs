use super::*;
use tenferro_ext::raw_exec::raw_exec_counters;

fn l128_input() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L128/namelist.def")
}

fn close(name: &str, actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
            "{name}[{i}]: {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "requires CUDA; independent Julia initialization/observable/SR fixture"]
fn resident_opt_matches_julia_l128_stages() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let model = Model::prepare(&l128_input(), &config).unwrap();
    assert_julia_stages(
        model,
        config,
        include_str!("../../tests/fixtures/opt_julia_l128.txt"),
    );
}

#[test]
#[ignore = "requires CUDA; all pair overlays and nonzero correlations against Julia"]
fn resident_opt_matches_julia_general_operators() {
    use mvmc_expert_parsers::types::{
        CoulombInterTerm, ExchangeTerm, HundTerm, InterAllTerm, PairHopTerm,
    };
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L64/namelist.def");
    let base = Model::prepare(&input, &config).unwrap();
    let mut data = base.data;
    let mut parameters = mvmc_core::pack_parameters(&data).unwrap();
    for (p, x) in parameters
        .iter_mut()
        .take(data.projection_layout().n_proj)
        .enumerate()
    {
        x.re = (((p + 1) % 7) as f64 - 3.) / 100.;
    }
    mvmc_core::unpack_parameters(&mut data, &parameters).unwrap();
    data.coulomb_inter_terms.push(CoulombInterTerm {
        site1: 0,
        site2: 1,
        value: 0.125,
    });
    data.hund_terms.push(HundTerm {
        site1: 0,
        site2: 1,
        value: 0.0625,
    });
    data.pair_hop_terms.extend([
        PairHopTerm {
            site1: 0,
            site2: 1,
            value: -0.03125,
        },
        PairHopTerm {
            site1: 1,
            site2: 0,
            value: -0.03125,
        },
    ]);
    data.exchange_terms.push(ExchangeTerm {
        site1: 0,
        site2: 1,
        value: 0.25,
    });
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    for s in 0..2 {
                        for t in 0..2 {
                            data.inter_all_terms.push(InterAllTerm {
                                site0: i,
                                spin0: s,
                                site1: j,
                                spin1: s,
                                site2: k,
                                spin2: t,
                                site3: l,
                                spin3: t,
                                value: num_complex::Complex64::new(0.015625, 0.),
                                is_complex: false,
                            });
                        }
                    }
                }
            }
        }
    }
    let model = Model::from_data(data, 79).unwrap();
    assert_julia_stages(
        model,
        config,
        include_str!("../../tests/fixtures/opt_julia_operators.txt"),
    );
}

fn assert_julia_stages(model: Model, config: OptConfig, fixture_text: &str) {
    let mut backend = runtime(0).unwrap();
    let mut state = backend
        .with_backend_session(|s| State::prepare(s, model, &config))
        .unwrap();
    backend
        .with_backend_session(|s| state.initialize(s))
        .unwrap();
    let mut tokens = fixture_text.split_whitespace();
    let mut fixture = HashMap::new();
    while let Some(name) = tokens.next() {
        let n: usize = tokens.next().unwrap().parse().unwrap();
        fixture.insert(
            name,
            (0..n)
                .map(|_| tokens.next().unwrap().parse::<f64>().unwrap())
                .collect::<Vec<_>>(),
        );
    }
    let (derivatives, energy) = backend
        .with_backend_session(|s| {
            let state = &mut state;
            with_raw_exec(s, "opt.julia.stages", |exec| {
                let module = exec.module(SOURCE, KERNELS)?;
                state.observe(exec, &module)?;
                let b = &state.buffers;
                let s = &state.resident.sampler;
                let mut values = HashMap::new();
                for (name, buffer) in [
                    ("parameters", &b.parameters),
                    ("pf", &s.pf),
                    ("derivatives", &b.derivatives),
                    ("energy", &b.local_energy),
                    ("overlaps", &b.overlaps),
                    ("terms", &b.values),
                ] {
                    let Some(expected) = fixture.get(name) else {
                        continue;
                    };
                    let mut actual = vec![0.; expected.len()];
                    unsafe {
                        exec.download(&mut actual, exec.ptr(buffer)?)?;
                    }
                    let (atol, rtol) = if name == "parameters" {
                        (1e-14, 1e-14)
                    } else if name == "pf" || name == "overlaps" {
                        (1e-12, 5e-11)
                    } else {
                        (1e-10, 1e-10)
                    };
                    close(name, &actual, expected, atol, rtol);
                    values.insert(name, actual);
                }
                let expected: Vec<i32> = fixture["cfg"].iter().map(|&v| v as i32).collect();
                let mut cfg = vec![0; expected.len()];
                unsafe {
                    exec.download(&mut cfg, exec.ptr(&s.ele)?)?;
                }
                assert_eq!(
                    cfg, expected,
                    "exact Julia cuRAND initialization/shuffle contract"
                );
                Ok((
                    values.remove("derivatives").unwrap(),
                    values.remove("energy").unwrap(),
                ))
            })
            .map_err(|e| e.to_string())
        })
        .unwrap();
    let before = raw_exec_counters();
    backend.with_backend_session(|s| state.run(s)).unwrap();
    let traffic = raw_exec_counters() - before;
    assert_eq!(
        (
            traffic.upload_calls,
            traffic.download_calls,
            traffic.synchronizations
        ),
        (0, 0, 1)
    );
    let result = backend.with_backend_session(|s| state.result(s)).unwrap();
    close(
        "parameters_after",
        &result.parameters,
        &fixture["parameters_after"],
        1e-10,
        1e-10,
    );
    close(
        "energies",
        &result.energies,
        &fixture["energies"],
        1e-10,
        1e-10,
    );
    let update = backend
        .with_backend_session(|s| {
            let state = &mut state;
            with_raw_exec(s, "opt.julia.residual", |exec| {
                let mut update = vec![0.; state.model.np];
                unsafe {
                    exec.download(&mut update, exec.ptr(&state.buffers.rhs)?)?;
                }
                Ok(update)
            })
            .map_err(|e| e.to_string())
        })
        .unwrap();
    // Independently reconstruct the mathematical SR system on CPU and measure
    // its normalized solve residual, rather than checking library success alone.
    let np = state.model.np;
    let w = energy.len();
    let mean: Vec<f64> = (0..np)
        .map(|p| (0..w).map(|b| derivatives[b * np + p]).sum::<f64>() / w as f64)
        .collect();
    let covariance = |i: usize, j: usize| {
        (0..w)
            .map(|b| derivatives[b * np + i] * derivatives[b * np + j])
            .sum::<f64>()
            / w as f64
            - mean[i] * mean[j]
    };
    let variance: Vec<_> = (0..np).map(|p| covariance(p, p)).collect();
    let mp = &state.model.data.modpara;
    let cutoff = variance.iter().map(|v| v.abs()).fold(0f64, f64::max) * mp.dsr_opt_red_cut;
    let active: Vec<_> = (0..np)
        .map(|p| state.model.flags[p] != 0 && variance[p] > cutoff)
        .collect();
    let mean_e = energy.iter().sum::<f64>() / w as f64;
    let mut residual = 0f64;
    let mut norm_a = 0f64;
    let mut norm_b = 0f64;
    for row in 0..np {
        let rhs = if active[row] {
            -2. * mp.dsr_opt_step_dt
                * ((0..w)
                    .map(|b| derivatives[b * np + row] * energy[b])
                    .sum::<f64>()
                    / w as f64
                    - mean_e * mean[row])
        } else {
            0.
        };
        let mut value = 0.;
        let mut row_norm = 0.;
        for col in 0..np {
            let a = if active[row] && active[col] {
                covariance(row, col)
                    * if row == col {
                        1. + mp.dsr_opt_sta_del
                    } else {
                        1.
                    }
            } else if row == col {
                1.
            } else {
                0.
            };
            value += a * update[col];
            row_norm += a.abs();
        }
        residual = residual.max((value - rhs).abs());
        norm_a = norm_a.max(row_norm);
        norm_b = norm_b.max(rhs.abs());
    }
    let relative =
        residual / (norm_a * update.iter().map(|v| v.abs()).fold(0f64, f64::max) + norm_b);
    assert!(relative < 1e-11, "SR backward error {relative:e}");
    println!("independent Julia stages passed; SR backward error={relative:e}");
}

#[test]
#[ignore = "requires CUDA; complete 20-step repeatability and runtime isolation"]
fn resident_opt_repeats_twenty_steps() {
    let config = OptConfig {
        nsteps: 20,
        walkers: Some(320),
        warmup: Some(128),
        interval: Some(128),
        seed: Some(79),
        ..Default::default()
    };
    let first = run_gpu_para_opt_from_namelist(l128_input(), config.clone()).unwrap();
    // Interleaving a differently seeded run must not share walker/RNG state.
    let other = OptConfig {
        device: usize::from(cuda_devices().unwrap().len() > 1),
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(23),
        ..config.clone()
    };
    run_gpu_para_opt_from_namelist(l128_input(), other).unwrap();
    let second = run_gpu_para_opt_from_namelist(l128_input(), config).unwrap();
    close(
        "repeat energies",
        &first.energies,
        &second.energies,
        1e-12,
        1e-12,
    );
    close(
        "repeat parameters",
        &first.parameters,
        &second.parameters,
        1e-12,
        1e-12,
    );
}

#[test]
#[ignore = "requires CUDA; failed native solve must never publish parameters"]
fn resident_opt_rejects_failed_sr() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(1),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let model = Model::prepare(&input, &config).unwrap();
    let expected = model.parameters.clone();
    let n = model.np;
    let mut backend = runtime(0).unwrap();
    let mut state = backend
        .with_backend_session(|s| State::prepare(s, model, &config))
        .unwrap();
    backend
        .with_backend_session(|s| {
            let state = &mut state;
            with_raw_exec(s, "opt.failed-solve", |exec| {
                let mut matrix = vec![0.; n * n];
                for p in 0..n {
                    matrix[p * n + p] = 1.;
                }
                matrix[0] = -1.;
                let b = &state.buffers;
                let module = exec.module(SOURCE, KERNELS)?;
                unsafe {
                    exec.upload(exec.ptr(&b.gram)?, &matrix)?;
                    exec.upload(exec.ptr(&b.rhs)?, &vec![1.; n])?;
                    exec.dpotrf_solve(
                        exec.ptr(&b.gram)?,
                        exec.ptr(&b.rhs)?,
                        n,
                        exec.ptr(&b.workspace)?,
                        b.workspace_len,
                        exec.ptr(&b.info)?,
                    )?;
                    exec.launch(
                        module.function("opt_update")?,
                        (n.div_ceil(128) as u32, 1, 1),
                        128,
                        0,
                        &[
                            Arg::P(exec.ptr(&b.parameters)?),
                            Arg::P(exec.ptr(&b.rhs)?),
                            Arg::P(exec.ptr(&b.info)?),
                            Arg::P(exec.ptr(&state.resident.error)?),
                            i(n),
                        ],
                    )?;
                    let mut info = [0i32; 2];
                    exec.download(&mut info, exec.ptr(&b.info)?)?;
                    assert!(
                        info[0] > 0,
                        "indefinite matrix must fail native Cholesky: {info:?}"
                    );
                    let mut actual = vec![0.; n];
                    exec.download(&mut actual, exec.ptr(&b.parameters)?)?;
                    close(
                        "unpublished failed update",
                        &actual,
                        &expected,
                        1e-14,
                        1e-14,
                    );
                }
                Ok(())
            })
            .map_err(|e| e.to_string())
        })
        .unwrap();
    for _ in 0..2 {
        let before = raw_exec_counters();
        let error = backend
            .with_backend_session(|s| state.result(s))
            .unwrap_err();
        assert!(error.contains("device status 3"), "{error}");
        let traffic = raw_exec_counters() - before;
        assert_eq!(
            (traffic.download_calls, traffic.download_bytes),
            (1, 4),
            "failure exposes status only"
        );
    }
}

#[test]
#[ignore = "requires CUDA; zero covariance must not activate roundoff-only SR rows"]
fn resident_opt_single_walker_has_zero_sr_update() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(1),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let expected = Model::prepare(&input, &config).unwrap().parameters;
    let result = run_gpu_para_opt_from_namelist(&input, config).unwrap();
    close(
        "zero covariance",
        &result.parameters,
        &expected,
        1e-13,
        1e-13,
    );
}

#[test]
#[ignore = "requires CUDA; numerical and transfer diagnostics"]
fn resident_opt_pipeline() {
    let input = std::env::var("MVMC_RESIDENT_INPUT").unwrap_or_else(|_| {
        format!(
            "{}/../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let config = OptConfig {
        nsteps: env("MVMC_RESIDENT_STEPS", 20),
        walkers: Some(env("MVMC_RESIDENT_WALKERS", 8)),
        warmup: Some(env("MVMC_RESIDENT_WARMUP", 16)),
        interval: Some(env("MVMC_RESIDENT_INTERVAL", 16)),
        seed: Some(79),
        ..Default::default()
    };
    let model = Model::prepare(Path::new(&input), &config).unwrap();
    let mut backend = CudaBackend::new(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = backend
        .with_backend_session(|s| State::prepare(s, model, &config))
        .unwrap();
    backend
        .with_backend_session(|s| state.initialize(s))
        .unwrap();
    // Explicit diagnostic downloads are outside the production run's counters.
    if let Ok(path) = std::env::var("MVMC_RESIDENT_DUMP") {
        let dump = backend
            .with_backend_session(|s| {
                let state = &mut state;
                with_raw_exec(s, "opt.diagnostic", |exec| {
                    let module = exec.module(SOURCE, KERNELS)?;
                    state.observe(exec, &module)?;
                    let mut text = String::new();
                    let mut floats = |name: &str,
                                      buffer: &DeviceBuffer,
                                      n: usize|
                     -> tenferro_tensor::Result<()> {
                        let mut data = vec![0f64; n];
                        unsafe {
                            exec.download(&mut data, exec.ptr(buffer)?)?;
                        }
                        text.push_str(&format!("{name} {n}\n"));
                        for x in data {
                            text.push_str(&format!("{x:.17e}\n"));
                        }
                        Ok(())
                    };
                    let b = &state.buffers;
                    let m = &state.model;
                    let s = &state.resident.sampler;
                    let w = s.geom.walkers;
                    floats("parameters", &b.parameters, m.np)?;
                    floats("slater", &s.slater, m.nq * 4 * m.ns * m.ns)?;
                    floats("pf", &s.pf, w * m.nq)?;
                    floats("inverse", &s.inv, w * m.nq * 4 * m.ne * m.ne)?;
                    floats("derivatives", &b.derivatives, w * m.np)?;
                    floats("energy", &b.local_energy, w)?;
                    floats("terms", &b.values, w * m.energies.len())?;
                    floats("overlaps", &b.overlaps, w)?;
                    let mut cfg = vec![0i32; w * 2 * m.ne];
                    unsafe {
                        exec.download(&mut cfg, exec.ptr(&s.ele)?)?;
                    }
                    text.push_str(&format!("cfg {}\n", cfg.len()));
                    for x in cfg {
                        text.push_str(&format!("{x}\n"));
                    }
                    Ok(text)
                })
                .map_err(|e| e.to_string())
            })
            .unwrap();
        std::fs::write(path, dump).unwrap();
    }
    let before = raw_exec_counters();
    let start = std::time::Instant::now();
    backend.with_backend_session(|s| state.run(s)).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let traffic = raw_exec_counters() - before;
    assert_eq!((traffic.upload_calls, traffic.download_calls), (0, 0));
    assert_eq!((traffic.synchronizations, traffic.scopes), (1, 1));
    let result = backend.with_backend_session(|s| state.result(s)).unwrap();
    if let Ok(path) = std::env::var("MVMC_RESIDENT_DUMP") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        writeln!(file, "parameters_after {}", result.parameters.len()).unwrap();
        for x in &result.parameters {
            writeln!(file, "{x:.17e}").unwrap();
        }
        writeln!(file, "energies {}", result.energies.len()).unwrap();
        for x in &result.energies {
            writeln!(file, "{x:.17e}").unwrap();
        }
    }
    assert!(result
        .energies
        .iter()
        .chain(&result.parameters)
        .all(|x| x.is_finite()));
    println!(
        "resident run={seconds:.9}s traffic={traffic:?} energies={:?}",
        result.energies
    );
}
