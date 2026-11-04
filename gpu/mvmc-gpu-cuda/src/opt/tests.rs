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
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = {
        let s: &mut CudaEngine = &mut backend;
        State::prepare(s, model, &config)
    }
    .unwrap();
    {
        let s: &mut CudaEngine = &mut backend;
        state.initialize(s)
    }
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
    let (derivatives, energy) = {
        let s: &mut CudaEngine = &mut backend;
        {
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
        }
    }
    .unwrap();
    let before = raw_exec_counters();
    {
        let s: &mut CudaEngine = &mut backend;
        state.run(s)
    }
    .unwrap();
    let traffic = raw_exec_counters() - before;
    assert_eq!(
        (
            traffic.upload_calls,
            traffic.download_calls,
            traffic.synchronizations
        ),
        (0, 0, 1)
    );
    let result = {
        let s: &mut CudaEngine = &mut backend;
        state.result(s)
    }
    .unwrap();
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
    let update = {
        let s: &mut CudaEngine = &mut backend;
        {
            let state = &mut state;
            with_raw_exec(s, "opt.julia.residual", |exec| {
                let mut update = vec![0.; state.model.np];
                unsafe {
                    exec.download(&mut update, exec.ptr(&state.buffers.rhs)?)?;
                }
                Ok(update)
            })
            .map_err(|e| e.to_string())
        }
    }
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
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = {
        let s: &mut CudaEngine = &mut backend;
        State::prepare(s, model, &config)
    }
    .unwrap();
    {
        let s: &mut CudaEngine = &mut backend;
        {
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
        }
    }
    .unwrap();
    for _ in 0..2 {
        let before = raw_exec_counters();
        let error = {
            let s: &mut CudaEngine = &mut backend;
            state.result(s)
        }
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

fn l16_input() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def")
}

fn initialized_state(backend: &mut CudaEngine, config: &OptConfig) -> State {
    let model = Model::prepare(&l16_input(), config).unwrap();
    let mut state = {
        let s: &mut CudaEngine = backend;
        State::prepare(s, model, config)
    }
    .unwrap();
    {
        let s: &mut CudaEngine = backend;
        state.initialize(s)
    }
    .unwrap();
    state
}

/// Observation buffers after one `observe`: (terms, local energy, derivatives,
/// overlaps, valid mask, configurations).
type Observed = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>);

/// Optionally replace the Slater table and refactor every walker (the
/// parameter-refresh commit), optionally override the valid mask, then observe.
fn observe_with(
    backend: &mut CudaEngine,
    state: &mut State,
    slater: Option<&[f64]>,
    valid: Option<&[i32]>,
) -> Observed {
    let s: &mut CudaEngine = backend;
    with_raw_exec(s, "opt.observe.check", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        let interval = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
        let r = &state.resident.sampler;
        let w = r.geom.walkers;
        unsafe {
            if let Some(slater) = slater {
                exec.upload(exec.ptr(&r.slater)?, slater)?;
                exec.upload(exec.ptr(&r.rebuild)?, &vec![1i32; w])?;
            }
        }
        if slater.is_some() {
            state.resident.enqueue_rebuild(exec, &interval)?;
        }
        unsafe {
            if let Some(valid) = valid {
                exec.upload(exec.ptr(&state.resident.valid)?, valid)?;
            }
        }
        state.observe(exec, &module)?;
        let m = &state.model;
        let b = &state.buffers;
        let mut out: Observed = (
            vec![0.; w * m.energies.len()],
            vec![0.; w],
            vec![0.; w * m.np],
            vec![0.; w],
            vec![0; w * m.nq],
            vec![0; w * 2 * m.ne],
        );
        let mut error = [0i32];
        unsafe {
            exec.download(&mut out.0, exec.ptr(&b.values)?)?;
            exec.download(&mut out.1, exec.ptr(&b.local_energy)?)?;
            exec.download(&mut out.2, exec.ptr(&b.derivatives)?)?;
            exec.download(&mut out.3, exec.ptr(&b.overlaps)?)?;
            exec.download(&mut out.4, exec.ptr(&state.resident.valid)?)?;
            exec.download(&mut out.5, exec.ptr(&r.ele)?)?;
            exec.download(&mut error, exec.ptr(&state.resident.error)?)?;
        }
        assert_eq!(error, [0], "observation latched a failure");
        Ok(out)
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

fn download_slater(backend: &mut CudaEngine, state: &State) -> Vec<f64> {
    let s: &mut CudaEngine = backend;
    with_raw_exec(s, "opt.slater", |exec| {
        let m = &state.model;
        let mut slater = vec![0.; m.nq * 4 * m.ns * m.ns];
        unsafe {
            exec.download(&mut slater, exec.ptr(&state.resident.sampler.slater)?)?;
        }
        Ok(slater)
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Independent CPU (PFAPACK) Pfaffian of the gathered plane `A(r,c) =
/// -S_q[site_c][site_r]` over the kept electron slots, as `opt_cofactors` and
/// `interval_resident_factor` gather it. An exact zero pivot gives 0.
fn cpu_plane_pfaffian(slater: &[f64], ns: usize, q: usize, sites: &[usize]) -> f64 {
    use mvmc_gpu::{pfaffian_inverse_batched, Backend, PlaneStatus};
    let m = sites.len();
    if m == 0 {
        return 1.;
    }
    let mut plane = vec![0.; m * m];
    for c in 0..m {
        for r in 0..m {
            plane[r + c * m] = -slater[q * 4 * ns * ns + sites[c] * 2 * ns + sites[r]];
        }
    }
    let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &plane, m, 1, 1).unwrap();
    if cpu.status[0] == PlaneStatus::Ok {
        cpu.pf[0]
    } else {
        0.
    }
}

/// CPU oracle of the Slater-parameter derivatives, from exact (n-2)-minor
/// cofactors of every plane (valid or not) and the CPU overlap.
fn cpu_minor_derivatives(model: &Model, slater: &[f64], cfg: &[i32]) -> Vec<f64> {
    let (ns, ne, nq, np) = (model.ns, model.ne, model.nq, model.np);
    let n = 2 * ne;
    let nproj = model.ng + model.nj;
    let walkers = cfg.len() / n;
    let mut out = vec![0.; walkers * np];
    for w in 0..walkers {
        let sites: Vec<usize> = (0..n)
            .map(|i| cfg[w * n + i] as usize + if i >= ne { ns } else { 0 })
            .collect();
        let mut lookup = vec![-1i32; 2 * ns];
        for (i, &r) in sites.iter().enumerate() {
            lookup[r] = i as i32;
        }
        let overlap = (0..nq)
            .map(|q| model.weights[q] * cpu_plane_pfaffian(slater, ns, q, &sites))
            .sum::<f64>();
        let mut cofactor = HashMap::new();
        for p in nproj..np {
            let mut sum = 0.;
            for k in model.columns[p] as usize..model.columns[p + 1] as usize {
                let (r, c, q) = (
                    model.coordinates[3 * k] as usize,
                    model.coordinates[3 * k + 1] as usize,
                    model.coordinates[3 * k + 2] as usize,
                );
                let (i, j) = (lookup[c], lookup[r]);
                if i < 0 || i >= j {
                    continue;
                }
                let (i, j) = (i as usize, j as usize);
                let value = *cofactor.entry((q, i, j)).or_insert_with(|| {
                    let kept: Vec<usize> = (0..n)
                        .filter(|&x| x != i && x != j)
                        .map(|x| sites[x])
                        .collect();
                    let sign = if (i + j + 1) % 2 == 1 { -1. } else { 1. };
                    sign * cpu_plane_pfaffian(slater, ns, q, &kept)
                });
                sum += model.coefficients[k] * (-model.weights[q] * value / overlap);
            }
            out[w * np + p] = sum;
        }
    }
    out
}

/// Julia `nodal_derivatives.jl`: a QP plane that is exactly singular on a
/// walker's configuration (zero row/column of an occupied site) still has
/// nonzero derivatives through exact minors, while the total overlap is
/// nonzero. GPU derivatives for valid (Pf * inverse) and invalid (minor)
/// planes match an independent CPU minor oracle.
#[test]
#[ignore = "requires CUDA; singular-plane derivatives against CPU minor Pfaffians"]
fn resident_opt_nodal_derivatives_match_cpu_minors() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = initialized_state(&mut backend, &config);
    let (ns, nq, ne) = (state.model.ns, state.model.nq, state.model.ne);
    let n = 2 * ne;
    let before = observe_with(&mut backend, &mut state, None, None);
    let mut slater = download_slater(&mut backend, &state);
    // Walker 0's up-spin electron 0 occupies site r0; zero that row/column of
    // QP plane 0. Every walker occupying r0 has an invalid plane 0.
    let r0 = before.5[0] as usize;
    for x in 0..2 * ns {
        slater[r0 * 2 * ns + x] = 0.;
        slater[x * 2 * ns + r0] = 0.;
    }
    let after = observe_with(&mut backend, &mut state, Some(&slater), None);
    assert_eq!(after.5, before.5, "observation does not move walkers");
    let invalid: Vec<_> = (0..config.walkers.unwrap())
        .filter(|&w| after.5[w * n..w * n + ne].contains(&(r0 as i32)))
        .collect();
    assert!(invalid.contains(&0));
    for w in 0..config.walkers.unwrap() {
        let expected: Vec<i32> = (0..nq)
            .map(|q| i32::from(q != 0 || !invalid.contains(&w)))
            .collect();
        assert_eq!(after.4[w * nq..(w + 1) * nq], expected, "walker {w} mask");
    }
    let oracle = cpu_minor_derivatives(&state.model, &slater, &after.5);
    let np = state.model.np;
    let nproj = state.model.ng + state.model.nj;
    let scale = oracle.iter().fold(0f64, |a, x| a.max(x.abs()));
    let mut worst = 0f64;
    for (index, (&a, &b)) in after.2.iter().zip(&oracle).enumerate() {
        if index % np < nproj {
            continue;
        }
        worst = worst.max((a - b).abs() / scale);
    }
    // Both sides evaluate n=16 Pfaffians or inverses of O(1) skew matrices; the
    // max relative-to-scale deviation is roundoff (observed ~1e-15).
    assert!(worst <= 1e-12, "max derivative deviation {worst:e}");
    let walker0: Vec<_> = (nproj..np).map(|p| after.2[p]).collect();
    assert!(
        walker0.iter().any(|&x| x != 0.),
        "nodal derivatives are nonzero"
    );
    assert!(after.1.iter().all(|x| x.is_finite()));
    println!("nodal derivatives: invalid walkers {invalid:?}; max deviation {worst:e}");
}

/// Julia's `:invalid` observable scenario: flagging nonsingular planes invalid
/// routes the walker through the full-Pfaffian energy fallback and the
/// exact-minor derivatives. Both match the inverse path.
#[test]
#[ignore = "requires CUDA; observable fallbacks against the inverse path"]
fn resident_opt_fallback_observables_match_inverse_path() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(16),
        interval: Some(0),
        seed: Some(79),
        ..Default::default()
    };
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = initialized_state(&mut backend, &config);
    let nq = state.model.nq;
    let walkers = config.walkers.unwrap();
    let reference = observe_with(&mut backend, &mut state, None, None);
    let mut mask = vec![1i32; walkers * nq];
    for w in 0..walkers / 2 {
        mask[w * nq + w % nq] = 0;
    }
    let flagged = observe_with(&mut backend, &mut state, None, Some(&mask));
    let compare = |name: &str, a: &[f64], b: &[f64]| {
        let scale = b.iter().fold(0f64, |m, x| m.max(x.abs()));
        let worst = a
            .iter()
            .zip(b)
            .fold(0f64, |m, (x, y)| m.max((x - y).abs() / scale));
        // n=16 Pfaffians/minors versus inverse ratios of O(1) skew planes; a
        // 1e-11 scale-relative bound is ~1e4 eps, far above observed roundoff.
        assert!(worst <= 1e-11, "{name}: max deviation {worst:e}");
        println!("{name}: max fallback deviation {worst:e}");
    };
    compare("energy terms", &flagged.0, &reference.0);
    compare("local energy", &flagged.1, &reference.1);
    compare("derivatives", &flagged.2, &reference.2);
}

/// An Opt run whose QP plane 0 becomes exactly singular for every sampled
/// configuration recovers per plane (#573): walkers keep sampling with
/// full Pfaffians, observables use the fallbacks, and the next parameter
/// refresh restores every plane. The run is repeatable and transfer-free.
#[test]
#[ignore = "requires CUDA; singular sampling plane recovers through a full Opt step"]
fn resident_opt_recovers_singular_sampling_plane() {
    let config = OptConfig {
        nsteps: 2,
        walkers: Some(4),
        warmup: Some(16),
        interval: Some(16),
        seed: Some(79),
        ..Default::default()
    };
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut results = vec![];
    for _ in 0..2 {
        let mut state = initialized_state(&mut backend, &config);
        let ns = state.model.ns;
        {
            let s: &mut CudaEngine = &mut backend;
            let state = &mut state;
            with_raw_exec(s, "opt.singular-plane", |exec| unsafe {
                exec.zero_bytes(exec.ptr(&state.resident.sampler.slater)?, 4 * ns * ns * 8)
            })
            .map_err(|e| e.to_string())
        }
        .unwrap();
        let before = raw_exec_counters();
        {
            let s: &mut CudaEngine = &mut backend;
            state.run(s)
        }
        .unwrap();
        let traffic = raw_exec_counters() - before;
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.download_calls,
                traffic.synchronizations
            ),
            (0, 0, 1)
        );
        let result = {
            let s: &mut CudaEngine = &mut backend;
            state.result(s)
        }
        .unwrap();
        assert!(result
            .energies
            .iter()
            .chain(&result.parameters)
            .all(|x| x.is_finite()));
        let valid = {
            let s: &mut CudaEngine = &mut backend;
            with_raw_exec(s, "opt.valid", |exec| {
                let mut valid = vec![0i32; 4 * state.model.nq];
                unsafe {
                    exec.download(&mut valid, exec.ptr(&state.resident.valid)?)?;
                }
                Ok(valid)
            })
            .map_err(|e| e.to_string())
        }
        .unwrap();
        assert!(
            valid.iter().all(|&v| v == 1),
            "parameter refresh restores planes"
        );
        results.push(result);
    }
    close(
        "repeat energies",
        &results[1].energies,
        &results[0].energies,
        0.,
        0.,
    );
    close(
        "repeat parameters",
        &results[1].parameters,
        &results[0].parameters,
        0.,
        0.,
    );
}

/// A zero total overlap cannot be recovered: the derivative stage latches,
/// and the result boundary downloads only the status word (#573).
#[test]
#[ignore = "requires CUDA; zero overlap must not publish"]
fn resident_opt_zero_overlap_publishes_nothing() {
    let config = OptConfig {
        nsteps: 1,
        walkers: Some(4),
        warmup: Some(16),
        interval: Some(16),
        seed: Some(79),
        ..Default::default()
    };
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = initialized_state(&mut backend, &config);
    let expected = state.model.parameters.clone();
    let nq = state.model.nq;
    {
        let s: &mut CudaEngine = &mut backend;
        let state = &mut state;
        with_raw_exec(s, "opt.zero-weights", |exec| unsafe {
            exec.zero_bytes(exec.ptr(&state.resident.sampler.weights)?, nq * 8)
        })
        .map_err(|e| e.to_string())
    }
    .unwrap();
    {
        let s: &mut CudaEngine = &mut backend;
        state.run(s)
    }
    .unwrap();
    for _ in 0..2 {
        let before = raw_exec_counters();
        let error = {
            let s: &mut CudaEngine = &mut backend;
            state.result(s)
        }
        .unwrap_err();
        assert!(error.contains("device status 6 (derivatives"), "{error}");
        let traffic = raw_exec_counters() - before;
        assert_eq!(
            (traffic.download_calls, traffic.download_bytes),
            (1, 4),
            "failure exposes status only"
        );
    }
    let parameters = {
        let s: &mut CudaEngine = &mut backend;
        with_raw_exec(s, "opt.zero-weights.check", |exec| {
            let mut parameters = vec![0.; state.model.np];
            unsafe {
                exec.download(&mut parameters, exec.ptr(&state.buffers.parameters)?)?;
            }
            Ok(parameters)
        })
        .map_err(|e| e.to_string())
    }
    .unwrap();
    close("unpublished parameters", &parameters, &expected, 0., 0.);
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
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = {
        let s: &mut CudaEngine = &mut backend;
        State::prepare(s, model, &config)
    }
    .unwrap();
    {
        let s: &mut CudaEngine = &mut backend;
        state.initialize(s)
    }
    .unwrap();
    // Explicit diagnostic downloads are outside the production run's counters.
    if let Ok(path) = std::env::var("MVMC_RESIDENT_DUMP") {
        let dump = {
            let s: &mut CudaEngine = &mut backend;
            {
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
            }
        }
        .unwrap();
        std::fs::write(path, dump).unwrap();
    }
    let before = raw_exec_counters();
    let start = std::time::Instant::now();
    {
        let s: &mut CudaEngine = &mut backend;
        state.run(s)
    }
    .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let traffic = raw_exec_counters() - before;
    assert_eq!((traffic.upload_calls, traffic.download_calls), (0, 0));
    assert_eq!((traffic.synchronizations, traffic.scopes), (1, 1));
    let result = {
        let s: &mut CudaEngine = &mut backend;
        state.result(s)
    }
    .unwrap();
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
