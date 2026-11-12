//! Window reductions (#729): CPU C-order per-configuration energies first,
//! then independent host sums of the same device operator values. The latter
//! isolate reduction rounding from Pfaffian/inverse rounding (#656).
use super::*;
use crate::opt::complex_tests::{budget, download};
use num_complex::Complex64;

#[test]
#[ignore = "requires CUDA; reported square AP/P configuration and PhysCal second moment"]
fn resident_physcal_square_ap_p_matches_cpu_configuration() {
    use crate::opt::antiperiodic_tests::{assert_square_snapshot, square_config, square_input};
    let path = square_input();
    let config = square_config();
    let model =
        Model::prepare_physcal(&path, &config, Some(&path.with_file_name("zqp_opt.dat"))).unwrap();
    let ops = Operators::prepare(&model).unwrap();
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = State::prepare_with_derivatives(&mut backend, model, &config, false).unwrap();
    state.resident.delayed_updates = false;
    state.initialize(&mut backend).unwrap();
    let mut measurement = Measurement::prepare(&mut backend, ops, &state, false).unwrap();
    measurement.run(&mut backend, &mut state).unwrap();
    let result = measurement.result(&mut backend, &state).unwrap();
    let (ip, values) = with_raw_exec(&mut backend, "square.ap.physcal", |exec| {
        Ok((
            download::<f64>(exec, &state.buffers.overlaps, 1)?[0],
            download::<f64>(exec, &measurement.values, state.model.energies.len())?,
        ))
    })
    .unwrap();
    let scale = state
        .model
        .energies
        .iter()
        .zip(&values)
        .map(|(c, v)| (c * v).abs())
        .sum();
    assert_square_snapshot(&mut backend, &state, ip, result.energy, scale);
    assert!(
        (result.energy_second_moment - result.energy * result.energy).abs()
            <= 8. * f64::EPSILON * (result.energy * result.energy).max(1.)
    );
}

#[test]
fn window_green_layout_preserves_operator_order_and_complex_components() {
    let ops = Operators {
        ops: vec![],
        n_energy: 0,
        n_one: 1,
        n_two: 2,
        n_ex: 1,
    };
    let g = GreenWindow::from_flat(&[1., 11., 2., 12., 3., 13., 4., 14.], &ops, 2);
    assert_eq!(g.green_one, [1.]);
    assert_eq!(g.green_two, [2., 3.]);
    assert_eq!(g.green_two_ex, [4.]);
    assert_eq!(g.green_one_imag, [11.]);
    assert_eq!(g.green_two_imag, [12., 13.]);
    assert_eq!(g.green_two_ex_imag, [14.]);
    let g = GreenWindow::from_flat(&[1., 2., 3., 4.], &ops, 1);
    assert_eq!(g.green_two, [2., 3.]);
    assert_eq!(g.green_two_imag, [0., 0.]);
}

fn real_input() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def")
}

fn sampling() -> OptConfig {
    OptConfig {
        nsteps: 3,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(4),
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

fn as_complex(values: &[f64], lanes: usize) -> Vec<Complex64> {
    values
        .chunks_exact(lanes)
        .map(|v| Complex64::new(v[0], if lanes == 2 { v[1] } else { 0. }))
        .collect()
}

fn green_rows(g: &GreenWindow) -> Vec<Complex64> {
    g.green_one
        .iter()
        .chain(&g.green_two)
        .chain(&g.green_two_ex)
        .zip(
            g.green_one_imag
                .iter()
                .chain(&g.green_two_imag)
                .chain(&g.green_two_ex_imag),
        )
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect()
}

fn check_windows(input: PathBuf) {
    let config = sampling();
    let model = Model::prepare_physcal(&input, &config, None).unwrap();
    let mut real_cpu =
        (!model.complex).then(|| crate::opt::dh_tests::Cpu::new(&model, &model.parameters));
    let mut complex_cpu = model
        .complex
        .then(|| crate::opt::complex_tests::Cpu::new(&model, &model.parameters));
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let ops = Operators::prepare(&model).unwrap();
    let mut state = State::prepare_with_derivatives(&mut backend, model, &config, false).unwrap();
    state.resident.delayed_updates = false;
    state.initialize(&mut backend).unwrap();
    let measurement = Measurement::prepare(&mut backend, ops, &state, true).unwrap();
    let g = state.resident.sampler.geom;
    let terms = measurement.operators.terms();
    let lanes = state.model.lanes();
    let snapshots = with_raw_exec(&mut backend, "window.regression", |exec| {
        state.resident.enqueue_random(exec, state.warmup)?;
        let mut snapshots = Vec::new();
        for window in 0..state.steps {
            state.resident.enqueue_random(exec, state.interval)?;
            measurement.observe(exec, &state, window)?;
            snapshots.push((
                download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)?,
                download::<i32>(exec, &state.resident.sampler.aw, g.walkers)?,
                download::<f64>(exec, &measurement.values, g.walkers * terms * lanes)?,
            ));
        }
        Ok(snapshots)
    })
    .unwrap();
    let result = measurement.result(&mut backend, &state).unwrap();
    let windows = result.green_windows.as_ref().unwrap();
    let coefficients = as_complex(&state.model.energies, lanes);
    let ops = &measurement.operators;
    let mut max_variance = 0f64;
    for (window, (cfg, aw, raw)) in snapshots.iter().enumerate() {
        let values = as_complex(raw, lanes);
        let mut energies = Vec::new();
        let mut greens = vec![Complex64::new(0., 0.); ops.rows() - 1];
        let mut scale = vec![0f64; ops.rows() - 1];
        for w in 0..g.walkers {
            let cfg = &cfg[w * g.n..(w + 1) * g.n];
            let (cpu, kappa) = if let Some(cpu) = real_cpu.as_mut() {
                let e = cpu.evaluate(cfg);
                (Complex64::new(e.energy, 0.), e.kappa)
            } else {
                let e = complex_cpu.as_mut().unwrap().evaluate(cfg);
                (e.energy, e.kappa)
            };
            let v = &values[w * terms..(w + 1) * terms];
            let energy = (0..ops.n_energy)
                .map(|t| coefficients[t] * v[t])
                .sum::<Complex64>();
            let energy_scale = (0..ops.n_energy)
                .map(|t| (coefficients[t] * v[t]).norm())
                .sum::<f64>();
            // Existing per-configuration #656 bound, before any statistics:
            // factor/inverse update error propagated through Hamiltonian ratios.
            let bound = 1e-13 + budget(aw[w], g.n, kappa) * energy_scale;
            assert!(
                (energy - cpu).norm() <= bound,
                "CPU energy {cpu} vs {energy}, bound={bound:e}"
            );
            energies.push(energy);
            for row in 0..ops.rows() - 1 {
                let value = if row < ops.n_one + ops.n_two {
                    v[ops.n_energy + row]
                } else {
                    let k =
                        ops.n_energy + ops.n_one + ops.n_two + 2 * (row - ops.n_one - ops.n_two);
                    v[k] * if lanes == 2 {
                        v[k + 1].conj()
                    } else {
                        v[k + 1]
                    }
                };
                greens[row] += value;
                scale[row] += value.norm();
            }
        }
        let count = g.walkers as f64;
        let expected = energies.iter().sum::<Complex64>() / count;
        let second = energies.iter().map(|e| e.norm_sqr()).sum::<f64>() / count;
        let e = &result.energy_windows[window];
        // Warp-tree vs chronological sums of the same input values; includes
        // products, divisions and complex additions (<= 8 W eps of the scale).
        let e_bound = 1e-15
            + 8. * count * f64::EPSILON * energies.iter().map(|e| e.norm()).sum::<f64>() / count;
        let m_bound = 1e-15 + 8. * count * f64::EPSILON * second;
        assert!((Complex64::new(e.energy, e.energy_imag) - expected).norm() <= e_bound);
        assert!(
            (e.energy_second_moment - second).abs() <= m_bound,
            "second moment {} vs {second}, bound={m_bound:e}",
            e.energy_second_moment
        );
        for (row, actual) in green_rows(&windows[window]).iter().enumerate() {
            assert!(
                (*actual - greens[row] / count).norm()
                    <= 1e-15 + 8. * count * f64::EPSILON * scale[row] / count
            );
        }
        let variance = e.energy_second_moment - e.energy * e.energy - e.energy_imag * e.energy_imag;
        assert!(variance >= -64. * count * f64::EPSILON * e.energy_second_moment.max(1.));
        max_variance = max_variance.max(variance);
    }
    assert!(
        max_variance > 1e-6,
        "must distinguish mean(E²) from mean(E)²"
    );
    let n = state.steps as f64;
    let mean = result
        .energy_windows
        .iter()
        .map(|e| Complex64::new(e.energy, e.energy_imag))
        .sum::<Complex64>()
        / n;
    assert!(
        (mean - Complex64::new(result.energy, result.energy_imag)).norm()
            <= 8.
                * n
                * f64::EPSILON
                * result
                    .energy_windows
                    .iter()
                    .map(|e| e.energy.hypot(e.energy_imag))
                    .fold(1., f64::max)
    );
    let second = result
        .energy_windows
        .iter()
        .map(|e| e.energy_second_moment)
        .sum::<f64>()
        / n;
    assert!((second - result.energy_second_moment).abs() <= 8. * n * f64::EPSILON * second.max(1.));
    let overall = GreenWindow {
        green_one: result.green_one.clone(),
        green_two: result.green_two.clone(),
        green_two_ex: result.green_two_ex.clone(),
        green_one_imag: result.green_one_imag.clone(),
        green_two_imag: result.green_two_imag.clone(),
        green_two_ex_imag: result.green_two_ex_imag.clone(),
    };
    for (row, value) in green_rows(&overall).iter().enumerate() {
        let values: Vec<_> = windows.iter().map(|w| green_rows(w)[row]).collect();
        let average = values.iter().sum::<Complex64>() / n;
        let bound = 8. * n * f64::EPSILON * values.iter().map(|x| x.norm()).fold(1., f64::max);
        assert!((*value - average).norm() <= bound);
    }
    // Disabling Green history changes neither the sampler nor any mean.
    let off = run_gpu_phys_cal_from_namelist(
        &input,
        PhysCalConfig {
            walkers: config.walkers,
            nsamples: Some(config.nsteps),
            warmup: config.warmup,
            interval: config.interval,
            seed: config.seed,
            initial_def: InitialDef::None,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(off.green_windows.is_none());
    crate::opt::tests::close(
        "history flag energy",
        &[off.energy, off.energy_second_moment],
        &[result.energy, result.energy_second_moment],
        1e-12,
        1e-12,
    );
    for (a, b) in off.energy_windows.iter().zip(&result.energy_windows) {
        crate::opt::tests::close(
            "history flag window",
            &[a.energy, a.energy_imag, a.energy_second_moment],
            &[b.energy, b.energy_imag, b.energy_second_moment],
            1e-12,
            1e-12,
        );
    }
    for (a, b) in [
        (&off.green_one, &result.green_one),
        (&off.green_two, &result.green_two),
        (&off.green_two_ex, &result.green_two_ex),
        (&off.green_one_imag, &result.green_one_imag),
        (&off.green_two_imag, &result.green_two_imag),
        (&off.green_two_ex_imag, &result.green_two_ex_imag),
    ] {
        crate::opt::tests::close("history flag Green mean", a, b, 1e-12, 1e-12);
    }
    assert_eq!((off.warmup, off.interval), (0, 4));
}

#[test]
#[ignore = "requires CUDA; real window moments and Green means against CPU C-order"]
fn physcal_windows_real_match_cpu_and_overall_means() {
    check_windows(real_input());
}

#[test]
#[ignore = "requires CUDA; L128 window statistics against CPU C-order"]
fn physcal_windows_l128_match_cpu_and_overall_means() {
    check_windows(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmark/cpu_scaling_l128/inputs/phys/namelist.def"),
    );
}

#[test]
#[ignore = "requires CUDA; complex second moment is C mean(conj(E)*E)"]
fn physcal_windows_complex_match_cpu_and_overall_means() {
    check_windows(crate::opt::complex_tests::input("hubbard_complex_L16"));
}

#[test]
#[ignore = "requires CUDA; nonzero exact eigenvalue of the fixed-particle-number operator"]
fn physcal_window_variance_vanishes_for_number_eigenstate() {
    use mvmc_expert_parsers::types::{Spin, TransferTerm};
    let cfg = sampling();
    let model = Model::prepare_physcal(&real_input(), &cfg, None).unwrap();
    let mut data = model.data;
    data.coulomb_intra_terms.clear();
    data.transfer_terms = (0..16)
        .flat_map(|site| {
            [Spin::Up, Spin::Down].map(|spin| TransferTerm {
                site1: site,
                spin1: spin,
                site2: site,
                spin2: spin,
                value: Complex64::new(-0.5, 0.),
            })
        })
        .collect();
    data.green_one_terms.clear();
    data.green_two_terms.clear();
    data.green_two_ex_terms.clear();
    let model = Model::from_data(data, 79).unwrap();
    let expected = model.ne as f64; // H = (N_up+N_down)/2 = Ne.
    let ops = Operators::prepare(&model).unwrap();
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = State::prepare_with_derivatives(&mut backend, model, &cfg, false).unwrap();
    state.resident.delayed_updates = false;
    state.initialize(&mut backend).unwrap();
    let mut measurement = Measurement::prepare(&mut backend, ops, &state, true).unwrap();
    measurement.run(&mut backend, &mut state).unwrap();
    let result = measurement.result(&mut backend, &state).unwrap();
    for e in result.energy_windows {
        let bound = 64. * 8. * f64::EPSILON * expected * expected;
        assert!((e.energy - expected).abs() <= bound);
        assert!((e.energy_second_moment - expected * expected).abs() <= bound);
        assert!((e.energy_second_moment - e.energy * e.energy).abs() <= bound);
    }
    assert!(result
        .green_windows
        .unwrap()
        .iter()
        .all(|g| g.green_one.is_empty()));
}
