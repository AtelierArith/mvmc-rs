//! AP translation projection regression (#730); CPU C-order physics reference.
use super::complex_tests::download;
use super::scope_tests::{cfg, edited_from};
use super::tests::close;
use super::*;

pub(crate) fn square_input() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/antiperiodic/ap_p_L8_U4_SP/namelist.def")
}

fn square_opt_input(tag: &str) -> std::path::PathBuf {
    edited_from(
        &square_input(),
        tag,
        &[("modpara.def", "NVMCCalMode    1", "NVMCCalMode    0")],
    )
}

pub(crate) fn square_config() -> OptConfig {
    OptConfig {
        nsteps: 1,
        walkers: Some(1),
        warmup: Some(0),
        interval: Some(0),
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

#[test]
fn resident_square_ap_p_sign_tables_match_original_input() {
    let path = square_input();
    let mut expected = mvmc_expert_parsers::parse_expert_mode_files(&path).unwrap();
    expected.ensure_orbital_idx_matrix();
    assert_eq!(
        (expected.modpara.nsite, expected.modpara.nmp_trans),
        (64, -4)
    );
    assert_eq!(
        expected
            .qp_trans_entries
            .iter()
            .flat_map(|t| (0..64).map(move |r| t.boundary_sign(r, true)))
            .filter(|&s| s == -1)
            .count(),
        16
    );
    let opt_path = square_opt_input("square-ap-host");
    let params = path.with_file_name("zqp_opt.dat");
    let mut cfg = square_config();
    cfg.initial_def = InitialDef::Path(params.clone());
    for model in [
        Model::prepare(&opt_path, &cfg).unwrap(),
        Model::prepare_physcal(&path, &square_config(), Some(&params)).unwrap(),
    ] {
        assert_eq!(
            (model.ns, model.nq, model.data.modpara.nmp_trans),
            (64, 32, -4)
        );
        let signs = expected.orbital_sgn_matrix.as_ref().unwrap();
        for q in 0..model.nq {
            for r in 0..model.ns {
                for s in 0..model.ns {
                    let (i, a) = mvmc_core::qp::translated_site(&expected, r, 0, q / 8, true);
                    let (j, b) = mvmc_core::qp::translated_site(&expected, s, 0, q / 8, true);
                    let k = (q * model.ns + r) * model.ns + s;
                    assert_eq!(model.sa[k], (signs[i][j] * a * b) as i32);
                    assert_eq!(model.sb[k], (signs[j][i] * a * b) as i32);
                }
            }
        }
    }
    std::fs::remove_dir_all(opt_path.parent().unwrap()).unwrap();
}

/// Reference tables are rebuilt by CPU C-order kernels from the original
/// negative translation flag, independently of the resident model's sign tables.
pub(crate) fn assert_square_snapshot(
    backend: &mut CudaEngine,
    state: &State,
    overlap: f64,
    energy: f64,
    energy_scale: f64,
) {
    let cfg = with_raw_exec(backend, "square.ap.cfg", |exec| {
        download::<i32>(exec, &state.resident.sampler.ele, 2 * state.model.ne)
    })
    .unwrap();
    let mut data = state.model.data.clone();
    data.modpara.nmp_trans = -4; // Original, hash-verified AP/P input contract.
    let reference = Model::from_data(data, state.model.seed).unwrap();
    let mut cpu = super::dh_tests::Cpu::new(&reference, &state.model.parameters);
    let e = cpu.evaluate(&cfg);
    let width = 2 * state.model.ns;
    let (slater, pf) = with_raw_exec(backend, "square.ap.first_divergence", |exec| {
        Ok((
            download::<f64>(
                exec,
                &state.resident.sampler.slater,
                state.model.nq * width * width,
            )?,
            download::<f64>(exec, &state.resident.sampler.pf, state.model.nq)?,
        ))
    })
    .unwrap();
    let mut first_slater = None;
    let mut slater_delta = 0f64;
    let mut slater_scale = 1f64;
    for q in 0..state.model.nq {
        for r in 0..width {
            for c in 0..width {
                let a = slater[(q * width + r) * width + c];
                let b = cpu.state.slater_matrix.slater_elm_real.get(q, r, c);
                let d = (a - b).abs();
                if d > 0. && first_slater.is_none() {
                    first_slater = Some((q, r, c, a, b));
                }
                slater_delta = slater_delta.max(d);
                slater_scale = slater_scale.max(a.abs()).max(b.abs());
            }
        }
    }
    assert!(slater_delta <= 4. * f64::EPSILON * slater_scale);
    let first_pf = pf
        .iter()
        .zip(&cpu.state.slater_matrix.pf_m_real)
        .position(|(a, b)| (a - b).abs() > 0.);
    let pf_delta = pf
        .iter()
        .zip(&cpu.state.slater_matrix.pf_m_real)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    println!("First FP stage: Slater max |delta|={slater_delta:e}, first={first_slater:?}; Pfaffian max |delta|={pf_delta:e}, first plane={first_pf:?}");
    let weights = &cpu.data.qp_weights.as_ref().unwrap().qp_full_weight;
    let scale = weights
        .iter()
        .zip(&cpu.state.slater_matrix.pf_m_real)
        .map(|(w, p)| (w.re * p).abs())
        .sum::<f64>();
    let rounding = super::complex_tests::budget(0, 2 * state.model.ne, e.kappa);
    let ip_bound = rounding * scale;
    assert!(
        ip_bound.is_finite() && ip_bound < 1e-5 * e.ip.abs(),
        "ill-conditioned projected overlap: kappa={}, scale={scale:e}, ip={:e}",
        e.kappa,
        e.ip
    );
    assert!(
        (overlap - e.ip).abs() <= ip_bound,
        "AP/P overlap {overlap:e} vs {:e}, bound={ip_bound:e}",
        e.ip
    );
    let energy_bound = 1e-13 + rounding * energy_scale;
    assert!(
        (energy - e.energy).abs() <= energy_bound,
        "AP/P energy {energy} vs {}, bound={energy_bound:e}",
        e.energy
    );
    // Reproduce the old normalization defect on the independent CPU path.
    // Require a physical effect larger than rounding on this very configuration.
    let mut unsigned = reference.data.clone();
    unsigned.modpara.nmp_trans = 4;
    let unsigned = Model::from_data(unsigned, state.model.seed).unwrap();
    let mut wrong = super::dh_tests::Cpu::new(&unsigned, &state.model.parameters);
    let idx: Vec<i64> = cfg.iter().map(|&r| i64::from(r)).collect();
    let wrong_ip = wrong.factor(&idx);
    let changed_ip = (wrong_ip - e.ip).abs() > 10. * ip_bound;
    if !changed_ip {
        let wrong_energy = wrong.evaluate(&cfg).energy;
        assert!((wrong_energy - e.energy).abs() > 10. * energy_bound);
        println!(
            "Dropping AP signs changes E by {:.17e}",
            wrong_energy - e.energy
        );
    } else {
        println!("Dropping AP signs changes IP by {:.17e}", wrong_ip - e.ip);
    }
    println!("AP/P square L8: ip={overlap:.17e}, CPU={:.17e}, |delta|={:.3e}, bound={ip_bound:.3e}; E={energy:.17e}, CPU={:.17e}, |delta|={:.3e}, bound={energy_bound:.3e}; kappa={:.3e}",e.ip,(overlap-e.ip).abs(),e.energy,(energy-e.energy).abs(),e.kappa);
}

#[test]
#[ignore = "requires CUDA; exact reported square AP/P input against CPU C-order"]
fn resident_opt_square_ap_p_matches_cpu_configuration() {
    let path = square_opt_input("square-ap-opt");
    let mut config = square_config();
    config.initial_def = InitialDef::Path(square_input().with_file_name("zqp_opt.dat"));
    let model = Model::prepare(&path, &config).unwrap();
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = State::prepare(&mut backend, model, &config).unwrap();
    state.initialize(&mut backend).unwrap();
    let (ip, energy, values) = with_raw_exec(&mut backend, "square.ap.opt", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        let error = download::<i32>(exec, &state.resident.error, 1)?;
        assert_eq!(error, [0]);
        Ok((
            download::<f64>(exec, &state.buffers.overlaps, 1)?[0],
            download::<f64>(exec, &state.buffers.local_energy, 1)?[0],
            download::<f64>(exec, &state.buffers.values, state.model.energies.len())?,
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
    assert_square_snapshot(&mut backend, &state, ip, energy, scale);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

fn input(tag: &str) -> std::path::PathBuf {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let path = edited_from(
        &base,
        tag,
        &[("modpara.def", "NMPTrans       -1", "NMPTrans       -4")],
    );
    // Four translations of a 16-site AP ring, by 0, 4, 8 and 12 sites.
    // Crossing the boundary contributes -1 (C QPTransSgn).
    let mut text = String::from("===\nNQPTrans 4\n===\n===\n===\n");
    for q in 0..4 {
        text.push_str(&format!("{q} 1.0\n"));
    }
    for q in 0..4 {
        for r in 0..16 {
            let t = r + 4 * q;
            text.push_str(&format!(
                "{q} {r} {} {}\n",
                t % 16,
                if t < 16 { 1 } else { -1 }
            ));
        }
    }
    std::fs::write(path.with_file_name("qptransidx.def"), text).unwrap();
    path
}

#[test]
fn resident_antiperiodic_tables_preserve_input_signs() {
    let path = input("ap-tables");
    let data = mvmc_expert_parsers::parse_expert_mode_files(&path).unwrap();
    assert_eq!(data.modpara.nmp_trans, -4);
    for model in [
        Model::prepare(&path, &cfg()).unwrap(),
        Model::prepare_physcal(&path, &cfg(), None).unwrap(),
    ] {
        assert_eq!(model.data.modpara.nmp_trans, -4);
        let nsp = data.modpara.nsp_gauss_leg as usize;
        let os = model.data.orbital_sgn_matrix.as_ref().unwrap();
        let mut negative = 0;
        for q in 0..model.nq {
            for r in 0..model.ns {
                for s in 0..model.ns {
                    let (i, a) = mvmc_core::qp::translated_site(&data, r, 0, q / nsp, true);
                    let (j, b) = mvmc_core::qp::translated_site(&data, s, 0, q / nsp, true);
                    let k = (q * model.ns + r) * model.ns + s;
                    assert_eq!(model.sa[k], (os[i][j] * a * b) as i32);
                    assert_eq!(model.sb[k], (os[j][i] * a * b) as i32);
                    negative += usize::from(a * b < 0);
                }
            }
        }
        assert!(negative > 0, "fixture must exercise AP boundary signs");
    }
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn resident_antiperiodic_fsz_and_opttrans_tables_preserve_signs() {
    for (name, base, old) in [
        (
            "ap-fsz",
            super::scope_tests::fsz_input("hubbard_fsz_L16"),
            "NMPTrans       2",
        ),
        (
            "ap-opttrans",
            super::opttrans_tests::input("hubbard_opttrans_L16"),
            "NMPTrans       1",
        ),
    ] {
        let path = edited_from(&base, name, &[("modpara.def", old, "NMPTrans       -4")]);
        let ap = input(&format!("{name}-translations"));
        std::fs::copy(
            ap.with_file_name("qptransidx.def"),
            path.with_file_name("qptransidx.def"),
        )
        .unwrap();
        for model in [
            Model::prepare(&path, &cfg()).unwrap(),
            Model::prepare_physcal(&path, &cfg(), None).unwrap(),
        ] {
            assert_eq!(model.data.modpara.nmp_trans, -4);
            let d = &model.data;
            let nsp = d.modpara.nsp_gauss_leg as usize;
            let nmp = 4;
            let width = model.ns * if model.fsz { 2 } else { 1 };
            let os = d.orbital_sgn_matrix.as_ref().unwrap();
            let mut negative = 0;
            for q in 0..model.nq {
                for r in 0..width {
                    for s in 0..width {
                        let (i, a) = mvmc_core::qp::translated_site(
                            d,
                            r % model.ns,
                            q / (nmp * nsp),
                            (q / nsp) % nmp,
                            true,
                        );
                        let (j, b) = mvmc_core::qp::translated_site(
                            d,
                            s % model.ns,
                            q / (nmp * nsp),
                            (q / nsp) % nmp,
                            true,
                        );
                        let (i, j) = (i + r / model.ns * model.ns, j + s / model.ns * model.ns);
                        let k = (q * width + r) * width + s;
                        assert_eq!(model.sa[k], (os[i][j] * a * b) as i32);
                        assert_eq!(model.sb[k], (os[j][i] * a * b) as i32);
                        negative += usize::from(a * b < 0);
                    }
                }
            }
            assert!(negative > 0);
        }
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
        std::fs::remove_dir_all(ap.parent().unwrap()).unwrap();
    }
}

#[test]
#[ignore = "requires CUDA; AP projected overlap, energy and derivatives against CPU C-order"]
fn resident_antiperiodic_observables_match_cpu() {
    let path = input("ap-cuda");
    let config = OptConfig {
        walkers: Some(1),
        warmup: Some(0),
        interval: Some(0),
        ..cfg()
    };
    let mut model = Model::prepare(&path, &config).unwrap();
    // Break the periodic seed orbital's translation symmetry: its AP
    // projection can otherwise sit on a node at initialization.
    for (k, p) in model.parameters.iter_mut().enumerate().skip(model.nproj) {
        *p = ((k * 37 % 101) as f64 - 50.) / 53.;
    }
    let mut cpu = super::dh_tests::Cpu::new(&model, &model.parameters);
    let mut periodic_data = model.data.clone();
    periodic_data.modpara.nmp_trans = 4;
    let periodic_model = Model::from_data(periodic_data, model.seed).unwrap();
    let mut periodic = super::dh_tests::Cpu::new(&periodic_model, &model.parameters);
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut state = State::prepare(&mut backend, model, &config).unwrap();
    state.initialize(&mut backend).unwrap();
    let m = &state.model;
    let (cfgs, overlaps, energy, derivatives) = with_raw_exec(&mut backend, "ap.observe", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        Ok((
            download::<i32>(exec, &state.resident.sampler.ele, 2 * m.ne)?,
            download::<f64>(exec, &state.buffers.overlaps, 1)?,
            download::<f64>(exec, &state.buffers.local_energy, 1)?,
            download::<f64>(exec, &state.buffers.derivatives, m.np)?,
        ))
    })
    .unwrap();
    for w in 0..1 {
        let e = cpu.evaluate(&cfgs[w * 2 * m.ne..(w + 1) * 2 * m.ne]);
        let wrong = periodic.evaluate(&cfgs[w * 2 * m.ne..(w + 1) * 2 * m.ne]);
        assert!(
            (wrong.energy - e.energy).abs() > 1e-4,
            "AP signs must affect this configuration"
        );
        println!(
            "walker {w}: ip GPU={} CPU={} E GPU={} CPU={} kappa={}",
            overlaps[w], e.ip, energy[w], e.energy, e.kappa
        );
        // Fresh FP64 factorization; condition-scaled rounding in overlap,
        // inverse and Hamiltonian ratios (same CPU comparison as DH fixtures).
        let rtol = 64. * (2 * m.ne) as f64 * f64::EPSILON * e.kappa;
        assert!(
            rtol < 1e-7,
            "ill-conditioned regression fixture: {}",
            e.kappa
        );
        close("AP overlap", &[overlaps[w]], &[e.ip], 1e-12, rtol);
        close("AP energy", &[energy[w]], &[e.energy], 1e-10, rtol);
        close(
            "AP derivatives",
            &derivatives[w * m.np..(w + 1) * m.np],
            &e.derivatives,
            1e-10,
            rtol,
        );
    }
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}
