//! AP translation projection regression (#730); CPU C-order physics reference.
use super::complex_tests::download;
use super::scope_tests::{cfg, edited_from};
use super::tests::close;
use super::*;

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
