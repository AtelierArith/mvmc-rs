//! Host-only checks of the resident GPU Opt/PhysCal support matrix (#572).
//!
//! `Model::prepare*` runs before `with_runtime`, so these tests open no CUDA
//! context and allocate no device memory. Each unsupported family must fail
//! there with its resident-scope error; the CPU validators must accept the
//! same input, so the rejection is GPU-specific rather than a generic error.
//! See `docs/design/gpu-resident-model-support.md`.
use super::*;
use mvmc_expert_parsers::types::Spin;
use std::fs;
use std::path::PathBuf;

/// `(file, from, to)` textual edit of a copied baseline input.
type Edit = (&'static str, &'static str, &'static str);

const SCOPE: &str = "resident GPU requires real normal orbitals";

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

/// Self-contained real six-site Hubbard chain accepted by the resident runners.
fn baseline() -> PathBuf {
    repo("tests/fixtures/complex_uhf/hubbard_chain_real/input/namelist.def")
}

/// Copy the baseline and apply `(file, from, to)` textual edits. An empty
/// `from` appends `to` (creating the file if it does not exist).
fn edited(tag: &str, edits: &[Edit]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-scope-572-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(baseline().parent().unwrap()).unwrap() {
        let source = entry.unwrap().path();
        if source.is_file() {
            fs::copy(&source, dir.join(source.file_name().unwrap())).unwrap();
        }
    }
    for (file, from, to) in edits {
        let path = dir.join(file);
        let text = fs::read_to_string(&path).unwrap_or_default();
        if from.is_empty() {
            fs::write(&path, text + to).unwrap();
            continue;
        }
        assert!(text.contains(from), "{tag}: {file} lacks {from:?}");
        fs::write(&path, text.replacen(from, to, 1)).unwrap();
    }
    dir.join("namelist.def")
}

fn cfg() -> OptConfig {
    OptConfig {
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

fn parse(path: &Path) -> mvmc_core::ExpertModeData {
    mvmc_expert_parsers::parse_expert_mode_files(path).unwrap()
}

#[test]
fn resident_scope_accepts_real_normal_baseline() {
    let input = baseline();
    let opt = Model::prepare(&input, &cfg()).unwrap();
    let phys = Model::prepare_physcal(&input, &cfg(), None).unwrap();
    assert_eq!((opt.ns, opt.ne), (6, 3));
    assert_eq!(opt.nq, 8, "NSPGaussLeg=8 x |NMPTrans|=1 QP sectors");
    assert_eq!(phys.np, opt.np);
}

#[test]
fn resident_scope_rejects_unsupported_families_before_cuda() {
    let edits: &[(&str, &[Edit])] = &[
        (
            "two-sz",
            &[("modpara.def", "2Sz            0", "2Sz            2")],
        ),
        (
            "nsplit",
            &[("modpara.def", "NSplitSize     1", "NSplitSize     2")],
        ),
        (
            "complex-orbital",
            &[(
                "orbitalidx.def",
                "ComplexType          0",
                "ComplexType          1",
            )],
        ),
    ];
    let mut cases: Vec<(String, PathBuf, bool)> = edits
        .iter()
        .map(|(tag, e)| (tag.to_string(), edited(tag, e), true))
        .collect();
    // Fixture inputs whose only resident-unsupported feature is the named
    // family (real six-site chains), except: the FSZ Hubbard chain
    // (OrbitalParallel selects general orbitals, complex orbital
    // coefficients); and the FSZ Kondo chain (NExUpdatePath=3).
    for (tag, rel) in [
        (
            "opttrans",
            "tests/fixtures/opttrans/run_opt_real/namelist.def",
        ),
        (
            "fsz-general-orbital",
            "tests/fixtures/native13_initialization/inputs/HubbardChain_fsz/namelist.def",
        ),
        (
            "kondo-local-spin-flip",
            "tests/fixtures/native13_initialization/inputs/KondoChain_fsz/namelist.def",
        ),
    ] {
        cases.push((tag.into(), repo(rel), false));
    }
    for (tag, input, temporary) in cases {
        let data = parse(&input);
        // The ordinary C-order validators accept every case.
        mvmc_core::validation::validate_para_opt(&data)
            .unwrap_or_else(|e| panic!("{tag}: CPU Opt rejects: {e}"));
        mvmc_core::validation::validate_phys_cal(&data)
            .unwrap_or_else(|e| panic!("{tag}: CPU PhysCal rejects: {e}"));
        for (route, result) in [
            ("Opt", Model::prepare(&input, &cfg()).map(|_| ())),
            (
                "PhysCal",
                Model::prepare_physcal(&input, &cfg(), None).map(|_| ()),
            ),
        ] {
            let error = result.expect_err(&format!("{tag}: resident {route} accepted"));
            assert!(error.contains(SCOPE), "{tag}/{route}: {error}");
        }
        if temporary {
            fs::remove_dir_all(input.parent().unwrap()).unwrap();
        }
    }
}

/// Julia GPU PhysCal accepts and ignores `NSRCG` (`ext/cuda_runners.jl:27`):
/// PhysCal never runs SR. Resident Opt keeps rejecting it (direct SR only).
#[test]
fn resident_physcal_ignores_nsrcg_while_opt_rejects_it() {
    let input = edited(
        "sr-cg",
        &[("modpara.def", "NSRCG          0", "NSRCG          1")],
    );
    let data = parse(&input);
    mvmc_core::validation::validate_para_opt(&data).unwrap();
    mvmc_core::validation::validate_phys_cal(&data).unwrap();
    let error = Model::prepare(&input, &cfg()).map(|_| ()).unwrap_err();
    assert!(error.contains(SCOPE), "Opt: {error}");
    let phys = Model::prepare_physcal(&input, &cfg(), None).unwrap();
    let base = Model::prepare_physcal(&baseline(), &cfg(), None).unwrap();
    assert_eq!(phys.data.modpara.nsrcg, 1);
    // NSRCG changes nothing that PhysCal uploads.
    assert_eq!(phys.parameters, base.parameters);
    assert_eq!(
        (phys.np, phys.nq, phys.energies.len()),
        (base.np, base.nq, base.energies.len())
    );
    fs::remove_dir_all(input.parent().unwrap()).unwrap();
}

/// #660: local spins and NExUpdatePath 1/2 follow Julia GPU
/// (`cuda_state.jl:47`, `cuda_runners.jl:52-58`, `get_loc_spn_array`).
#[test]
fn resident_scope_accepts_local_spins_and_exchange() {
    let exchange = edited(
        "exchange-path",
        &[("modpara.def", "NExUpdatePath  0", "NExUpdatePath  1")],
    );
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    for (input, probability, nlocal, ne) in [
        (exchange.clone(), 0.5, 0, 3),
        (
            repo("tests/fixtures/original_heisenberg_parser_184/namelist.def"),
            1.0,
            16,
            8,
        ),
        (
            local("tests/fixtures/localspin/heisenberg_chain_L64/namelist.def"),
            1.0,
            64,
            32,
        ),
        (
            local("tests/fixtures/localspin/kondo_chain_L64/namelist.def"),
            0.5,
            64,
            64,
        ),
    ] {
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            assert_eq!(model.exchange, probability, "{input:?}");
            assert_eq!(model.ne, ne, "{input:?}");
            assert_eq!(model.localspins.len(), model.ns);
            assert_eq!(model.localspins.iter().filter(|&&x| x == 1).count(), nlocal);
        }
    }
    let kondo = Model::prepare(
        &local("tests/fixtures/localspin/kondo_chain_L64/namelist.def"),
        &cfg(),
    )
    .unwrap();
    assert!(kondo.localspins[..64].iter().all(|&x| x == 0));
    assert!(kondo.localspins[64..].iter().all(|&x| x == 1));
    fs::remove_dir_all(exchange.parent().unwrap()).unwrap();
}

/// Julia `initial_gpu_state`: `nlocal <= 2Ne <= 2Nsite - nlocal`
/// (`cuda_initial_sampling.jl:87`), checked on the host before CUDA.
#[test]
fn resident_scope_rejects_infeasible_local_spin_filling() {
    let model = Model::prepare(&baseline(), &cfg()).unwrap();
    let mut data = model.data.clone();
    // Six local spins cannot all be singly occupied by 2Ne = 4 electrons.
    data.modpara.nelec = 2;
    data.locspin_terms = (0..6)
        .map(|site| mvmc_expert_parsers::types::LocSpinTerm {
            site,
            spin_value: 1,
        })
        .collect();
    let error = Model::from_data(data, 79).map(|_| ()).unwrap_err();
    assert!(error.contains("infeasible local-spin filling"), "{error}");
}

#[test]
fn resident_physcal_rejects_lanczos_before_cuda() {
    let input = edited(
        "lanczos",
        &[
            ("modpara.def", "NVMCCalMode    0", "NVMCCalMode    1"),
            ("modpara.def", "NLanczosMode   0", "NLanczosMode   1"),
        ],
    );
    mvmc_core::validation::validate_phys_cal(&parse(&input)).unwrap();
    let error = Model::prepare_physcal(&input, &cfg(), None)
        .map(|_| ())
        .unwrap_err();
    assert!(error.contains("NLanczosMode=0"), "{error}");
    fs::remove_dir_all(input.parent().unwrap()).unwrap();
}

#[test]
fn resident_model_rejects_spin_changing_operators_before_cuda() {
    let model = Model::prepare(&baseline(), &cfg()).unwrap();
    let mut data = model.data.clone();
    let t = &mut data.transfer_terms[0];
    (t.spin1, t.spin2) = (Spin::Up, Spin::Down);
    let error = Model::from_data(data, 79).map(|_| ()).unwrap_err();
    assert!(
        error.contains("Transfer requires spin conservation"),
        "{error}"
    );

    let mut data = model.data.clone();
    data.inter_all_terms
        .push(mvmc_expert_parsers::types::InterAllTerm {
            site0: 0,
            spin0: 0,
            site1: 1,
            spin1: 1,
            site2: 1,
            spin2: 1,
            site3: 0,
            spin3: 0,
            value: num_complex::Complex64::new(0.25, 0.),
            is_complex: false,
        });
    let error = Model::from_data(data, 79).map(|_| ()).unwrap_err();
    assert!(
        error.contains("InterAll requires spin-conserving"),
        "{error}"
    );
}

/// #665: DH2/DH4 follow the C counters. The model keeps the C parameter layout
/// (Gutzwiller, Jastrow, DH2, DH4, Slater), the neighbor tables in center
/// order, and C SetFlagShift.
#[test]
fn resident_scope_accepts_doublon_holon_factors() {
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    for (input, n2, n4, shifts) in [
        (
            repo("tests/fixtures/dh2/production_real/namelist.def"),
            1,
            0,
            (true, false),
        ),
        (
            repo("tests/fixtures/dh4/production_dh4_real/namelist.def"),
            0,
            1,
            (false, true),
        ),
        (
            local("tests/fixtures/dh/hubbard_dh_L16/namelist.def"),
            2,
            1,
            (true, true),
        ),
        (
            local("tests/fixtures/dh/hubbard_dh_L128/namelist.def"),
            1,
            1,
            (true, true),
        ),
    ] {
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            let data = &model.data;
            assert_eq!((model.n2, model.n4), (n2, n4), "{input:?}");
            assert_eq!(model.nproj, data.projection_layout().n_proj);
            assert_eq!(model.nproj, model.ng + model.nj + 6 * n2 + 10 * n4);
            assert_eq!((model.shift_dh2, model.shift_dh4), shifts, "{input:?}");
            let expected: Vec<i32> = data.doublon_holon_2site_indices[..n2]
                .iter()
                .flat_map(|t| t.neighbors.iter().flatten())
                .chain(
                    data.doublon_holon_4site_indices[..n4]
                        .iter()
                        .flat_map(|t| t.neighbors.iter().flatten()),
                )
                .map(|&r| r as i32)
                .collect();
            assert_eq!(model.dh, expected);
            let projection: Vec<f64> = data.projection_parameters().iter().map(|p| p.re).collect();
            assert_eq!(model.parameters[..model.nproj], projection[..]);
            // InDH2/InDH4 overlays survive the initial C-order shift.
            assert!(model.parameters[model.ng + model.nj..model.nproj]
                .iter()
                .any(|&p| p != 0.0));
            // Slater parameter indices start after the DH block.
            assert!(model
                .pa
                .iter()
                .chain(&model.pb)
                .all(|&p| p as usize >= model.nproj));
        }
    }
}

const CHARGE_RBM: &str = "===\nNChargeRBM_PhysLayer 1\nComplexType 0\n===\n===\n\
                          0 0\n1 0\n2 0\n3 0\n4 0\n5 0\n0 1\n";

/// Host emulation of `resident_rbm_make` (C MakeRBMCnt) from the device tables.
fn table_theta(model: &Model, num: &[i64]) -> Vec<f64> {
    let (ns, h) = (model.ns, model.nneuron);
    let t = &model.rbm;
    let w = &model.parameters[model.nproj..model.nproj + model.nrbm];
    (0..h)
        .map(|k| {
            let family = t[4 * ns + k];
            let bias = t[4 * ns + h + k];
            let c = &t[4 * ns + 2 * h + k * 2 * ns..][..2 * ns];
            let mut theta = 0.0;
            if bias >= 0 {
                theta += w[bias as usize];
            }
            if c[0] >= 0 {
                let width = if family == 2 { 2 * ns } else { ns };
                let mut sum = 0.0;
                for rs in 0..width {
                    let x = match family {
                        0 => num[rs] + num[rs + ns] - 1,
                        1 => num[rs] - num[rs + ns],
                        _ => 2 * num[rs] - 1,
                    };
                    sum += w[c[rs] as usize] * x as f64;
                }
                theta += sum;
            }
            theta
        })
        .collect()
}

/// #667: RBM factors keep the C parameter layout (projection, RBM, Slater).
/// The device tables reproduce C MakeRBMCnt exactly on random occupations,
/// and the In*RBM overlays reach the uploaded coefficients.
#[test]
fn resident_scope_accepts_rbm_factors() {
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    // Only a physical layer and one charge neuron without hidden or coupling
    // sections: its field stays 0. (C's real initializer divides by Nneuron,
    // so zero neurons give nonfinite coefficients, which are rejected.)
    let minimal = edited(
        "rbm-minimal",
        &[
            ("namelist.def", "", "ChargeRBM_PhysLayer chargerbm.def\n"),
            ("chargerbm.def", "", CHARGE_RBM),
            ("modpara.def", "", "\nNneuronCharge  1\n"),
        ],
    );
    for (input, nrbm, nneuron, overlays) in [
        (
            local("tests/fixtures/rbm/hubbard_rbm_L16/namelist.def"),
            76,
            10,
            true,
        ),
        (
            local("tests/fixtures/rbm/hubbard_rbm_L128/namelist.def"),
            524,
            10,
            true,
        ),
        (minimal.clone(), 1, 1, false),
    ] {
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            let data = &model.data;
            assert_eq!((model.nrbm, model.nneuron), (nrbm, nneuron), "{input:?}");
            assert_eq!(model.nrbm, data.count_rbm_parameters());
            assert_eq!(
                model.rbm.len(),
                4 * model.ns + 2 * nneuron + 2 * model.ns * nneuron
            );
            // Slater parameter indices start after the RBM block.
            assert!(model
                .pa
                .iter()
                .chain(&model.pb)
                .all(|&p| p as usize >= model.nproj + model.nrbm));
            let rbm: Vec<f64> = data.rbm_parameters().iter().map(|p| p.re).collect();
            assert_eq!(
                model.parameters[model.nproj..model.nproj + model.nrbm],
                rbm[..]
            );
            assert!(
                !overlays || rbm.iter().all(|&p| p != 0.0),
                "In*RBM overlays"
            );
            let config = mvmc_core::sampling::rbm::RbmConfig::from(data);
            let mut state = 0x9e3779b97f4a7c15u64;
            for _ in 0..16 {
                let mut num = vec![0i64; 2 * model.ns];
                for x in &mut num {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *x = (state >> 63) as i64;
                }
                let cnt = mvmc_core::sampling::rbm::make_rbm_cnt(&num, &config);
                let expected: Vec<u64> = cnt[cnt.len() - nneuron..]
                    .iter()
                    .map(|z| z.re.to_bits())
                    .collect();
                let actual: Vec<u64> = table_theta(&model, &num)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                assert_eq!(
                    actual, expected,
                    "{input:?}: MakeRBMCnt from the device tables"
                );
            }
        }
    }
    fs::remove_dir_all(minimal.parent().unwrap()).unwrap();
}

/// Complex RBM coefficients are rejected before CUDA, as other complex parameters.
#[test]
fn resident_scope_rejects_complex_rbm_coefficients() {
    let input = edited(
        "rbm-complex",
        &[
            (
                "namelist.def",
                "",
                "ChargeRBM_PhysLayer chargerbm.def\nInChargeRBM_PhysLayer inchargerbm.def\n",
            ),
            ("chargerbm.def", "", CHARGE_RBM),
            ("chargerbm.def", "ComplexType 0", "ComplexType 1"),
            (
                "inchargerbm.def",
                "",
                "===\nNChargeRBM_PhysLayer 1\n===\n===\n===\n0 0.125 0.0625\n",
            ),
        ],
    );
    let data = parse(&input);
    mvmc_core::validation::validate_para_opt(&data).unwrap();
    for result in [
        Model::prepare(&input, &cfg()).map(|_| ()),
        Model::prepare_physcal(&input, &cfg(), None).map(|_| ()),
    ] {
        let error = result.expect_err("complex RBM accepted");
        assert!(error.contains("finite and real"), "{error}");
    }
    fs::remove_dir_all(input.parent().unwrap()).unwrap();
}
