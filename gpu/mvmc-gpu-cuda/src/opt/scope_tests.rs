//! Host-only checks of the resident GPU Opt/PhysCal support matrix (#572).
//!
//! `Model::prepare*` runs before `with_runtime`, so these tests open no CUDA
//! context and allocate no device memory. Each unsupported family must fail
//! there with its resident-scope error; the CPU validators must accept the
//! same input, so the rejection is GPU-specific rather than a generic error.
//! See `docs/design/gpu-resident-model-support.md`.
use super::*;
use crate::opt::model::flag_contract;
use crate::opt::rbm_tests::assert_theta_close;
use mvmc_core::ExpertModeData;
use mvmc_expert_parsers::types::Spin;
use std::fs;
use std::path::PathBuf;

/// `(file, from, to)` textual edit of a copied baseline input.
type Edit = (&'static str, &'static str, &'static str);

const SCOPE: &str = "resident GPU requires NExUpdatePath<=2";

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
pub(crate) fn edited(tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
    edited_from(&baseline(), tag, edits)
}

/// [`edited`] for another input directory (`input` is its `namelist.def`).
pub(crate) fn edited_from(input: &Path, tag: &str, edits: &[(&str, &str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-scope-572-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(input.parent().unwrap()).unwrap() {
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

pub(crate) fn cfg() -> OptConfig {
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
    ];
    let mut cases: Vec<(String, PathBuf, bool)> = edits
        .iter()
        .map(|(tag, e)| (tag.to_string(), edited(tag, e), true))
        .collect();
    // Fixture inputs whose only resident-unsupported feature is the named
    // family: the FSZ Kondo chain (NExUpdatePath=3, general orbitals). The
    // normal "two-sz" edit keeps normal orbitals with 2Sz=2, which C rejects
    // (readdef.c:691-696) and the resident scope rejects with it.
    cases.push((
        "kondo-local-spin-flip".into(),
        repo("tests/fixtures/native13_initialization/inputs/KondoChain_fsz/namelist.def"),
        false,
    ));
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
/// PhysCal never runs SR. Resident Opt accepts it and runs C SR-CG (#670);
/// `NSRCG` changes no uploaded model data on either route.
#[test]
fn resident_accepts_nsrcg_for_physcal_and_opt() {
    let input = edited(
        "sr-cg",
        &[("modpara.def", "NSRCG          0", "NSRCG          1")],
    );
    let data = parse(&input);
    mvmc_core::validation::validate_para_opt(&data).unwrap();
    mvmc_core::validation::validate_phys_cal(&data).unwrap();
    let opt = Model::prepare(&input, &cfg()).unwrap();
    let opt_base = Model::prepare(&baseline(), &cfg()).unwrap();
    assert_eq!(opt.data.modpara.nsrcg, 1);
    assert_eq!(opt.parameters, opt_base.parameters);
    assert_eq!(opt.flags, opt_base.flags);
    let phys = Model::prepare_physcal(&input, &cfg(), None).unwrap();
    let base = Model::prepare_physcal(&baseline(), &cfg(), None).unwrap();
    assert_eq!(phys.data.modpara.nsrcg, 1);
    assert_eq!(phys.parameters, base.parameters);
    assert_eq!(
        (phys.np, phys.nq, phys.energies.len()),
        (base.np, base.nq, base.energies.len())
    );
    fs::remove_dir_all(input.parent().unwrap()).unwrap();
}

/// The enqueued CG bound is C's `max_iter` when `NSROptCGMaxIter > 0`, else
/// `NPara` (an upper bound of `nSmat`); out-of-range limits fail before allocation.
#[test]
fn resident_cg_iteration_bound_follows_c_max_iter() {
    assert_eq!(cg::iteration_limit(0, 120).unwrap(), 120);
    assert_eq!(cg::iteration_limit(-3, 120).unwrap(), 120);
    assert_eq!(cg::iteration_limit(7, 120).unwrap(), 7);
    assert_eq!(cg::iteration_limit(500, 120).unwrap(), 500);
    assert!(cg::iteration_limit(i32::MAX as i64, 120).is_err());
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
/// The device tables reproduce C MakeRBMCnt on random occupations within the
/// `rbm_tests::theta_bound` rounding bound (#699),
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
                let expected: Vec<f64> = cnt[cnt.len() - nneuron..].iter().map(|z| z.re).collect();
                // A fresh MakeRBMCnt on both sides: the rounding bound of
                // `rbm_tests::theta_bound` with no incremental updates (#699).
                assert_theta_close(
                    &format!("{input:?}: MakeRBMCnt from the device tables"),
                    &model,
                    &rbm,
                    &table_theta(&model, &num),
                    &expected,
                    &[0],
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

const OPT_TRANS: &str = "=============================================
NQPOptTrans 2
ComplexType 0
=============================================
=============================================
0 0.75
1 -0.5
0 0 0 1
0 1 1 1
0 2 2 1
0 3 3 1
0 4 4 1
0 5 5 1
1 0 5 1
1 1 4 1
1 2 3 1
1 3 2 1
1 4 1 1
1 5 0 1
";

/// OptTrans (#674): the weights follow C's layout after projection, RBM and
/// Slater, the planes are `NQPFix * NQPOptTrans` with the full weights of C
/// UpdateQPWeight, and the setup normalization of C SyncModifiedParameter.
/// A complex OptTrans weight is rejected before CUDA.
#[test]
fn resident_scope_accepts_opttrans() {
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let two = edited(
        "opttrans-two",
        &[
            ("namelist.def", "", "OptTrans opttrans.def\n"),
            ("opttrans.def", "", OPT_TRANS),
        ],
    );
    for (input, nopt, nfix) in [
        (
            repo("tests/fixtures/opttrans/run_opt_real/namelist.def"),
            3,
            8,
        ),
        (two.clone(), 2, 8),
        (
            local("tests/fixtures/opttrans/hubbard_opttrans_L16/namelist.def"),
            3,
            8,
        ),
        (
            local("tests/fixtures/opttrans/hubbard_opttrans_L128/namelist.def"),
            3,
            8,
        ),
    ] {
        let data = parse(&input);
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        mvmc_core::validation::validate_phys_cal(&data).unwrap();
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            let tail = model.np - model.nopt;
            assert_eq!((model.nopt, model.fix.len()), (nopt, nfix), "{input:?}");
            assert_eq!(model.nq, nopt * nfix);
            assert_eq!(
                tail,
                model.nproj + model.nrbm + model.data.slater_params.len()
            );
            // Slater parameter indices end before the OptTrans block.
            assert!(model
                .pa
                .iter()
                .chain(&model.pb)
                .all(|&p| (p as usize) >= model.nproj + model.nrbm && (p as usize) < tail));
            // Normalized to maximum amplitude one; the definition flags
            // optimize every real component.
            let opt = &model.parameters[tail..];
            // C normalization multiplies by 1/max: the largest amplitude is
            // 1 within the two roundings of the reciprocal and the product.
            let largest = opt.iter().fold(0f64, |a, x| a.max(x.abs()));
            assert!((largest - 1.0).abs() <= 2.0 * f64::EPSILON, "{largest}");
            assert!(model.flags[tail..].iter().all(|&f| f == 1));
            for (i, &o) in opt.iter().enumerate() {
                for (j, &f) in model.fix.iter().enumerate() {
                    // One rounded product on each side (C UpdateQPWeight):
                    // at most one ulp of relative difference.
                    let (a, b) = (model.weights[i * nfix + j], o * f);
                    assert!(
                        a.is_finite() && (a - b).abs() <= f64::EPSILON * a.abs().max(b.abs()),
                        "C UpdateQPWeight: {a} vs {b}"
                    );
                }
            }
        }
    }
    fs::remove_dir_all(two.parent().unwrap()).unwrap();
    let complex = edited(
        "opttrans-complex",
        &[
            (
                "namelist.def",
                "",
                "OptTrans opttrans.def\nInOptTrans inopttrans.def\n",
            ),
            ("opttrans.def", "", OPT_TRANS),
            (
                "inopttrans.def",
                "",
                "=============================================
NQPOptTrans 2
=============================================
=============================================
=============================================
0 0.75 0.0
1 -0.5 0.25
",
            ),
        ],
    );
    for result in [
        Model::prepare(&complex, &cfg()).map(|_| ()),
        Model::prepare_physcal(&complex, &cfg(), None).map(|_| ()),
    ] {
        let error = result.expect_err("complex OptTrans weight accepted");
        assert!(error.contains("finite and real"), "{error}");
    }
    fs::remove_dir_all(complex.parent().unwrap()).unwrap();
}

const COMPLEX_ORBITAL: Edit = (
    "orbitalidx.def",
    "ComplexType          0",
    "ComplexType          1",
);

/// Complex normal orbitals (#678): C `AllComplexFlag != 0` selects the
/// complex path, with interleaved complex parameters, C's 2*NPara component
/// flags, complex QP weights and complex Hamiltonian coefficients.
#[test]
fn resident_scope_accepts_complex_normal_orbitals() {
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let edited_baseline = edited("complex-orbital", &[COMPLEX_ORBITAL]);
    for (input, complex_weights) in [
        (edited_baseline.clone(), false),
        (
            local("tests/fixtures/complex/hubbard_complex_L16/namelist.def"),
            true,
        ),
        (
            local("tests/fixtures/complex/hubbard_complex_L128/namelist.def"),
            true,
        ),
    ] {
        let data = parse(&input);
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        mvmc_core::validation::validate_phys_cal(&data).unwrap();
        assert!(mvmc_core::get_all_complex_flag(&data).unwrap());
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            assert!(model.complex, "{input:?}");
            assert_eq!(model.lanes(), 2);
            assert_eq!(model.parameters.len(), 2 * model.np);
            assert_eq!(model.flags.len(), 2 * model.np);
            assert_eq!(model.weights.len(), 2 * model.nq);
            assert_eq!(model.energies.len(), 2 * model.operators.len() / 7);
            assert_eq!(model.sr_dimension(), 2 * model.np);
            // Packed C parameters, in interleaved (re, im) order.
            let packed = mvmc_core::pack_parameters(&model.data).unwrap();
            for (p, z) in packed.iter().enumerate() {
                assert_eq!(
                    [model.parameters[2 * p], model.parameters[2 * p + 1]],
                    [z.re, z.im]
                );
            }
            assert_eq!(
                model.projection(),
                (0..model.nproj).map(|p| packed[p].re).collect::<Vec<_>>()
            );
            // Random complex Slater initialization (C InitParameter).
            assert!((model.nproj..model.np).any(|p| model.parameters[2 * p + 1] != 0.0));
            let imaginary = |v: &[f64]| v.iter().skip(1).step_by(2).any(|&x| x != 0.0);
            assert_eq!(imaginary(&model.weights), complex_weights, "{input:?}");
            if complex_weights {
                assert!(imaginary(&model.energies), "complex Transfer/InterAll");
            }
        }
    }
    fs::remove_dir_all(edited_baseline.parent().unwrap()).unwrap();
}

/// Families without complex resident kernels stay rejected before CUDA in
/// complex mode: RBM, OptTrans, DH2/DH4, local spins and SR-CG for Opt
/// (PhysCal ignores NSRCG). A complex `ParaQPTrans` weight is still rejected
/// in real mode, where C takes its real part (qp_real.c).
#[test]
fn resident_scope_rejects_complex_mode_families_before_cuda() {
    const FAMILIES: &str = "resident GPU complex mode does not support";
    let opttrans: [Edit; 2] = [
        ("namelist.def", "", "OptTrans opttrans.def\n"),
        ("opttrans.def", "", OPT_TRANS),
    ];
    let rbm: [Edit; 3] = [
        (
            "namelist.def",
            "",
            "ChargeRBM_PhysLayer chargerbm.def\nInChargeRBM_PhysLayer inchargerbm.def\n",
        ),
        ("chargerbm.def", "", CHARGE_RBM),
        (
            "inchargerbm.def",
            "",
            "===\nNChargeRBM_PhysLayer 1\n===\n===\n===\n0 0.125 0.0\n",
        ),
    ];
    let cases: Vec<(&str, Vec<Edit>)> = vec![
        (
            "complex-opttrans",
            [&[COMPLEX_ORBITAL][..], &opttrans].concat(),
        ),
        ("complex-rbm", [&[COMPLEX_ORBITAL][..], &rbm].concat()),
    ];
    for (tag, edits) in cases {
        let input = edited(tag, &edits);
        let data = parse(&input);
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        for result in [
            Model::prepare(&input, &cfg()).map(|_| ()),
            Model::prepare_physcal(&input, &cfg(), None).map(|_| ()),
        ] {
            let error = result.expect_err(&format!("{tag} accepted"));
            assert!(error.contains(FAMILIES), "{tag}: {error}");
        }
        fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
    // DH2 and local spins: complex variants of fixtures the real path accepts.
    let dh = repo("tests/fixtures/dh2/production_cmp/namelist.def");
    let data = parse(&dh);
    assert!(mvmc_core::get_all_complex_flag(&data).unwrap());
    let error = Model::prepare(&dh, &cfg())
        .map(|_| ())
        .expect_err("complex DH2");
    assert!(error.contains(FAMILIES), "{error}");
    let local = |rel: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let kondo = local("tests/fixtures/localspin/kondo_chain_L64/namelist.def");
    let dir = std::env::temp_dir().join(format!("mvmc-scope-678-kondo-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(kondo.parent().unwrap()).unwrap() {
        let source = entry.unwrap().path();
        if source.is_file() {
            fs::copy(&source, dir.join(source.file_name().unwrap())).unwrap();
        }
    }
    let path = dir.join("orbitalidx.def");
    let text = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        text.replacen("ComplexType          0", "ComplexType          1", 1),
    )
    .unwrap();
    let input = dir.join("namelist.def");
    assert!(mvmc_core::get_all_complex_flag(&parse(&input)).unwrap());
    let error = Model::prepare(&input, &cfg())
        .map(|_| ())
        .expect_err("complex Kondo");
    assert!(error.contains(FAMILIES), "{error}");
    fs::remove_dir_all(dir).unwrap();
    // SR-CG: rejected for complex Opt, ignored by complex PhysCal.
    let cg = edited(
        "complex-cg",
        &[
            COMPLEX_ORBITAL,
            ("modpara.def", "NSRCG          0", "NSRCG          1"),
        ],
    );
    let error = Model::prepare(&cg, &cfg())
        .map(|_| ())
        .expect_err("complex CG Opt");
    assert!(error.contains("complex Opt requires direct SR"), "{error}");
    assert!(Model::prepare_physcal(&cg, &cfg(), None).unwrap().complex);
    fs::remove_dir_all(cg.parent().unwrap()).unwrap();
    // Real mode keeps rejecting a complex TransSym weight.
    let weight = edited(
        "real-complex-weight",
        &[("qptransidx.def", "0    1.00000", "0 0.6 0.8")],
    );
    let data = parse(&weight);
    assert!(!mvmc_core::get_all_complex_flag(&data).unwrap());
    assert!(data.para_qp_trans.iter().any(|w| w.im != 0.0));
    for result in [
        Model::prepare(&weight, &cfg()).map(|_| ()),
        Model::prepare_physcal(&weight, &cfg(), None).map(|_| ()),
    ] {
        let error = result.expect_err("real-mode complex weight accepted");
        assert!(error.contains("complex or nonfinite"), "{error}");
    }
    fs::remove_dir_all(weight.parent().unwrap()).unwrap();
}

/// CPU shift decisions observed from the C-order sync itself: with nonzero
/// bin means, `sync_modified_parameter_local` moves DH2/DH4 only when it
/// applies the DH gauge, and Jastrow only when it applies the GJ shift.
fn cpu_shifts(data: &ExpertModeData) -> (bool, bool, bool) {
    let mut seeded = data.clone();
    for (k, p) in seeded.doublon_holon_2site_params.iter_mut().enumerate() {
        p.re = 0.003 + 0.01 * k as f64;
    }
    for (k, p) in seeded.doublon_holon_4site_params.iter_mut().enumerate() {
        p.re = -0.002 + 0.007 * k as f64;
    }
    for (k, t) in seeded.jastrow_terms.iter_mut().enumerate() {
        t.value.re = 0.05 + 0.01 * k as f64;
    }
    let mut synced = seeded.clone();
    mvmc_core::sync::sync_modified_parameter_local(&mut synced, true);
    let jastrow = |d: &ExpertModeData| d.jastrow_terms.iter().map(|t| t.value).collect::<Vec<_>>();
    // Exact comparisons of copied values: an unshifted parameter is untouched.
    (
        !seeded.jastrow_terms.is_empty() && jastrow(&seeded) != jastrow(&synced),
        seeded.doublon_holon_2site_params != synced.doublon_holon_2site_params,
        seeded.doublon_holon_4site_params != synced.doublon_holon_4site_params,
    )
}

/// #700: resident SR activity and the GJ/DH2/DH4 gauge predicates follow the
/// raw exact-one C OptFlag contract (stcopt.c:112, parameter.c:255-313), as the
/// CPU `component_is_optimized` and `sync_modified_parameter` do. Signed flags
/// `2` and `-1` stay parsed and mean "fixed", like `0`.
#[test]
fn resident_flags_follow_cpu_exact_one_semantics() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dh/hubbard_dh_L16/namelist.def");
    let base = Model::prepare(&input, &cfg()).unwrap();
    let (ng, nj, n2, n4) = (base.ng, base.nj, base.n2, base.n4);
    assert!(ng > 0 && nj > 0 && n2 > 0 && n4 > 0);
    let (dh2, dh4) = (ng + nj, ng + nj + 6 * n2);
    // (label, real-component parameter indices, raw flag)
    let mut cases: Vec<(String, Vec<usize>, i64)> = vec![("all one".into(), vec![], 1)];
    for flag in [0, 1, 2, -1] {
        cases.push((format!("gutzwiller 0 = {flag}"), vec![0], flag));
        cases.push((format!("jastrow 0 = {flag}"), vec![ng], flag));
        cases.push((format!("dh2 first = {flag}"), vec![dh2], flag));
        cases.push((format!("dh2 last = {flag}"), vec![dh4 - 1], flag));
        cases.push((format!("dh4 middle = {flag}"), vec![dh4 + 5], flag));
        // Mixed active/fixed: one DH block fixed, the other still active.
        cases.push((format!("dh2 block = {flag}"), (dh2..dh4).collect(), flag));
        cases.push((
            format!("dh4 block = {flag}"),
            (dh4..base.nproj).collect(),
            flag,
        ));
    }
    for (label, params, flag) in cases {
        let mut data = base.data.clone();
        for &p in &params {
            data.optimization_flags[2 * p] = flag;
        }
        let model = Model::from_data(data.clone(), base.seed).unwrap();
        assert_eq!(model.flags.len(), model.np, "{label}");
        for p in 0..model.np {
            let cpu = mvmc_core::sr::component_is_optimized(&data, 2 * p);
            assert!(matches!(model.flags[p], 0 | 1), "{label}");
            assert_eq!(model.flags[p] == 1, cpu, "{label}: SR activity of {p}");
        }
        assert_eq!(
            (model.shift, model.shift_dh2, model.shift_dh4),
            cpu_shifts(&data),
            "{label}: GJ/DH2/DH4 shift"
        );
        // Independent statement of the expected decisions.
        let fixed = flag != 1 && !params.is_empty();
        let gutz = fixed && params.contains(&0);
        let jast = fixed && params.contains(&ng);
        let dh2_fixed = fixed && params.iter().any(|p| (dh2..dh4).contains(p));
        let dh4_fixed = fixed && params.iter().any(|&p| p >= dh4);
        assert_eq!(model.shift, !gutz && !jast, "{label}");
        assert_eq!(model.shift_dh2, !gutz && !dh2_fixed, "{label}");
        assert_eq!(model.shift_dh4, !gutz && !dh4_fixed, "{label}");
    }
}

/// #700: the complex flag path keeps every real/imaginary component's raw
/// exact-one flag (C stcopt_cg_impl.c:161, `OptFlag[pi] != 1`); a missing flag
/// is fixed, as in the CPU `component_is_optimized`.
#[test]
fn resident_complex_flags_follow_cpu_exact_one_semantics() {
    let raw = [1, 1, 2, 1, -1, 0, 1, 2, 0, 1, 1];
    let complex = flag_contract(&raw, 6, true, 2, 2, 0, 0);
    assert_eq!(complex.flags, [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]);
    assert!(!complex.shift, "Gutzwiller 1 has real flag 2");
    let real = flag_contract(&raw, 6, false, 2, 2, 0, 0);
    assert_eq!(real.flags, [1, 0, 0, 1, 0, 1]);
    let gj = flag_contract(&[1, 0, 1, 0, 1, 0, 1, 0], 4, false, 2, 2, 0, 0);
    assert!(gj.shift && !gj.shift_dh2 && !gj.shift_dh4);
    let missing = flag_contract(&[1, 0, 1, 0, 1, 0], 4, false, 2, 2, 0, 0);
    assert_eq!(missing.flags, [1, 1, 1, 0]);
    assert!(!missing.shift, "a missing Jastrow flag is fixed");
}

/// #701 minimal real RBM inputs on the six-site baseline: hidden-layer-only
/// sections with one neuron per family coefficient, so the family tag (1 spin,
/// 2 general) is not below the coefficient count. `(label, input, nrbm,
/// nneuron, family)`; `prefix` keeps concurrent tests' directories apart.
pub(crate) fn minimal_rbm_inputs(prefix: &str) -> Vec<(&'static str, PathBuf, usize, usize, i32)> {
    let hidden = |key: &str, count: usize, rows: &str| {
        format!("===\n{key} {count}\nComplexType 0\n===\n===\n{rows}")
    };
    let cases: [(&'static str, &str, &str, usize, &str, usize, i32); 4] = [
        (
            "charge1",
            "ChargeRBM_HiddenLayer",
            "NneuronCharge",
            1,
            "0 0\n0 1\n",
            1,
            0,
        ),
        (
            "spin1",
            "SpinRBM_HiddenLayer",
            "NneuronSpin",
            1,
            "0 0\n0 1\n",
            1,
            1,
        ),
        (
            "general1",
            "GeneralRBM_HiddenLayer",
            "NneuronGeneral",
            1,
            "0 0\n0 1\n",
            1,
            2,
        ),
        (
            "general2",
            "GeneralRBM_HiddenLayer",
            "NneuronGeneral",
            2,
            "0 0\n1 1\n0 1\n1 1\n",
            2,
            2,
        ),
    ];
    cases
        .into_iter()
        .map(|(label, key, neurons, nneuron, rows, nrbm, family)| {
            let file = hidden(key, nrbm, rows);
            let path = edited(
                &format!("{prefix}-{label}"),
                &[
                    ("namelist.def", "", &format!("{key} rbmhidden.def\n")),
                    ("rbmhidden.def", "", &file),
                    ("modpara.def", "", &format!("\n{neurons}  {nneuron}\n")),
                ],
            );
            (label, path, nrbm, nneuron, family)
        })
        .collect()
}

/// #701: the resident RBM validator checks the family-tag segment against
/// 0..=2 and the coefficient segments against [-1, nrbm) separately, so the
/// minimal Spin/General tables are accepted for Opt and PhysCal, and malformed
/// tables are rejected with deterministic errors (before any allocation).
#[test]
fn resident_rbm_validator_separates_family_tags_from_coefficients() {
    use crate::interval::resident::validate_rbm_tables;
    for (label, input, nrbm, nneuron, family) in minimal_rbm_inputs("rbm-validate") {
        for model in [
            Model::prepare(&input, &cfg()).unwrap(),
            Model::prepare_physcal(&input, &cfg(), None).unwrap(),
        ] {
            assert_eq!((model.nrbm, model.nneuron), (nrbm, nneuron), "{label}");
            let ns = model.ns;
            let t = &model.rbm;
            let w = &model.parameters[model.nproj..model.nproj + model.nrbm];
            assert!(w.iter().all(|x| x.is_finite()), "{label}");
            assert!(t[4 * ns..4 * ns + nneuron].iter().all(|&f| f == family));
            // The hidden-layer coefficient h maps neuron h; all others absent.
            for (k, &x) in t.iter().enumerate() {
                let bias = k.checked_sub(4 * ns + nneuron).filter(|&h| h < nneuron);
                if let Some(h) = bias {
                    assert_eq!(x, h as i32, "{label}: hidden index of neuron {h}");
                } else if !(4 * ns..4 * ns + nneuron).contains(&k) {
                    assert_eq!(x, -1, "{label}: entry {k}");
                }
            }
            validate_rbm_tables(ns, t, w, nneuron).unwrap();
            let reject = |tables: &[i32], weights: &[f64]| {
                validate_rbm_tables(ns, tables, weights, nneuron).unwrap_err()
            };
            let tag = 4 * ns;
            for bad in [-1, 3] {
                let mut x = t.clone();
                x[tag] = bad;
                assert_eq!(
                    reject(&x, w),
                    format!("invalid resident RBM tables: family tag {bad} of neuron 0")
                );
            }
            for (entry, bad) in [(0, nrbm as i32), (ns, -2), (t.len() - 1, nrbm as i32)] {
                let mut x = t.clone();
                x[entry] = bad;
                assert_eq!(
                    reject(&x, w),
                    format!(
                        "invalid resident RBM tables: coefficient index {bad} at entry \
                         {entry} for {nrbm} coefficients"
                    )
                );
            }
            assert_eq!(
                reject(&t[..t.len() - 1], w),
                format!(
                    "invalid resident RBM tables: length {} for nsite {ns} and {nneuron} neurons",
                    t.len() - 1
                )
            );
            for weights in [vec![], vec![f64::NAN; nrbm], vec![f64::INFINITY; nrbm]] {
                assert_eq!(
                    reject(t, &weights),
                    "invalid resident RBM tables: coefficients must be nonempty and finite"
                );
            }
        }
        fs::remove_dir_all(input.parent().unwrap()).unwrap();
    }
}

pub(crate) fn fsz_input(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fsz")
        .join(name)
        .join("namelist.def")
}

/// General orbitals / FSZ (#714). The model follows C's FSZ path: walker slots
/// hold `rs`, the Slater tables cover the full `2ns x 2ns` orbital matrix of C
/// `UpdateSlaterElm_fsz`, the derivative basis is `dS/dF` of the same tables,
/// operators are encoded on `rs` (spin flips allowed), and the initial spins
/// follow C `makeInitialSample_fsz` (`Ne + 2Sz/2` up).
#[test]
fn resident_scope_accepts_fsz_general_orbitals() {
    for (name, spin_free, nup, nlocal) in [
        ("hubbard_fsz_L16", true, 6, 0),
        ("hubbard_fsz_L128", true, 64, 0),
        ("kondo_fsz_2sz2", false, 8, 8),
    ] {
        let input = fsz_input(name);
        let data = parse(&input);
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        mvmc_core::validation::validate_phys_cal(&data).unwrap();
        assert_eq!(data.i_flg_orbital_general, 1, "{name}");
        assert!(!mvmc_core::get_all_complex_flag(&data).unwrap());
        for (route, model) in [
            ("Opt", Model::prepare(&input, &cfg()).unwrap()),
            (
                "PhysCal",
                Model::prepare_physcal(&input, &cfg(), None).unwrap(),
            ),
        ] {
            let (ns, nq) = (model.ns, model.nq);
            assert!(model.fsz && !model.complex, "{name} {route}");
            assert_eq!((model.spin_free, model.nup), (spin_free, nup), "{name}");
            assert_eq!(model.split(), 2 * model.ne);
            assert_eq!(nq, 2, "{name}: NSPGaussLeg=1 x NMPTrans=2");
            assert_eq!(model.localspins.iter().filter(|&&x| x == 1).count(), nlocal);
            assert_eq!(model.pa.len(), nq * 4 * ns * ns);
            // The device table (opt_slater_f) against the CPU C-order
            // update_slater_elm_fsz: one subtraction of +-1 multiples.
            let mut cpu = model.data.clone();
            let mut state = mvmc_core::VmcOptimizationState::zeros(
                ns,
                model.ne,
                model.nproj,
                model.np,
                nq,
                1,
                false,
                true,
            );
            state.validate_declared_mode(&cpu).unwrap();
            mvmc_core::slater_update::update_slater_elm_fsz(&mut cpu, &mut state);
            let p = &model.parameters;
            let n2 = 2 * ns;
            for k in 0..nq * n2 * n2 {
                let device = p[model.pa[k] as usize] * model.sa[k] as f64
                    - p[model.pb[k] as usize] * model.sb[k] as f64;
                let host = state
                    .slater_matrix
                    .slater_elm
                    .get(k / (n2 * n2), (k / n2) % n2, k % n2);
                assert_eq!(host.im, 0.0);
                assert!(
                    (device - host.re).abs() <= f64::EPSILON * host.re.abs(),
                    "{name} {route} entry {k}: {device} vs {}",
                    host.re
                );
            }
            // Operators on rs positions: the Sz-free chains carry spin flips.
            let ops = model.operators.as_chunks::<7>().0;
            assert!(ops.iter().all(|op| op[2] == 0 && op[5] == 0));
            assert!(ops
                .iter()
                .all(|op| op[..2].iter().all(|&x| (0..n2 as i32).contains(&x))));
            let flips = ops
                .iter()
                .filter(|op| op[0] / ns as i32 != op[1] / ns as i32);
            assert_eq!(flips.count() > 0, spin_free, "{name}: spin-flip operators");
            if route == "Opt" {
                // The derivative basis reproduces the table: S = sum_p F_p dS/dF_p.
                let mut rebuilt = vec![0.0; nq * n2 * n2];
                for (column, value) in p.iter().enumerate() {
                    let range = model.columns[column] as usize..model.columns[column + 1] as usize;
                    for e in range {
                        let [row, col, q] = model.coordinates[3 * e..3 * e + 3] else {
                            unreachable!()
                        };
                        let k = (q as usize * n2 + row as usize) * n2 + col as usize;
                        rebuilt[k] += value * model.coefficients[e];
                    }
                }
                for (k, &value) in rebuilt.iter().enumerate() {
                    let host = state
                        .slater_matrix
                        .slater_elm
                        .get(k / (n2 * n2), (k / n2) % n2, k % n2)
                        .re;
                    assert!(
                        (value - host).abs() <= 4.0 * f64::EPSILON * host.abs(),
                        "{name}: basis entry {k}: {value} vs {host}"
                    );
                }
            }
        }
    }
}

/// FSZ inputs outside the supported subset fail before CUDA (#714): complex
/// general orbitals, local spin flips (NExUpdatePath=2 with 2Sz=-1, and
/// NExUpdatePath=3), NSPGaussLeg>1, SR-CG for Opt, and RBM/DH/OptTrans. The
/// CPU validators accept each input, so the rejection is GPU-specific.
#[test]
fn resident_scope_rejects_fsz_families_before_cuda() {
    const FAMILIES: &str = "resident GPU FSZ supports real Gutzwiller/Jastrow";
    let l16 = fsz_input("hubbard_fsz_L16");
    let ns = 16;
    let rbm = format!(
        "===\nNChargeRBM_PhysLayer 1\nComplexType 0\n===\n===\n{}0 1\n",
        (0..ns).map(|i| format!("{i} 0\n")).collect::<String>()
    );
    let opttrans = format!(
        "===\nNQPOptTrans 2\nComplexType 0\n===\n===\n0 0.75\n1 -0.5\n{}{}",
        (0..ns)
            .map(|i| format!("0 {i} {i} 1\n"))
            .collect::<String>(),
        (0..ns)
            .map(|i| format!("1 {i} {} 1\n", ns - 1 - i))
            .collect::<String>()
    );
    let cases: Vec<(&str, PathBuf, &str, bool)> = vec![
        (
            "fsz-complex",
            repo("tests/fixtures/native13_initialization/inputs/HubbardChain_fsz/namelist.def"),
            FAMILIES,
            false,
        ),
        (
            "fsz-local-flip",
            edited_from(
                &l16,
                "fsz-local-flip",
                &[("modpara.def", "NExUpdatePath  1", "NExUpdatePath  2")],
            ),
            "local spin flips",
            true,
        ),
        (
            "fsz-spin-projection",
            edited_from(
                &l16,
                "fsz-spin-projection",
                &[("modpara.def", "NSPGaussLeg    1", "NSPGaussLeg    4")],
            ),
            "NSPGaussLeg=1",
            true,
        ),
        (
            "fsz-rbm",
            edited_from(
                &l16,
                "fsz-rbm",
                &[
                    ("namelist.def", "", "ChargeRBM_PhysLayer chargerbm.def\n"),
                    ("chargerbm.def", "", &rbm),
                ],
            ),
            FAMILIES,
            true,
        ),
        (
            "fsz-opttrans",
            edited_from(
                &l16,
                "fsz-opttrans",
                &[
                    ("namelist.def", "", "OptTrans opttrans.def\n"),
                    ("opttrans.def", "", &opttrans),
                ],
            ),
            FAMILIES,
            true,
        ),
    ];
    for (tag, input, message, temporary) in cases {
        let data = parse(&input);
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
            assert!(error.contains(message), "{tag}/{route}: {error}");
        }
        if temporary {
            fs::remove_dir_all(input.parent().unwrap()).unwrap();
        }
    }
    // SR-CG: rejected for FSZ Opt, ignored by FSZ PhysCal.
    let cg = edited_from(
        &l16,
        "fsz-cg",
        &[("modpara.def", "NSRCG          0", "NSRCG          1")],
    );
    let error = Model::prepare(&cg, &cfg())
        .map(|_| ())
        .expect_err("FSZ CG Opt");
    assert!(error.contains("FSZ Opt requires direct SR"), "{error}");
    assert!(Model::prepare_physcal(&cg, &cfg(), None).unwrap().fsz);
    fs::remove_dir_all(cg.parent().unwrap()).unwrap();
    // C rejects an odd 2Sz other than -1 (readdef.c:697); so does the model.
    let odd = edited_from(
        &fsz_input("kondo_fsz_2sz2"),
        "fsz-odd",
        &[("modpara.def", "2Sz            2", "2Sz            3")],
    );
    let error = Model::prepare(&odd, &cfg())
        .map(|_| ())
        .expect_err("odd 2Sz");
    assert!(error.contains("even 2Sz"), "{error}");
    fs::remove_dir_all(odd.parent().unwrap()).unwrap();
}
