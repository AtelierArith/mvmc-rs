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
            "exchange-path",
            &[("modpara.def", "NExUpdatePath  0", "NExUpdatePath  1")],
        ),
        (
            "nsplit",
            &[("modpara.def", "NSplitSize     1", "NSplitSize     2")],
        ),
        (
            "rbm",
            &[
                ("namelist.def", "", "ChargeRBM_PhysLayer chargerbm.def\n"),
                (
                    "chargerbm.def",
                    "",
                    "===\nNChargeRBM_PhysLayer 1\nComplexType 0\n===\n===\n\
                     0 0\n1 0\n2 0\n3 0\n4 0\n5 0\n0 1\n",
                ),
            ],
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
    // family (real six-site chains), except: the local-spin Heisenberg chain,
    // which C requires to use NExUpdatePath=2 (readdef.c:621-628); the FSZ
    // Hubbard chain (OrbitalParallel selects general orbitals, complex
    // orbital coefficients); and the FSZ Kondo chain (NExUpdatePath=3).
    for (tag, rel) in [
        ("dh2", "tests/fixtures/dh2/production_real/namelist.def"),
        ("dh4", "tests/fixtures/dh4/production_dh4_real/namelist.def"),
        (
            "opttrans",
            "tests/fixtures/opttrans/run_opt_real/namelist.def",
        ),
        (
            "local-spin",
            "tests/fixtures/original_heisenberg_parser_184/namelist.def",
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
