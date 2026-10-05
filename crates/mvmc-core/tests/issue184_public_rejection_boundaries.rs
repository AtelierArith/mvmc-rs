//! Existing Rust/Julia feature guards, NOT claims that C rejects these inputs.
//! Calls public production entrypoints with exact S200/S205/S214 settings.
use mvmc_core::{
    prepare_phys_cal_from_namelist_with_reducer, vmc_para_opt, vmc_phys_cal_in_place,
    ExpertModeData, OptimizationOptions, Reducer, SingleProcessReducer, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::{
    cell::Cell,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct OwnedDir(PathBuf);
impl OwnedDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue184-boundary-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
}
impl Drop for OwnedDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn copied_input(dir: &OwnedDir) -> (PathBuf, PathBuf) {
    // Checked-in offline copy, byte-identical to the original PhysCal input
    // and fixed record. No vendored/reference runtime is needed by Cargo.
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let inputs = dir.0.join("inputs");
    fs::create_dir(&inputs).unwrap();
    for entry in fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let fixed = dir.0.join("fixed.dat");
    fs::copy(source.join("zqp_opt.dat"), &fixed).unwrap();
    (inputs.join("namelist.def"), fixed)
}

fn boundary(physcal: bool, configure: impl FnOnce(&mut ExpertModeData), diagnostic: &str) {
    let dir = OwnedDir::new();
    let (input, _) = copied_input(&dir);
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(input).unwrap();
    configure(&mut data);
    let before_data = format!("{data:?}");
    // Nonempty, multiple QP planes and both real/complex buffers: rejection
    // must not replace an existing caller-owned state with initialized state.
    let mut state = VmcOptimizationState::zeros(2, 1, 2, 3, 3, 2, false, false);
    state.energy.etot = Complex64::new(7.0, -2.0);
    state
        .slater_matrix
        .slater_elm
        .as_mut_slice()
        .fill(Complex64::new(3.0, 1.0));
    state
        .slater_matrix
        .inv_m
        .as_mut_slice()
        .fill(Complex64::new(4.0, 2.0));
    state.slater_matrix.slater_elm_real.as_mut_slice().fill(5.0);
    state.slater_matrix.inv_m_real.as_mut_slice().fill(6.0);
    let planes = state.slater_matrix.clone();
    // Materialized copies remain independent even if a tensor Clone aliases storage.
    let plane_values = (
        state.slater_matrix.slater_elm.as_slice().to_vec(),
        state.slater_matrix.inv_m.as_slice().to_vec(),
        state.slater_matrix.slater_elm_real.as_slice().to_vec(),
        state.slater_matrix.inv_m_real.as_slice().to_vec(),
    );
    let energy = state.energy;
    let sr = state.sr_opt.clone();
    let config = state.electron_config.clone();
    let before_state = format!("{state:?}");
    let mut rng = Sfmt19937Rng::new(11272);
    rng.gen_rand32(); // Nonzero pre-existing count must be preserved too.
    let mut expected_rng = rng.clone();
    let count = rng.words_consumed();
    let out = dir.0.join("never-created");
    assert!(!out.exists());
    let error = if physcal {
        vmc_phys_cal_in_place(
            &mut data,
            &mut state,
            &mut rng,
            Some(&out),
            &SingleProcessReducer,
            None,
        )
        .unwrap_err()
    } else {
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&out),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap_err()
    };
    assert!(
        error.contains(diagnostic),
        "wrong rejection boundary: {error}"
    );
    assert!(!out.exists());
    assert_eq!(format!("{data:?}"), before_data); // Every parameter, flag and setting.
    assert_eq!(state.slater_matrix, planes); // Full storage including every QP plane/pad.
    assert_eq!(state.slater_matrix.slater_elm.as_slice(), plane_values.0);
    assert_eq!(state.slater_matrix.inv_m.as_slice(), plane_values.1);
    assert_eq!(
        state.slater_matrix.slater_elm_real.as_slice(),
        plane_values.2
    );
    assert_eq!(state.slater_matrix.inv_m_real.as_slice(), plane_values.3);
    assert_eq!(state.energy, energy);
    assert_eq!(state.sr_opt, sr);
    assert_eq!(state.electron_config, config);
    assert_eq!(format!("{state:?}"), before_state); // Workspace/history/physical/caches too.
    assert_eq!(rng.words_consumed(), count);
    for _ in 0..624 {
        assert_eq!(rng.gen_rand32(), expected_rng.gen_rand32());
    }
    assert_eq!(rng.words_consumed(), count + 624);
}

#[test]
fn s200_paraopt_split_three_cg_one_rejects_before_output_state_and_rng() {
    boundary(
        false,
        |d| {
            d.modpara.nsplit_size = 3;
            d.modpara.nsrcg = 1;
        },
        "NSplitSize > 1 with SR-CG",
    );
}
#[test]
fn s200_physcal_split_three_general_rejects_before_output_state_and_rng() {
    boundary(
        true,
        |d| {
            d.modpara.nsplit_size = 3;
            d.i_flg_orbital_general = 1;
        },
        "FSZ / general-orbital PhysCal",
    );
}
#[test]
fn s205_paraopt_cg_two_without_store_rejects_before_output_state_and_rng() {
    boundary(
        false,
        |d| {
            d.modpara.nsrcg = 2;
            d.modpara.nstore_o = 0;
        },
        "undefined in mVMC C",
    );
}
#[test]
fn s214_paraopt_lanczos_one_rejects_before_output_state_and_rng() {
    boundary(
        false,
        |d| d.modpara.lanczos_mode = 1,
        "parameter optimization",
    );
}
#[test]
fn s214_physcal_lanczos_two_general_rejects_before_output_state_and_rng() {
    boundary(
        true,
        |d| {
            d.modpara.lanczos_mode = 2;
            d.i_flg_orbital_general = 1;
        },
        "FSZ/general orbitals",
    );
}

#[test]
fn s195_s196_grouped_opttrans_public_entries_reject_before_mutation() {
    // Original payloads, not a count-only synthetic rejection. These are
    // current Rust/Julia feature guards, not proof of C-invalid input.
    for physcal in [false, true] {
        boundary(
            physcal,
            |d| {
                d.modpara.nsplit_size = 2;
                d.modpara.nsp_gauss_leg = 1;
                d.modpara.nmp_trans = 1;
                d.modpara.lanczos_mode = 0;
                d.i_flg_orbital_general = 0;
                d.n_qp_opt_trans = 2;
                d.opt_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
                d.qp_opt_trans = vec![vec![0], vec![0]];
            },
            "NSplitSize > 1 with NQPOptTrans > 1 / OptTrans",
        );
    }
}

#[test]
fn s197_grouped_physcal_general_and_lanczos_reject_before_mutation() {
    boundary(
        true,
        |d| {
            d.modpara.nsplit_size = 2;
            d.modpara.lanczos_mode = 0;
            d.i_flg_orbital_general = 1;
        },
        "FSZ / general-orbital PhysCal",
    );
    boundary(
        true,
        |d| {
            d.modpara.nsplit_size = 2;
            d.modpara.lanczos_mode = 2;
            d.i_flg_orbital_general = 0;
        },
        "NSplitSize > 1 with NLanczosMode > 0",
    );
}

#[test]
fn s198_grouped_general_standard_projection_rejects_each_original_combination() {
    for (nsp, nmp) in [(2, 1), (1, 2), (1, -2)] {
        boundary(
            false,
            |d| {
                d.modpara.nsplit_size = 2;
                d.modpara.nsrcg = 0;
                d.modpara.nsp_gauss_leg = nsp;
                d.modpara.nmp_trans = nmp;
                d.n_qp_opt_trans = 1;
                d.i_flg_orbital_general = 1;
            },
            "NSplitSize > 1 with FSZ standard-projection NQPFull > 1",
        );
    }
}

#[test]
fn s204_unported_cg_submodes_reject_public_paraopt_before_mutation() {
    boundary(false, |d| d.modpara.use_diag_scale = 1, "useDiagScale != 0");
    boundary(false, |d| d.modpara.rescale_smat = 1, "RescaleSmat != 0");
}

#[derive(Default)]
struct SeedObserver(Cell<usize>);
impl Reducer for SeedObserver {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {}
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {}
    fn allreduce_sum_i64(&self, _: &mut [i64]) {}
    fn broadcast_i64(&self, _: usize, _: &mut [i64]) -> Result<(), String> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}

#[test]
fn missing_and_malformed_fixed_records_reject_public_preparation_before_seed_and_output() {
    let dir = OwnedDir::new();
    let (input, fixed) = copied_input(&dir);
    // Valid time-based C seed setting exposes the actual seed-resolution stage.
    // Only the owned input copy is changed; fixed data and vendor remain intact.
    let namelist = fs::read_to_string(&input).unwrap();
    let modpara_name = namelist
        .lines()
        .find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == "ModPara").then(|| words.next().unwrap())
        })
        .unwrap();
    let modpara_path = input.parent().unwrap().join(modpara_name);
    let original_modpara = fs::read_to_string(&modpara_path).unwrap();
    let mut time_seed_modpara = original_modpara
        .lines()
        .filter(|line| line.split_whitespace().next() != Some("RndSeed"))
        .collect::<Vec<_>>()
        .join("\n");
    time_seed_modpara.push_str("\nRndSeed -1\n");
    fs::write(&modpara_path, time_seed_modpara).unwrap();
    let valid = fs::read_to_string(&fixed).unwrap();
    let observer = SeedObserver::default();
    let prepared =
        prepare_phys_cal_from_namelist_with_reducer(&input, &fixed, "real", None, &observer)
            .unwrap();
    assert_eq!(prepared.rng.words_consumed(), 0);
    assert!(
        observer.0.get() > 0,
        "control must reach actual seed broadcast"
    );
    let tokens: Vec<_> = valid.split_whitespace().collect();
    assert_eq!(tokens.len(), 48); // Independent stored C record: 6 + 3*14.
    for (case, text, diagnostic) in [
        ("missing", None, "file not found"),
        ("short", Some(tokens[..47].join(" ")), "too short"),
        (
            "mid-token",
            Some({
                let mut t = tokens.clone();
                t[27] = "malformed";
                t.join(" ")
            }),
            "non-numeric token 'malformed' at field 28",
        ),
        ("trailing", Some(format!("{valid} 1")), "trailing floats"),
    ] {
        let path = dir.0.join(format!("{case}.dat"));
        if let Some(text) = text {
            fs::write(&path, text).unwrap();
        }
        let out = dir.0.join(format!("output-{case}"));
        let observer = SeedObserver::default();
        let error = match prepare_phys_cal_from_namelist_with_reducer(
            &input, &path, "real", None, &observer,
        ) {
            Ok(_) => panic!("{case}: invalid record reached runnable preparation"),
            Err(e) => e,
        };
        assert!(error.contains(diagnostic), "{case}: {error}");
        assert_eq!(observer.0.get(), 0, "{case}: reached seed resolution");
        // Preparation has no output-dir argument; it fails before the subsequent
        // public core can receive this path. Do not label this a CLI subprocess test.
        assert!(!out.exists());
        assert_eq!(fs::read_to_string(&fixed).unwrap(), valid);
    }
}
