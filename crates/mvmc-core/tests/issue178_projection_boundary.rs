//! C manual: no translation projection means NMPTrans=1, not zero.
//! C's reader accepts zero as an integer; these tests do not claim a native-C
//! explicit error. They prevent a silent zero-to-identity Rust runtime repair.
use mvmc_core::{
    read_opt_para_file, vmc_para_opt, vmc_phys_cal_in_place, ExpertModeData, OptimizationOptions,
    SingleProcessReducer, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::{
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
                "issue178-projection-{}-{}",
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
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn loaded() -> ExpertModeData {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files(root.join("inputs/namelist.def")).unwrap();
    read_opt_para_file(&mut data, root.join("zqp_opt.dat")).unwrap();
    // Same supported model on both APIs; TwoBodyGEx is measurement-only.
    data.namelist.retain(|(kind, _)| kind != "TwoBodyGEx");
    data.green_two_ex_terms.clear();
    data.modpara.vmc_calc_mode = 0;
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    data.modpara.nvmc_warmup = 1;
    data.modpara.nvmc_sample = 3;
    data.modpara.nvmc_interval = 1;
    data
}

fn zero_boundary(physcal: bool) {
    let parent = OwnedDir::new();
    let output = parent.0.join("must-not-exist");
    let mut data = loaded();
    data.modpara.nmp_trans = 0;
    let before_data = format!("{data:?}");
    let mut state = VmcOptimizationState::zeros(2, 1, 2, 3, 3, 2, false, true);
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
    let before_state = format!("{state:?}");
    let before_planes = (
        state.slater_matrix.slater_elm.as_slice().to_vec(),
        state.slater_matrix.inv_m.as_slice().to_vec(),
        state.slater_matrix.slater_elm_real.as_slice().to_vec(),
        state.slater_matrix.inv_m_real.as_slice().to_vec(),
    );
    let mut rng = Sfmt19937Rng::new(11272);
    rng.gen_rand32();
    let before_rng = rng.state_snapshot();
    let before_count = rng.words_consumed();
    let result = if physcal {
        vmc_phys_cal_in_place(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &SingleProcessReducer,
            None,
        )
        .map(|_| ())
    } else {
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
    };
    // Preserve the old repair's actual mutation/RNG/output evidence before
    // asserting rejection; no expected values are derived from this report.
    println!("ISSUE178_NMP physcal={physcal} result={result:?} nmp={} words_before={before_count} words_after={} cursor_before={} cursor_after={} state_changed={} output_exists={}",
        data.modpara.nmp_trans, rng.words_consumed(), before_rng.1, rng.state_snapshot().1,
        format!("{state:?}") != before_state, output.exists());
    let error = result.unwrap_err();
    assert_eq!(
        error,
        "NMPTrans must be nonzero; use 1 for no translation projection (mVMC C contract)"
    );
    assert_eq!(format!("{data:?}"), before_data);
    assert_eq!(format!("{state:?}"), before_state);
    assert_eq!(state.slater_matrix.slater_elm.as_slice(), before_planes.0);
    assert_eq!(state.slater_matrix.inv_m.as_slice(), before_planes.1);
    assert_eq!(
        state.slater_matrix.slater_elm_real.as_slice(),
        before_planes.2
    );
    assert_eq!(state.slater_matrix.inv_m_real.as_slice(), before_planes.3);
    assert_eq!(rng.state_snapshot(), before_rng);
    assert_eq!(rng.words_consumed(), before_count);
    let mut actual = [0; 624];
    let mut expected = [0; 624];
    rng.dump_rand32(&mut actual);
    let mut untouched = Sfmt19937Rng::new(11272);
    untouched.gen_rand32();
    untouched.dump_rand32(&mut expected);
    assert_eq!(actual, expected);
    assert_eq!(rng.state_snapshot(), before_rng);
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&parent.0).unwrap().count(), 0);
}

#[test]
fn zero_translation_paraopt_rejects_before_data_state_rng_and_output() {
    zero_boundary(false);
}
#[test]
fn zero_translation_physcal_rejects_before_data_state_rng_and_output() {
    zero_boundary(true);
}

#[test]
fn signed_unit_translation_physcal_public_core_remains_accepted() {
    for nmp in [1, -1] {
        let parent = OwnedDir::new();
        let mut data = loaded();
        data.modpara.nmp_trans = nmp;
        let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
        let mut rng = Sfmt19937Rng::new(11272);
        assert_eq!(
            vmc_phys_cal_in_place(
                &mut data,
                &mut state,
                &mut rng,
                Some(&parent.0.join("out")),
                &SingleProcessReducer,
                None
            )
            .unwrap(),
            1
        );
        assert!(rng.words_consumed() > 0);
        assert!(parent.0.join("out/zvo_out_001.dat").is_file());
        // This API preserves the existing negative-count normalization; CLI
        // parsing separately establishes the actual anti-periodic flag.
        assert_eq!(data.modpara.nmp_trans, 1);
    }
}
