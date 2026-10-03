//! Independent discrete trajectory gates for the canonical GeneralRBM ctest input.
//! Numerical SR/output comparisons remain in the existing runner gates (#190).
use std::fs;
use std::path::Path;

mod support;

struct OutputDirectory(std::path::PathBuf);

impl OutputDirectory {
    fn create() -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("mvmc-ctest-prefix-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for OutputDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "50-step independent reference gate; MVMC_RS_CTEST_PREFIXES=1 required"]
fn general_rbm_ctest_short_prefix_and_fifty_step_discrete_trajectory() {
    support::require_gate("ctest-prefixes", "MVMC_RS_CTEST_PREFIXES");
    let root = support::julia_mvmc_root().expect("Julia reference checkout required");
    let input = root.join("test/integration/reference/general_rbm_cmp/inputs/namelist.def");
    let fixtures =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/c_kernel_order");
    for (cg, store, directory) in [
        (0, 1, "sr_direct/rbm_reference_cmp_store_runner"),
        (1, 0, "sr_cg/rbm_reference_cmp_runner"),
    ] {
        for steps in [1, 2, 3, 50] {
            let mut data =
                mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false)
                    .unwrap();
            mvmc_core::validation::validate_para_opt(&data).unwrap();
            assert_eq!(data.modpara.rnd_seed, 12395);
            let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
            mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng);
            assert!(mvmc_core::read_initial_def(
                &mut data,
                input.parent().unwrap().join("initial.def"),
            )
            .unwrap());
            mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
                &mut data, &input,
            )
            .unwrap();
            mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
            mvmc_core::qp::init_qp_weight(&mut data);
            data.modpara.nsr_opt_itr_step = steps;
            data.modpara.nsr_opt_itr_smp = steps;
            data.modpara.nsrcg = cg;
            data.modpara.nstore_o = store;
            let mut state = mvmc_core::VmcOptimizationState::zeros(
                data.modpara.nsite as usize,
                data.modpara.nelec as usize,
                data.projection_layout().n_proj,
                data.count_variational_parameters(),
                data.modpara.nsp_gauss_leg.max(1) as usize
                    * data.modpara.nmp_trans.unsigned_abs() as usize
                    * data.n_qp_opt_trans.max(1) as usize,
                data.modpara.nvmc_sample as usize,
                mvmc_core::get_all_complex_flag(&data),
                data.i_flg_orbital_general != 0,
            );
            let output = OutputDirectory::create();
            mvmc_core::vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&output.0),
                &mvmc_core::SingleProcessReducer,
                mvmc_core::OptimizationOptions::default(),
            )
            .unwrap();
            let reference = fixtures.join(directory);
            let configurations =
                fs::read_to_string(reference.join(format!("step-{steps}-configs.txt"))).unwrap();
            let actual = [
                state.electron_config.ele_idx.as_slice(),
                state.electron_config.ele_cfg.as_slice(),
                state.electron_config.ele_num.as_slice(),
                state.electron_config.ele_proj_cnt.as_slice(),
                state.electron_config.burn_ele_idx.as_slice(),
                state.electron_config.counter.as_slice(),
            ];
            assert_eq!(configurations.lines().count(), actual.len());
            for (index, (line, actual)) in configurations.lines().zip(actual).enumerate() {
                let expected: Vec<i64> = line
                    .split_whitespace()
                    .map(|word| word.parse().unwrap())
                    .collect();
                assert_eq!(
                    actual, expected,
                    "{directory} step {steps} configuration {index}"
                );
            }
            let expected: Vec<u32> =
                fs::read_to_string(reference.join(format!("step-{steps}-rng.txt")))
                    .unwrap()
                    .split_whitespace()
                    .map(|word| word.parse().unwrap())
                    .collect();
            assert_eq!(expected.len(), 624);
            let actual: Vec<u32> = (0..624).map(|_| rng.gen_rand32()).collect();
            assert_eq!(actual, expected, "{directory} step {steps} full SFMT block");
        }
    }
}
