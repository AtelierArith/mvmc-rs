//! Declaration errors are preflight failures, not numerical/model evidence.
//! Ordinary peer controls simulate only error agreement. The separately
//! ignored MPI control requires exactly two actual ranks and is NOT RUN.

use mvmc_core::{
    vmc_para_opt, vmc_phys_cal_in_place, ExpertModeData, OptimizationOptions, Reducer,
    SingleProcessReducer, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::cell::RefCell;

fn loaded() -> ExpertModeData {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real/inputs/namelist.def");
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
    // This test exercises preflight only; the unsupported optimization GEx
    // branch is not part of either selected entrypoint's control input.
    data.namelist.retain(|(kind, _)| kind != "TwoBodyGEx");
    data.green_two_ex_terms.clear();
    data.modpara.nsplit_size = 1;
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    data.modpara.nsrcg = 0;
    data.modpara.lanczos_mode = 0;
    data.modpara.vmc_calc_mode = 0;
    assert_eq!(mvmc_core::get_all_complex_flag(&data), Ok(false));
    data
}

fn state(data: &ExpertModeData, complex: bool) -> VmcOptimizationState {
    VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        1,
        data.modpara.nvmc_sample as usize,
        complex,
        false,
    )
}

fn invalidate_metadata(data: &mut ExpertModeData) {
    data.native_complex_headers.insert("DH2".to_string(), 0);
    data.native_complex_declarations
        .insert("DH2".to_string(), false);
    data.doublon_holon_2site_complex = true;
}

fn rejected_without_mutation(
    mut data: ExpertModeData,
    complex_state: bool,
    physcal: bool,
    reducer: &impl Reducer,
) -> String {
    data.modpara.vmc_calc_mode = i64::from(physcal);
    let mut state = state(&data, complex_state);
    let mut rng = Sfmt19937Rng::new(12395);
    // Nonzero draw history proves preservation of both cursor and count.
    let _ = rng.gen_rand32();
    let raw = rng.state_snapshot();
    let count = rng.words_consumed();
    let data_before = format!("{data:?}");
    let state_before = format!("{state:?}");
    let result = if physcal {
        vmc_phys_cal_in_place(&mut data, &mut state, &mut rng, None, reducer, None).map(|_| ())
    } else {
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            None,
            reducer,
            OptimizationOptions::default(),
        )
    };
    let error = result.expect_err("preflight must fail before entering any numerical kernel");
    assert_eq!(format!("{data:?}"), data_before);
    assert_eq!(format!("{state:?}"), state_before);
    assert_eq!(rng.state_snapshot(), raw);
    assert_eq!(rng.words_consumed(), count);
    error
}

struct PeerPreflight {
    rank: usize,
    peer_failed: bool,
    calls: RefCell<Vec<bool>>,
}

impl Reducer for PeerPreflight {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {
        panic!("numerical reduction reached after preflight failure");
    }
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {
        panic!("numerical reduction reached after preflight failure");
    }
    fn allreduce_sum_i64(&self, _: &mut [i64]) {
        panic!("unexpected integer reduction outside the explicit preflight control");
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn any_failure(&self, failed: bool) -> bool {
        self.calls.borrow_mut().push(failed);
        failed || self.peer_failed
    }
}

#[test]
fn local_and_peer_invalid_metadata_reach_the_same_first_collective_without_mutation() {
    for bad_rank in [0, 1] {
        for rank in [0, 1] {
            for physcal in [false, true] {
                let mut data = loaded();
                let faulty = rank == bad_rank;
                if faulty {
                    invalidate_metadata(&mut data);
                }
                let reducer = PeerPreflight {
                    rank,
                    peer_failed: !faulty,
                    calls: RefCell::new(Vec::new()),
                };
                let error = rejected_without_mutation(data, false, physcal, &reducer);
                assert_eq!(*reducer.calls.borrow(), [faulty]);
                if faulty {
                    assert!(error.contains("loaded complex declaration changed"));
                } else {
                    assert!(error.contains("failed on another MPI rank"));
                }
            }
        }
    }
}

#[test]
fn local_and_peer_state_mismatch_reach_the_same_collective_without_mutation() {
    for bad_rank in [0, 1] {
        for rank in [0, 1] {
            let faulty = rank == bad_rank;
            let reducer = PeerPreflight {
                rank,
                peer_failed: !faulty,
                calls: RefCell::new(Vec::new()),
            };
            let error = rejected_without_mutation(loaded(), faulty, false, &reducer);
            assert_eq!(*reducer.calls.borrow(), [faulty]);
            if faulty {
                assert!(error.contains("allocated mode differs"));
            } else {
                assert!(error.contains("failed on another MPI rank"));
            }
        }
    }
}

#[test]
fn allocated_mode_mismatch_returns_error_before_data_state_or_raw_rng_changes() {
    let data = loaded();
    assert!(state(&data, false).validate_declared_mode(&data).is_ok());
    assert!(state(&data, true).validate_declared_mode(&data).is_err());
    let error = rejected_without_mutation(data, true, false, &SingleProcessReducer);
    assert!(error.contains("allocated mode differs"));
}

#[test]
fn direct_caller_guard_keeps_coefficient_mutation_separate_from_declaration_drift() {
    let mut data = loaded();
    let state = state(&data, false);
    let before = format!("{state:?}");
    let rng = Sfmt19937Rng::new(12395);
    let raw = rng.state_snapshot();
    let count = rng.words_consumed();
    assert!(state.validate_declared_mode(&data).is_ok());
    data.gutzwiller_terms[0].value.im = 0.25;
    assert!(state.validate_declared_mode(&data).is_ok());
    data.gutzwiller_terms[0].is_complex = true;
    assert!(state.validate_declared_mode(&data).is_err());
    // This is the direct caller's explicit error boundary, not a claim that
    // an unguarded infallible kernel autonomously validates metadata.
    assert_eq!(format!("{state:?}"), before);
    assert_eq!(rng.state_snapshot(), raw);
    assert_eq!(rng.words_consumed(), count);
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "explicit two-rank metadata preflight gate; requires external bounded MPI launch"]
fn actual_two_rank_invalid_metadata_and_healthy_peer_both_return_without_rng_changes() {
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    assert_eq!(world.world_size(), 2, "this gate is exactly two ranks");
    for bad_rank in [0, 1] {
        for physcal in [false, true] {
            let mut data = loaded();
            if world.rank() == bad_rank {
                invalidate_metadata(&mut data);
            }
            let error = rejected_without_mutation(data, false, physcal, &world);
            if world.rank() == bad_rank {
                assert!(error.contains("loaded complex declaration changed"));
            } else {
                assert!(error.contains("failed on another MPI rank"));
            }
            println!("ISSUE283_METADATA_PREFLIGHT rank={} world=2 bad_rank={bad_rank} physcal={physcal} unchanged=true", world.rank());
            world.barrier();
        }
    }
    for bad_rank in [0, 1] {
        let error = rejected_without_mutation(loaded(), world.rank() == bad_rank, false, &world);
        if world.rank() == bad_rank {
            assert!(error.contains("allocated mode differs"));
        } else {
            assert!(error.contains("failed on another MPI rank"));
        }
        println!(
            "ISSUE283_STATE_PREFLIGHT rank={} world=2 bad_rank={bad_rank} unchanged=true",
            world.rank()
        );
        world.barrier();
    }
}
