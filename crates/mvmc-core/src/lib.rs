//! VMC parameter-optimization engine.
//!
//! Port target: `extern/Julia-mVMC/MVMCOptimizers.jl/src/`.
//!
//! Phase 4 status: Phase 4 closes here. Sub-steps 4.1–4.8 are now
//! ported (state + storage-order newtypes, Pfaffian setup, sampling
//! drivers, observables, SR, sync/qp/average/counter/io, run
//! entrypoints + initial.def loader). The QPTrans-aware
//! `slater_elm_diff` and FSZ / BackFlow paths remain stretch goals
//! for Phase 7.
//!
//! License: GPL-3.0-or-later.

#![warn(missing_docs)]

#[cfg(test)]
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;

pub mod accel_validation;
pub mod average;
pub mod backend;
mod c_complex;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod c_complex_gnu;
pub mod c_timer;
pub mod counter;
pub mod initial_params;
pub mod io;
pub mod lanczos;
pub mod measurement_batch;
#[cfg(feature = "mpi")]
pub mod mpi;
pub mod multichain;
pub mod multidef;
pub use pfapack::julia_complex;
pub mod device_sampler;
pub mod observables;
pub mod output_files;
pub mod parallel;
pub mod parallel_scalar;
pub mod parameter_diagnostics;
pub mod parameters;
pub mod pfaffian;
pub mod qp;
pub mod reducer;
pub mod run;
pub mod sampling;
mod serial_blas;
mod slater_derivative;
pub mod slater_update;
pub mod sr;
pub mod sr_accumulator;
pub mod sr_backend;
pub mod sr_cg;
pub mod stage_backend;
pub mod state;
pub mod sync;
pub mod threading;
pub mod validation;

pub use initial_params::{read_initial_def, read_opt_para_file};
pub use mvmc_expert_parsers::ExpertModeData;
pub use parameters::{
    get_parameter_value, pack_parameters, set_parameter_value, unpack_parameters,
    ParameterAccessError,
};
pub use pfaffian::{
    calc_m_all_complex, calc_m_all_fsz_complex, calc_m_all_fsz_real, calc_m_all_real,
    use_julia_complex_kernel, CalcMAllError, JuliaComplexKernelGuard,
};
pub use reducer::{Reducer, SingleProcessReducer};
pub use run::{
    get_all_complex_flag, prepare_phys_cal_from_namelist,
    prepare_phys_cal_from_namelist_with_reducer,
    prepare_phys_cal_from_namelist_with_reducer_and_opt_trans, prepare_phys_cal_with_seed_offset,
    prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans, resolve_rnd_seed,
    run_para_opt_from_namelist, run_para_opt_from_namelist_observed,
    run_para_opt_from_namelist_with_reducer, vmc_para_opt, vmc_para_opt_timed, vmc_phys_cal,
    vmc_phys_cal_in_place, vmc_phys_cal_in_place_timed, vmc_phys_cal_in_place_with_options_timed,
    vmc_phys_cal_to_dir, vmc_phys_cal_to_dir_with_callback, vmc_phys_cal_with_callback,
    vmc_phys_cal_with_reducer, vmc_phys_cal_with_reducer_and_callback,
    vmc_phys_cal_with_reducer_and_callback_timed, vmc_phys_cal_with_reducer_timed, InitialDef,
    OptimizationOptions, PhysCalCallback, PhysCalPreparation, PhysCalResult, PhysCalRunOptions,
    RunConfig, RunSummary, StepCallback, FALLBACK_SEED,
};
pub use sampling::driver::{vmc_make_sample, vmc_make_sample_real, SampleStats};
pub use state::{
    ElectronConfiguration, EnergyData, InvMColMajor, OptDataPoint, PfaPackMode, PfaPackWorkspace,
    PhysicalQuantities, SROptData, SamplingWorkspace, SlaterElmFlat, SlaterMatrixData,
    ThreadedPfaPackWorkspace, VmcOptimizationState,
};
#[cfg(test)]
#[path = "../../../tests/support/c_general_fsz.rs"]
mod c_general_fsz;
#[cfg(test)]
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[cfg(test)]
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

#[cfg(test)]
#[path = "../../../tests/support/historical_optimization_flags.rs"]
mod historical_optimization_flags;
