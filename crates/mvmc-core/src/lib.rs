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

pub mod average;
pub mod counter;
pub mod io;
pub mod observables;
pub mod pfaffian;
pub mod qp;
pub mod reducer;
pub mod run;
pub mod sampling;
pub mod slater_update;
pub mod sr;
pub mod state;
pub mod sync;

pub use mvmc_expert_parsers::ExpertModeData;
pub use pfaffian::{calc_m_all_complex, calc_m_all_real, CalcMAllError};
pub use reducer::{Reducer, SingleProcessReducer};
pub use run::{
    read_initial_def, run_para_opt_from_namelist, vmc_para_opt, RunSummary, FALLBACK_SEED,
};
pub use sampling::driver::{vmc_make_sample, vmc_make_sample_real, SampleStats};
pub use state::{
    ElectronConfiguration, EnergyData, InvMColMajor, OptDataPoint, PfaPackMode, PfaPackWorkspace,
    PhysicalQuantities, SROptData, SamplingWorkspace, SlaterElmFlat, SlaterMatrixData,
    ThreadedPfaPackWorkspace, VmcOptimizationState,
};
