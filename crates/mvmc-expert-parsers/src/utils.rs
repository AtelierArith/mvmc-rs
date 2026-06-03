//! Shared parser / initialization utilities.
//!
//! Phase 4 status: `file`, `qp_weight`, `parameter_init` and
//! `read_input_parameters` carry real ports. `validation` /
//! `opt_flag` / `orbital_qptrans` stay as empty stubs because the
//! Phase 4.8 driver does not need them yet.

pub mod file;

/// Post-parse cross-field validation (port of `utils/validation.jl`).
/// Not yet implemented — the round-trip parser path does not need it.
pub mod validation {}

/// Variational-parameter initialisation (port of `utils/parameter_init.jl`).
pub mod parameter_init;

/// `In*.def` overlay reader (port of `utils/read_input_parameters.jl`).
pub mod read_input_parameters;

/// Quantum-projection weight init + `gauss_legendre` (port of `utils/qp_weight.jl`).
pub mod qp_weight;

/// `OptFlag` accessors (port of `utils/opt_flag_utils.jl`).
pub mod opt_flag {}

/// Orbital + QPTrans matrix builders (port of `utils/orbital_qptrans_utils.jl`).
pub mod orbital_qptrans {}
