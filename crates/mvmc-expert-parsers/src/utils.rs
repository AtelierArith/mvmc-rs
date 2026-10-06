//! Shared parser / initialization utilities.
//!
//! Structured validation is an explicit read-only API. Optimization
//! flags use declared projection/orbital widths and Julia's component rules.
//! The In*.def overlay integration remains pending.

pub(crate) mod c_numeric;
pub mod file;

/// Post-parse cross-field validation (port of `utils/validation.jl`).
pub mod validation;

/// Variational-parameter initialisation (port of `utils/parameter_init.jl`).
pub mod parameter_init;

/// `In*.def` overlay reader (port of `utils/read_input_parameters.jl`).
pub mod read_input_parameters;

/// Quantum-projection weight init + `gauss_legendre` (port of `utils/qp_weight.jl`).
pub mod qp_weight;

/// Scalar libm functions as C calls them, with the Julia emulation as an explicit opt-in.
pub mod c_math;
pub mod julia_hypot;
/// Julia Float64 logarithmic operations for deterministic kernels.
pub mod julia_log;
/// Julia Float64 trigonometric and hyperbolic operations for deterministic kernels.
pub mod julia_trig;

/// Julia's Float64 exponential operation for deterministic projection ratios.
pub mod julia_exp;

/// `OptFlag` accessors (port of `utils/opt_flag_utils.jl`).
pub mod opt_flag;

/// Orbital + QPTrans matrix builders (port of `utils/orbital_qptrans_utils.jl`).
pub mod orbital_qptrans {}
