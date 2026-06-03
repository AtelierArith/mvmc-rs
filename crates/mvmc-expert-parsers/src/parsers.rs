//! Per-file `.def` parsers (Phase 3 of the port plan).
//!
//! Phase 3 covers the parsers needed to round-trip the four upstream
//! `examples/inputs/*` namelists (Heisenberg chain real / complex /
//! FSZ, Hubbard chain real). The remaining parsers (`pairhop`,
//! `interall`, 9-channel `rbm`, `doublon_holon`) stay as empty stubs
//! and will land alongside Phase 4 when `mvmc-core` needs them.

pub mod coulomb;
pub mod exchange;
pub mod green;
pub mod gutzwiller;
pub mod hund;
pub mod jastrow;
pub mod locspin;
pub mod modpara;
pub mod orbital;
pub mod qptrans;
pub mod trans;

/// `pairhop.def` parser (port of `pairhop_parser.jl`). Empty for now.
pub mod pairhop {}
/// `interall.def` parser (port of `interall_parser.jl`). Empty for now.
pub mod interall {}
/// 9-channel RBM parsers (port of `rbm_parser.jl`). Empty for now.
pub mod rbm {}
/// 2-site / 4-site doublon-holon parsers (port of `doublon_holon_parser.jl`).
/// Empty for now.
pub mod doublon_holon {}
