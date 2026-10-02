//! Per-file `.def` parsers (Phase 3 of the port plan).
//!
//! Phase 3 covers the parsers needed to round-trip the four upstream
//! `examples/inputs/*` namelists (Heisenberg chain real / complex /
//! FSZ, Hubbard chain real), and the InterAll/PairHop/DH2/DH4 input contracts. The remaining
//! RBM and OptTrans index sections are implemented.

pub mod coulomb;
pub mod exchange;
pub mod green;
pub mod gutzwiller;
pub mod hund;
pub mod interall;
pub mod jastrow;
pub mod locspin;
pub mod modpara;
pub mod opttrans;
pub mod orbital;
pub mod pairhop;
pub mod qptrans;
pub mod trans;

/// Strict DH2/DH4 neighbor-table parsers.
pub mod doublon_holon;
/// Nine-channel index/optimization-flag RBM parsers.
pub mod rbm;
