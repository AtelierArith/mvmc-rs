//! Pure-Rust port of the mVMC StdFace (Standard mode): read a `stan.in`-style keyword file and
//! generate the Expert-mode input files (`namelist.def`, `modpara.def`, `trans.def`, ...).
//!
//! The module layout mirrors `extern/mVMC-1.3.0/src/StdFace/src/`:
//!
//! | Rust module | C source |
//! | --- | --- |
//! | [`vals`] | `StdFace_vals.h` and `StdFace_ResetVals` |
//! | [`stdface_main()`](fn@stdface_main) | `StdFace_main.c` (keyword reader, parameter checks, file writers) |
//! | [`model_util`] | `StdFace_ModelUtil.c` (helpers, super-cell setup, Jastrow/orbital/projection) |
//! | [`chain_lattice`] | `ChainLattice.c` |
//!
//! Only the mVMC (`_mVMC`) solver branches are ported. Output text is byte-identical to the C
//! program: see `tests/stdface_c_fixtures.rs` and `tests/fixtures/stdface/PROVENANCE.md`.
//! Remaining lattices are tracked in issues #354-#357.

// The C loops are index loops over parallel arrays (`Cell`, `Orb`, `box`, ...); they are kept
// as `for i in 0..n` so each routine can be audited line by line against the C source.
#![allow(clippy::needless_range_loop)]

pub mod ccomplex;
pub mod cexpr;
pub mod cfmt;
pub mod chain_lattice;
pub mod honeycomb_lattice;
pub mod kagome;
pub mod ladder;
pub mod model_util;
pub mod out;
pub mod square_lattice;
pub mod stdface_main;
pub mod triangular_lattice;
pub mod vals;

pub use out::{Out, StdFaceError};
pub use stdface_main::{stdface_main, stdface_main_bytes, StdFaceFailure, StdFaceReport};
