//! Pfaffian and LTL-decomposition routines for dense skew-symmetric matrices.
//!
//! Phase 2 of the port plan delivers a **BLAS-free, bit-deterministic
//! scalar port** of the pure-Julia subset of
//! [`PfaPack.jl`](../../../Julia-mVMC/PfaPack.jl):
//!
//! | Rust module | Upstream Julia file | License |
//! |---|---|---|
//! | [`pfaffian`] | `PfaPack.jl/src/pfaffian.jl`          | BSD-3-Clause |
//! | [`ltl`]      | `PfaPack.jl/src/ltl_decomposition.jl` | BSD-3-Clause |
//! | [`utu2`]     | `PfaPack.jl/src/utu2.jl`              | **MPL-2.0**  |
//!
//! The C++ / Fortran FFI shims (`c_wrapper.jl`, `fortran_wrapper.jl`)
//! and their bundled C++ / Fortran sources are intentionally not
//! ported — the upstream optimizer's hot path uses the pure-Julia
//! routines (`calculate_m_all.jl:202` calls `utu2inv!`, not
//! `cimpl_utu2inv!`), so the FFI surface is dead code there.
//!
//! ## Storage convention
//!
//! All matrices are **column-major** square `n x n` (`lda == n`). This
//! matches the upstream Julia / Fortran convention, which the
//! `mVMC-rs` engine inherits. Construct views via [`SqMat::new`].
//!
//! ## Numerical contract
//!
//! * `pfaffian_ltl_{real,complex}` returns a numeric zero (not
//!   `Option`) when `n` is odd or a pivot is zero.
//! * `dsktf2` / `zsktf2` return `Err(zero_pivot_row)` (1-based) when a
//!   pivot column is exactly zero — same `INFO > 0` convention as
//!   Fortran.
//! * Pivot arrays are **1-based** ([`PivotIndex1Based`]) to mirror the
//!   Fortran `IPIV` convention that the post-decomposition routines
//!   (`utu2pfa` / `utu2inv`) rely on.

// Pure-Rust default forbids unsafe. The `blas-backend` feature turns the
// `backend` module into a thin FFI shim around the `blas` / `lapack`
// crates, which require `unsafe`; that module overrides the deny with
// its own `#![allow(unsafe_code)]`.
#![cfg_attr(not(feature = "blas-backend"), forbid(unsafe_code))]
#![cfg_attr(feature = "blas-backend", deny(unsafe_code))]
#![warn(missing_docs)]

mod mat;

pub(crate) mod backend;

pub mod ltl;
pub mod pfaffian;
pub mod utu2;

pub use ltl::{dsktf2, zsktf2};
pub use mat::SqMat;
pub use pfaffian::{pfaffian_ltl_complex, pfaffian_ltl_real};
pub use utu2::{utu2inv_complex, utu2inv_real, utu2pfa_complex, utu2pfa_real};

/// One-based pivot index newtype.
///
/// The `zsktf2` / `dsktf2` / `utu2pfa` / `utu2inv` algorithms count
/// swaps using 1-based pivot arrays (matches Wimmer's Fortran `IPIV`
/// convention). We isolate that convention here so the rest of the
/// workspace can stay 0-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PivotIndex1Based(pub u32);
