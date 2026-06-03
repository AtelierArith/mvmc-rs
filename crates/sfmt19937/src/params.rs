//! SFMT19937 compile-time parameters.
//!
//! Directly transcribed from `extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT-params19937.h`
//! and the generic SFMT layout constants in `SFMT-params.h`. The values
//! must match the upstream C #defines bit-for-bit; otherwise the
//! 32-bit stream diverges silently at the first `gen_rand_all` call.

/// Mersenne exponent (period = 2^MEXP - 1).
pub const MEXP: usize = 19937;

/// Number of 128-bit blocks in the internal state.
///
/// Same as the C `#define N (MEXP / 128 + 1)`.
pub const N: usize = MEXP / 128 + 1;

/// Internal state size when viewed as a `u32` array (`N * 4`).
pub const N32: usize = N * 4;

/// Internal state size when viewed as a `u64` array (`N * 2`).
pub const N64: usize = N * 2;

/// Pick-up position for the recursion (`SFMT-params19937.h`).
pub const POS1: usize = 122;

/// Left shift amount applied to each 32-bit lane (`SL1`).
pub const SL1: u32 = 18;

/// 128-bit left shift in bytes (`SL2 * 8` actual bits, see `lshift128`).
pub const SL2: u32 = 1;

/// Right shift amount applied to each 32-bit lane (`SR1`).
pub const SR1: u32 = 11;

/// 128-bit right shift in bytes (`SR2 * 8` actual bits, see `rshift128`).
pub const SR2: u32 = 1;

/// Per-lane mask 1 (`MSK1`).
pub const MSK1: u32 = 0xdfff_ffef;
/// Per-lane mask 2 (`MSK2`).
pub const MSK2: u32 = 0xddfe_cb7f;
/// Per-lane mask 3 (`MSK3`).
pub const MSK3: u32 = 0xbffa_ffff;
/// Per-lane mask 4 (`MSK4`).
pub const MSK4: u32 = 0xbfff_fff6;

/// Period-certification parity vector (`PARITY1..4`).
pub const PARITY: [u32; 4] = [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x13c9_e684];

/// SFMT identification string (`IDSTR` from `SFMT-params19937.h`).
pub const IDSTR: &str = "SFMT-19937:122-18-1-11-1:dfffffef-ddfecb7f-bffaffff-bffffff6";
