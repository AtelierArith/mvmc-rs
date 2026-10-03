//! SFMT19937 PRNG — pure-Rust port of
//! `extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT.{h,c}`.
//!
//! ## Bit-parity guarantee
//!
//! For every public entry point below, the output is byte-equal to the
//! upstream C reference compiled with no `-DHAVE_SSE2` /
//! `-DHAVE_ALTIVEC` / `-DONLY64` / `-DBIG_ENDIAN64` flags (which is
//! what `SFMT.jl/deps/sfmt/Makefile` actually builds). The C library
//! is then statically linked into `libsfmt.dylib` and exposed to Julia
//! via `ccall`, so matching the C output also matches Julia's output,
//! which in turn matches the mVMC C optimizer's RNG draws.
//!
//! The chain we replicate is:
//!
//! ```text
//!   mVMC C optimizer ---uses---> SFMT (vendored C, scalar build)
//!                                  ^
//!                                  |
//!                                  |---also-used-by-- Julia-mVMC v0.1 (via SFMT.jl ccall)
//!                                                                                  ^
//!                                                                                  |
//!                                                          this Rust port targets <-+
//! ```
//!
//! ## Endianness
//!
//! LITTLE-ENDIAN only. The C reference has two parallel
//! implementations of `rshift128` / `lshift128` (and a `swap()` step in
//! `fill_array64`) selected by `BIG_ENDIAN64` / `ONLY64`; we port the
//! LE branches because every Julia-mVMC and mVMC build target is LE
//! (x86_64-linux, x86_64-macos, aarch64-apple-darwin). The crate fails
//! to compile on a big-endian host so the bit-parity claim above stays
//! honest — see `state.rs`.
//!
//! ## Range sampling
//!
//! `Sfmt19937Rng` implements [`rand_core::RngCore`] so it composes with
//! the wider Rust ecosystem, but the canonical mVMC range pattern is
//! `gen_rand32() % len` (biased, no rejection sampling). The C
//! optimizer uses it at e.g. `vmc_sampling.jl:1019`. We expose that
//! pattern explicitly as [`Sfmt19937Rng::gen_rand_mod`] so port code
//! does not accidentally use `rand`'s rejection-sampled `Uniform`
//! distribution and silently desynchronise from C.
//!
//! License: BSD-3-Clause (matches `SFMT.jl`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use rand_core::{Error, RngCore, SeedableRng};

mod init;
mod params;
mod real;
mod state;

pub use params::{IDSTR, MEXP, N, N32, N64};
pub use real::{to_real1, to_real2, to_real3, to_res53, to_res53_mix};

/// C-parity fallback seed (`11272`).
///
/// Matches `vmc_para_opt!` / `run_para_opt_from_namelist`'s convention
/// of substituting `11272` when `modpara.def`'s `RndSeed` is `<= 0`.
pub const FALLBACK_SEED: u32 = 11_272;

/// SFMT19937 random number generator.
///
/// Each instance owns its full internal state (no thread-local globals,
/// unlike the upstream C reference). Constructing with [`Self::new`] is
/// equivalent to `init_gen_rand(seed)`; constructing with
/// [`Self::from_init_key`] is equivalent to `init_by_array(key, len)`.
#[derive(Debug, Clone)]
pub struct Sfmt19937Rng {
    inner: state::SfmtState,
    words_consumed: u128,
}

impl Sfmt19937Rng {
    /// Construct + seed (`init_gen_rand`).
    ///
    /// `seed == 0` is allowed and matches the C behaviour. The
    /// period-certification step adjusts the internal state so a
    /// 0 seed still produces the full 2^MEXP-period stream.
    pub fn new(seed: u32) -> Self {
        let mut s = state::SfmtState::default();
        init::init_gen_rand(&mut s, seed);
        Self {
            inner: s,
            words_consumed: 0,
        }
    }

    /// Construct + seed from a `u32` key array (`init_by_array`).
    ///
    /// Provided for completeness; the Julia-mVMC optimizer always uses
    /// the single-`u32` [`Self::new`] path, but the C SFMT exposes
    /// `init_by_array` and downstream Rust consumers may want to
    /// reproduce array-seeded test vectors.
    pub fn from_init_key(key: &[u32]) -> Self {
        let mut s = state::SfmtState::default();
        init::init_by_array(&mut s, key);
        Self {
            inner: s,
            words_consumed: 0,
        }
    }

    /// Re-seed in place (`init_gen_rand`).
    pub fn seed(&mut self, seed: u32) {
        init::init_gen_rand(&mut self.inner, seed);
        self.words_consumed = 0;
    }

    /// Number of 32-bit words consumed since initialization or reseeding.
    /// Cloned generators have independent counts; non-consuming dumps do not
    /// advance this generator's count or numerical state.
    pub fn words_consumed(&self) -> u128 {
        self.words_consumed
    }

    /// `gen_rand32` from `SFMT.c`.
    ///
    /// Refills the buffer via `gen_rand_all` when exhausted (i.e. when
    /// `idx >= N32`) and returns the next 32-bit word.
    #[inline]
    pub fn gen_rand32(&mut self) -> u32 {
        if self.inner.idx >= N32 {
            state::gen_rand_all(&mut self.inner);
            self.inner.idx = 0;
        }
        let r = self.inner.word(self.inner.idx);
        self.inner.idx += 1;
        self.words_consumed += 1;
        r
    }

    /// `gen_rand64` from `SFMT.c`.
    ///
    /// **Caller contract**: `gen_rand64` must not be interleaved with
    /// `gen_rand32` on the same generator without a re-seed; the C
    /// reference asserts `idx % 2 == 0`. This port preserves the
    /// contract via a `debug_assert!`. In release builds we silently
    /// follow the C behaviour (skip the parity check, read the next
    /// two `u32` words as a `u64`).
    #[inline]
    pub fn gen_rand64(&mut self) -> u64 {
        debug_assert!(
            self.inner.idx.is_multiple_of(2),
            "gen_rand64 requires idx % 2 == 0 (C SFMT contract); \
             did you interleave gen_rand32 and gen_rand64?"
        );
        if self.inner.idx >= N32 {
            state::gen_rand_all(&mut self.inner);
            self.inner.idx = 0;
        }
        // LE-only: psfmt64[idx / 2] == (psfmt32[idx]) | (psfmt32[idx+1] << 32).
        let lo = self.inner.word(self.inner.idx) as u64;
        let hi = self.inner.word(self.inner.idx + 1) as u64;
        self.inner.idx += 2;
        self.words_consumed += 2;
        lo | (hi << 32)
    }

    /// `genrand_real2()` from `SFMT-real.c` (uniform on `[0, 1)`).
    ///
    /// This is the function the mVMC sampler actually consumes; the
    /// `Sz` branch in `vmc_sampling.jl:1019` is `s = (r<0.5) ? 0 : 1`
    /// using exactly this value.
    #[inline]
    pub fn genrand_real2(&mut self) -> f64 {
        real::to_real2(self.gen_rand32())
    }

    /// `genrand_real1()` from `SFMT-real.c` (uniform on `[0, 1]`).
    #[inline]
    pub fn genrand_real1(&mut self) -> f64 {
        real::to_real1(self.gen_rand32())
    }

    /// `genrand_real3()` from `SFMT-real.c` (uniform on `(0, 1)`).
    #[inline]
    pub fn genrand_real3(&mut self) -> f64 {
        real::to_real3(self.gen_rand32())
    }

    /// `genrand_res53()` from `SFMT-real.c` (53-bit precision from a single `u64` draw).
    #[inline]
    pub fn genrand_res53(&mut self) -> f64 {
        real::to_res53(self.gen_rand64())
    }

    /// `genrand_res53_mix()` from `SFMT-real.c` (53-bit precision from
    /// two consecutive `u32` draws).
    #[inline]
    pub fn genrand_res53_mix(&mut self) -> f64 {
        let x = self.gen_rand32();
        let y = self.gen_rand32();
        real::to_res53_mix(x, y)
    }

    /// `sfmt_dump_rand32(out, n)` from `SFMT.c`.
    ///
    /// **Non-destructive**: the C helper saves a copy of the state
    /// before generating into `out` and restores it afterwards, so the
    /// caller's RNG cursor is unchanged. We reproduce that contract by
    /// cloning `self.inner` for the duration of the dump. This matches
    /// what `SFMT.jl/src/SFMT.jl::sfmt_dump_rand32` does and what the
    /// Phase-0.5 fixture extractor relies on for offline diffing.
    pub fn dump_rand32(&self, out: &mut [u32]) {
        let mut shadow = self.clone();
        for slot in out.iter_mut() {
            *slot = shadow.gen_rand32();
        }
    }

    /// `gen_rand32() % n` with C's biased modulo (no rejection sampling).
    ///
    /// Use this anywhere the upstream code says
    /// `gen_rand32() % SomeLength`; `rand`'s `Uniform` distribution
    /// would do rejection sampling and break bit parity.
    ///
    /// # Panics (debug)
    /// Panics in debug builds if `n == 0`; release builds preserve the
    /// C undefined-behaviour (which on x86 traps via `#DE`).
    #[inline]
    pub fn gen_rand_mod(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0, "gen_rand_mod requires n > 0");
        self.gen_rand32() % n
    }

    /// SFMT identification string (`get_idstring` from `SFMT.c`).
    #[inline]
    pub fn id_string() -> &'static str {
        IDSTR
    }

    /// Minimum array size for [`Self::fill_array32`] (`get_min_array_size32`).
    #[inline]
    pub const fn min_array_size32() -> usize {
        N32
    }

    /// Minimum array size for [`Self::fill_array64`] (`get_min_array_size64`).
    #[inline]
    pub const fn min_array_size64() -> usize {
        N64
    }

    /// `fill_array32(array, size)` from `SFMT.c`.
    ///
    /// Bulk variant of [`Self::gen_rand32`] that fills `out` with
    /// `out.len()` consecutive `u32` draws. The C contract requires
    /// `len % 4 == 0`, `len >= N32`, and `idx == N32` (the buffer must
    /// already be exhausted) — the same constraints apply here and are
    /// enforced via debug assertions, matching the C `assert()`s.
    pub fn fill_array32(&mut self, out: &mut [u32]) {
        let size = out.len();
        debug_assert!(
            self.inner.idx == N32,
            "fill_array32 requires the buffer to be empty"
        );
        debug_assert!(
            size.is_multiple_of(4),
            "fill_array32 size must be a multiple of 4"
        );
        debug_assert!(size >= N32, "fill_array32 size must be >= N32 ({N32})");

        // The C version casts `array` to `w128_t *` and calls
        // `gen_rand_array(... size / 4)`. We do the same via a stack
        // intermediate so we don't have to bitcast the caller's `&mut [u32]`
        // to `&mut [W128]` (which would be sound for repr(transparent)
        // but requires unsafe).
        let mut blocks = vec![state::W128::default(); size / 4];
        state::gen_rand_array(&mut self.inner, &mut blocks);
        self.inner.idx = N32;
        self.words_consumed += size as u128;

        for (i, blk) in blocks.iter().enumerate() {
            out[4 * i] = blk.0[0];
            out[4 * i + 1] = blk.0[1];
            out[4 * i + 2] = blk.0[2];
            out[4 * i + 3] = blk.0[3];
        }
    }

    /// `fill_array64(array, size)` from `SFMT.c` (LE-only).
    pub fn fill_array64(&mut self, out: &mut [u64]) {
        let size = out.len();
        debug_assert!(
            self.inner.idx == N32,
            "fill_array64 requires the buffer to be empty"
        );
        debug_assert!(
            size.is_multiple_of(2),
            "fill_array64 size must be a multiple of 2"
        );
        debug_assert!(size >= N64, "fill_array64 size must be >= N64 ({N64})");

        // C: gen_rand_array((w128_t*) array, size / 2) treats `array`
        // as size/2 128-bit blocks; on LE the per-lane u32 ordering
        // collapses to `u64 = lo | (hi << 32)` (no `swap()` needed,
        // since !BIG_ENDIAN64).
        let mut blocks = vec![state::W128::default(); size / 2];
        state::gen_rand_array(&mut self.inner, &mut blocks);
        self.inner.idx = N32;
        self.words_consumed += 2 * size as u128;

        for (i, blk) in blocks.iter().enumerate() {
            let lo0 = blk.0[0] as u64;
            let hi0 = blk.0[1] as u64;
            let lo1 = blk.0[2] as u64;
            let hi1 = blk.0[3] as u64;
            out[2 * i] = lo0 | (hi0 << 32);
            out[2 * i + 1] = lo1 | (hi1 << 32);
        }
    }
}

// ---------------------------------------------------------------------------
// rand_core glue
// ---------------------------------------------------------------------------

impl RngCore for Sfmt19937Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.gen_rand32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // NOTE: we deliberately do NOT call gen_rand64() here. The C
        // contract for gen_rand64 forbids interleaving with gen_rand32,
        // which `rand` callers can't reasonably uphold (rand_core's API
        // explicitly allows mixing next_u32 / next_u64). We synthesize
        // the u64 from two u32 draws instead, which is what the C
        // `genrand_res53_mix` does and matches the byte order callers
        // get on LE hosts.
        let lo = self.gen_rand32() as u64;
        let hi = self.gen_rand32() as u64;
        lo | (hi << 32)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for Sfmt19937Rng {
    type Seed = [u8; 4];

    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(u32::from_le_bytes(seed))
    }

    fn seed_from_u64(seed: u64) -> Self {
        // Matches the (de-facto) Julia convention of casting `Int` to
        // `UInt32(seed)` before calling init_gen_rand; truncation is
        // intentional so callers stay in sync if they ever pass an
        // i64 / u64 seed downstream.
        Self::new(seed as u32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_seed_matches_upstream() {
        assert_eq!(FALLBACK_SEED, 11_272);
    }

    #[test]
    fn id_string_matches_c() {
        assert_eq!(
            Sfmt19937Rng::id_string(),
            "SFMT-19937:122-18-1-11-1:dfffffef-ddfecb7f-bffaffff-bffffff6"
        );
    }

    #[test]
    fn min_array_sizes_match_c() {
        // From the C reference: N == MEXP/128 + 1 == 156; N32 == 624; N64 == 312.
        assert_eq!(N, 156);
        assert_eq!(N32, 624);
        assert_eq!(N64, 312);
        assert_eq!(Sfmt19937Rng::min_array_size32(), 624);
        assert_eq!(Sfmt19937Rng::min_array_size64(), 312);
    }

    #[test]
    fn dump_rand32_is_non_destructive() {
        let mut rng = Sfmt19937Rng::new(FALLBACK_SEED);
        let mut a = [0u32; 8];
        let mut b = [0u32; 8];
        rng.dump_rand32(&mut a);
        rng.dump_rand32(&mut b);
        // Two consecutive dumps from an unmodified state must agree.
        assert_eq!(a, b);
        // And a destructive draw afterwards must equal the first dumped value.
        let next = rng.gen_rand32();
        assert_eq!(next, a[0]);
    }

    #[test]
    fn gen_rand32_advances_state() {
        let mut rng = Sfmt19937Rng::new(1);
        let x = rng.gen_rand32();
        let y = rng.gen_rand32();
        assert_ne!(
            x, y,
            "two consecutive draws should differ for non-degenerate seed"
        );
    }

    #[test]
    fn genrand_real2_is_in_unit_interval() {
        let mut rng = Sfmt19937Rng::new(42);
        for _ in 0..1024 {
            let v = rng.genrand_real2();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn fill_array32_matches_per_call_path() {
        // The C contract guarantees that fill_array32 + later draws is
        // equivalent to N32+ consecutive gen_rand32() calls (modulo the
        // batched-recursion shortcut). Verify that on the first batch.
        let mut a = Sfmt19937Rng::new(7);
        let mut b = Sfmt19937Rng::new(7);

        let mut buf = vec![0u32; N32];
        a.fill_array32(&mut buf);

        for (i, &v) in buf.iter().enumerate() {
            let from_per_call = b.gen_rand32();
            assert_eq!(
                v, from_per_call,
                "fill_array32 / gen_rand32 disagree at index {i}"
            );
        }
    }

    #[test]
    fn gen_rand64_matches_two_u32_draws_on_le() {
        // On LE, gen_rand64 must equal (next_u32 as u64) | ((next_u32 as u64) << 32)
        // when called from a freshly seeded state with idx == N32.
        let mut a = Sfmt19937Rng::new(11_272);
        let mut b = Sfmt19937Rng::new(11_272);

        let r64 = a.gen_rand64();

        // Re-derive from two u32 draws on the parallel state.
        let lo = b.gen_rand32() as u64;
        let hi = b.gen_rand32() as u64;
        assert_eq!(r64, lo | (hi << 32));
    }

    #[test]
    fn rng_core_next_u64_uses_two_u32_draws() {
        let mut a = Sfmt19937Rng::new(11_272);
        let mut b = Sfmt19937Rng::new(11_272);
        let via_next = a.next_u64();
        let lo = b.gen_rand32() as u64;
        let hi = b.gen_rand32() as u64;
        assert_eq!(via_next, lo | (hi << 32));
    }

    #[test]
    fn seed_zero_is_accepted() {
        // The C reference's period_certification step prevents the
        // 0-seed degenerate cycle; we should produce a non-zero draw.
        let mut rng = Sfmt19937Rng::new(0);
        let mut any_nonzero = false;
        for _ in 0..N32 {
            if rng.gen_rand32() != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(
            any_nonzero,
            "period_certification should have fixed up the 0 seed"
        );
    }

    #[test]
    fn seed_from_u64_truncates_like_julia() {
        // Julia: Random.seed!(SFMT19937RNG(), seed::Int) calls init_gen_rand(UInt32(seed)),
        // which is a *checked* narrowing in Julia but a truncation here.
        // The two paths should agree on values that fit in u32.
        let a = Sfmt19937Rng::seed_from_u64(11_272);
        let b = Sfmt19937Rng::new(11_272);
        // Same internal state -> same first draw.
        let mut a = a;
        let mut b = b;
        assert_eq!(a.gen_rand32(), b.gen_rand32());
    }
}
