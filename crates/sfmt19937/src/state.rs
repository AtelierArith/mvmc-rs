//! Internal SFMT19937 state + 128-bit recursion.
//!
//! Direct line-by-line port of the scalar (non-SSE2 / non-AltiVec) C
//! reference at `extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT.c`. The C
//! file selects between SSE2, AltiVec, and a portable scalar fallback;
//! the bundled `Makefile` of `SFMT.jl/deps/sfmt` builds the scalar
//! fallback (no `-DHAVE_SSE2` / `-DHAVE_ALTIVEC`), so we port the
//! scalar branch.
//!
//! Endianness: this port is LITTLE-ENDIAN only — it mirrors the C
//! `#else` branches of `rshift128` / `lshift128` (the non-`ONLY64`,
//! non-`BIG_ENDIAN64` paths). The upstream Julia / C build targets are
//! all LE (x86_64-linux, x86_64-macos, aarch64-apple-darwin); on a
//! hypothetical BE host the stream would diverge silently. We surface
//! that as a compile-time error to keep bit-parity claims honest.

use crate::params::{MSK1, MSK2, MSK3, MSK4, N, N32, POS1, SL1, SL2, SR1, SR2};

#[cfg(target_endian = "big")]
compile_error!(
    "sfmt19937 currently only supports little-endian targets; \
     the C reference branches we port (`#else` arms of rshift128/lshift128) \
     are LITTLE_ENDIAN-only. Big-endian support would require porting the \
     `ONLY64` / `BIG_ENDIAN64` branches and a `swap()` step in fill_array64."
);

/// A 128-bit SFMT lane, viewed as four 32-bit words.
///
/// Matches the C `union W128_T { uint32_t u[4]; ... }` layout. The
/// recursion accesses lane `u[0..3]` directly, so this type is just a
/// thin newtype around `[u32; 4]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct W128(pub [u32; 4]);

/// The full SFMT19937 internal state.
///
/// * `state[0..N]` — the 128-bit blocks (the C `static w128_t sfmt[N]`).
/// * `idx` — the 32-bit index counter (the C `static int idx`).
///
/// The C reference treats the state as a flat array of `u32` via the
/// `psfmt32` pointer. Because we restrict to little-endian, `psfmt32[i]`
/// corresponds to `state[i / 4].0[i % 4]` here. We expose this via
/// [`Self::word`] / [`Self::word_mut`] helpers so the seeding / parity
/// code reads like the C source.
#[derive(Debug, Clone)]
pub(crate) struct SfmtState {
    pub(crate) state: [W128; N],
    /// `u32` cursor into `state` (range `0..=N32`; `N32` means "buffer empty, refill on next draw").
    pub(crate) idx: usize,
}

impl Default for SfmtState {
    fn default() -> Self {
        Self {
            state: [W128::default(); N],
            // C reference sets idx = N32 after init / fill so the next
            // gen_rand32 triggers a gen_rand_all first. We mirror that.
            idx: N32,
        }
    }
}

impl SfmtState {
    /// Read the `i`-th 32-bit word (`psfmt32[i]` in the C reference).
    #[inline]
    pub(crate) fn word(&self, i: usize) -> u32 {
        // LE-only: idxof(i) == i. See params.rs / lib.rs comments.
        self.state[i >> 2].0[i & 3]
    }

    /// Write the `i`-th 32-bit word (`psfmt32[i] = v` in the C reference).
    #[inline]
    pub(crate) fn word_mut(&mut self, i: usize, v: u32) {
        self.state[i >> 2].0[i & 3] = v;
    }
}

/// 128-bit left shift by `shift` BYTES (`shift * 8` bits), LE flavour.
///
/// Ports the `#else` (non-`ONLY64`) branch of the C `lshift128`
/// function. The original is a 128-bit shift implemented via two 64-bit
/// halves; we reproduce it verbatim so the output bit pattern is
/// identical (in particular, the C version mixes 32-bit halves via
/// `(uint32_t)(ol >> 32)` etc., which we keep).
#[inline]
fn lshift128(out: &mut W128, x: &W128, shift: u32) {
    let th: u64 = ((x.0[3] as u64) << 32) | (x.0[2] as u64);
    let tl: u64 = ((x.0[1] as u64) << 32) | (x.0[0] as u64);

    let shift_bits = shift * 8;
    let oh: u64 = (th << shift_bits) | (tl >> (64 - shift_bits));
    let ol: u64 = tl << shift_bits;

    out.0[1] = (ol >> 32) as u32;
    out.0[0] = ol as u32;
    out.0[3] = (oh >> 32) as u32;
    out.0[2] = oh as u32;
}

/// 128-bit right shift by `shift` BYTES (`shift * 8` bits), LE flavour.
///
/// Ports the `#else` (non-`ONLY64`) branch of the C `rshift128`.
#[inline]
fn rshift128(out: &mut W128, x: &W128, shift: u32) {
    let th: u64 = ((x.0[3] as u64) << 32) | (x.0[2] as u64);
    let tl: u64 = ((x.0[1] as u64) << 32) | (x.0[0] as u64);

    let shift_bits = shift * 8;
    let oh: u64 = th >> shift_bits;
    let ol: u64 = (tl >> shift_bits) | (th << (64 - shift_bits));

    out.0[1] = (ol >> 32) as u32;
    out.0[0] = ol as u32;
    out.0[3] = (oh >> 32) as u32;
    out.0[2] = oh as u32;
}

/// The SFMT recursion `r = a XOR lshift(a) XOR ((b >> SR1) & MSK) XOR rshift(c) XOR (d << SL1)`.
///
/// Ports the scalar `#else` (non-`ONLY64`, non-`HAVE_SSE2`,
/// non-`HAVE_ALTIVEC`) branch of `do_recursion`. The per-lane MSK
/// assignment matches the LE (non-`ONLY64`) branch:
///   lane 0 -> MSK1, lane 1 -> MSK2, lane 2 -> MSK3, lane 3 -> MSK4.
///
/// `a`, `b`, `c`, `d` are taken by value (they are `Copy`) so the
/// caller can pass independent indices into the same backing array
/// without aliasing-related borrow-checker fights.
#[inline]
fn do_recursion(a: W128, b: W128, c: W128, d: W128) -> W128 {
    let mut x = W128::default();
    let mut y = W128::default();
    lshift128(&mut x, &a, SL2);
    rshift128(&mut y, &c, SR2);

    let r0 = a.0[0] ^ x.0[0] ^ ((b.0[0] >> SR1) & MSK1) ^ y.0[0] ^ (d.0[0] << SL1);
    let r1 = a.0[1] ^ x.0[1] ^ ((b.0[1] >> SR1) & MSK2) ^ y.0[1] ^ (d.0[1] << SL1);
    let r2 = a.0[2] ^ x.0[2] ^ ((b.0[2] >> SR1) & MSK3) ^ y.0[2] ^ (d.0[2] << SL1);
    let r3 = a.0[3] ^ x.0[3] ^ ((b.0[3] >> SR1) & MSK4) ^ y.0[3] ^ (d.0[3] << SL1);
    W128([r0, r1, r2, r3])
}

/// Refill the entire internal state (`gen_rand_all`).
///
/// Direct port of the scalar `gen_rand_all` in `SFMT.c`. Splits the
/// loop into the two C-style halves (`i < N - POS1` and `i >= N - POS1`)
/// so the `sfmt[i + POS1]` vs `sfmt[i + POS1 - N]` indexing matches.
pub(crate) fn gen_rand_all(s: &mut SfmtState) {
    let mut r1: W128 = s.state[N - 2];
    let mut r2: W128 = s.state[N - 1];

    for i in 0..(N - POS1) {
        let a = s.state[i];
        let b = s.state[i + POS1];
        let new_i = do_recursion(a, b, r1, r2);
        s.state[i] = new_i;
        r1 = r2;
        r2 = new_i;
    }
    for i in (N - POS1)..N {
        let a = s.state[i];
        let b = s.state[i + POS1 - N];
        let new_i = do_recursion(a, b, r1, r2);
        s.state[i] = new_i;
        r1 = r2;
        r2 = new_i;
    }
}

/// Bulk variant of [`gen_rand_all`] used by `fill_array32` /
/// `fill_array64`.
///
/// Ports the scalar `gen_rand_array(w128_t *array, int size)` from
/// `SFMT.c`. `array` here is `&mut [W128]` and its length is `size` (in
/// 128-bit blocks). The C contract requires `size >= N` and (for
/// `fill_array32`) `size % 4 == 0`; we re-assert that.
///
/// `#[allow(clippy::needless_range_loop)]`: the four `for i in ...`
/// loops are direct line-by-line ports of the upstream C
/// `for (i = 0; i < ...; i++)` loops; rewriting via iterators would
/// obscure the 1:1 correspondence and make the array-aliasing pattern
/// (state vs. array) harder to audit.
#[allow(clippy::needless_range_loop)]
pub(crate) fn gen_rand_array(s: &mut SfmtState, array: &mut [W128]) {
    let size = array.len();
    assert!(size >= N, "gen_rand_array: size must be >= N ({N})");

    let mut r1: W128 = s.state[N - 2];
    let mut r2: W128 = s.state[N - 1];

    // Phase 1: array[i] = recur(sfmt[i], sfmt[i+POS1], r1, r2), i in [0, N-POS1)
    for i in 0..(N - POS1) {
        let a = s.state[i];
        let b = s.state[i + POS1];
        let new_i = do_recursion(a, b, r1, r2);
        array[i] = new_i;
        r1 = r2;
        r2 = new_i;
    }
    // Phase 2: array[i] = recur(sfmt[i], array[i+POS1-N], r1, r2), i in [N-POS1, N)
    for i in (N - POS1)..N {
        let a = s.state[i];
        let b = array[i + POS1 - N];
        let new_i = do_recursion(a, b, r1, r2);
        array[i] = new_i;
        r1 = r2;
        r2 = new_i;
    }
    // Phase 3: array[i] = recur(array[i-N], array[i+POS1-N], r1, r2), i in [N, size-N)
    let mut i = N;
    while i < size.saturating_sub(N) {
        let a = array[i - N];
        let b = array[i + POS1 - N];
        let new_i = do_recursion(a, b, r1, r2);
        array[i] = new_i;
        r1 = r2;
        r2 = new_i;
        i += 1;
    }
    // Phase 4: copy the trailing 2N-size blocks back into sfmt[],
    // continuing the recursion for the final `size - i` blocks.
    // Translated verbatim from the C `for (j = 0; j < 2*N - size; j++)`
    // / `for (; i < size; i++, j++)` pair.
    let mut j = 0usize;
    while j < 2 * N - size {
        s.state[j] = array[j + size - N];
        j += 1;
    }
    while i < size {
        let a = array[i - N];
        let b = array[i + POS1 - N];
        let new_i = do_recursion(a, b, r1, r2);
        array[i] = new_i;
        r1 = r2;
        r2 = new_i;
        s.state[j] = new_i;
        i += 1;
        j += 1;
    }
}
