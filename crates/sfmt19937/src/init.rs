//! Seeding routines: `init_gen_rand` and `init_by_array`.
//!
//! Direct port of the equivalent C functions in
//! `extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT.c`. The arithmetic uses
//! 32-bit wrapping semantics throughout, so all `+`/`-`/`*` operations
//! on `u32`s are written as `wrapping_*` here.

use crate::params::{N, N32, PARITY};
use crate::state::SfmtState;

/// `psfmt32[idxof(0)] = seed; psfmt32[i] = 1812433253 * (prev ^ (prev >> 30)) + i;`
///
/// Line-by-line equivalent of the C `init_gen_rand`. Note that we use
/// `SfmtState::word{,_mut}` which already collapse `idxof(i) == i` for
/// the little-endian build (the only build we support).
pub(crate) fn init_gen_rand(s: &mut SfmtState, seed: u32) {
    s.word_mut(0, seed);
    for i in 1..N32 {
        let prev = s.word(i - 1);
        let v = 1_812_433_253u32
            .wrapping_mul(prev ^ (prev >> 30))
            .wrapping_add(i as u32);
        s.word_mut(i, v);
    }
    s.idx = N32;
    period_certification(s);
}

#[inline]
fn func1(x: u32) -> u32 {
    (x ^ (x >> 27)).wrapping_mul(1_664_525)
}

#[inline]
fn func2(x: u32) -> u32 {
    (x ^ (x >> 27)).wrapping_mul(1_566_083_941)
}

/// `init_by_array(init_key, key_length)` from `SFMT.c`.
///
/// Used by `init_by_array` callers in the upstream C tests; the Julia
/// optimizer never uses this entry point (`run_para_opt_from_namelist`
/// always seeds with a single `u32` via `init_gen_rand`), but we port
/// it so the crate's API surface matches the C library 1:1 and so
/// downstream consumers can reproduce array-seeded test vectors.
pub(crate) fn init_by_array(s: &mut SfmtState, init_key: &[u32]) {
    let key_length = init_key.len();
    let size = N * 4; // == N32

    let lag: usize = if size >= 623 {
        11
    } else if size >= 68 {
        7
    } else if size >= 39 {
        5
    } else {
        3
    };
    let mid = (size - lag) / 2;

    // memset(sfmt, 0x8b, sizeof(sfmt))  --  every byte = 0x8b, so every u32 = 0x8b8b8b8b.
    for blk in s.state.iter_mut() {
        blk.0 = [0x8b8b_8b8b; 4];
    }

    let count = if key_length + 1 > N32 {
        key_length + 1
    } else {
        N32
    };

    // First mixing pass (the unconditional one before the `count--`).
    let r0 = func1(s.word(0) ^ s.word(mid) ^ s.word(N32 - 1));
    let v_mid = s.word(mid).wrapping_add(r0);
    s.word_mut(mid, v_mid);
    let r0b = r0.wrapping_add(key_length as u32);
    let v_mid_lag = s.word(mid + lag).wrapping_add(r0b);
    s.word_mut(mid + lag, v_mid_lag);
    s.word_mut(0, r0b);

    let count = count - 1;
    let mut i: usize = 1;
    let mut j: usize = 0;
    while j < count && j < key_length {
        let r = func1(s.word(i) ^ s.word((i + mid) % N32) ^ s.word((i + N32 - 1) % N32));
        let p = s.word((i + mid) % N32).wrapping_add(r);
        s.word_mut((i + mid) % N32, p);
        let r = r.wrapping_add(init_key[j]).wrapping_add(i as u32);
        let q = s.word((i + mid + lag) % N32).wrapping_add(r);
        s.word_mut((i + mid + lag) % N32, q);
        s.word_mut(i, r);
        i = (i + 1) % N32;
        j += 1;
    }
    while j < count {
        let r = func1(s.word(i) ^ s.word((i + mid) % N32) ^ s.word((i + N32 - 1) % N32));
        let p = s.word((i + mid) % N32).wrapping_add(r);
        s.word_mut((i + mid) % N32, p);
        let r = r.wrapping_add(i as u32);
        let q = s.word((i + mid + lag) % N32).wrapping_add(r);
        s.word_mut((i + mid + lag) % N32, q);
        s.word_mut(i, r);
        i = (i + 1) % N32;
        j += 1;
    }
    for _ in 0..N32 {
        let r = func2(
            s.word(i)
                .wrapping_add(s.word((i + mid) % N32))
                .wrapping_add(s.word((i + N32 - 1) % N32)),
        );
        let p = s.word((i + mid) % N32) ^ r;
        s.word_mut((i + mid) % N32, p);
        let r = r.wrapping_sub(i as u32);
        let q = s.word((i + mid + lag) % N32) ^ r;
        s.word_mut((i + mid + lag) % N32, q);
        s.word_mut(i, r);
        i = (i + 1) % N32;
    }

    s.idx = N32;
    period_certification(s);
}

/// Period-certification (`period_certification` in `SFMT.c`).
///
/// Computes the parity of `(state[0..4] & PARITY[0..4])` and, if even,
/// flips the lowest set bit of the first matching word so the
/// 2^MEXP period is guaranteed. Port is bit-faithful to the C source.
///
/// `#[allow(clippy::needless_range_loop)]`: the `for i in 0..4 { ... PARITY[i] ... }`
/// loops mirror the C `for (i = 0; i < 4; i++) ... parity[i] ...` shape
/// 1:1; rewriting via `.iter().enumerate()` would obscure that
/// correspondence.
#[allow(clippy::needless_range_loop)]
fn period_certification(s: &mut SfmtState) {
    let mut inner: u32 = 0;
    for i in 0..4 {
        inner ^= s.word(i) & PARITY[i];
    }
    // Fold to a single bit via the same shift-by-shift cascade as C.
    let mut i = 16u32;
    while i > 0 {
        inner ^= inner >> i;
        i >>= 1;
    }
    inner &= 1;
    if inner == 1 {
        return;
    }
    for i in 0..4 {
        let mut work: u32 = 1;
        for _ in 0..32 {
            if (work & PARITY[i]) != 0 {
                let v = s.word(i) ^ work;
                s.word_mut(i, v);
                return;
            }
            work <<= 1;
        }
    }
}
