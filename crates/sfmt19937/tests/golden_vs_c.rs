//! C-reference golden vectors.
//!
//! These are byte-for-byte copies of `gen_rand32` / `gen_rand64` output
//! captured from the upstream C SFMT library bundled with the Julia
//! port (`extern/Julia-mVMC/SFMT.jl/deps/sfmt/libsfmt.dylib`, scalar
//! `-DMEXP=19937` build, no SSE2 / AltiVec / ONLY64 / BIG_ENDIAN64).
//!
//! Reproduce the vectors with the small dumper at the bottom of this
//! file's accompanying comment; the test then asserts strict equality.
//! Any divergence here is a bit-parity regression.
//!
//! Provenance (commit-time):
//!
//! ```text
//! $ cat > dump_sfmt.c <<'EOF'
//! #include <stdio.h>
//! #include <stdint.h>
//! #include <stdlib.h>
//! extern void init_gen_rand(uint32_t);
//! extern uint32_t gen_rand32(void);
//! extern uint64_t gen_rand64(void);
//! int main(int argc, char **argv) {
//!     init_gen_rand((uint32_t)strtoul(argv[1], NULL, 0));
//!     int n = atoi(argv[2]);
//!     for (int i = 0; i < n; i++) printf("0x%08x\n", gen_rand32());
//!     return 0;
//! }
//! EOF
//! $ clang -O2 dump_sfmt.c -L extern/Julia-mVMC/SFMT.jl/deps/sfmt -lsfmt -o dump_sfmt \
//!      -Wl,-rpath,$(pwd)/extern/Julia-mVMC/SFMT.jl/deps/sfmt
//! $ DYLD_LIBRARY_PATH=$(pwd)/extern/Julia-mVMC/SFMT.jl/deps/sfmt \
//!      ./dump_sfmt 11272 16
//! ```

use sfmt19937::Sfmt19937Rng;

/// Golden 32-bit prefix per seed.
///
/// 8 consecutive `gen_rand32()` calls right after `init_gen_rand(seed)`.
/// Covers degenerate seeds (0, u32::MAX) and the C-mVMC fallback (11272).
const GOLDEN_FIRST8_U32: &[(u32, [u32; 8])] = &[
    (
        0,
        [
            0x2e0c_aa58,
            0x0fcf_240a,
            0x3e79_6292,
            0x5f81_4e26,
            0xc91a_29cd,
            0xcb79_02b7,
            0xbf59_0dae,
            0x3377_7fa1,
        ],
    ),
    (
        1,
        [
            0x56a0_faa4,
            0x99cb_63cf,
            0xd9af_b700,
            0xf56a_5f31,
            0x4187_d5cf,
            0x8e86_2e8a,
            0xfeab_0285,
            0x5ef9_6ead,
        ],
    ),
    (
        42,
        [
            0x4446_29bc,
            0x5218_0135,
            0xa52c_d8db,
            0x4690_f18b,
            0x6a51_cc95,
            0x2a7c_0f11,
            0x370e_2409,
            0xb5c4_f336,
        ],
    ),
    (
        1_234_567,
        [
            0xc5dc_8d0e,
            0x6b7a_ea72,
            0xfa95_9863,
            0xd4b2_f0f5,
            0xc2b9_74c2,
            0x0fd3_6855,
            0x5b9c_dba8,
            0x8fa5_c27b,
        ],
    ),
    (
        11_272,
        [
            0x56a6_be38,
            0x961c_03ac,
            0xf106_6bc8,
            0x7e97_a109,
            0x76de_252f,
            0x11e9_7f30,
            0x9113_f692,
            0x48b7_1f76,
        ],
    ),
    (
        u32::MAX,
        [
            0x4990_5cb1,
            0x9a45_8c3c,
            0x5940_dccc,
            0x6ea5_5fa4,
            0xf9b1_320c,
            0xb474_fbf0,
            0x0cca_d353,
            0x6cca_d3fb,
        ],
    ),
];

/// Golden values *around the gen_rand_all refill boundary* (idx == N32 == 624).
///
/// Triggers the recursion path; without this we couldn't tell a buggy
/// `gen_rand_all` from a working scalar pickup. The values were captured
/// for seed 11272 (the C-mVMC fallback), positions 1..=2 (pre-refill),
/// 623..=626 (around the first refill, 1-based positions → 0-based
/// indices 622..=625), 1247..=1249 (around the second refill), 1872..=1873,
/// and 2000 as a sanity tail.
const GOLDEN_BOUNDARY_SEED: u32 = 11_272;
const GOLDEN_BOUNDARY: &[(usize, u32)] = &[
    (0, 0x56a6_be38),
    (1, 0x961c_03ac),
    // last word of the initial buffer (index N32 - 1 == 623, 1-based: 624).
    (622, 0x72b4_0f46),
    (623, 0xab50_5314),
    // first words of the refilled buffer (index N32 .. N32+1).
    (624, 0x22a8_3b85),
    (625, 0x545c_7c27),
    // around the second refill.
    (1246, 0x4e0d_5775),
    (1247, 0xb905_695f),
    (1248, 0x9fbd_0890),
    // around the third refill + tail.
    (1871, 0xebf6_b4d9),
    (1872, 0x4a65_3a5e),
    (1999, 0xae2b_1d66),
];

/// Golden 64-bit prefix for seed 11272.
///
/// Same provenance as above but via `gen_rand64()` on a fresh state.
/// Verifies the LE u64 packing matches the upstream `psfmt64` view.
const GOLDEN_FIRST8_U64_SEED: u32 = 11_272;
const GOLDEN_FIRST8_U64: [u64; 8] = [
    0x961c_03ac_56a6_be38,
    0x7e97_a109_f106_6bc8,
    0x11e9_7f30_76de_252f,
    0x48b7_1f76_9113_f692,
    0xeddd_a83f_b07c_6262,
    0xbc14_c517_b13c_8705,
    0xf31c_2771_3276_e8dd,
    0x9826_d3a1_4a22_c243,
];

#[test]
fn first8_u32_matches_c_for_every_seed() {
    for (seed, golden) in GOLDEN_FIRST8_U32 {
        let mut rng = Sfmt19937Rng::new(*seed);
        for (i, &want) in golden.iter().enumerate() {
            let got = rng.gen_rand32();
            assert_eq!(
                got, want,
                "seed=0x{seed:08x} (={seed}), draw #{i}: got=0x{got:08x}, want=0x{want:08x}"
            );
        }
    }
}

#[test]
fn boundary_indices_match_c_refill_path() {
    let mut rng = Sfmt19937Rng::new(GOLDEN_BOUNDARY_SEED);
    // Generate up to the deepest checkpoint, asserting each one as we pass it.
    let max_idx = GOLDEN_BOUNDARY.iter().map(|(i, _)| *i).max().unwrap();
    let mut checkpoints = GOLDEN_BOUNDARY.iter().peekable();
    for i in 0..=max_idx {
        let v = rng.gen_rand32();
        if let Some(&&(j, want)) = checkpoints.peek() {
            if j == i {
                assert_eq!(
                    v, want,
                    "seed=0x{GOLDEN_BOUNDARY_SEED:08x}, index {i}: got=0x{v:08x}, want=0x{want:08x}"
                );
                checkpoints.next();
            }
        }
    }
    assert!(
        checkpoints.peek().is_none(),
        "some golden checkpoints were not asserted"
    );
}

#[test]
fn first8_u64_matches_c() {
    let mut rng = Sfmt19937Rng::new(GOLDEN_FIRST8_U64_SEED);
    for (i, &want) in GOLDEN_FIRST8_U64.iter().enumerate() {
        let got = rng.gen_rand64();
        assert_eq!(
            got, want,
            "seed=0x{GOLDEN_FIRST8_U64_SEED:08x}, u64 draw #{i}: got=0x{got:016x}, want=0x{want:016x}"
        );
    }
}

#[test]
fn u64_equals_two_u32_draws_on_le() {
    // Cross-check (independent of the goldens above): on the LE port we
    // require gen_rand64() == (gen_rand32() as u64) | ((gen_rand32() as u64) << 32).
    // If this breaks, the most likely cause is a stray `idxof()` flip
    // in `SfmtState::word{,_mut}`.
    let mut a = Sfmt19937Rng::new(11_272);
    let mut b = Sfmt19937Rng::new(11_272);
    for _ in 0..16 {
        let r64 = a.gen_rand64();
        let lo = b.gen_rand32() as u64;
        let hi = b.gen_rand32() as u64;
        assert_eq!(r64, lo | (hi << 32));
    }
}

// ---------------------------------------------------------------------------
// genrand_real2 golden vectors (the VMC sampler hot path).
//
// Captured the same way as the u32 goldens above, but via `genrand_real2()`,
// printed as the raw IEEE 754 bit pattern so the comparison is byte-exact:
//
// ```c
// #include <stdio.h>
// #include <stdint.h>
// extern void init_gen_rand(uint32_t);
// extern double genrand_real2(void);
// int main(int argc, char **argv) {
//     init_gen_rand((uint32_t) atoi(argv[1]));
//     int n = atoi(argv[2]);
//     for (int i = 0; i < n; i++) {
//         double v = genrand_real2();
//         printf("0x%016llx\n", *(unsigned long long*)&v);
//     }
//     return 0;
// }
// ```
//
// `genrand_real2()` is the function the mVMC optimizer actually consumes
// (`vmc_sampling.jl:1019` etc.), so a divergence here would silently
// move every Metropolis acceptance decision off C. We pin the bit
// pattern, not the f64 value, because IEEE 754 round-to-nearest-even
// could otherwise hide low-bit drift in the divisor folding.
// ---------------------------------------------------------------------------

const GOLDEN_REAL2_BITS: &[(u32, [u64; 4])] = &[
    (
        0,
        [
            0x3fc7_0655_2c00_0000,
            0x3faf_9e48_1400_0000,
            0x3fcf_3cb1_4900_0000,
            0x3fd7_e053_8980_0000,
        ],
    ),
    (
        1,
        [
            0x3fd5_a83e_a900_0000,
            0x3fe3_396c_79e0_0000,
            0x3feb_35f6_e000_0000,
            0x3fee_ad4b_e620_0000,
        ],
    ),
    (
        42,
        [
            0x3fd1_118a_6f00_0000,
            0x3fd4_8600_4d40_0000,
            0x3fe4_a59b_1b60_0000,
            0x3fd1_a43c_62c0_0000,
        ],
    ),
    (
        11_272,
        [
            0x3fd5_a9af_8e00_0000,
            0x3fe2_c380_7580_0000,
            0x3fee_20cd_7900_0000,
            0x3fdf_a5e8_4240_0000,
        ],
    ),
];

#[test]
fn genrand_real2_matches_c_bit_pattern() {
    for (seed, golden) in GOLDEN_REAL2_BITS {
        let mut rng = Sfmt19937Rng::new(*seed);
        for (i, &want_bits) in golden.iter().enumerate() {
            let v = rng.genrand_real2();
            let got_bits = v.to_bits();
            assert_eq!(
                got_bits,
                want_bits,
                "seed={seed}, genrand_real2 draw #{i}: got=0x{got_bits:016x} (={v:.17e}), \
                 want=0x{want_bits:016x} (={want_f:.17e})",
                want_f = f64::from_bits(want_bits)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// init_by_array goldens.
//
// Captured by hard-coding three small key arrays in a C dumper and
// printing the first 8 gen_rand32() values per case. Covers:
//   case 0: [1, 2, 3, 4]                          -- short ordered key
//   case 1: [0x12345678, 0x9abcdef0]              -- 2-word, high bits set
//   case 2: [11272]                               -- single-element parity check
//                                                 (matches init_gen_rand(11272)? NO -- different path)
// ---------------------------------------------------------------------------

const GOLDEN_INIT_BY_ARRAY: &[(&[u32], [u32; 8])] = &[
    (
        &[1, 2, 3, 4],
        [
            0xa096_a2bb,
            0xaecd_7807,
            0x58ae_d979,
            0x49d1_49cc,
            0xc985_47d8,
            0x78a8_c739,
            0x1f65_18a3,
            0x2c4c_7404,
        ],
    ),
    (
        &[0x1234_5678, 0x9abc_def0],
        [
            0x28e6_bd5f,
            0x7e4c_c835,
            0x233c_df77,
            0xa369_e23e,
            0xe0ae_b36f,
            0xb80d_3afd,
            0x05dd_aaa3,
            0x0e3d_d28c,
        ],
    ),
    (
        &[11_272],
        [
            0x0694_67c2,
            0xe0de_327d,
            0x894f_c536,
            0xe767_cf4d,
            0x4725_5d88,
            0x2bec_ce02,
            0xb1c3_15a5,
            0xe924_beb1,
        ],
    ),
];

#[test]
fn init_by_array_matches_c() {
    for (key, golden) in GOLDEN_INIT_BY_ARRAY {
        let mut rng = Sfmt19937Rng::from_init_key(key);
        for (i, &want) in golden.iter().enumerate() {
            let got = rng.gen_rand32();
            assert_eq!(
                got, want,
                "init_by_array(key={key:#x?}), draw #{i}: got=0x{got:08x}, want=0x{want:08x}"
            );
        }
    }
}

#[test]
fn init_by_array_single_element_differs_from_init_gen_rand() {
    // Sanity check: init_by_array([seed]) is NOT the same as init_gen_rand(seed).
    // They use completely different mixing schedules (see SFMT.c). If the
    // two ever coincidentally produced the same stream, the test above
    // could be silently testing init_gen_rand twice.
    let mut a = Sfmt19937Rng::from_init_key(&[11_272]);
    let mut b = Sfmt19937Rng::new(11_272);
    let mut differ = false;
    for _ in 0..8 {
        if a.gen_rand32() != b.gen_rand32() {
            differ = true;
            break;
        }
    }
    assert!(
        differ,
        "init_by_array([seed]) unexpectedly matches init_gen_rand(seed) -- one of them is wrong"
    );
}
