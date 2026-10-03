//! Independent literals from SFMT.jl/test/runtests.jl, submodule revision
//! 1526553009f318ae78338151460fda78beadddc2, source SHA256
//! 179890c4bb99c8f099960cd27189625d65af4cdcf43b9fc7ee37b039e763eaba.
//! The source labels these upstream C SFMT seed-11272 values. No oracle is
//! executed by these tests. Rust owns state instead of Julia's global wrapper.
use sfmt19937::Sfmt19937Rng;

const WORDS: [u32; 5] = [0x56a6be38, 0x961c03ac, 0xf1066bc8, 0x7e97a109, 0x76de252f];
const REAL2: [f64; 5] = [
    0.33848179690539837,
    0.586364964954555,
    0.9415042269974947,
    0.49450117559172213,
    0.46432716748677194,
];

#[test]
fn five_literal_c_words_and_real2_values_match_public_methods() {
    let mut rng = Sfmt19937Rng::new(11272);
    for (index, expected) in WORDS.into_iter().enumerate() {
        let actual: u32 = rng.gen_rand32();
        assert_eq!(actual, expected);
        assert_eq!(rng.words_consumed(), index as u128 + 1);
    }
    rng.seed(11272);
    assert_eq!(rng.words_consumed(), 0);
    for (index, expected) in REAL2.into_iter().enumerate() {
        let actual: f64 = rng.genrand_real2();
        // RNG conversion is an exact discrete contract, not computed-kernel
        // floating-point parity subject to a numerical tolerance.
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(rng.words_consumed(), index as u128 + 1);
    }
}

#[test]
fn two_actual_reseeds_repeat_all_five_real2_draws_and_reset_count() {
    let mut rng = Sfmt19937Rng::new(11272);
    let first: [f64; 5] = std::array::from_fn(|_| rng.genrand_real2());
    assert_eq!(first, REAL2);
    for _ in 0..2 {
        rng.seed(11272);
        assert_eq!(rng.words_consumed(), 0);
        for (index, expected) in REAL2.into_iter().enumerate() {
            let actual: f64 = rng.genrand_real2();
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(actual.to_bits(), first[index].to_bits());
        }
        assert_eq!(rng.words_consumed(), 5);
    }
}

#[test]
fn public_u32_types_after_five_draws_preserve_one_word_consumption() {
    let mut rng = Sfmt19937Rng::new(11272);
    for expected in WORDS {
        assert_eq!(rng.gen_rand32(), expected);
    }
    let _: u32 = rng.gen_rand32();
    assert_eq!(rng.words_consumed(), 6);
    let _: u32 = rng.gen_rand32();
    assert_eq!(rng.words_consumed(), 7);
}

#[test]
fn dump_three_literal_words_is_nonconsuming_and_next_three_match() {
    let mut rng = Sfmt19937Rng::new(11272);
    let mut peeked = [0_u32; 3];
    rng.dump_rand32(&mut peeked);
    assert_eq!(peeked, [0x56a6be38, 0x961c03ac, 0xf1066bc8]);
    assert_eq!(rng.words_consumed(), 0);
    for (index, expected) in [0x56a6be38, 0x961c03ac, 0xf1066bc8].into_iter().enumerate() {
        assert_eq!(rng.gen_rand32(), expected);
        assert_eq!(rng.words_consumed(), index as u128 + 1);
    }
}
