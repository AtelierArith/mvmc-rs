use sfmt19937::Sfmt19937Rng;

#[test]
fn counts_actual_words_without_counting_cloned_peeks() {
    let mut rng = Sfmt19937Rng::new(1234);
    assert_eq!(rng.words_consumed(), 0);
    rng.gen_rand32();
    rng.genrand_real2();
    rng.gen_rand64();
    assert_eq!(rng.words_consumed(), 4);
    let mut shadow = rng.clone();
    let mut peek = [0; 624];
    rng.dump_rand32(&mut peek);
    assert_eq!(rng.words_consumed(), 4);
    assert_eq!(shadow.words_consumed(), 4);
    for word in peek {
        assert_eq!(shadow.gen_rand32(), word);
    }
    assert_eq!(shadow.words_consumed(), 628);
    assert_eq!(rng.words_consumed(), 4);
    rng.seed(1234);
    assert_eq!(rng.words_consumed(), 0);
    let seeded = Sfmt19937Rng::new(1234);
    let mut expected = [0; 624];
    seeded.dump_rand32(&mut expected);
    rng.dump_rand32(&mut peek);
    assert_eq!(peek, expected);
}

#[test]
fn bulk_counts_words_once_and_array_initialization_starts_at_zero() {
    let mut rng = Sfmt19937Rng::from_init_key(&[1, 2, 3, 4]);
    assert_eq!(rng.words_consumed(), 0);
    let mut words = vec![0; Sfmt19937Rng::min_array_size32()];
    rng.fill_array32(&mut words);
    assert_eq!(rng.words_consumed(), words.len() as u128);
    rng.gen_rand32();
    assert_eq!(rng.words_consumed(), words.len() as u128 + 1);
    rng.seed(1);
    let mut pairs = vec![0; Sfmt19937Rng::min_array_size64()];
    rng.fill_array64(&mut pairs);
    assert_eq!(rng.words_consumed(), 2 * pairs.len() as u128);
}
