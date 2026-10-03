use sfmt19937::Sfmt19937Rng;

#[test]
fn actual_snapshot_matches_independent_native_c_initialization_capture() {
    // Native C InitParameter/SFMT capture; compiler, source and seed provenance
    // are checked in beside these fixtures. No oracle is invoked by this test.
    const STATE: &str = include_str!("../../../tests/fixtures/reviewed_parameter_c_audit/canonical_general_rbm/group-1-initialized-state.txt");
    const COUNT: &str = include_str!("../../../tests/fixtures/reviewed_parameter_c_audit/canonical_general_rbm/group-1-initialized-draw-count.txt");
    const NEXT: &str = include_str!("../../../tests/fixtures/reviewed_parameter_c_audit/canonical_general_rbm/group-1-initialized-next624.txt");
    let lines: Vec<_> = STATE.lines().collect();
    assert_eq!(lines.len(), 3);
    let expected: Vec<u32> = lines[0]
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(expected.len(), 624);
    let count: u128 = COUNT.trim().parse().unwrap();
    assert_eq!(count, 192);
    let mut rng = Sfmt19937Rng::new(12395);
    for _ in 0..count {
        rng.gen_rand32();
    }
    let before = rng.state_snapshot();
    assert_eq!(before.0.as_slice(), expected);
    assert_eq!(before.1, lines[1].parse::<usize>().unwrap());
    assert_eq!(rng.words_consumed(), lines[2].parse::<u128>().unwrap());
    assert_eq!(rng.words_consumed(), count);
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
    let expected_next: Vec<u32> = NEXT
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(next.as_slice(), expected_next);
    assert_eq!(rng.state_snapshot(), before);
    assert_eq!(rng.words_consumed(), count);
}

#[test]
fn snapshots_preserve_cursor_count_and_future_stream_across_refill_boundaries() {
    for consumed in [0, 1, 623, 624, 625, 2048] {
        let mut rng = Sfmt19937Rng::new(12395);
        for _ in 0..consumed {
            rng.gen_rand32();
        }
        let mut untouched = rng.clone();
        let before = rng.state_snapshot();
        assert_eq!(rng.state_snapshot(), before);
        assert_eq!(rng.words_consumed(), consumed);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), untouched.gen_rand32());
        }
        assert_eq!(rng.state_snapshot(), untouched.state_snapshot());
        assert_eq!(rng.words_consumed(), untouched.words_consumed());
    }
}
