use sfmt19937::Sfmt19937Rng;

/// Numbers of the JSON array `"key" : [ ... ]` that follows byte offset `from`.
fn array_after(text: &str, from: usize, key: &str) -> Vec<u64> {
    let start = from + text[from..].find(&format!("\"{key}\" : [")).unwrap();
    let open = start + text[start..].find('[').unwrap();
    let close = open + text[open..].find(']').unwrap();
    text[open + 1..close]
        .split(',')
        .map(|v| v.trim().trim_matches('"').parse().unwrap())
        .collect()
}

#[test]
fn actual_snapshot_matches_independent_native_c_initialization_capture() {
    // Native C 1.4.0 InitParameter/SFMT capture of the canonical GeneralRBM model
    // (seed 12395) after `InitParameter`: tests/fixtures/native13_initialization,
    // provenance.json. No oracle is invoked by this test.
    const CAPTURE: &str =
        include_str!("../../../tests/fixtures/native13_initialization/GeneralRBM_cmp.json");
    let stage = CAPTURE.find("\"stage\" : \"random-initialized\"").unwrap();
    // Canonical JSON: the stage object starts at its first key, `cursor`.
    let object = CAPTURE[..stage].rfind("\"cursor\" :").unwrap();
    let cursor: usize = CAPTURE[object..]
        .split_once(':')
        .unwrap()
        .1
        .split(',')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let count: u128 = CAPTURE[stage..]
        .split("\"word_count\" : \"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(count, 192);
    assert_eq!(cursor, 192);
    let expected: Vec<u32> = array_after(CAPTURE, object, "raw624")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let expected_next: Vec<u32> = array_after(CAPTURE, object, "next624")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    assert_eq!(expected.len(), 624);
    assert_eq!(expected_next.len(), 624);
    let mut rng = Sfmt19937Rng::new(12395);
    for _ in 0..count {
        rng.gen_rand32();
    }
    let before = rng.state_snapshot();
    assert_eq!(before.0.as_slice(), expected);
    assert_eq!(before.1, cursor);
    assert_eq!(rng.words_consumed(), count);
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
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
