//! MultiDef (`vmc.out -m N`, issue #348) communicator-split arithmetic against the native C
//! formula.
//!
//! `tests/fixtures/multidef_348/split_c_probe.txt` holds the colour computed by the verbatim
//! statements of `vmcmain.c:752-759` for every `1 <= N <= size <= 48` and every rank
//! (`c_toolbox/multidef_348/generate_split.sh`, provenance in `split_PROVENANCE.txt`). The
//! Rust rank-to-group map must reproduce each line exactly (integer arithmetic).

use std::fs;
use std::path::PathBuf;

use mvmc_core::multidef::{check_world, group_of_rank, group_size};

#[test]
fn rank_to_group_map_equals_the_c_formula_for_all_size_and_n() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/multidef_348/split_c_probe.txt");
    let text = fs::read_to_string(path).unwrap();
    let mut lines = 0;
    for line in text.lines() {
        let fields: Vec<usize> = line
            .split_whitespace()
            .map(|field| field.parse().unwrap())
            .collect();
        let (size, n, groups) = (fields[0], fields[1], &fields[2..]);
        assert_eq!(groups.len(), size, "{line}");
        for (rank, &expected) in groups.iter().enumerate() {
            assert_eq!(
                group_of_rank(rank, size, n),
                expected,
                "size {size} N {n} rank {rank}"
            );
        }
        for group in 0..n {
            let members = groups.iter().filter(|&&g| g == group).count();
            assert_eq!(
                group_size(group, size, n),
                members,
                "size {size} N {n} group {group}"
            );
        }
        assert!(check_world(size, n as i64).is_ok());
        lines += 1;
    }
    assert_eq!(lines, (1..=48).sum::<usize>());
}

#[test]
fn world_smaller_than_n_is_the_c_error_and_uneven_split_the_c_warning() {
    let messages = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/multidef_348/c_messages");
    let c_error = fs::read_to_string(messages.join("size_lt_n.first_stderr_line")).unwrap();
    assert_eq!(check_world(1, 2).unwrap_err(), c_error.trim_end());
    let c_warning = fs::read_to_string(messages.join("r3_stderr.txt")).unwrap();
    assert_eq!(
        check_world(3, 2).unwrap().warning.unwrap(),
        c_warning.trim_end()
    );
}
