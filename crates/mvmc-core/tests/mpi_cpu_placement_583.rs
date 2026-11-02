#![cfg(all(feature = "mpi", target_os = "linux"))]
use mpi::traits::*;
use mvmc_core::{mpi::MpiContext, Reducer};
use std::collections::BTreeSet;

fn affinity() -> BTreeSet<usize> {
    let status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
    let list = status
        .lines()
        .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
        .unwrap();
    let mut result = BTreeSet::new();
    for part in list.trim().split(',') {
        let values: Vec<usize> = part.split('-').map(|v| v.parse().unwrap()).collect();
        result.extend(values[0]..=*values.last().unwrap());
    }
    result
}

#[test]
#[ignore = "requires explicit Linux MPICH 2/4-rank launch with timeout"]
fn node_cpu_placement_applies_restores_and_recovers() {
    let world = MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let before = affinity();
    let bad = std::env::var_os("MVMC_PLACEMENT_BAD_LAST").is_some();
    if bad {
        let request = usize::from(world.rank() + 1 != world.world_size());
        let failed = world.restrict_cpu_placement(request).is_err();
        assert!(!world.any_failure(!failed || affinity() != before));
    }
    let requested: usize = std::env::var("MVMC_PLACEMENT_REQUEST")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    let mut guard = world.restrict_cpu_placement(requested).unwrap();
    let actual = affinity();
    let inherited = std::thread::spawn(affinity).join().unwrap();
    let valid = inherited == actual
        && actual == *guard.cpus()
        && actual.len() == requested
        && actual.is_subset(&before);
    assert!(!world.any_failure(!valid));
    let local: Vec<u8> = (0..1024)
        .map(|cpu| u8::from(actual.contains(&cpu)))
        .collect();
    // CPU identifiers are node-local: identical ids on different hosts are valid.
    let node = world.world().split_shared(world.world().rank());
    let mut all = vec![0u8; local.len() * node.size() as usize];
    node.all_gather_into(&local[..], &mut all[..]);
    let disjoint = (0..1024).all(|cpu| {
        all.as_chunks::<1024>()
            .0
            .iter()
            .map(|rank| usize::from(rank[cpu]))
            .sum::<usize>()
            <= 1
    });
    assert!(!world.any_failure(!disjoint));
    println!(
        "rank={} assigned={actual:?} before={before:?} recovered={bad}",
        world.rank()
    );
    let restored = guard.restore().is_ok() && affinity() == before;
    assert!(!world.any_failure(!restored));
}
