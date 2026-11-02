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

fn set_affinity(cpus: &BTreeSet<usize>) {
    assert!(!cpus.is_empty() && cpus.iter().all(|&cpu| cpu < libc::CPU_SETSIZE as usize));
    // A zeroed cpu_set_t is valid mask storage and every index was bounded above.
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    for &cpu in cpus {
        unsafe { libc::CPU_SET(cpu, &mut mask) };
    }
    // This test changes only the calling OS worker's affinity, using a valid buffer.
    assert_eq!(
        unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) },
        0
    );
}

#[test]
#[ignore = "requires explicit Linux MPICH 2/4-rank launch with timeout"]
fn node_cpu_placement_applies_restores_and_recovers() {
    if std::env::var_os("MVMC_PLACEMENT_EXISTING_POOL").is_some() {
        mvmc_core::threading::install(run_placement_case);
    } else {
        run_placement_case();
    }
}

fn run_placement_case() {
    let world = MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let before = affinity();
    let before_pool = std::env::var_os("MVMC_PLACEMENT_EXISTING_POOL")
        .map(|_| mvmc_core::threading::install(|| rayon::broadcast(|_| affinity())));
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
    if std::env::var_os("MVMC_PLACEMENT_WORKER_FAILURE").is_some() {
        let original = before_pool
            .as_ref()
            .expect("failure test needs an existing pool");
        assert!(original.len() >= 2);
        let mut probe = world.restrict_cpu_placement(requested).unwrap();
        let planned = probe.cpus().clone();
        assert!(!world.any_failure(probe.restore().is_err()));
        let caller = rayon::current_thread_index().unwrap();
        let worker = (caller + 1) % original.len();
        let mut narrowed = BTreeSet::from([*original[worker].last().unwrap()]);
        if narrowed == planned {
            narrowed = BTreeSet::from([*original[worker].first().unwrap()]);
        }
        assert!(
            !planned.is_subset(&narrowed),
            "needs at least two allowed CPUs"
        );
        let last = world.rank() + 1 == world.world_size();
        if last {
            rayon::broadcast(|context| {
                if context.index() == worker {
                    set_affinity(&narrowed);
                }
            });
        }
        // The initializing/calling worker is unchanged, so planning still succeeds.
        // Only the other worker on the last rank rejects the selected subset.
        let failed = world.restrict_cpu_placement(requested).is_err();
        let restored = rayon::broadcast(|_| affinity())
            .iter()
            .enumerate()
            .all(|(index, mask)| {
                mask == if last && index == worker {
                    &narrowed
                } else {
                    &original[index]
                }
            });
        assert!(!world.any_failure(!failed || !restored || affinity() != before));
        if last {
            rayon::broadcast(|context| {
                if context.index() == worker {
                    set_affinity(&original[worker]);
                }
            });
        }
        assert!(!world.any_failure(rayon::broadcast(|_| affinity()) != *original));
        println!(
            "rank={} worker_application_failure_recovered=true",
            world.rank()
        );
    }
    let mut guard = world.restrict_cpu_placement(requested).unwrap();
    let actual = affinity();
    let inherited = std::thread::spawn(affinity).join().unwrap();
    let pool_valid = before_pool.as_ref().is_none_or(|_| {
        mvmc_core::threading::install(|| rayon::broadcast(|_| affinity()))
            .iter()
            .all(|mask| mask == &actual)
    });
    let valid = inherited == actual
        && pool_valid
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
    let pool_restored = before_pool.is_none_or(|original| {
        original == mvmc_core::threading::install(|| rayon::broadcast(|_| affinity()))
    });
    assert!(!world.any_failure(!pool_restored));
}
