//! Deterministic CPU-slot feasibility planning; applying affinity is separate.
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(crate) fn plan(
    affinity: &[BTreeSet<usize>],
    required: &[usize],
) -> Result<Vec<BTreeSet<usize>>, String> {
    if affinity.len() != required.len() || required.contains(&0) {
        return Err("each rank needs a positive CPU request and an affinity mask".into());
    }
    let union: BTreeSet<_> = affinity.iter().flat_map(|s| s.iter().copied()).collect();
    let total = required
        .iter()
        .try_fold(0usize, |n, x| n.checked_add(*x))
        .ok_or("CPU request count overflow")?;
    if total > union.len() {
        return Err("requested CPUs exceed the available affinity union".into());
    }
    let slots: Vec<_> = required
        .iter()
        .enumerate()
        .flat_map(|(rank, count)| std::iter::repeat_n(rank, *count))
        .collect();
    let mut owner = BTreeMap::<usize, usize>::new();
    for start in 0..slots.len() {
        let mut queue = VecDeque::from([start]);
        let mut seen_slots = BTreeSet::from([start]);
        let mut seen_cpus = BTreeSet::new();
        let mut previous = BTreeMap::new();
        let mut free = None;
        'search: while let Some(slot) = queue.pop_front() {
            for &cpu in &affinity[slots[slot]] {
                if !seen_cpus.insert(cpu) {
                    continue;
                }
                if let Some(&other) = owner.get(&cpu) {
                    if seen_slots.insert(other) {
                        previous.insert(other, (slot, cpu));
                        queue.push_back(other);
                    }
                } else {
                    free = Some((slot, cpu));
                    break 'search;
                }
            }
        }
        let (mut slot, cpu) = free.ok_or("rank affinity masks cannot satisfy CPU requests")?;
        owner.insert(cpu, slot);
        while let Some(&(parent, cpu)) = previous.get(&slot) {
            owner.insert(cpu, parent);
            slot = parent;
        }
    }
    let mut result = vec![BTreeSet::new(); affinity.len()];
    for (cpu, slot) in owner {
        result[slots[slot]].insert(cpu);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sets(values: &[&[usize]]) -> Vec<BTreeSet<usize>> {
        values.iter().map(|v| v.iter().copied().collect()).collect()
    }
    #[test]
    fn reassigns_flexible_rank_to_preserve_constrained_rank() {
        assert_eq!(
            plan(&sets(&[&[0, 1], &[0]]), &[1, 1]).unwrap(),
            sets(&[&[1], &[0]])
        );
    }
    #[test]
    fn disjoint_masks_keep_full_per_rank_capacity() {
        assert_eq!(
            plan(&sets(&[&[0, 1], &[4, 5]]), &[2, 2]).unwrap(),
            sets(&[&[0, 1], &[4, 5]])
        );
    }
    #[test]
    fn union_capacity_alone_does_not_prove_feasibility() {
        assert!(plan(&sets(&[&[0], &[0], &[1, 2]]), &[1, 1, 1]).is_err());
        assert!(plan(&sets(&[&[0], &[1]]), &[usize::MAX, 1]).is_err());
    }
}

#[cfg(test)]
mod exhaustive_tests {
    use super::*;

    #[test]
    fn matches_independent_enumeration_for_three_cpus_and_three_ranks() {
        for masks in 0usize..512 {
            let affinity: Vec<BTreeSet<usize>> = (0..3)
                .map(|rank| {
                    (0..3)
                        .filter(|cpu| masks & (1 << (rank * 3 + cpu)) != 0)
                        .collect()
                })
                .collect();
            for demands in 0usize..8 {
                let required: Vec<_> = (0..3).map(|rank| 1 + ((demands >> rank) & 1)).collect();
                // Independently enumerate each CPU's owner: 0/1/2, or 3 = unused.
                let feasible = (0usize..64).any(|assignment| {
                    let mut counts = [0usize; 3];
                    for cpu in 0..3 {
                        let rank = (assignment >> (cpu * 2)) & 3;
                        if rank < 3 {
                            if !affinity[rank].contains(&cpu) {
                                return false;
                            }
                            counts[rank] += 1;
                        }
                    }
                    counts.as_slice() == required.as_slice()
                });
                let actual = plan(&affinity, &required);
                assert_eq!(actual.is_ok(), feasible, "masks={masks} demands={demands}");
                if let Ok(assigned) = actual {
                    let mut used = BTreeSet::new();
                    for rank in 0..3 {
                        assert_eq!(assigned[rank].len(), required[rank]);
                        assert!(assigned[rank].is_subset(&affinity[rank]));
                        for cpu in &assigned[rank] {
                            assert!(used.insert(*cpu));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn handles_multiple_slots_per_rank_against_independent_enumeration() {
        for masks in 0usize..256 {
            let affinity: Vec<BTreeSet<usize>> = (0..2)
                .map(|rank| {
                    (0..4)
                        .filter(|cpu| masks & (1 << (rank * 4 + cpu)) != 0)
                        .collect()
                })
                .collect();
            for a in 1..=3 {
                for b in 1..=3 {
                    let required = [a, b];
                    let feasible = (0usize..81).any(|mut code| {
                        let mut counts = [0; 2];
                        for cpu in 0..4 {
                            let rank = code % 3;
                            code /= 3;
                            if rank < 2 {
                                if !affinity[rank].contains(&cpu) {
                                    return false;
                                }
                                counts[rank] += 1;
                            }
                        }
                        counts == required
                    });
                    assert_eq!(
                        plan(&affinity, &required).is_ok(),
                        feasible,
                        "masks={masks} required={required:?}"
                    );
                }
            }
        }
    }
}
