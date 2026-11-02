//! Report node-local MPI rank affinity and overlaps without changing placement.
//! Optional `--plan=N` prints disjoint N-CPU candidate masks; it does not apply them
//! or account for cgroup quotas, physical-core topology or other process threads.
//! Linux reads `/proc/self/status`; other platforms report affinity as unavailable.

#[path = "../src/cpu_placement.rs"]
mod cpu_placement;

use std::collections::BTreeSet;

use mpi::collective::SystemOperation;
use mpi::traits::*;

fn parse_cpu_list(value: &str) -> Result<BTreeSet<usize>, String> {
    let mut cpus = BTreeSet::new();
    for part in value.trim().split(',') {
        let mut ends = part.split('-');
        let start: usize = ends
            .next()
            .ok_or("missing CPU")?
            .parse()
            .map_err(|_| format!("invalid CPU range: {part}"))?;
        let end: usize = match ends.next() {
            Some(end) => end
                .parse()
                .map_err(|_| format!("invalid CPU range: {part}"))?,
            None => start,
        };
        if ends.next().is_some() || end < start || end > 1_048_575 {
            return Err(format!("invalid CPU range: {part}"));
        }
        cpus.extend(start..=end);
    }
    if cpus.is_empty() {
        return Err("empty CPU affinity".into());
    }
    Ok(cpus)
}

#[cfg(target_os = "linux")]
fn affinity() -> Result<BTreeSet<usize>, String> {
    let status = std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or("Cpus_allowed_list missing from /proc/self/status")?;
    parse_cpu_list(value)
}

#[cfg(not(target_os = "linux"))]
fn affinity() -> Result<BTreeSet<usize>, String> {
    Err("CPU affinity inspection is supported only on Linux".into())
}

fn main() {
    let Some(universe) = mpi::initialize() else {
        eprintln!("could not initialize MPI");
        std::process::exit(1);
    };
    let world = universe.world();
    let node = world.split_shared(world.rank());
    let local = match affinity() {
        Ok(cpus) => format!(
            "ok:{}",
            cpus.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Err(error) => format!("error:{error}"),
    };
    let mut width = 0_i32;
    let local_width = i32::try_from(local.len()).expect("affinity description too long");
    node.all_reduce_into(&local_width, &mut width, SystemOperation::max());
    let width = usize::try_from(width).expect("nonnegative affinity description length");
    let mut encoded = vec![0_u8; width];
    encoded[..local.len()].copy_from_slice(local.as_bytes());
    let mut gathered = vec![0_u8; width * node.size() as usize];
    node.all_gather_into(&encoded[..], &mut gathered[..]);
    let mut ranks = vec![0_i32; node.size() as usize];
    node.all_gather_into(&world.rank(), &mut ranks[..]);
    if node.rank() == 0 {
        let mut sets = Vec::new();
        for (rank, bytes) in ranks.iter().zip(gathered.chunks_exact(width)) {
            let length = bytes.iter().position(|byte| *byte == 0).unwrap_or(width);
            let value = String::from_utf8_lossy(&bytes[..length]);
            println!("world_rank={rank} affinity={value}");
            sets.push(
                value
                    .strip_prefix("ok:")
                    .and_then(|list| parse_cpu_list(list).ok()),
            );
        }
        for i in 0..ranks.len() {
            for j in (i + 1)..ranks.len() {
                match (&sets[i], &sets[j]) {
                    (Some(a), Some(b)) => println!(
                        "world_ranks={},{} shared_cpus={}",
                        ranks[i],
                        ranks[j],
                        a.intersection(b).count()
                    ),
                    _ => println!(
                        "world_ranks={},{} shared_cpus=unavailable",
                        ranks[i], ranks[j]
                    ),
                }
            }
        }
        if let Some(argument) = std::env::args().nth(1) {
            let request = argument
                .strip_prefix("--plan=")
                .and_then(|v| v.parse::<usize>().ok());
            match (request, sets.into_iter().collect::<Option<Vec<_>>>()) {
                (Some(count), Some(masks)) => {
                    match cpu_placement::plan(&masks, &vec![count; masks.len()]) {
                        Ok(assigned) => {
                            for (rank, cpus) in ranks.iter().zip(assigned) {
                                println!(
                                    "world_rank={rank} planned_cpus={}",
                                    cpus.iter()
                                        .map(usize::to_string)
                                        .collect::<Vec<_>>()
                                        .join(",")
                                );
                            }
                        }
                        Err(error) => println!("plan_error={error}"),
                    }
                }
                _ => println!(
                    "plan_error=requires --plan=POSITIVE_CPUS and available affinity masks"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_list;

    #[test]
    fn parses_ranges_and_deduplicates_cpu_ids() {
        assert_eq!(
            parse_cpu_list(" 0-2,2,8,10-11 ")
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 8, 10, 11]
        );
    }

    #[test]
    fn rejects_malformed_or_unbounded_ranges() {
        for bad in ["", "3-1", "1-2-3", "1,", "-1", "1048576"] {
            assert!(parse_cpu_list(bad).is_err(), "{bad}");
        }
    }
}
