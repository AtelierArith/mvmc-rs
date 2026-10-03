//! Deterministic rank and range partitioning shared by MPI execution paths.
//!
//! This module contains the rank arithmetic independently of the MPI backend.
//! It is deliberately usable in ordinary single-process tests so partition
//! boundaries can be verified without requiring an MPI launcher.

use std::ops::Range;

/// Rank information detected from common MPI launcher environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchContext {
    /// Zero-based process rank.
    pub rank: usize,
    /// Number of processes in the world.
    pub world_size: usize,
}

impl LaunchContext {
    /// Detect Open MPI, MPICH/PMI, or explicit Rust test variables.
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let rank = [
            "MVMC_RS_MPI_RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
        ]
        .iter()
        .find_map(|key| get(key).and_then(|value| value.parse().ok()))?;
        let world_size = [
            "MVMC_RS_MPI_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
        ]
        .iter()
        .find_map(|key| get(key).and_then(|value| value.parse().ok()))?;
        (world_size > 0 && rank < world_size).then_some(Self { rank, world_size })
    }

    /// Whether this process is the world root.
    pub fn is_root(self) -> bool {
        self.rank == 0
    }
}

/// Group assignment for `NSplitSize > 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupAssignment {
    /// Zero-based group index.
    pub group: usize,
    /// Rank within the group.
    pub local_rank: usize,
    /// Number of ranks in each group.
    pub group_size: usize,
}

impl GroupAssignment {
    /// Range assigned to this group when `length` work items are split across
    /// `nsplit` groups.
    pub fn group_range(self, length: usize, nsplit: usize) -> Range<usize> {
        partition_range(length, nsplit, self.group)
    }

    /// Range assigned to this rank within its group. The returned range is
    /// relative to the group's start and is therefore suitable for indexing a
    /// group-local QP/sample buffer.
    pub fn local_range(self, group_length: usize) -> Range<usize> {
        partition_range(group_length, self.group_size, self.local_rank)
    }
}

/// Validate and calculate the upstream comm0/comm1/comm2 group arithmetic.
pub fn assign_group(context: LaunchContext, nsplit: usize) -> Result<GroupAssignment, String> {
    if nsplit == 0 {
        return Err("NSplitSize must be positive".into());
    }
    if !context.world_size.is_multiple_of(nsplit) {
        return Err(format!(
            "MPI world size {} must be divisible by NSplitSize {}",
            context.world_size, nsplit
        ));
    }
    let group_size = context.world_size / nsplit;
    Ok(GroupAssignment {
        group: context.rank / group_size,
        local_rank: context.rank % group_size,
        group_size,
    })
}

/// Split a half-open range into balanced contiguous pieces.
pub fn partition_range(length: usize, parts: usize, index: usize) -> Range<usize> {
    assert!(parts > 0, "partition count must be positive");
    assert!(index < parts, "partition index must be in range");
    let base = length / parts;
    let remainder = length % parts;
    let start = index * base + index.min(remainder);
    let end = start + base + usize::from(index < remainder);
    start..end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_launcher_variables_in_priority_order() {
        let context = LaunchContext::from_env(|key| match key {
            "OMPI_COMM_WORLD_RANK" => Some("3".into()),
            "OMPI_COMM_WORLD_SIZE" => Some("8".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            context,
            LaunchContext {
                rank: 3,
                world_size: 8
            }
        );
        assert!(!context.is_root());
    }

    #[test]
    fn assigns_balanced_groups_and_rejects_incompatible_worlds() {
        assert_eq!(
            assign_group(
                LaunchContext {
                    rank: 5,
                    world_size: 8
                },
                2
            ),
            Ok(GroupAssignment {
                group: 1,
                local_rank: 1,
                group_size: 4,
            })
        );
        assert!(assign_group(
            LaunchContext {
                rank: 3,
                world_size: 7
            },
            2
        )
        .is_err());
    }

    #[test]
    fn partitions_cover_range_without_overlap() {
        let ranges: Vec<_> = (0..3).map(|index| partition_range(8, 3, index)).collect();
        assert_eq!(ranges, vec![0..3, 3..6, 6..8]);
    }

    #[test]
    fn grouped_ranges_cover_sample_work_without_overlap_within_each_group() {
        let assignments: Vec<_> = (0..8)
            .map(|rank| {
                assign_group(
                    LaunchContext {
                        rank,
                        world_size: 8,
                    },
                    2,
                )
                .unwrap()
            })
            .collect();
        let sample_ranges: Vec<_> = assignments
            .iter()
            .map(|assignment| {
                let group = assignment.group_range(10, 2);
                let local = assignment.local_range(group.len());
                (group.start + local.start)..(group.start + local.end)
            })
            .collect();
        assert_eq!(
            sample_ranges,
            vec![0..2, 2..3, 3..4, 4..5, 5..7, 7..8, 8..9, 9..10]
        );
    }
}
