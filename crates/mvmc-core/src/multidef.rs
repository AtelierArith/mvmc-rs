//! C MultiDef mode (`vmc.out -m N`, issue #348): communicator split arithmetic, the
//! directory-list file and the per-group directory change.
//!
//! Port of `initMultiDefMode` (`extern/mVMC-1.3.0/src/mVMC/vmcmain.c:727-800`). The
//! functions here are independent of the MPI backend so they can be checked in ordinary
//! single-process tests; the optional `mpi` feature adds `MpiContext::split_multi_def`.

use std::path::Path;

/// Group index of `rank` in a world of `size` ranks split into `n` groups
/// (`vmcmain.c:752-760`).
///
/// The first `size % n` groups hold `size / n + 1` ranks, the others `size / n`. C
/// divides by `n` unchecked (`n == 0` raises `SIGFPE`, a negative `n` gives an
/// unspecified color), so the caller must validate `n` first ([`check_world`]).
pub fn group_of_rank(rank: usize, size: usize, n: usize) -> usize {
    let div = size / n;
    let modulo = size % n;
    let threshold = (div + 1) * modulo;
    if rank < threshold {
        rank / (div + 1)
    } else {
        modulo + (rank - threshold) / div
    }
}

/// Number of ranks of group `group` (`vmcmain.c:752-760`).
pub fn group_size(group: usize, size: usize, n: usize) -> usize {
    size / n + usize::from(group < size % n)
}

/// Validation of `-m N` against the world size (`vmcmain.c:743-749`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldCheck {
    /// Printed by rank 0 on stderr when the groups are not equally sized.
    pub warning: Option<String>,
}

/// C message when the world is smaller than `N` (`vmcmain.c:744`).
pub const SIZE_ERROR: &str = "error: -m: N should be smaller than MPI size.";

/// Rust-side message for `N < 1`: C performs an unchecked division by `N`.
pub const NONPOSITIVE_ERROR: &str = "error: -m: N should be a positive integer.";

/// Check `n` against the world size. `Err` carries the message C prints on rank 0
/// before `MPI_Finalize(); exit(EXIT_FAILURE)`.
pub fn check_world(size: usize, n: i64) -> Result<WorldCheck, String> {
    let n = usize::try_from(n)
        .ok()
        .filter(|&n| n >= 1)
        .ok_or_else(|| NONPOSITIVE_ERROR.to_owned())?;
    if size < n {
        return Err(SIZE_ERROR.to_owned());
    }
    let warning = (!size.is_multiple_of(n))
        .then(|| format!("warning: load imbalance. MPI_size={size} nMultiDef={n}"));
    Ok(WorldCheck { warning })
}

/// Read the first `n` whitespace-separated names of the directory-list file
/// (`vmcmain.c:766-782`: `fscanf("%s\n")` per entry). Errors carry C's messages.
pub fn read_dir_list(path: &Path, n: usize) -> Result<Vec<String>, String> {
    let text = std::fs::read(path).map_err(|_| "error: DirListFile does not exist.".to_owned())?;
    let text = String::from_utf8_lossy(&text);
    let names: Vec<String> = text
        .split_ascii_whitespace()
        .take(n)
        .map(str::to_owned)
        .collect();
    if names.len() < n {
        return Err(format!("error: {} is incomplete.", path.display()));
    }
    Ok(names)
}

/// C `perror`-style text of an I/O error (no Rust ` (os error N)` suffix).
fn strerror(error: &std::io::Error) -> String {
    let text = error.to_string();
    match text.find(" (os error ") {
        Some(position) => text[..position].to_owned(),
        None => text,
    }
}

/// `chdir(dirName)` (`vmcmain.c:791-798`). The Err text is C's `error: chdir(): %s: ` plus
/// `perror("")`.
///
/// C runs one process per MPI rank, so the process-global working directory is the
/// per-group directory. The Rust CLI follows C: every definition, initial-parameter and
/// default output path is resolved after this call. Unlike C, which changes directory
/// only on the group's rank 0 (`group2 == 0`), the CLI changes it on every rank of the
/// group because every Rust rank parses the definition files.
pub fn change_directory(dir: &str) -> Result<(), String> {
    std::env::set_current_dir(dir)
        .map_err(|error| format!("error: chdir(): {dir}: {}", strerror(&error)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_follow_the_c_div_mod_rule() {
        // size 7, N 3: groups of 3, 2, 2.
        let groups: Vec<_> = (0..7).map(|rank| group_of_rank(rank, 7, 3)).collect();
        assert_eq!(groups, [0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(
            (0..3).map(|g| group_size(g, 7, 3)).collect::<Vec<_>>(),
            [3, 2, 2]
        );
        assert_eq!(
            (0..4).map(|r| group_of_rank(r, 4, 2)).collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
    }

    #[test]
    fn group_sizes_are_consistent_with_the_rank_map() {
        for size in 1..=64 {
            for n in 1..=size {
                let mut counts = vec![0; n];
                let mut last = 0;
                for rank in 0..size {
                    let group = group_of_rank(rank, size, n);
                    assert!(group >= last && group < n, "{size} {n} {rank}");
                    last = group;
                    counts[group] += 1;
                }
                for (group, count) in counts.iter().enumerate() {
                    assert_eq!(*count, group_size(group, size, n), "{size} {n} {group}");
                    assert!(*count >= 1);
                }
            }
        }
    }

    #[test]
    fn world_check_uses_the_c_messages() {
        assert_eq!(check_world(4, 2), Ok(WorldCheck { warning: None }));
        assert_eq!(
            check_world(3, 2).unwrap().warning.as_deref(),
            Some("warning: load imbalance. MPI_size=3 nMultiDef=2")
        );
        assert_eq!(check_world(1, 2), Err(SIZE_ERROR.to_owned()));
        assert_eq!(check_world(4, 0), Err(NONPOSITIVE_ERROR.to_owned()));
        assert_eq!(check_world(4, -1), Err(NONPOSITIVE_ERROR.to_owned()));
    }

    #[test]
    fn dir_list_reads_whitespace_separated_names() {
        let dir = std::env::temp_dir().join(format!("mvmc-multidef-list-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let list = dir.join("dirs.txt");
        std::fs::write(&list, "a\n  b\tc\nextra\n").unwrap();
        assert_eq!(read_dir_list(&list, 3).unwrap(), ["a", "b", "c"]);
        assert_eq!(read_dir_list(&list, 2).unwrap(), ["a", "b"]);
        assert_eq!(
            read_dir_list(&list, 5),
            Err(format!("error: {} is incomplete.", list.display()))
        );
        assert_eq!(
            read_dir_list(&dir.join("missing"), 1),
            Err("error: DirListFile does not exist.".into())
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
