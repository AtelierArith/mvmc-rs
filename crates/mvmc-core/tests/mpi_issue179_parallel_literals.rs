//! Original Julia parallel literals M0518–M0522, M0562–M0567 and M0578.
//! C owns partition/group/seed semantics; Julia ranges are one-based adaptations.
//! Uses offline PhysCal input only to exercise the public seed preparation path.
use mvmc_core::Reducer;
use mvmc_core::{
    parallel::{assign_group, GroupAssignment, LaunchContext},
    prepare_phys_cal_from_namelist, SingleProcessReducer,
};
use sfmt19937::Sfmt19937Rng;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

#[test]
fn m0518_serial_qp4_is_full() {
    assert_eq!(SingleProcessReducer.sampling_qp_range(4), 0..4);
    // Julia (1,5), not Rust (1,5).
}

#[test]
fn m0519_qp4_local_rank0_width2() {
    assert_eq!(
        GroupAssignment {
            group: 0,
            local_rank: 0,
            group_size: 2
        }
        .local_range(4),
        0..2
    );
}

#[test]
fn m0520_qp4_local_rank1_width2() {
    assert_eq!(
        GroupAssignment {
            group: 0,
            local_rank: 1,
            group_size: 2
        }
        .local_range(4),
        2..4
    );
}

#[test]
fn m0521_qp1_local_rank3_width4_is_empty() {
    assert_eq!(
        GroupAssignment {
            group: 0,
            local_rank: 3,
            group_size: 4
        }
        .local_range(1),
        1..1
    );
}

#[test]
fn m0522_world3_width2_has_two_groups_with_short_final_group() {
    let groups: Vec<_> = (0..3)
        .map(|rank| {
            assign_group(
                LaunchContext {
                    rank,
                    world_size: 3,
                },
                2,
            )
            .unwrap()
        })
        .collect();
    assert_eq!(
        groups.iter().map(|g| g.group).collect::<Vec<_>>(),
        [0, 0, 1]
    );
    assert_eq!(
        groups.iter().map(|g| g.local_rank).collect::<Vec<_>>(),
        [0, 1, 0]
    );
    assert_eq!(
        groups.iter().map(|g| g.group_size).collect::<Vec<_>>(),
        [2, 2, 1]
    );
}

struct Input(PathBuf);
impl Input {
    fn new(seed: Option<i64>) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "issue179-seed-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        };
        let original = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
        for entry in fs::read_dir(original.join("inputs")).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        fs::copy(original.join("zqp_opt.dat"), path.join("fixed.dat")).unwrap();
        let modpara = path.join("modpara.def");
        let mut text = fs::read_to_string(&modpara)
            .unwrap()
            .lines()
            .filter(|line| line.split_whitespace().next() != Some("RndSeed"))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        if let Some(seed) = seed {
            text.push_str(&format!("RndSeed {seed}\n"));
        }
        fs::write(modpara, text).unwrap();
        Self(path)
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn assert_stream(input_seed: Option<i64>, explicit: Option<i64>, expected: u32) {
    let input = Input::new(input_seed);
    let prepared = prepare_phys_cal_from_namelist(
        input.0.join("namelist.def"),
        input.0.join("fixed.dat"),
        "real",
        explicit,
    )
    .unwrap();
    assert_eq!(prepared.data.modpara.rnd_seed, input_seed.unwrap_or(11272));
    assert_eq!(prepared.rng.words_consumed(), 0);
    let mut actual = prepared.rng.clone();
    let mut expected = Sfmt19937Rng::new(expected);
    for _ in 0..624 {
        assert_eq!(actual.gen_rand32(), expected.gen_rand32());
    }
    assert_eq!(
        prepared.rng.words_consumed(),
        0,
        "peek must not consume original"
    );
}

#[test]
fn m0562_m0578_missing_seed_uses_parser_default11272() {
    assert_stream(None, None, 11272);
}
#[test]
fn m0563_zero_seed_is_not_default_or_clock() {
    assert_stream(Some(0), None, 0);
}
#[test]
fn m0564_positive_seed123() {
    assert_stream(Some(123), None, 123);
}
#[test]
fn m0566_explicit777_overrides_input123() {
    assert_stream(Some(123), Some(777), 777);
}

#[test]
fn m0565_negative_seed_uses_positive_current_unix_seconds() {
    let input = Input::new(Some(-1));
    let unix = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let before = unix();
    let prepared = prepare_phys_cal_from_namelist(
        input.0.join("namelist.def"),
        input.0.join("fixed.dat"),
        "real",
        None,
    )
    .unwrap();
    let after = unix();
    assert!(before > 0 && after >= before);
    let mut actual = prepared.rng.clone();
    let observed: Vec<_> = (0..624).map(|_| actual.gen_rand32()).collect();
    assert!((before..=after).any(|seed| {
        let mut expected = Sfmt19937Rng::new(u32::try_from(seed).unwrap());
        (0..624)
            .map(|_| expected.gen_rand32())
            .eq(observed.iter().copied())
    }));
    assert_eq!(prepared.rng.words_consumed(), 0);
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "requires isolated mpiexec -n 4; actual group3 seed resolution"]
fn m0567_actual_mpi_group3_base100_is103() {
    use mvmc_core::{mpi::MpiContext, prepare_phys_cal_from_namelist_with_reducer};
    let world = MpiContext::initialize().unwrap();
    assert_eq!(world.world_size(), 4);
    let group = world.split_groups(1).unwrap();
    assert_eq!(group.seed_offset(), world.rank());
    let input = Input::new(Some(100));
    let prepared = prepare_phys_cal_from_namelist_with_reducer(
        input.0.join("namelist.def"),
        input.0.join("fixed.dat"),
        "real",
        None,
        &group,
    )
    .unwrap();
    let expected_seed = [100, 101, 102, 103][world.rank()];
    let mut expected = Sfmt19937Rng::new(expected_seed);
    let mut actual = prepared.rng.clone();
    for _ in 0..624 {
        assert_eq!(actual.gen_rand32(), expected.gen_rand32());
    }
    assert_eq!(prepared.rng.words_consumed(), 0);
}
