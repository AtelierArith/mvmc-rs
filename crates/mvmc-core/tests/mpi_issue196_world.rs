//! Explicit runtime diagnostic, not a numerical/reference parity test.
#![cfg(feature = "mpi")]

#[test]
#[ignore = "explicit developer gate: launch with MPI -n 2 or -n 4"]
fn issue196_genuine_world() {
    use mvmc_core::{mpi::MpiContext, Reducer};
    let world = MpiContext::initialize().unwrap();
    let expected: usize = std::env::var("MPI196_EXPECT_RANKS")
        .unwrap()
        .parse()
        .unwrap();
    assert!(matches!(expected, 2 | 4));
    assert_eq!(
        world.world_size(),
        expected,
        "launcher processes are not proof of an MPI world"
    );
    let mut total = [world.rank() as i64 + 1];
    world.allreduce_sum_i64(&mut total);
    assert_eq!(total, [(expected * (expected + 1) / 2) as i64]);
    let mut payload = [if world.rank() == expected - 1 {
        196
    } else {
        -1
    }];
    world.broadcast_i64(expected - 1, &mut payload).unwrap();
    assert_eq!(payload, [196]);
    println!(
        "WORLD {} {} {} {}",
        world.rank(),
        expected,
        total[0],
        payload[0]
    );
}
