#![cfg(feature = "mpi")]

use mpi::traits::*;
use mvmc_core::mpi::MpiContext;
use mvmc_core::parallel_scalar::{ParallelScalarOperations, ScalarCommunicator};
use num_complex::Complex64;

#[test]
#[ignore = "requires explicit native MPI world2/world4 launch"]
fn scalar_domains_use_actual_membership_and_signed_negative_max() {
    let world = MpiContext::initialize().unwrap();
    let rank = world.world().rank() as usize;
    let size = world.world().size() as usize;
    assert!(size == 2 || size == 4);
    assert!(world.max_integer(ScalarCommunicator::Sampling, -1).is_err());
    assert!(world.sum_real(ScalarCommunicator::CrossGroup, 1.0).is_err());
    let world_sum = (size * (size + 1) / 2) as f64;
    for _ in 0..2 {
        assert_eq!(
            world
                .sum_real(ScalarCommunicator::World, (rank + 1) as f64)
                .unwrap(),
            world_sum
        );
        assert_eq!(
            world
                .sum_complex(
                    ScalarCommunicator::World,
                    Complex64::new((rank + 1) as f64, -0.5 * (rank + 1) as f64)
                )
                .unwrap(),
            Complex64::new(world_sum, -0.5 * world_sum)
        );
        assert_eq!(
            world
                .max_integer(ScalarCommunicator::World, rank as i32 + 1)
                .unwrap(),
            size as i32
        );
    }
    for width in [1, 2, 3, size] {
        let group = world.split_groups(width).unwrap();
        for _ in 0..2 {
            for domain in [
                ScalarCommunicator::World,
                ScalarCommunicator::Sampling,
                ScalarCommunicator::CrossGroup,
            ] {
                // Independent arithmetic membership: floor division for comm1,
                // equal remainders for comm2, including the short final group.
                let members: Vec<_> = (0..size)
                    .filter(|&other| match domain {
                        ScalarCommunicator::World => true,
                        ScalarCommunicator::Sampling => other / width == rank / width,
                        ScalarCommunicator::CrossGroup => other % width == rank % width,
                    })
                    .collect();
                let sum: f64 = members.iter().map(|&r| (r + 1) as f64).sum();
                let expected_max = -(members[0] as i32) - 1;
                assert_eq!(group.sum_real(domain, (rank + 1) as f64).unwrap(), sum);
                assert_eq!(
                    group
                        .sum_complex(
                            domain,
                            Complex64::new((rank + 1) as f64, -0.5 * (rank + 1) as f64)
                        )
                        .unwrap(),
                    Complex64::new(sum, -0.5 * sum)
                );
                assert_eq!(
                    group.max_integer(domain, -(rank as i32) - 1).unwrap(),
                    expected_max
                );
                assert_eq!(group.max_integer(domain, i32::MIN).unwrap(), i32::MIN);
            }
        }
    }
}
