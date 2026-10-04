use mvmc_core::parallel_scalar::{ParallelScalarOperations, ScalarCommunicator};
use mvmc_core::SingleProcessReducer;
use num_complex::Complex64;

#[test]
fn original_serial_scalar_literals_and_negative_max_are_identities() {
    let serial = SingleProcessReducer;
    for domain in [
        ScalarCommunicator::World,
        ScalarCommunicator::Sampling,
        ScalarCommunicator::CrossGroup,
    ] {
        assert_eq!(serial.sum_real(domain, 4.5).unwrap(), 4.5);
        assert_eq!(
            serial
                .sum_complex(domain, Complex64::new(2.0, 3.0))
                .unwrap(),
            Complex64::new(2.0, 3.0)
        );
        assert_eq!(serial.max_integer(domain, 7).unwrap(), 7);
        assert_eq!(serial.max_integer(domain, -7).unwrap(), -7);
        assert_eq!(serial.max_integer(domain, i32::MIN).unwrap(), i32::MIN);
    }
}
