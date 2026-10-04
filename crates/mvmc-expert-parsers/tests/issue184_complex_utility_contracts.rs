//! F040 utility contracts, not C definition-loader parity.
//!
//! Julia-mVMC 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
//! MVMCExpertModeParsers.jl/src/utils/file_utils.jl:98-117,178-184;
//! test/test_utils.jl:56-59,74-76. Expected values are literal component values,
//! not generated Rust outputs. No reference runtime is used by these tests.

use mvmc_expert_parsers::utils::file::safe_parse_complex;
use num_complex::Complex64;

#[test]
fn complex_utility_preserves_component_and_exponent_signs() {
    let fallback = Complex64::new(19.0, 23.0);
    for (input, real, imaginary) in [
        ("1.0-2.0e-3i", 1.0, -0.002),
        ("1e-2-3e-4im", 0.01, -0.0003),
        ("-1e+2+3e-4j", -100.0, 0.0003),
        ("1e+2-3e+4i", 100.0, -30000.0),
        ("1.0+2.0im", 1.0, 2.0),
        ("-2.5e-3i", 0.0, -0.0025),
        ("-im", 0.0, -1.0),
        ("1.0", 1.0, 0.0),
        ("1.5", 1.5, 0.0),
        ("1.5 -2.0", 1.5, -2.0),
    ] {
        assert_eq!(
            safe_parse_complex(input, fallback),
            Complex64::new(real, imaginary),
            "{input}"
        );
    }
}

#[test]
fn complex_utility_returns_supplied_default_for_invalid_or_surplus_fields() {
    for default in [Complex64::default(), Complex64::new(1.0, 1.0)] {
        for input in [
            "invalid",
            "",
            "1 2 3",
            "1+2e-i",
            "1+-2i",
            "1+2imjunk",
            "1 2 rubbish",
        ] {
            assert_eq!(safe_parse_complex(input, default), default, "{input}");
        }
    }
}

#[test]
fn complex_utility_default_does_not_change_valid_values() {
    for default in [Complex64::default(), Complex64::new(-7.0, 11.0)] {
        assert_eq!(
            safe_parse_complex("1.0+2.0i", default),
            Complex64::new(1.0, 2.0)
        );
        assert_eq!(
            safe_parse_complex("1.0+2.0j", default),
            Complex64::new(1.0, 2.0)
        );
    }
}
