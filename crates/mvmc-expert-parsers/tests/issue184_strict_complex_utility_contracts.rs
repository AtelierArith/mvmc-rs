//! A135 / S084 / M0426-M0429 strict public utility contracts.
//!
//! Original source: Julia-mVMC 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1,
//! MVMCExpertModeParsers.jl/src/utils/file_utils.jl:98-117,178-184;
//! test/test_utils.jl:56-59. These are independent literal expectations, not
//! C loader contracts or runtime oracle calls. Rust exposes Julia's throw/catch
//! utility boundary as a typed Result/default boundary.

use mvmc_expert_parsers::utils::file::{
    parse_complex_value, safe_parse_complex, ComplexParseError,
};
use num_complex::Complex64;

#[test]
fn strict_complex_utility_original_real_m0426() {
    assert_eq!(parse_complex_value("1.0"), Ok(Complex64::new(1.0, 0.0)));
}

#[test]
fn strict_complex_utility_original_i_m0427() {
    assert_eq!(
        parse_complex_value("1.0+2.0i"),
        Ok(Complex64::new(1.0, 2.0))
    );
}

#[test]
fn strict_complex_utility_original_two_reals_m0428() {
    assert_eq!(parse_complex_value("1.0 2.0"), Ok(Complex64::new(1.0, 2.0)));
}

#[test]
fn strict_complex_utility_original_j_m0429() {
    assert_eq!(
        parse_complex_value("1.0+2.0j"),
        Ok(Complex64::new(1.0, 2.0))
    );
}

#[test]
fn strict_complex_utility_malformed_inputs_return_typed_error() {
    for input in ["invalid", "", "1 2 3", "1+-2i", "1+2e-i", "1+2imjunk"] {
        assert_eq!(
            parse_complex_value(input),
            Err(ComplexParseError),
            "{input}"
        );
    }
    // The error implements the ordinary Rust error boundary; callers need not
    // parse a fallback value or panic payload to detect failure.
    let error: &dyn std::error::Error = &ComplexParseError;
    assert!(error.to_string().contains("invalid complex literal"));
}

#[test]
fn strict_complex_utility_zero_is_success_not_parse_failure() {
    for input in ["0", "0 0", "0+0i", "0+0im"] {
        assert_eq!(parse_complex_value(input), Ok(Complex64::default()));
    }
}

#[test]
fn safe_complex_utility_catches_only_the_fallible_boundary() {
    let default = Complex64::new(7.0, -11.0);
    for input in ["invalid", "1 2 3"] {
        assert_eq!(parse_complex_value(input), Err(ComplexParseError));
        assert_eq!(safe_parse_complex(input, default), default);
    }
    assert_eq!(parse_complex_value("0+0i"), Ok(Complex64::default()));
    assert_eq!(safe_parse_complex("0+0i", default), Complex64::default());
    assert_eq!(
        parse_complex_value("1e-2-3e-4im"),
        Ok(Complex64::new(0.01, -0.0003))
    );
    assert_eq!(
        safe_parse_complex("1e-2-3e-4im", default),
        Complex64::new(0.01, -0.0003)
    );
}
