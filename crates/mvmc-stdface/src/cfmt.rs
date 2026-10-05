//! C `printf` conversion helpers (`%f`, `%e`, `%d`, `%s` with width/precision/flags).
//!
//! Rust's float formatting is exact (correctly rounded, ties-to-even on the exact binary value),
//! which is what glibc's `printf` produces, so decimal digits agree. These helpers add the
//! C-specific pieces: `nan`/`inf` spelling, two-digit minimum exponents and width padding.

/// Pad `s` to `width` (right-aligned unless `left`).
pub fn pad(s: String, width: usize, left: bool) -> String {
    if s.len() >= width {
        return s;
    }
    let fill = " ".repeat(width - s.len());
    if left {
        format!("{s}{fill}")
    } else {
        format!("{fill}{s}")
    }
}

fn nonfinite(v: f64) -> Option<String> {
    if v.is_nan() {
        Some(if v.is_sign_negative() { "-nan" } else { "nan" }.to_string())
    } else if v.is_infinite() {
        Some(if v < 0.0 { "-inf" } else { "inf" }.to_string())
    } else {
        None
    }
}

/// `%.{prec}f` without padding.
pub fn f_raw(v: f64, prec: usize) -> String {
    nonfinite(v).unwrap_or_else(|| format!("{v:.prec$}"))
}

/// `%{width}.{prec}f`.
pub fn f(v: f64, width: usize, prec: usize) -> String {
    pad(f_raw(v, prec), width, false)
}

/// `%-{width}.{prec}f`.
pub fn f_left(v: f64, width: usize, prec: usize) -> String {
    pad(f_raw(v, prec), width, true)
}

/// `%.{prec}e` without padding (C exponent: sign and at least two digits).
pub fn e_raw(v: f64, prec: usize) -> String {
    if let Some(s) = nonfinite(v) {
        return s;
    }
    let s = format!("{v:.prec$e}");
    let (mantissa, exponent) = s.split_once('e').expect("Rust exponent format");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

/// `%{width}.{prec}e`.
pub fn e(v: f64, width: usize, prec: usize) -> String {
    pad(e_raw(v, prec), width, false)
}

/// `%{width}d`.
pub fn d(v: i32, width: usize) -> String {
    pad(v.to_string(), width, false)
}

/// `%-{width}d`.
pub fn d_left(v: i32, width: usize) -> String {
    pad(v.to_string(), width, true)
}

/// `%-{width}s`.
pub fn s_left(v: &str, width: usize) -> String {
    pad(v.to_string(), width, true)
}

/// `%{width}s`.
pub fn s(v: &str, width: usize) -> String {
    pad(v.to_string(), width, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_formats_match_c() {
        assert_eq!(f(1.0, 25, 15), "        1.000000000000000");
        assert_eq!(f(-0.0, 25, 15), "       -0.000000000000000");
        assert_eq!(f_left(0.003, 10, 5), "0.00300   ");
        assert_eq!(f_raw(0.5, 0), "0");
        assert_eq!(f_raw(1.5, 0), "2");
        assert_eq!(f_raw(f64::NAN, 3), "nan");
        assert_eq!(f_raw(-f64::INFINITY, 3), "-inf");
    }

    #[test]
    fn exponent_formats_match_c() {
        assert_eq!(e_raw(1.0, 15), "1.000000000000000e+00");
        assert_eq!(e_raw(-0.0, 3), "-0.000e+00");
        assert_eq!(e_raw(1.25e-120, 2), "1.25e-120");
        assert_eq!(e(0.5, 25, 15), "    5.000000000000000e-01");
    }

    #[test]
    fn integer_formats_match_c() {
        assert_eq!(d(-3, 5), "   -3");
        assert_eq!(d_left(7, 5), "7    ");
        assert_eq!(s_left("zqp", 12), "zqp         ");
    }
}
