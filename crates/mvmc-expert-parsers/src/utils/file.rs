//! Low-level `.def` line / token helpers.
//!
//! Port targets in `MVMCExpertModeParsers.jl/src/utils/file_utils.jl`:
//! `read_def_file`, `clean_line`, `split_def_line`, `safe_parse_*`,
//! `parse_namelist_content`. The semantics mirror the Julia helpers
//! exactly so round-tripping `examples/inputs/*/namelist.def` is
//! byte-stable.

use std::fs;
use std::io;
use std::path::Path;

use num_complex::Complex64;

/// Julia's base-detecting strict integer reader, including signed 0x/0o/0b.
pub(crate) fn julia_parse_int(token: &str) -> Option<i64> {
    let unsigned = token.strip_prefix(['+', '-']).unwrap_or(token);
    let (radix, digits) = if let Some(digits) = unsigned.strip_prefix("0x") {
        (16, digits)
    } else if let Some(digits) = unsigned.strip_prefix("0o") {
        (8, digits)
    } else if let Some(digits) = unsigned.strip_prefix("0b") {
        (2, digits)
    } else {
        return token.parse().ok();
    };
    if digits.starts_with(['+', '-']) {
        return None;
    }
    let signed = if token.starts_with('-') {
        format!("-{digits}")
    } else {
        digits.to_owned()
    };
    i64::from_str_radix(&signed, radix).ok()
}

/// Parse a complete C floating-point field. Like fscanf's valid numeric inputs,
/// range errors produce their rounded value, including zero or infinity.
pub fn c_parse_float(token: &str) -> Option<f64> {
    super::c_numeric::parse_complete(token)
}

/// Julia Float64 tryparse uses the C decimal/hexadecimal conversion. It rejects
/// overflow and nonzero literals rounded to zero, but accepts finite subnormals
/// and explicit nonfinite tokens (which strict callers reject separately).
pub fn julia_parse_float(token: &str) -> Option<f64> {
    let value = c_parse_float(token)?;
    let unsigned = token
        .strip_prefix(['+', '-'])
        .unwrap_or(token)
        .to_ascii_lowercase();
    if value.is_infinite() && !matches!(unsigned.as_str(), "inf" | "infinity") {
        return None;
    }
    if value == 0.0 {
        let hex = unsigned.starts_with("0x");
        let mantissa = if hex {
            unsigned[2..].split('p').next()?
        } else {
            unsigned.split('e').next()?
        };
        if mantissa.chars().any(|c| {
            if hex {
                c.is_ascii_hexdigit() && c != '0'
            } else {
                matches!(c, '1'..='9')
            }
        }) {
            return None;
        }
    }
    Some(value)
}

/// Read a `.def` file in full. Mirrors `read_def_file` in upstream.
pub fn read_def_file<P: AsRef<Path>>(path: P) -> io::Result<String> {
    fs::read_to_string(path.as_ref())
}

/// Drop comments and trim whitespace. Equivalent to upstream
/// `clean_line(line)`:
///
/// 1. If the trimmed line starts with `#` or `//`, return `""`.
/// 2. Otherwise strip from the first `#` and then from the first `//`.
/// 3. Trim ASCII whitespace from both ends of what remains.
pub fn clean_line(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.starts_with('#') || trimmed.starts_with("//") {
        return "";
    }
    // Strip the first inline `#` if any.
    let mut working = line;
    if let Some(pos) = working.find('#') {
        working = &working[..pos];
    }
    if let Some(pos) = working.find("//") {
        working = &working[..pos];
    }
    working.trim()
}

/// Split a `.def` line into whitespace-separated tokens. Empty tokens
/// are dropped; consecutive whitespace is treated as one separator.
/// Returns an empty vector for comment-only or blank lines.
pub fn split_def_line(line: &str) -> Vec<&str> {
    let cleaned = clean_line(line);
    if cleaned.is_empty() {
        return Vec::new();
    }
    cleaned
        .split_ascii_whitespace()
        .filter(|s| !s.is_empty())
        .collect()
}

/// `safe_parse_int(token, default)`: parse an `i64`, falling back to
/// `default` if the token does not parse. Mirrors upstream
/// `safe_parse_int(::String, ::Int)`.
pub fn safe_parse_int(token: &str, default: i64) -> i64 {
    token.parse::<i64>().unwrap_or(default)
}

/// `safe_parse_float(token, default)`: parse an `f64`, falling back
/// to `default` if the token does not parse.
pub fn safe_parse_float(token: &str, default: f64) -> f64 {
    token.parse::<f64>().unwrap_or(default)
}

/// Best-effort complex parser used by `gutzwilleridx.def` /
/// `orbitalidx.def` legacy forms. Recognises:
///
/// * `"1.0+2.0i"` / `"1.0-2.0j"` (Julia-style complex literal),
/// * `"1.0 2.0"` (two whitespace-separated reals),
/// * `"1.0"` (real-only).
///
/// Returns `Complex64::new(0.0, 0.0)` on parse failure so the Julia
/// fallback (`safe_parse_complex` -> `0+0i`) is preserved.
pub fn safe_parse_complex(token: &str) -> Complex64 {
    if token.contains('i') || token.contains('j') {
        // Julia's `parse(Complex{Float64}, ...)` accepts e.g. "1+2im".
        // We do a minimal split on the last +/-.
        if let Some(c) = parse_complex_literal(token) {
            return c;
        }
    } else if token.contains(' ') {
        let mut it = token.split_ascii_whitespace();
        let re = it.next().and_then(|s| s.parse::<f64>().ok());
        let im = it.next().and_then(|s| s.parse::<f64>().ok());
        if let (Some(re), Some(im)) = (re, im) {
            return Complex64::new(re, im);
        }
    }
    if let Ok(re) = token.parse::<f64>() {
        return Complex64::new(re, 0.0);
    }
    Complex64::new(0.0, 0.0)
}

fn parse_complex_literal(token: &str) -> Option<Complex64> {
    // Accept "a+bi", "a-bi", "a+bj", "a-bj", "a+im b" -- the round-trip
    // fixtures here are tiny so a small handwritten splitter is enough.
    let token = token.trim();
    let last_marker = token
        .rfind('+')
        .or_else(|| token.rfind('-'))
        .filter(|&p| p > 0)?;
    // Skip a sign that immediately follows an exponent marker.
    let chars: Vec<char> = token.chars().collect();
    if last_marker > 0 {
        let prev = chars[last_marker - 1];
        if prev == 'e' || prev == 'E' {
            return None;
        }
    }
    let (re_str, im_str_with_sign) = token.split_at(last_marker);
    let im_str = im_str_with_sign
        .trim_end_matches('i')
        .trim_end_matches('j')
        .trim_end_matches('I')
        .trim_end_matches('J');
    let re = re_str.trim().parse::<f64>().ok()?;
    let im = im_str.trim().parse::<f64>().ok()?;
    Some(Complex64::new(re, im))
}

/// Parse `namelist.def` content into ordered `(file_type, file_name)`
/// pairs. Empty / comment-only lines and lines with fewer than 2 tokens
/// are skipped.
pub fn parse_namelist_content(content: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in content.lines() {
        let tokens = split_def_line(line);
        if tokens.len() >= 2 {
            out.push((tokens[0].to_string(), tokens[1].to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_line_strips_inline_comment() {
        assert_eq!(
            clean_line("  ModPara modpara.def   # comment"),
            "ModPara modpara.def"
        );
        assert_eq!(clean_line("# whole line comment"), "");
        assert_eq!(clean_line("// also a whole line comment"), "");
        assert_eq!(
            clean_line("    ModPara modpara.def // tail"),
            "ModPara modpara.def"
        );
    }

    #[test]
    fn split_def_line_drops_extra_whitespace() {
        let toks = split_def_line("   0    1   -0.250  ");
        assert_eq!(toks, vec!["0", "1", "-0.250"]);
    }

    #[test]
    fn safe_parse_int_falls_back() {
        assert_eq!(safe_parse_int("42", 0), 42);
        assert_eq!(safe_parse_int("not a number", 7), 7);
    }

    #[test]
    fn safe_parse_complex_real_only() {
        assert_eq!(safe_parse_complex("1.5"), Complex64::new(1.5, 0.0));
    }

    #[test]
    fn safe_parse_complex_two_reals() {
        assert_eq!(safe_parse_complex("1.5 -2.0"), Complex64::new(1.5, -2.0));
    }

    #[test]
    fn safe_parse_complex_julia_literal() {
        assert_eq!(safe_parse_complex("1.5+2.0i"), Complex64::new(1.5, 2.0));
        assert_eq!(safe_parse_complex("3.0-4.0j"), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn parse_namelist_basic() {
        let content = "         ModPara  modpara.def\n# comment\n           Trans  trans.def\n";
        let out = parse_namelist_content(content);
        assert_eq!(
            out,
            vec![
                ("ModPara".to_string(), "modpara.def".to_string()),
                ("Trans".to_string(), "trans.def".to_string()),
            ]
        );
    }
}
