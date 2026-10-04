//! Low-level `.def` line / token helpers.
//!
//! Port targets in `MVMCExpertModeParsers.jl/src/utils/file_utils.jl`:
//! `read_def_file`, `clean_line`, `split_def_line`, `safe_parse_*`,
//! `parse_namelist_content`. The semantics mirror the Julia helpers
//! exactly so round-tripping `examples/inputs/*/namelist.def` is
//! byte-stable.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Filesystem metadata, distinct from content parsing or model validation.
/// Paths retain their caller-supplied identity; regular symlinks follow targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    /// Final lexical path component, preserving non-UTF-8 bytes.
    pub filename: OsString,
    /// Caller-supplied path, without canonicalization or symlink resolution.
    pub filepath: PathBuf,
    /// Whether the target is a regular file, not merely an existing path.
    pub exists: bool,
    /// Regular-file byte length; zero for missing or non-regular paths.
    pub size_bytes: u64,
    /// Regular-file modification time; Unix epoch for missing or non-regular paths.
    pub last_modified: SystemTime,
}

/// Query a regular file without reading or decoding its contents.
/// Missing/broken-symlink and nonregular paths return false/zero/UNIX_EPOCH.
/// Other metadata or modification-time errors propagate; this is not an atomic
/// existence/open guarantee. Julia architecture, not a C numerical contract.
pub fn get_file_info<P: AsRef<Path>>(path: P) -> io::Result<FileInfo> {
    let path = path.as_ref();
    let metadata = match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Some(metadata),
        Ok(_) => None,
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let exists = metadata.is_some();
    let (size_bytes, last_modified) = match metadata {
        Some(metadata) => (metadata.len(), metadata.modified()?),
        None => (0, UNIX_EPOCH),
    };
    Ok(FileInfo {
        filename: path.file_name().unwrap_or_default().to_os_string(),
        filepath: path.to_path_buf(),
        exists,
        size_bytes,
        last_modified,
    })
}

/// Conventional regular-file predicate: false for missing/nonregular/error.
/// Following symlinks matches Julia's `isfile`; no file contents are loaded.
pub fn validate_file_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
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

/// A complex utility input could not be parsed in a supported literal format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComplexParseError;

impl std::fmt::Display for ComplexParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid complex literal: expected a real, two reals, or i/j/im form")
    }
}

impl std::error::Error for ComplexParseError {}

/// Parse a real-only, exactly-two-real, or `i`/`j`/`im` complex utility literal.
///
/// This is the fallible Rust counterpart of Julia's `parse_complex_value`;
/// [`safe_parse_complex`] alone supplies a fallback on error. It is not a C
/// definition reader: C parameter overlays use separate numeric fields.
///
/// # Errors
///
/// Returns [`ComplexParseError`] for malformed components or surplus fields.
pub fn parse_complex_value(token: &str) -> Result<Complex64, ComplexParseError> {
    parse_complex_literal(token).ok_or(ComplexParseError)
}

/// Best-effort public complex utility. This is not the strict C definition
/// reader: C parameter overlays contain separate index, real and imaginary fields.
/// Recognises:
///
/// * `"1.0+2.0i"` / `"1.0-2.0j"` / `"1.0+2.0im"`, including exponent signs,
/// * `"1.0 2.0"` (two whitespace-separated reals),
/// * `"1.0"` (real-only).
///
/// Returns the caller's `default` on parse failure, matching Julia's public
/// `safe_parse_complex(str, default)` contract. Pass zero for Julia's usual default.
pub fn safe_parse_complex(token: &str, default: Complex64) -> Complex64 {
    parse_complex_value(token).unwrap_or(default)
}

fn parse_complex_literal(token: &str) -> Option<Complex64> {
    let token = token.trim();
    let imaginary = token.strip_suffix("im").or_else(|| {
        token
            .strip_suffix('i')
            .or_else(|| token.strip_suffix('j'))
            .or_else(|| token.strip_suffix('I'))
            .or_else(|| token.strip_suffix('J'))
    });
    if let Some(body) = imaginary {
        // A component sign must not be the sign of a decimal exponent.
        let separator = body.bytes().enumerate().rev().find_map(|(pos, byte)| {
            (pos > 0
                && matches!(byte, b'+' | b'-')
                && !matches!(body.as_bytes()[pos - 1], b'e' | b'E'))
            .then_some(pos)
        });
        let (real, imaginary) = match separator {
            Some(pos) => (body[..pos].trim().parse::<f64>().ok()?, &body[pos..]),
            None => (0.0, body),
        };
        let imaginary = match imaginary.trim() {
            "" | "+" => 1.0,
            "-" => -1.0,
            value => value.parse::<f64>().ok()?,
        };
        return Some(Complex64::new(real, imaginary));
    }
    let mut fields = token.split_ascii_whitespace();
    let real = fields.next()?.parse::<f64>().ok()?;
    let imaginary = fields.next().map(str::parse::<f64>).transpose().ok()?;
    if fields.next().is_some() {
        return None;
    }
    Some(Complex64::new(real, imaginary.unwrap_or(0.0)))
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
        assert_eq!(
            safe_parse_complex("1.5", Complex64::default()),
            Complex64::new(1.5, 0.0)
        );
    }

    #[test]
    fn safe_parse_complex_two_reals() {
        assert_eq!(
            safe_parse_complex("1.5 -2.0", Complex64::default()),
            Complex64::new(1.5, -2.0)
        );
    }

    #[test]
    fn safe_parse_complex_julia_literal() {
        assert_eq!(
            safe_parse_complex("1.5+2.0i", Complex64::default()),
            Complex64::new(1.5, 2.0)
        );
        assert_eq!(
            safe_parse_complex("3.0-4.0j", Complex64::default()),
            Complex64::new(3.0, -4.0)
        );
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
