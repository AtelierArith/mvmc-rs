//! Port of `StdFace_main.c` (mVMC branch): the keyword reader, parameter checks and the
//! Expert-file writers (`modpara.def`, `namelist.def`, `locspn.def`, `trans.def`, interaction
//! files, `greenone.def`, `greentwo.def`, Gutzwiller / Jastrow / orbital index files).
//!
//! The HPhi/UHF/HWAVE-only code (`PrintCalcMod`, `PrintExcitation`, time evolution, Boost, ...)
//! is not ported; its keywords are rejected like the C `_mVMC` build rejects them.
#![allow(non_snake_case)]

use crate::ccomplex::C64;
use crate::cfmt;
use crate::chain_lattice::std_face_chain;
use crate::ladder::std_face_ladder;
use crate::model_util as mu;
use crate::out::{exit, Out, Res, StdFaceError};
use crate::outf;
use crate::vals::{StdIntList, NAN_I, UNSET_STR};
use std::path::Path;

/// What a finished StdFace run printed.
#[derive(Debug, Clone)]
pub struct StdFaceReport {
    /// Everything C prints to `stdout`.
    pub log: String,
}

/// A failed StdFace run: the error plus everything printed before it.
#[derive(Debug, Clone)]
pub struct StdFaceFailure {
    /// Why it stopped.
    pub error: StdFaceError,
    /// Everything C prints to `stdout` before exiting.
    pub log: String,
}

// ---------------------------------------------------------------------------------------------
// Keyword reader helpers (C string handling reproduced exactly)
// ---------------------------------------------------------------------------------------------

fn text2lower(value: &str) -> String {
    value.chars().map(|c| c.to_ascii_lowercase()).collect()
}

/// `TrimSpaceQuote`: drop ` `, `:`, `;`, `"`, `\b`, `\\`, `\v` and `\n` (not tabs or `\r`).
fn trim_space_quote(value: &[u8]) -> Vec<u8> {
    value
        .iter()
        .copied()
        .filter(|&b| {
            !matches!(
                b,
                b' ' | b':' | b';' | b'"' | 0x08 | b'\\' | 0x0b | b'\n' | 0
            )
        })
        .collect()
}

/// `strtok` over a byte string: successive tokens split at `delim`, skipping leading delimiters.
struct Strtok<'a> {
    rest: &'a [u8],
}

impl<'a> Strtok<'a> {
    fn next(&mut self, delim: u8) -> Option<&'a [u8]> {
        let start = self.rest.iter().position(|&b| b != delim)?;
        let rest = &self.rest[start..];
        match rest.iter().position(|&b| b == delim) {
            Some(end) => {
                self.rest = &rest[end + 1..];
                Some(&rest[..end])
            }
            None => {
                self.rest = &[];
                Some(rest)
            }
        }
    }
}

fn lossy(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `sscanf(s, "%d", &v)`: `Some(v)` when a conversion happened.
fn scan_int(s: &[u8]) -> Option<i32> {
    let mut i = 0;
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let digits_start = i;
    let mut value: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        value = value
            .saturating_mul(10)
            .saturating_add((s[i] - b'0') as i64);
        i += 1;
    }
    if i == digits_start {
        return None;
    }
    if negative {
        value = -value;
    }
    // glibc stores `(int) strtol(...)`: saturated to the long range, then truncated.
    Some(value as i32)
}

/// `sscanf(s, "%lf", &v)` for decimal, `inf` and `nan` spellings.
fn scan_double(s: &[u8]) -> Option<f64> {
    let mut i = 0;
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    let start = i;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        i += 1;
    }
    let negative = s.get(start) == Some(&b'-');
    let lower: Vec<u8> = s[i..].iter().map(|b| b.to_ascii_lowercase()).collect();
    if lower.starts_with(b"infinity") || lower.starts_with(b"inf") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if lower.starts_with(b"nan") {
        return Some(if negative { -f64::NAN } else { f64::NAN });
    }
    let mantissa_start = i;
    let mut digits = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    let mut end = i;
    if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
        let mut k = i + 1;
        if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
            k += 1;
        }
        let exp_digits = k;
        while k < s.len() && s[k].is_ascii_digit() {
            k += 1;
        }
        if k > exp_digits {
            end = k;
        }
    }
    let text = std::str::from_utf8(&s[start..end]).ok()?;
    let _ = mantissa_start;
    text.trim_start_matches('+').parse::<f64>().ok()
}

fn dup_error(o: &mut Out, keyword: &str) -> Res<()> {
    outf!(o, "ERROR !  Keyword {keyword} is duplicated ! \n");
    exit(-1)
}

/// `StoreWithCheckDup_s`.
fn dup_s(o: &mut Out, keyword: &str, valuestring: &str, value: &mut String) -> Res<()> {
    if value != UNSET_STR {
        return dup_error(o, keyword);
    }
    *value = valuestring.to_string();
    Ok(())
}

/// `StoreWithCheckDup_sl`.
fn dup_sl(o: &mut Out, keyword: &str, valuestring: &str, value: &mut String) -> Res<()> {
    if value != UNSET_STR {
        return dup_error(o, keyword);
    }
    *value = text2lower(valuestring);
    Ok(())
}

/// `StoreWithCheckDup_i`.
fn dup_i(o: &mut Out, keyword: &str, valuestring: &str, value: &mut i32) -> Res<()> {
    if *value != NAN_I {
        return dup_error(o, keyword);
    }
    if let Some(v) = scan_int(valuestring.as_bytes()) {
        *value = v;
    }
    Ok(())
}

/// `StoreWithCheckDup_d`.
fn dup_d(o: &mut Out, keyword: &str, valuestring: &str, value: &mut f64) -> Res<()> {
    if !value.is_nan() {
        return dup_error(o, keyword);
    }
    if let Some(v) = scan_double(valuestring.as_bytes()) {
        *value = v;
    }
    Ok(())
}

/// `StoreWithCheckDup_c`: `real,imag` pair.
fn dup_c(o: &mut Out, keyword: &str, valuestring: &str, value: &mut C64) -> Res<()> {
    if !value.re.is_nan() {
        return dup_error(o, keyword);
    }
    let bytes = valuestring.as_bytes();
    let mut tok = Strtok { rest: bytes };
    let (r_str, i_str) = if bytes.first() == Some(&b',') {
        (None, tok.next(b','))
    } else {
        let r = tok.next(b',');
        (r, tok.next(b','))
    };
    let mut v = match r_str {
        None => C64::real(0.0),
        Some(r) => match scan_double(r) {
            Some(x) => C64::real(x),
            None => C64::real(0.0),
        },
    };
    v = match i_str {
        None => v + C64::I * C64::real(0.0),
        Some(i) => match scan_double(i) {
            Some(x) => v + C64::I * C64::real(x),
            None => v + C64::I * C64::real(0.0),
        },
    };
    *value = v;
    Ok(())
}

/// Dispatch one `keyword = value` pair (mVMC keyword set).
fn store_keyword(o: &mut Out, s: &mut StdIntList, keyword: &str, value: &str) -> Res<()> {
    match keyword {
        "a" => dup_d(o, keyword, value, &mut s.a)?,
        "a0h" => dup_i(o, keyword, value, &mut s.box_[0][2])?,
        "a0l" => dup_i(o, keyword, value, &mut s.box_[0][1])?,
        "a0w" => dup_i(o, keyword, value, &mut s.box_[0][0])?,
        "a1h" => dup_i(o, keyword, value, &mut s.box_[1][2])?,
        "a1l" => dup_i(o, keyword, value, &mut s.box_[1][1])?,
        "a1w" => dup_i(o, keyword, value, &mut s.box_[1][0])?,
        "a2h" => dup_i(o, keyword, value, &mut s.box_[2][2])?,
        "a2l" => dup_i(o, keyword, value, &mut s.box_[2][1])?,
        "a2w" => dup_i(o, keyword, value, &mut s.box_[2][0])?,
        "cutoff_j" => dup_d(o, keyword, value, &mut s.cutoff_j)?,
        "cutoff_jh" => dup_i(o, keyword, value, &mut s.cutoff_JR[2])?,
        "cutoff_jl" => dup_i(o, keyword, value, &mut s.cutoff_JR[1])?,
        "cutoff_jw" => dup_i(o, keyword, value, &mut s.cutoff_JR[0])?,
        "cutoff_j_a0w" => dup_d(o, keyword, value, &mut s.cutoff_JVec[0][0])?,
        "cutoff_j_a0l" => dup_d(o, keyword, value, &mut s.cutoff_JVec[0][1])?,
        "cutoff_j_a0h" => dup_d(o, keyword, value, &mut s.cutoff_JVec[0][2])?,
        "cutoff_j_a1w" => dup_d(o, keyword, value, &mut s.cutoff_JVec[1][0])?,
        "cutoff_j_a1l" => dup_d(o, keyword, value, &mut s.cutoff_JVec[1][1])?,
        "cutoff_j_a1h" => dup_d(o, keyword, value, &mut s.cutoff_JVec[1][2])?,
        "cutoff_j_a2w" => dup_d(o, keyword, value, &mut s.cutoff_JVec[2][0])?,
        "cutoff_j_a2l" => dup_d(o, keyword, value, &mut s.cutoff_JVec[2][1])?,
        "cutoff_j_a2h" => dup_d(o, keyword, value, &mut s.cutoff_JVec[2][2])?,
        "cutoff_length_j" => dup_d(o, keyword, value, &mut s.cutoff_length_J)?,
        "cutoff_length_u" => dup_d(o, keyword, value, &mut s.cutoff_length_U)?,
        "cutoff_length_t" => dup_d(o, keyword, value, &mut s.cutoff_length_t)?,
        "cutoff_t" => dup_d(o, keyword, value, &mut s.cutoff_t)?,
        "cutoff_th" => dup_i(o, keyword, value, &mut s.cutoff_tR[2])?,
        "cutoff_tl" => dup_i(o, keyword, value, &mut s.cutoff_tR[1])?,
        "cutoff_tw" => dup_i(o, keyword, value, &mut s.cutoff_tR[0])?,
        "cutoff_t_a0w" => dup_d(o, keyword, value, &mut s.cutoff_tVec[0][0])?,
        "cutoff_t_a0l" => dup_d(o, keyword, value, &mut s.cutoff_tVec[0][1])?,
        "cutoff_t_a0h" => dup_d(o, keyword, value, &mut s.cutoff_tVec[0][2])?,
        "cutoff_t_a1w" => dup_d(o, keyword, value, &mut s.cutoff_tVec[1][0])?,
        "cutoff_t_a1l" => dup_d(o, keyword, value, &mut s.cutoff_tVec[1][1])?,
        "cutoff_t_a1h" => dup_d(o, keyword, value, &mut s.cutoff_tVec[1][2])?,
        "cutoff_t_a2w" => dup_d(o, keyword, value, &mut s.cutoff_tVec[2][0])?,
        "cutoff_t_a2l" => dup_d(o, keyword, value, &mut s.cutoff_tVec[2][1])?,
        "cutoff_t_a2h" => dup_d(o, keyword, value, &mut s.cutoff_tVec[2][2])?,
        "cutoff_u" => dup_d(o, keyword, value, &mut s.cutoff_u)?,
        "cutoff_uh" => dup_i(o, keyword, value, &mut s.cutoff_UR[2])?,
        "cutoff_ul" => dup_i(o, keyword, value, &mut s.cutoff_UR[1])?,
        "cutoff_uw" => dup_i(o, keyword, value, &mut s.cutoff_UR[0])?,
        "cutoff_u_a0w" => dup_d(o, keyword, value, &mut s.cutoff_UVec[0][0])?,
        "cutoff_u_a0l" => dup_d(o, keyword, value, &mut s.cutoff_UVec[0][1])?,
        "cutoff_u_a0h" => dup_d(o, keyword, value, &mut s.cutoff_UVec[0][2])?,
        "cutoff_u_a1w" => dup_d(o, keyword, value, &mut s.cutoff_UVec[1][0])?,
        "cutoff_u_a1l" => dup_d(o, keyword, value, &mut s.cutoff_UVec[1][1])?,
        "cutoff_u_a1h" => dup_d(o, keyword, value, &mut s.cutoff_UVec[1][2])?,
        "cutoff_u_a2w" => dup_d(o, keyword, value, &mut s.cutoff_UVec[2][0])?,
        "cutoff_u_a2l" => dup_d(o, keyword, value, &mut s.cutoff_UVec[2][1])?,
        "cutoff_u_a2h" => dup_d(o, keyword, value, &mut s.cutoff_UVec[2][2])?,
        "lambda" => dup_d(o, keyword, value, &mut s.lambda)?,
        "lambda_u" => dup_d(o, keyword, value, &mut s.lambda_U)?,
        "lambda_j" => dup_d(o, keyword, value, &mut s.lambda_J)?,
        "alpha" => dup_d(o, keyword, value, &mut s.alpha)?,
        "d" => dup_d(o, keyword, value, &mut s.D[2][2])?,
        "doublecounting" => dup_sl(o, keyword, value, &mut s.double_counting_mode)?,
        "gamma" => dup_d(o, keyword, value, &mut s.Gamma)?,
        "h" => dup_d(o, keyword, value, &mut s.h)?,
        "gamma_y" => dup_d(o, keyword, value, &mut s.Gamma_y)?,
        "height" => dup_i(o, keyword, value, &mut s.Height)?,
        "hlength" => dup_d(o, keyword, value, &mut s.length[2])?,
        "hx" => dup_d(o, keyword, value, &mut s.direct[2][0])?,
        "hy" => dup_d(o, keyword, value, &mut s.direct[2][1])?,
        "hz" => dup_d(o, keyword, value, &mut s.direct[2][2])?,
        "j" => dup_d(o, keyword, value, &mut s.JAll)?,
        "jx" => dup_d(o, keyword, value, &mut s.J[0][0])?,
        "jxy" => dup_d(o, keyword, value, &mut s.J[0][1])?,
        "jxz" => dup_d(o, keyword, value, &mut s.J[0][2])?,
        "jy" => dup_d(o, keyword, value, &mut s.J[1][1])?,
        "jyx" => dup_d(o, keyword, value, &mut s.J[1][0])?,
        "jyz" => dup_d(o, keyword, value, &mut s.J[1][2])?,
        "jz" => dup_d(o, keyword, value, &mut s.J[2][2])?,
        "jzx" => dup_d(o, keyword, value, &mut s.J[2][0])?,
        "jzy" => dup_d(o, keyword, value, &mut s.J[2][1])?,
        "j0" => dup_d(o, keyword, value, &mut s.J0All)?,
        "j0x" => dup_d(o, keyword, value, &mut s.J0[0][0])?,
        "j0xy" => dup_d(o, keyword, value, &mut s.J0[0][1])?,
        "j0xz" => dup_d(o, keyword, value, &mut s.J0[0][2])?,
        "j0y" => dup_d(o, keyword, value, &mut s.J0[1][1])?,
        "j0yx" => dup_d(o, keyword, value, &mut s.J0[1][0])?,
        "j0yz" => dup_d(o, keyword, value, &mut s.J0[1][2])?,
        "j0z" => dup_d(o, keyword, value, &mut s.J0[2][2])?,
        "j0zx" => dup_d(o, keyword, value, &mut s.J0[2][0])?,
        "j0zy" => dup_d(o, keyword, value, &mut s.J0[2][1])?,
        "j0'" => dup_d(o, keyword, value, &mut s.J0pAll)?,
        "j0'x" => dup_d(o, keyword, value, &mut s.J0p[0][0])?,
        "j0'xy" => dup_d(o, keyword, value, &mut s.J0p[0][1])?,
        "j0'xz" => dup_d(o, keyword, value, &mut s.J0p[0][2])?,
        "j0'y" => dup_d(o, keyword, value, &mut s.J0p[1][1])?,
        "j0'yx" => dup_d(o, keyword, value, &mut s.J0p[1][0])?,
        "j0'yz" => dup_d(o, keyword, value, &mut s.J0p[1][2])?,
        "j0'z" => dup_d(o, keyword, value, &mut s.J0p[2][2])?,
        "j0'zx" => dup_d(o, keyword, value, &mut s.J0p[2][0])?,
        "j0'zy" => dup_d(o, keyword, value, &mut s.J0p[2][1])?,
        "j0''" => dup_d(o, keyword, value, &mut s.J0ppAll)?,
        "j0''x" => dup_d(o, keyword, value, &mut s.J0pp[0][0])?,
        "j0''xy" => dup_d(o, keyword, value, &mut s.J0pp[0][1])?,
        "j0''xz" => dup_d(o, keyword, value, &mut s.J0pp[0][2])?,
        "j0''y" => dup_d(o, keyword, value, &mut s.J0pp[1][1])?,
        "j0''yx" => dup_d(o, keyword, value, &mut s.J0pp[1][0])?,
        "j0''yz" => dup_d(o, keyword, value, &mut s.J0pp[1][2])?,
        "j0''z" => dup_d(o, keyword, value, &mut s.J0pp[2][2])?,
        "j0''zx" => dup_d(o, keyword, value, &mut s.J0pp[2][0])?,
        "j0''zy" => dup_d(o, keyword, value, &mut s.J0pp[2][1])?,
        "j1" => dup_d(o, keyword, value, &mut s.J1All)?,
        "j1x" => dup_d(o, keyword, value, &mut s.J1[0][0])?,
        "j1xy" => dup_d(o, keyword, value, &mut s.J1[0][1])?,
        "j1xz" => dup_d(o, keyword, value, &mut s.J1[0][2])?,
        "j1y" => dup_d(o, keyword, value, &mut s.J1[1][1])?,
        "j1yx" => dup_d(o, keyword, value, &mut s.J1[1][0])?,
        "j1yz" => dup_d(o, keyword, value, &mut s.J1[1][2])?,
        "j1z" => dup_d(o, keyword, value, &mut s.J1[2][2])?,
        "j1zx" => dup_d(o, keyword, value, &mut s.J1[2][0])?,
        "j1zy" => dup_d(o, keyword, value, &mut s.J1[2][1])?,
        "j1'" => dup_d(o, keyword, value, &mut s.J1pAll)?,
        "j1'x" => dup_d(o, keyword, value, &mut s.J1p[0][0])?,
        "j1'xy" => dup_d(o, keyword, value, &mut s.J1p[0][1])?,
        "j1'xz" => dup_d(o, keyword, value, &mut s.J1p[0][2])?,
        "j1'y" => dup_d(o, keyword, value, &mut s.J1p[1][1])?,
        "j1'yx" => dup_d(o, keyword, value, &mut s.J1p[1][0])?,
        "j1'yz" => dup_d(o, keyword, value, &mut s.J1p[1][2])?,
        "j1'z" => dup_d(o, keyword, value, &mut s.J1p[2][2])?,
        "j1'zx" => dup_d(o, keyword, value, &mut s.J1p[2][0])?,
        "j1'zy" => dup_d(o, keyword, value, &mut s.J1p[2][1])?,
        "j1''" => dup_d(o, keyword, value, &mut s.J1ppAll)?,
        "j1''x" => dup_d(o, keyword, value, &mut s.J1pp[0][0])?,
        "j1''xy" => dup_d(o, keyword, value, &mut s.J1pp[0][1])?,
        "j1''xz" => dup_d(o, keyword, value, &mut s.J1pp[0][2])?,
        "j1''y" => dup_d(o, keyword, value, &mut s.J1pp[1][1])?,
        "j1''yx" => dup_d(o, keyword, value, &mut s.J1pp[1][0])?,
        "j1''yz" => dup_d(o, keyword, value, &mut s.J1pp[1][2])?,
        "j1''z" => dup_d(o, keyword, value, &mut s.J1pp[2][2])?,
        "j1''zx" => dup_d(o, keyword, value, &mut s.J1pp[2][0])?,
        "j1''zy" => dup_d(o, keyword, value, &mut s.J1pp[2][1])?,
        "j2" => dup_d(o, keyword, value, &mut s.J2All)?,
        "j2x" => dup_d(o, keyword, value, &mut s.J2[0][0])?,
        "j2xy" => dup_d(o, keyword, value, &mut s.J2[0][1])?,
        "j2xz" => dup_d(o, keyword, value, &mut s.J2[0][2])?,
        "j2y" => dup_d(o, keyword, value, &mut s.J2[1][1])?,
        "j2yx" => dup_d(o, keyword, value, &mut s.J2[1][0])?,
        "j2yz" => dup_d(o, keyword, value, &mut s.J2[1][2])?,
        "j2z" => dup_d(o, keyword, value, &mut s.J2[2][2])?,
        "j2zx" => dup_d(o, keyword, value, &mut s.J2[2][0])?,
        "j2zy" => dup_d(o, keyword, value, &mut s.J2[2][1])?,
        "j2'" => dup_d(o, keyword, value, &mut s.J2pAll)?,
        "j2'x" => dup_d(o, keyword, value, &mut s.J2p[0][0])?,
        "j2'xy" => dup_d(o, keyword, value, &mut s.J2p[0][1])?,
        "j2'xz" => dup_d(o, keyword, value, &mut s.J2p[0][2])?,
        "j2'y" => dup_d(o, keyword, value, &mut s.J2p[1][1])?,
        "j2'yx" => dup_d(o, keyword, value, &mut s.J2p[1][0])?,
        "j2'yz" => dup_d(o, keyword, value, &mut s.J2p[1][2])?,
        "j2'z" => dup_d(o, keyword, value, &mut s.J2p[2][2])?,
        "j2'zx" => dup_d(o, keyword, value, &mut s.J2p[2][0])?,
        "j2'zy" => dup_d(o, keyword, value, &mut s.J2p[2][1])?,
        "j2''" => dup_d(o, keyword, value, &mut s.J2ppAll)?,
        "j2''x" => dup_d(o, keyword, value, &mut s.J2pp[0][0])?,
        "j2''xy" => dup_d(o, keyword, value, &mut s.J2pp[0][1])?,
        "j2''xz" => dup_d(o, keyword, value, &mut s.J2pp[0][2])?,
        "j2''y" => dup_d(o, keyword, value, &mut s.J2pp[1][1])?,
        "j2''yx" => dup_d(o, keyword, value, &mut s.J2pp[1][0])?,
        "j2''yz" => dup_d(o, keyword, value, &mut s.J2pp[1][2])?,
        "j2''z" => dup_d(o, keyword, value, &mut s.J2pp[2][2])?,
        "j2''zx" => dup_d(o, keyword, value, &mut s.J2pp[2][0])?,
        "j2''zy" => dup_d(o, keyword, value, &mut s.J2pp[2][1])?,
        "j'" => dup_d(o, keyword, value, &mut s.JpAll)?,
        "j'x" => dup_d(o, keyword, value, &mut s.Jp[0][0])?,
        "j'xy" => dup_d(o, keyword, value, &mut s.Jp[0][1])?,
        "j'xz" => dup_d(o, keyword, value, &mut s.Jp[0][2])?,
        "j'y" => dup_d(o, keyword, value, &mut s.Jp[1][1])?,
        "j'yx" => dup_d(o, keyword, value, &mut s.Jp[1][0])?,
        "j'yz" => dup_d(o, keyword, value, &mut s.Jp[1][2])?,
        "j'z" => dup_d(o, keyword, value, &mut s.Jp[2][2])?,
        "j'zx" => dup_d(o, keyword, value, &mut s.Jp[2][0])?,
        "j'zy" => dup_d(o, keyword, value, &mut s.Jp[2][1])?,
        "j''" => dup_d(o, keyword, value, &mut s.JppAll)?,
        "j''x" => dup_d(o, keyword, value, &mut s.Jpp[0][0])?,
        "j''xy" => dup_d(o, keyword, value, &mut s.Jpp[0][1])?,
        "j''xz" => dup_d(o, keyword, value, &mut s.Jpp[0][2])?,
        "j''y" => dup_d(o, keyword, value, &mut s.Jpp[1][1])?,
        "j''yx" => dup_d(o, keyword, value, &mut s.Jpp[1][0])?,
        "j''yz" => dup_d(o, keyword, value, &mut s.Jpp[1][2])?,
        "j''z" => dup_d(o, keyword, value, &mut s.Jpp[2][2])?,
        "j''zx" => dup_d(o, keyword, value, &mut s.Jpp[2][0])?,
        "j''zy" => dup_d(o, keyword, value, &mut s.Jpp[2][1])?,
        "k" => dup_d(o, keyword, value, &mut s.K)?,
        "l" => dup_i(o, keyword, value, &mut s.L)?,
        "lattice" => dup_sl(o, keyword, value, &mut s.lattice)?,
        "llength" => dup_d(o, keyword, value, &mut s.length[1])?,
        "lx" => dup_d(o, keyword, value, &mut s.direct[1][0])?,
        "ly" => dup_d(o, keyword, value, &mut s.direct[1][1])?,
        "lz" => dup_d(o, keyword, value, &mut s.direct[1][2])?,
        "model" => dup_sl(o, keyword, value, &mut s.model)?,
        "mu" => dup_d(o, keyword, value, &mut s.mu)?,
        "ncond" => dup_i(o, keyword, value, &mut s.ncond)?,
        "nelec" => dup_i(o, keyword, value, &mut s.ncond)?,
        "outputmode" => dup_sl(o, keyword, value, &mut s.outputmode)?,
        "phase0" => dup_d(o, keyword, value, &mut s.phase[0])?,
        "phase1" => dup_d(o, keyword, value, &mut s.phase[1])?,
        "phase2" => dup_d(o, keyword, value, &mut s.phase[2])?,
        "t" => dup_c(o, keyword, value, &mut s.t)?,
        "t0" => dup_c(o, keyword, value, &mut s.t0)?,
        "t0'" => dup_c(o, keyword, value, &mut s.t0p)?,
        "t0''" => dup_c(o, keyword, value, &mut s.t0pp)?,
        "t1" => dup_c(o, keyword, value, &mut s.t1)?,
        "t1'" => dup_c(o, keyword, value, &mut s.t1p)?,
        "t1''" => dup_c(o, keyword, value, &mut s.t1pp)?,
        "t2" => dup_c(o, keyword, value, &mut s.t2)?,
        "t2'" => dup_c(o, keyword, value, &mut s.t2p)?,
        "t2''" => dup_c(o, keyword, value, &mut s.t2pp)?,
        "t'" => dup_c(o, keyword, value, &mut s.tp)?,
        "t''" => dup_c(o, keyword, value, &mut s.tpp)?,
        "u" => dup_d(o, keyword, value, &mut s.U)?,
        "v" => dup_d(o, keyword, value, &mut s.V)?,
        "v0" => dup_d(o, keyword, value, &mut s.V0)?,
        "v0'" => dup_d(o, keyword, value, &mut s.V0p)?,
        "v0''" => dup_d(o, keyword, value, &mut s.V0pp)?,
        "v1" => dup_d(o, keyword, value, &mut s.V1)?,
        "v1'" => dup_d(o, keyword, value, &mut s.V1p)?,
        "v1''" => dup_d(o, keyword, value, &mut s.V1pp)?,
        "v2" => dup_d(o, keyword, value, &mut s.V2)?,
        "v2'" => dup_d(o, keyword, value, &mut s.V2p)?,
        "v2''" => dup_d(o, keyword, value, &mut s.V2pp)?,
        "v'" => dup_d(o, keyword, value, &mut s.Vp)?,
        "v''" => dup_d(o, keyword, value, &mut s.Vpp)?,
        "w" => dup_i(o, keyword, value, &mut s.W)?,
        "wlength" => dup_d(o, keyword, value, &mut s.length[0])?,
        "wx" => dup_d(o, keyword, value, &mut s.direct[0][0])?,
        "wy" => dup_d(o, keyword, value, &mut s.direct[0][1])?,
        "wz" => dup_d(o, keyword, value, &mut s.direct[0][2])?,
        "2sz" => dup_i(o, keyword, value, &mut s.Sz2)?,
        "a0hsub" => dup_i(o, keyword, value, &mut s.boxsub[0][2])?,
        "a0lsub" => dup_i(o, keyword, value, &mut s.boxsub[0][1])?,
        "a0wsub" => dup_i(o, keyword, value, &mut s.boxsub[0][0])?,
        "a1hsub" => dup_i(o, keyword, value, &mut s.boxsub[1][2])?,
        "a1lsub" => dup_i(o, keyword, value, &mut s.boxsub[1][1])?,
        "a1wsub" => dup_i(o, keyword, value, &mut s.boxsub[1][0])?,
        "a2hsub" => dup_i(o, keyword, value, &mut s.boxsub[2][2])?,
        "a2lsub" => dup_i(o, keyword, value, &mut s.boxsub[2][1])?,
        "a2wsub" => dup_i(o, keyword, value, &mut s.boxsub[2][0])?,
        "complextype" => dup_i(o, keyword, value, &mut s.ComplexType)?,
        "cparafilehead" => dup_s(o, keyword, value, &mut s.CParaFileHead)?,
        "dsroptredcut" => dup_d(o, keyword, value, &mut s.DSROptRedCut)?,
        "dsroptstadel" => dup_d(o, keyword, value, &mut s.DSROptStaDel)?,
        "dsroptstepdt" => dup_d(o, keyword, value, &mut s.DSROptStepDt)?,
        "hsub" => dup_i(o, keyword, value, &mut s.Hsub)?,
        "lsub" => dup_i(o, keyword, value, &mut s.Lsub)?,
        "nvmccalmode" => dup_i(o, keyword, value, &mut s.NVMCCalMode)?,
        "ndataidxstart" => dup_i(o, keyword, value, &mut s.NDataIdxStart)?,
        "ndataqtysmp" => dup_i(o, keyword, value, &mut s.NDataQtySmp)?,
        "nlanczosmode" => dup_i(o, keyword, value, &mut s.NLanczosMode)?,
        "nmptrans" => dup_i(o, keyword, value, &mut s.NMPTrans)?,
        "nspgaussleg" => dup_i(o, keyword, value, &mut s.NSPGaussLeg)?,
        "nsplitsize" => dup_i(o, keyword, value, &mut s.NSplitSize)?,
        "nspstot" => dup_i(o, keyword, value, &mut s.NSPStot)?,
        "nsroptitrsmp" => dup_i(o, keyword, value, &mut s.NSROptItrSmp)?,
        "nsroptitrstep" => dup_i(o, keyword, value, &mut s.NSROptItrStep)?,
        "nstore" => dup_i(o, keyword, value, &mut s.NStore)?,
        "nsrcg" => dup_i(o, keyword, value, &mut s.NSRCG)?,
        "nvmcinterval" => dup_i(o, keyword, value, &mut s.NVMCInterval)?,
        "nvmcsample" => dup_i(o, keyword, value, &mut s.NVMCSample)?,
        "nvmcwarmup" => dup_i(o, keyword, value, &mut s.NVMCWarmUp)?,
        "rndseed" => dup_i(o, keyword, value, &mut s.RndSeed)?,
        "wsub" => dup_i(o, keyword, value, &mut s.Wsub)?,
        _ => {
            o.print("ERROR ! Unsupported Keyword in Standard mode!\n");
            return exit(-1);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------------------------

/// `PrintOrb`: `orbitalidx.def` (anti-parallel orbital index).
fn print_orb(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!("NOrbitalIdx {}\n", cfmt::d(s.NOrb, 10)));
    t.push_str(&format!("ComplexType {}\n", cfmt::d(s.ComplexType, 10)));
    t.push_str("=============================================\n");
    t.push_str("=============================================\n");
    let anti_any = s.AntiPeriod[0] == 1 || s.AntiPeriod[1] == 1 || s.AntiPeriod[2] == 1;
    let nsite = s.nsite as usize;
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if anti_any {
                t.push_str(&format!(
                    "{}  {}  {}  {}\n",
                    cfmt::d(isite as i32, 5),
                    cfmt::d(jsite as i32, 5),
                    cfmt::d(s.Orb[isite][jsite], 5),
                    cfmt::d(s.AntiOrb[isite][jsite], 5)
                ));
            } else {
                t.push_str(&format!(
                    "{}  {}  {}\n",
                    cfmt::d(isite as i32, 5),
                    cfmt::d(jsite as i32, 5),
                    cfmt::d(s.Orb[isite][jsite], 5)
                ));
            }
        }
    }
    for iorb in 0..s.NOrb {
        t.push_str(&format!("{}  {}\n", cfmt::d(iorb, 5), cfmt::d(1, 5)));
    }
    o.write_file("orbitalidx.def", &t)?;
    o.print("    orbitalidx.def is written.\n");
    s.Orb = Vec::new();
    Ok(())
}

/// `PrintOrbPara`: `orbitalidxpara.def` and `orbitalidxgen.def` (parallel-spin orbitals).
fn print_orb_para(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let nsite = s.nsite as usize;
    // (1) Copy from anti-parallel orbital index
    let mut orb_gc = s.Orb.clone();
    let mut reverse = s.AntiOrb.clone();
    // (2) Symmetrize
    for iorb in 0..s.NOrb {
        for isite in 0..nsite {
            for jsite in 0..nsite {
                if orb_gc[isite][jsite] == iorb {
                    orb_gc[jsite][isite] = orb_gc[isite][jsite];
                    reverse[jsite][isite] = -reverse[isite][jsite];
                }
            }
        }
    }
    let mut norb_gc = 0i32;
    for isite in 0..nsite {
        for jsite in 0..isite {
            if orb_gc[isite][jsite] >= 0 {
                let iorb_gc = orb_gc[isite][jsite];
                norb_gc -= 1;
                for row in orb_gc.iter_mut() {
                    for v in row.iter_mut() {
                        if *v == iorb_gc {
                            *v = norb_gc;
                        }
                    }
                }
            }
        }
    }
    norb_gc = -norb_gc;
    for row in orb_gc.iter_mut() {
        for v in row.iter_mut() {
            *v = -1 - *v;
        }
    }

    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!("NOrbitalIdx {}\n", cfmt::d(norb_gc, 10)));
    t.push_str(&format!("ComplexType {}\n", cfmt::d(s.ComplexType, 10)));
    t.push_str("=============================================\n");
    t.push_str("=============================================\n");
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if isite >= jsite {
                continue;
            }
            t.push_str(&format!(
                "{}  {}  {}  {}\n",
                cfmt::d(isite as i32, 5),
                cfmt::d(jsite as i32, 5),
                cfmt::d(orb_gc[isite][jsite], 5),
                cfmt::d(reverse[isite][jsite], 5)
            ));
        }
    }
    for iorb_gc in 0..norb_gc {
        t.push_str(&format!("{}  {}\n", cfmt::d(iorb_gc, 5), cfmt::d(1, 5)));
    }
    o.write_file("orbitalidxpara.def", &t)?;
    o.print("    orbitalidxpara.def is written.\n");

    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!(
        "NOrbitalIdx {}\n",
        cfmt::d(s.NOrb + 2 * norb_gc, 10)
    ));
    t.push_str(&format!("ComplexType {}\n", cfmt::d(s.ComplexType, 10)));
    t.push_str("=============================================\n");
    t.push_str("=============================================\n");
    let anti_any = s.AntiPeriod[0] == 1 || s.AntiPeriod[1] == 1 || s.AntiPeriod[2] == 1;
    // anti parallel
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if anti_any {
                t.push_str(&format!(
                    "{}  0  {}  1  {}  {}\n",
                    cfmt::d(isite as i32, 5),
                    cfmt::d(jsite as i32, 5),
                    cfmt::d(s.Orb[isite][jsite], 5),
                    cfmt::d(s.AntiOrb[isite][jsite], 5)
                ));
            } else {
                t.push_str(&format!(
                    "{}  0  {}  1  {}  1\n",
                    cfmt::d(isite as i32, 5),
                    cfmt::d(jsite as i32, 5),
                    cfmt::d(s.Orb[isite][jsite], 5)
                ));
            }
        }
    }
    // parallel
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if isite >= jsite {
                continue;
            }
            // Both branches of the C code print the same fields (`reverse` is always used).
            t.push_str(&format!(
                "{}  0  {}  0  {}  {}\n",
                cfmt::d(isite as i32, 5),
                cfmt::d(jsite as i32, 5),
                cfmt::d(orb_gc[isite][jsite] + s.NOrb, 5),
                cfmt::d(reverse[isite][jsite], 5)
            ));
            t.push_str(&format!(
                "{}  1  {}  1  {}  {}\n",
                cfmt::d(isite as i32, 5),
                cfmt::d(jsite as i32, 5),
                cfmt::d(orb_gc[isite][jsite] + s.NOrb + norb_gc, 5),
                cfmt::d(reverse[isite][jsite], 5)
            ));
        }
    }
    for iorb_gc in 0..s.NOrb {
        t.push_str(&format!("{}  {}\n", cfmt::d(iorb_gc, 5), cfmt::d(1, 5)));
    }
    for iorb_gc in 0..norb_gc * 2 {
        t.push_str(&format!(
            "{}  {}\n",
            cfmt::d(iorb_gc + s.NOrb, 5),
            cfmt::d(1, 5)
        ));
    }
    o.write_file("orbitalidxgen.def", &t)?;
    o.print("    orbitalidxgen.def is written.\n");
    Ok(())
}

/// `PrintGutzwiller`: `gutzwilleridx.def`.
fn print_gutzwiller(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let nsite = s.nsite as usize;
    let mut gutz = vec![0i32; nsite];
    let n_gutzwiller: i32;
    if s.NMPTrans.abs() == 1 || s.NMPTrans == NAN_I {
        let mut ng = if s.model == "hubbard" { 0 } else { -1 };
        for isite in 0..nsite {
            gutz[isite] = s.Orb[isite][isite];
        }
        for isite in 0..nsite {
            if s.locspinflag[isite] != 0 {
                gutz[isite] = -1;
                continue;
            }
            if gutz[isite] >= 0 {
                let igutz = gutz[isite];
                ng -= 1;
                for jsite in 0..nsite {
                    if gutz[jsite] == igutz {
                        gutz[jsite] = ng;
                    }
                }
            }
        }
        n_gutzwiller = -ng;
        for g in gutz.iter_mut() {
            *g = -1 - *g;
        }
    } else {
        let n_uc = s.NsiteUC;
        n_gutzwiller = if s.model == "hubbard" {
            n_uc
        } else if s.model == "spin" {
            1
        } else {
            n_uc + 1
        };
        for icell in 0..s.NCell {
            for isite in 0..n_uc {
                if s.model == "hubbard" {
                    gutz[(isite + n_uc * icell) as usize] = isite;
                } else if s.model == "spin" {
                    gutz[(isite + n_uc * icell) as usize] = 0;
                } else {
                    gutz[(isite + n_uc * icell) as usize] = 0;
                    gutz[(isite + n_uc * (icell + s.NCell)) as usize] = isite + 1;
                }
            }
        }
    }
    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!("NGutzwillerIdx {}\n", cfmt::d(n_gutzwiller, 10)));
    t.push_str(&format!("ComplexType {}\n", cfmt::d(0, 10)));
    t.push_str("=============================================\n");
    t.push_str("=============================================\n");
    for (isite, g) in gutz.iter().enumerate() {
        t.push_str(&format!(
            "{}  {}\n",
            cfmt::d(isite as i32, 5),
            cfmt::d(*g, 5)
        ));
    }
    for igutz in 0..n_gutzwiller {
        let flag = if s.model == "hubbard" || igutz > 0 {
            1
        } else {
            0
        };
        t.push_str(&format!("{}  {}\n", cfmt::d(igutz, 5), cfmt::d(flag, 5)));
    }
    o.write_file("gutzwilleridx.def", &t)?;
    o.print("    gutzwilleridx.def is written.\n");
    Ok(())
}

/// `PrintLocSpin`: `locspn.def`.
fn print_loc_spin(o: &mut Out, s: &StdIntList) -> Res<()> {
    let nlocspin = s.locspinflag.iter().filter(|&&f| f != 0).count();
    let mut t = String::new();
    t.push_str("================================ \n");
    t.push_str(&format!("NlocalSpin {}  \n", cfmt::d(nlocspin as i32, 5)));
    t.push_str("================================ \n");
    t.push_str("========i_1LocSpn_0IteElc ====== \n");
    t.push_str("================================ \n");
    for (isite, f) in s.locspinflag.iter().enumerate() {
        t.push_str(&format!(
            "{}  {}\n",
            cfmt::d(isite as i32, 5),
            cfmt::d(*f, 5)
        ));
    }
    o.write_file("locspn.def", &t)?;
    o.print("    locspn.def is written.\n");
    Ok(())
}

/// `PrintTrans`: merge equivalent transfers and write `trans.def`.
fn print_trans(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let n = s.trans.len();
    for j in 0..n {
        for k in j + 1..n {
            if s.transindx[j] == s.transindx[k] {
                s.trans[j] = s.trans[j] + s.trans[k];
                s.trans[k] = C64::real(0.0);
            }
        }
    }
    let ntrans0 = s.trans.iter().filter(|c| c.abs() > 0.000001).count();
    let mut t = String::new();
    t.push_str("======================== \n");
    t.push_str(&format!("NTransfer {}  \n", cfmt::d(ntrans0 as i32, 7)));
    t.push_str("======================== \n");
    t.push_str("========i_j_s_tijs====== \n");
    t.push_str("======================== \n");
    for k in 0..n {
        if s.trans[k].abs() > 0.000001 {
            let idx = s.transindx[k];
            t.push_str(&format!(
                "{} {} {} {} {} {}\n",
                cfmt::d(idx[0], 5),
                cfmt::d(idx[1], 5),
                cfmt::d(idx[2], 5),
                cfmt::d(idx[3], 5),
                cfmt::f(s.trans[k].re, 25, 15),
                cfmt::f(s.trans[k].im, 25, 15)
            ));
        }
    }
    o.write_file("trans.def", &t)?;
    o.print("      trans.def is written.\n");
    Ok(())
}

/// `PrintNamelist`: `namelist.def`.
fn print_namelist(o: &mut Out, s: &StdIntList) -> Res<()> {
    let mut t = String::new();
    t.push_str("         ModPara  modpara.def\n");
    t.push_str("         LocSpin  locspn.def\n");
    t.push_str("           Trans  trans.def\n");
    if s.LCintra == 1 {
        t.push_str("    CoulombIntra  coulombintra.def\n");
    }
    if s.LCinter == 1 {
        t.push_str("    CoulombInter  coulombinter.def\n");
    }
    if s.LHund == 1 {
        t.push_str("            Hund  hund.def\n");
    }
    if s.LEx == 1 {
        t.push_str("        Exchange  exchange.def\n");
    }
    if s.LPairLift == 1 {
        t.push_str("        PairLift  pairlift.def\n");
    }
    if s.LPairHopp == 1 {
        t.push_str("         PairHop  pairhopp.def\n");
    }
    if s.Lintr == 1 {
        t.push_str("        InterAll  interall.def\n");
    }
    if s.ioutputmode != 0 {
        t.push_str("        OneBodyG  greenone.def\n");
        t.push_str("        TwoBodyG  greentwo.def\n");
    }
    t.push_str("      Gutzwiller  gutzwilleridx.def\n");
    t.push_str("         Jastrow  jastrowidx.def\n");
    t.push_str("         Orbital  orbitalidx.def\n");
    if s.lGC == 1 || (s.Sz2 != 0 && s.Sz2 != NAN_I) {
        t.push_str(" OrbitalParallel  orbitalidxpara.def\n");
        t.push_str("# OrbitalGeneral  orbitalidxgen.def\n");
    }
    t.push_str("        TransSym  qptransidx.def\n");
    o.write_file("namelist.def", &t)?;
    o.print("    namelist.def is written.\n");
    Ok(())
}

/// `PrintModPara`: `modpara.def`.
fn print_mod_para(o: &mut Out, s: &StdIntList) -> Res<()> {
    let mut t = String::new();
    t.push_str("--------------------\n");
    t.push_str("Model_Parameters   0\n");
    t.push_str("--------------------\n");
    t.push_str("VMC_Cal_Parameters\n");
    t.push_str("--------------------\n");
    t.push_str(&format!("CDataFileHead  {}\n", s.CDataFileHead));
    t.push_str(&format!("CParaFileHead  {}\n", s.CParaFileHead));
    t.push_str("--------------------\n");
    t.push_str(&format!("NVMCCalMode    {}\n", s.NVMCCalMode));
    t.push_str(&format!("NLanczosMode   {}\n", s.NLanczosMode));
    t.push_str("--------------------\n");
    t.push_str(&format!("NDataIdxStart  {}\n", s.NDataIdxStart));
    t.push_str(&format!("NDataQtySmp    {}\n", s.NDataQtySmp));
    t.push_str("--------------------\n");
    t.push_str(&format!("Nsite          {}\n", s.nsite));
    t.push_str(&format!("Ncond          {}\n", cfmt::d_left(s.ncond, 5)));
    if s.Sz2 != NAN_I {
        t.push_str(&format!("2Sz            {}\n", s.Sz2));
    }
    if s.NSPGaussLeg != NAN_I {
        t.push_str(&format!("NSPGaussLeg    {}\n", s.NSPGaussLeg));
    }
    if s.NSPStot != NAN_I {
        t.push_str(&format!("NSPStot        {}\n", s.NSPStot));
    }
    t.push_str(&format!("NMPTrans       {}\n", s.NMPTrans));
    t.push_str(&format!("NSROptItrStep  {}\n", s.NSROptItrStep));
    t.push_str(&format!("NSROptItrSmp   {}\n", s.NSROptItrSmp));
    t.push_str(&format!(
        "DSROptRedCut   {}\n",
        cfmt::f_raw(s.DSROptRedCut, 10)
    ));
    t.push_str(&format!(
        "DSROptStaDel   {}\n",
        cfmt::f_raw(s.DSROptStaDel, 10)
    ));
    t.push_str(&format!(
        "DSROptStepDt   {}\n",
        cfmt::f_raw(s.DSROptStepDt, 10)
    ));
    t.push_str(&format!("NVMCWarmUp     {}\n", s.NVMCWarmUp));
    t.push_str(&format!("NVMCInterval   {}\n", s.NVMCInterval));
    t.push_str(&format!("NVMCSample     {}\n", s.NVMCSample));
    t.push_str(&format!("NExUpdatePath  {}\n", s.NExUpdatePath));
    t.push_str(&format!("RndSeed        {}\n", s.RndSeed));
    t.push_str(&format!("NSplitSize     {}\n", s.NSplitSize));
    t.push_str(&format!("NStore         {}\n", s.NStore));
    t.push_str(&format!("NSRCG          {}\n", s.NSRCG));
    o.write_file("modpara.def", &t)?;
    o.print("     modpara.def is written.\n");
    Ok(())
}

/// `Print1Green`: `greenone.def`.
fn print_1_green(o: &mut Out, s: &StdIntList) -> Res<()> {
    if s.ioutputmode == 0 {
        return Ok(());
    }
    let nsite = s.nsite;
    let xkondo = if s.model == "kondo" { 2 } else { 1 };
    let mut green: Vec<[i32; 4]> = Vec::new();
    let spin_max = |site: i32| {
        let f = s.locspinflag[site as usize];
        if f == 0 {
            1
        } else {
            f
        }
    };
    if s.ioutputmode == 1 {
        for isite in 0..s.NsiteUC * xkondo {
            let isite2 = if isite >= s.NsiteUC {
                isite - s.NsiteUC + nsite / 2
            } else {
                isite
            };
            let si_max = spin_max(isite2);
            for ispin in 0..=si_max {
                for jsite in 0..nsite {
                    let sj_max = spin_max(jsite);
                    for jspin in 0..=sj_max {
                        if isite2 != jsite
                            && (s.locspinflag[isite2 as usize] != 0
                                && s.locspinflag[jsite as usize] != 0)
                        {
                            continue;
                        }
                        if ispin == jspin {
                            green.push([isite2, ispin, jsite, jspin]);
                        }
                    }
                }
            }
        }
    } else {
        for isite in 0..nsite {
            let si_max = spin_max(isite);
            for ispin in 0..=si_max {
                for jsite in 0..nsite {
                    let sj_max = spin_max(jsite);
                    for jspin in 0..=sj_max {
                        if isite != jsite
                            && (s.locspinflag[isite as usize] != 0
                                && s.locspinflag[jsite as usize] != 0)
                        {
                            continue;
                        }
                        green.push([isite, ispin, jsite, jspin]);
                    }
                }
            }
        }
    }
    let mut t = String::new();
    t.push_str("===============================\n");
    t.push_str(&format!("NCisAjs {}\n", cfmt::d(green.len() as i32, 10)));
    t.push_str("===============================\n");
    t.push_str("======== Green functions ======\n");
    t.push_str("===============================\n");
    for g in &green {
        t.push_str(&format!(
            "{} {} {} {}\n",
            cfmt::d(g[0], 5),
            cfmt::d(g[1], 5),
            cfmt::d(g[2], 5),
            cfmt::d(g[3], 5)
        ));
    }
    o.write_file("greenone.def", &t)?;
    o.print("    greenone.def is written.\n");
    Ok(())
}

/// `Print2Green`: `greentwo.def`.
fn print_2_green(o: &mut Out, s: &StdIntList) -> Res<()> {
    if s.ioutputmode == 0 {
        return Ok(());
    }
    let nsite = s.nsite;
    let xkondo = if s.model == "kondo" { 2 } else { 1 };
    let mut green: Vec<[i32; 8]> = Vec::new();
    let spin_max = |site: i32| {
        let f = s.locspinflag[site as usize];
        if f == 0 {
            1
        } else {
            f
        }
    };
    if s.ioutputmode == 1 {
        for site1 in 0..s.NsiteUC * xkondo {
            let site1k = if site1 >= s.NsiteUC {
                site1 - s.NsiteUC + nsite / 2
            } else {
                site1
            };
            let s1_max = spin_max(site1k);
            for spin1 in 0..=s1_max {
                for spin2 in 0..=s1_max {
                    for site3 in 0..nsite {
                        let s3_max = spin_max(site3);
                        for spin3 in 0..=s3_max {
                            for spin4 in 0..=s3_max {
                                if spin1 - spin2 + spin3 - spin4 == 0 {
                                    if spin1 != spin2 || spin3 != spin4 {
                                        green.push([
                                            site1k, spin1, site3, spin4, site3, spin3, site1k,
                                            spin2,
                                        ]);
                                    } else {
                                        green.push([
                                            site1k, spin1, site1k, spin2, site3, spin3, site3,
                                            spin4,
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if s.ioutputmode == 2 {
        let lf = |site: i32| s.locspinflag[site as usize];
        for site1 in 0..nsite {
            let s1_max = spin_max(site1);
            for spin1 in 0..=s1_max {
                for site2 in 0..nsite {
                    if lf(site1) != 0 && lf(site2) != 0 && site1 != site2 {
                        continue;
                    }
                    let s2_max = spin_max(site2);
                    for spin2 in 0..=s2_max {
                        for site3 in 0..nsite {
                            let s3_max = spin_max(site3);
                            for spin3 in 0..=s3_max {
                                for site4 in 0..nsite {
                                    if lf(site3) != 0 && lf(site4) != 0 && site3 != site4 {
                                        continue;
                                    }
                                    let s4_max = spin_max(site4);
                                    for spin4 in 0..=s4_max {
                                        green.push([
                                            site1, spin1, site2, spin2, site3, spin3, site4, spin4,
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!(
        "NCisAjsCktAltDC {}\n",
        cfmt::d(green.len() as i32, 10)
    ));
    t.push_str("=============================================\n");
    t.push_str("======== Green functions for Sq AND Nq ======\n");
    t.push_str("=============================================\n");
    for g in &green {
        let cols: Vec<String> = g.iter().map(|&v| cfmt::d(v, 5)).collect();
        t.push_str(&cols.join(" "));
        t.push('\n');
    }
    o.write_file("greentwo.def", &t)?;
    o.print("    greentwo.def is written.\n");
    Ok(())
}

/// `UnsupportedSystem`.
fn unsupported_system(o: &mut Out, model: &str, lattice: &str) -> Res<()> {
    o.print("\nSorry, specified combination, \n");
    outf!(o, "    MODEL : {model}  \n");
    outf!(o, "  LATTICE : {lattice}, \n");
    o.print("is unsupported in the STANDARD MODE...\n");
    o.print("Please use the EXPART MODE, or write a NEW FUNCTION and post us.\n");
    exit(-1)
}

/// `CheckOutputMode`.
fn check_output_mode(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let tag = "  ######  DEFAULT VALUE IS USED  ######";
    match s.outputmode.as_str() {
        "non" | "none" | "off" => {
            s.ioutputmode = 0;
            outf!(
                o,
                "      ioutputmode = {}\n",
                cfmt::d_left(s.ioutputmode, 10)
            );
        }
        "cor" | "corr" | "correlation" => {
            s.ioutputmode = 1;
            outf!(
                o,
                "      ioutputmode = {}\n",
                cfmt::d_left(s.ioutputmode, 10)
            );
        }
        "****" => {
            s.ioutputmode = 1;
            outf!(
                o,
                "      ioutputmode = {}{tag}\n",
                cfmt::d_left(s.ioutputmode, 10)
            );
        }
        "raw" | "all" | "full" => {
            s.ioutputmode = 2;
            outf!(
                o,
                "      ioutputmode = {}\n",
                cfmt::d_left(s.ioutputmode, 10)
            );
        }
        other => {
            outf!(o, "\n ERROR ! Unsupported OutPutMode : {other}\n");
            return exit(-1);
        }
    }
    Ok(())
}

/// `CheckModPara` (mVMC branch): defaults and consistency of the numerical parameters.
fn check_mod_para(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    if s.CParaFileHead == UNSET_STR {
        s.CParaFileHead = "zqp".to_string();
        outf!(
            o,
            "    CParaFileHead = {}######  DEFAULT VALUE IS USED  ######\n",
            cfmt::s_left(&s.CParaFileHead, 12)
        );
    } else {
        outf!(o, "    CParaFileHead = {}\n", s.CParaFileHead);
    }

    mu::print_val_i(o, "NVMCCalMode", &mut s.NVMCCalMode, 0);
    mu::print_val_i(o, "NLanczosMode", &mut s.NLanczosMode, 0);
    mu::print_val_i(o, "NDataIdxStart", &mut s.NDataIdxStart, 1);

    if s.NVMCCalMode == 0 {
        mu::not_used_i(o, "NDataQtySmp", s.NDataQtySmp)?;
    }
    mu::print_val_i(o, "NDataQtySmp", &mut s.NDataQtySmp, 1);

    if s.lGC == 0 && (s.Sz2 == 0 || s.Sz2 == NAN_I) {
        mu::print_val_i(o, "NSPGaussLeg", &mut s.NSPGaussLeg, 8);
        mu::print_val_i(o, "NSPStot", &mut s.NSPStot, 0);
    } else {
        mu::not_used_i(o, "NSPGaussLeg", s.NSPGaussLeg)?;
        mu::not_used_i(o, "NSPStot", s.NSPStot)?;
    }

    mu::print_val_i(o, "NMPTrans", &mut s.NMPTrans, -1);

    mu::print_val_i(o, "NSROptItrStep", &mut s.NSROptItrStep, 1000);

    if s.NVMCCalMode == 1 {
        mu::not_used_i(o, "NSROptItrSmp", s.NSROptItrSmp)?;
    }
    mu::print_val_i(o, "NSROptItrSmp", &mut s.NSROptItrSmp, s.NSROptItrStep / 10);

    mu::print_val_i(o, "NVMCWarmUp", &mut s.NVMCWarmUp, 10);
    mu::print_val_i(o, "NVMCInterval", &mut s.NVMCInterval, 1);
    mu::print_val_i(o, "NVMCSample", &mut s.NVMCSample, 1000);

    if s.model == "hubbard" {
        s.NExUpdatePath = 0;
    } else if s.model == "spin" {
        s.NExUpdatePath = 2;
    } else if s.model == "kondo" {
        s.NExUpdatePath = if s.lGC == 0 { 1 } else { 3 };
    }
    outf!(
        o,
        "  {} = {}\n",
        cfmt::s("NExUpdatePath", 15),
        cfmt::d_left(s.NExUpdatePath, 10)
    );

    mu::print_val_i(o, "RndSeed", &mut s.RndSeed, 123456789);
    mu::print_val_i(o, "NSplitSize", &mut s.NSplitSize, 1);
    mu::print_val_i(o, "NStore", &mut s.NStore, 1);
    mu::print_val_i(o, "NSRCG", &mut s.NSRCG, 0);

    mu::print_val_d(o, "DSROptRedCut", &mut s.DSROptRedCut, 0.001);
    mu::print_val_d(o, "DSROptStaDel", &mut s.DSROptStaDel, 0.02);
    mu::print_val_d(o, "DSROptStepDt", &mut s.DSROptStepDt, 0.02);

    // (Un)Conserved variables (Number of electrons, total Sz)
    if s.model == "hubbard" || s.model == "kondo" {
        mu::required_val_i(o, "ncond", s.ncond)?;
        if s.lGC == 0 {
            mu::print_val_i(o, "2Sz", &mut s.Sz2, 0);
        } else {
            mu::not_used_i(o, "2Sz", s.Sz2)?;
        }
    } else if s.model == "spin" {
        mu::not_used_i(o, "ncond", s.ncond)?;
        s.ncond = 0;
        if s.lGC == 0 {
            mu::required_val_i(o, "2Sz", s.Sz2)?;
        } else {
            mu::not_used_i(o, "2Sz", s.Sz2)?;
        }
    }
    Ok(())
}

/// Merge equivalent symmetric pair terms (`Cinter`, `Hund`, `Ex`, `PairLift`, `PairHopp`).
fn merge_pair_terms(idx: &[[i32; 2]], val: &mut [f64]) {
    let n = val.len();
    for k in 0..n {
        for j in k + 1..n {
            if (idx[j][0] == idx[k][0] && idx[j][1] == idx[k][1])
                || (idx[j][0] == idx[k][1] && idx[j][1] == idx[k][0])
            {
                val[k] += val[j];
                val[j] = 0.0;
            }
        }
    }
}

/// Write one pair-term file: merge, count, set the print flag and write if non-empty.
#[allow(clippy::too_many_arguments)]
fn write_pair_file(
    o: &mut Out,
    idx: &[[i32; 2]],
    val: &mut [f64],
    lboost: i32,
    flag: &mut i32,
    file: &str,
    count_label: &str,
    banner: &str,
) -> Res<()> {
    merge_pair_terms(idx, val);
    let nintr0 = val.iter().filter(|v| v.abs() > 0.000001).count() as i32;
    *flag = if nintr0 == 0 || lboost == 1 { 0 } else { 1 };
    if *flag == 1 {
        let mut t = String::new();
        t.push_str("=============================================\n");
        t.push_str(&format!("{count_label} {}\n", cfmt::d(nintr0, 10)));
        t.push_str("=============================================\n");
        t.push_str(banner);
        t.push('\n');
        t.push_str("=============================================\n");
        for k in 0..val.len() {
            if val[k].abs() > 0.000001 {
                t.push_str(&format!(
                    "{} {} {}\n",
                    cfmt::d(idx[k][0], 5),
                    cfmt::d(idx[k][1], 5),
                    cfmt::f(val[k], 25, 15)
                ));
            }
        }
        o.write_file(file, &t)?;
        outf!(o, "    {file} is written.\n");
    }
    Ok(())
}

/// `PrintInteractions`: merge and write the interaction files.
fn print_interactions(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // Coulomb INTRA
    let n = s.Cintra.len();
    for k in 0..n {
        for j in k + 1..n {
            if s.CintraIndx[j][0] == s.CintraIndx[k][0] {
                s.Cintra[k] += s.Cintra[j];
                s.Cintra[j] = 0.0;
            }
        }
    }
    let nintr0 = s.Cintra.iter().filter(|v| v.abs() > 0.000001).count() as i32;
    s.LCintra = if nintr0 == 0 || s.lBoost == 1 { 0 } else { 1 };
    if s.LCintra == 1 {
        let mut t = String::new();
        t.push_str("=============================================\n");
        t.push_str(&format!("NCoulombIntra {}\n", cfmt::d(nintr0, 10)));
        t.push_str("=============================================\n");
        t.push_str("================== CoulombIntra ================\n");
        t.push_str("=============================================\n");
        for k in 0..n {
            if s.Cintra[k].abs() > 0.000001 {
                t.push_str(&format!(
                    "{} {}\n",
                    cfmt::d(s.CintraIndx[k][0], 5),
                    cfmt::f(s.Cintra[k], 25, 15)
                ));
            }
        }
        o.write_file("coulombintra.def", &t)?;
        o.print("    coulombintra.def is written.\n");
    }
    let lboost = s.lBoost;
    write_pair_file(
        o,
        &s.CinterIndx,
        &mut s.Cinter,
        lboost,
        &mut s.LCinter,
        "coulombinter.def",
        "NCoulombInter",
        "================== CoulombInter ================",
    )?;
    write_pair_file(
        o,
        &s.HundIndx,
        &mut s.Hund,
        lboost,
        &mut s.LHund,
        "hund.def",
        "NHund",
        "=============== Hund coupling ===============",
    )?;
    write_pair_file(
        o,
        &s.ExIndx,
        &mut s.Ex,
        lboost,
        &mut s.LEx,
        "exchange.def",
        "NExchange",
        "====== ExchangeCoupling coupling ============",
    )?;
    write_pair_file(
        o,
        &s.PLIndx,
        &mut s.PairLift,
        lboost,
        &mut s.LPairLift,
        "pairlift.def",
        "NPairLift",
        "====== Pair-Lift term ============",
    )?;
    write_pair_file(
        o,
        &s.PHIndx,
        &mut s.PairHopp,
        lboost,
        &mut s.LPairHopp,
        "pairhopp.def",
        "NPairHopp",
        "====== Pair-Hopping term ============",
    )?;

    // InterAll: merge equivalent terms.
    let n = s.intr.len();
    for j in 0..n {
        for k in j + 1..n {
            let a = s.intrindx[j];
            let b = s.intrindx[k];
            let same = a == b;
            let swapped = a[0] == b[4]
                && a[1] == b[5]
                && a[2] == b[6]
                && a[3] == b[7]
                && a[4] == b[0]
                && a[5] == b[1]
                && a[6] == b[2]
                && a[7] == b[3]
                && !(a[0] == a[6] && a[1] == a[7])
                && !(a[2] == a[4] && a[3] == a[5]);
            if same || swapped {
                s.intr[j] = s.intr[j] + s.intr[k];
                s.intr[k] = C64::real(0.0);
            } else if (a[0] == b[4]
                && a[1] == b[5]
                && a[2] == b[2]
                && a[3] == b[3]
                && a[4] == b[0]
                && a[5] == b[1]
                && a[6] == b[6]
                && a[7] == b[7]
                && !(a[2] == a[0] && a[3] == a[1])
                && !(a[2] == a[4] && a[3] == a[5]))
                || (a[0] == b[0]
                    && a[1] == b[1]
                    && a[2] == b[6]
                    && a[3] == b[7]
                    && a[4] == b[4]
                    && a[5] == b[5]
                    && a[6] == b[2]
                    && a[7] == b[3]
                    && !(a[4] == a[2] && a[5] == a[3])
                    && !(a[4] == a[6] && a[5] == a[7]))
            {
                s.intr[j] = s.intr[j] - s.intr[k];
                s.intr[k] = C64::real(0.0);
            }
        }
    }
    // Force Hermite term as (c1+ c2 c3+ c4)+ = c4+ c3 c2+ c1
    for j in 0..n {
        for k in j + 1..n {
            let a = s.intrindx[j];
            let b = s.intrindx[k];
            let full_swap = [a[6], a[7], a[4], a[5], a[2], a[3], a[0], a[1]];
            if a[6] == b[4]
                && a[7] == b[5]
                && a[4] == b[6]
                && a[5] == b[7]
                && a[2] == b[0]
                && a[3] == b[1]
                && a[0] == b[2]
                && a[1] == b[3]
                && !(b[0] == b[6] && b[1] == b[7])
                && !(b[2] == b[4] && b[3] == b[5])
            {
                s.intrindx[k] = full_swap;
            } else if (a[6] == b[4]
                && a[7] == b[5]
                && a[4] == b[2]
                && a[5] == b[3]
                && a[2] == b[0]
                && a[3] == b[1]
                && a[0] == b[6]
                && a[1] == b[7]
                && !(b[2] == b[0] && b[3] == b[1])
                && !(b[2] == b[4] && b[3] == b[5]))
                || (a[6] == b[0]
                    && a[7] == b[1]
                    && a[4] == b[6]
                    && a[5] == b[7]
                    && a[2] == b[4]
                    && a[3] == b[5]
                    && a[0] == b[2]
                    && a[1] == b[3]
                    && !(b[4] == b[2] && b[5] == b[3])
                    && !(b[4] == b[6] && b[5] == b[7]))
            {
                s.intrindx[k] = full_swap;
                s.intr[k] = -s.intr[k];
            }
        }
    }
    for j in 0..n {
        let a = s.intrindx[j];
        if ((a[0] == a[4] && a[1] == a[5]) || (a[2] == a[6] && a[3] == a[7]))
            && !((a[0] == a[2] && a[1] == a[3])
                || (a[0] == a[6] && a[1] == a[7])
                || (a[4] == a[2] && a[5] == a[3])
                || (a[4] == a[6] && a[5] == a[7]))
        {
            s.intr[j] = C64::real(0.0);
        }
    }

    let nintr0 = s.intr.iter().filter(|c| c.abs() > 0.000001).count() as i32;
    s.Lintr = if nintr0 == 0 || s.lBoost == 1 { 0 } else { 1 };
    if s.Lintr == 1 {
        let mut t = String::new();
        t.push_str("====================== \n");
        t.push_str(&format!("NInterAll {}  \n", cfmt::d(nintr0, 7)));
        t.push_str("====================== \n");
        t.push_str("========zInterAll===== \n");
        t.push_str("====================== \n");
        for k in 0..n {
            if s.intr[k].abs() > 0.000001 {
                let ix = s.intrindx[k];
                let cols: Vec<String> = ix.iter().map(|&v| cfmt::d(v, 5)).collect();
                t.push_str(&format!(
                    "{} {}  {}\n",
                    cols.join(" "),
                    cfmt::f(s.intr[k].re, 25, 15),
                    cfmt::f(s.intr[k].im, 25, 15)
                ));
            }
        }
        o.write_file("interall.def", &t)?;
        o.print("    interall.def is written.\n");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

/// `StdFace_main` on an in-memory input file (`name` is only echoed in the log). `None` stands
/// for a file that cannot be opened.
pub fn stdface_main_bytes(
    name: &str,
    input: Option<&[u8]>,
    out_dir: &Path,
) -> Result<StdFaceReport, StdFaceFailure> {
    let mut o = Out::new(out_dir);
    match run(&mut o, name, input) {
        Ok(()) => Ok(StdFaceReport { log: o.log }),
        Err(error) => Err(StdFaceFailure { error, log: o.log }),
    }
}

/// `StdFace_main(fname)`: read the Standard-mode input `input` and write the Expert files into
/// `out_dir`.
pub fn stdface_main(input: &Path, out_dir: &Path) -> Result<StdFaceReport, StdFaceFailure> {
    let mut o = Out::new(out_dir);
    let data = std::fs::read(input).ok();
    let name = input.to_string_lossy().into_owned();
    match run(&mut o, &name, data.as_deref()) {
        Ok(()) => Ok(StdFaceReport { log: o.log }),
        Err(error) => Err(StdFaceFailure { error, log: o.log }),
    }
}

/// `fgets(buf, 256, fp)` over the whole file: chunks of at most 255 bytes ending at a newline.
fn fgets_chunks(data: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let mut end = pos;
        while end < data.len() && end - pos < 255 {
            end += 1;
            if data[end - 1] == b'\n' {
                break;
            }
        }
        chunks.push(&data[pos..end]);
        pos = end;
    }
    chunks
}

fn run(o: &mut Out, fname: &str, input: Option<&[u8]>) -> Res<()> {
    let mut s = StdIntList::reset_vals();
    o.print("\n######  Input Parameter of Standard Intarface  ######\n");
    let Some(input) = input else {
        outf!(o, "\n  ERROR !  Cannot open input file {fname} !\n\n");
        return exit(-1);
    };
    outf!(o, "\n  Open Standard-Mode Inputfile {fname} \n\n");

    for chunk in fgets_chunks(input) {
        // C strings end at the first NUL.
        let chunk = chunk.split(|&b| b == 0).next().unwrap_or(&[]);
        let line = trim_space_quote(chunk);
        if line.starts_with(b"//") || line.is_empty() {
            o.print("  Skipping a line.\n");
            continue;
        }
        let mut tok = Strtok { rest: &line };
        let keyword = tok.next(b'=');
        let value = tok.next(b'=');
        let (Some(keyword), Some(value)) = (keyword, value) else {
            o.print("\n  ERROR !  \"=\" is NOT found !\n\n");
            return exit(-1);
        };
        let keyword = text2lower(&lossy(keyword));
        let value = lossy(value);
        outf!(
            o,
            "  KEYWORD : {} | VALUE : {} \n",
            cfmt::s_left(&keyword, 20),
            value
        );
        store_keyword(o, &mut s, &keyword, &value)?;
    }
    o.print("\n");
    o.print("#######  Construct Model  #######\n");
    o.print("\n");

    if s.CDataFileHead == UNSET_STR {
        s.CDataFileHead = "zvo".to_string();
        outf!(
            o,
            "    CDataFileHead = {}######  DEFAULT VALUE IS USED  ######\n",
            cfmt::s_left(&s.CDataFileHead, 12)
        );
    } else {
        outf!(o, "    CDataFileHead = {}\n", s.CDataFileHead);
    }
    s.lGC = 0;
    s.lBoost = 0;
    match s.model.as_str() {
        "fermionhubbard" | "hubbard" => s.model = "hubbard".to_string(),
        "fermionhubbardgc" | "hubbardgc" => {
            s.model = "hubbard".to_string();
            s.lGC = 1;
        }
        "spin" => s.model = "spin".to_string(),
        "spingc" => {
            s.model = "spin".to_string();
            s.lGC = 1;
        }
        "kondolattice" | "kondo" => s.model = "kondo".to_string(),
        "kondolatticegc" | "kondogc" => {
            s.model = "kondo".to_string();
            s.lGC = 1;
        }
        _ => unsupported_system(o, &s.model.clone(), &s.lattice.clone())?,
    }

    // Generate Hamiltonian definition files
    match s.lattice.as_str() {
        "chain" | "chainlattice" => std_face_chain(o, &mut s)?,
        "ladder" | "ladderlattice" => std_face_ladder(o, &mut s)?,
        "face-centeredorthorhombic"
        | "fcorthorhombic"
        | "fco"
        | "face-centeredcubic"
        | "fccubic"
        | "fcc"
        | "honeycomb"
        | "honeycomblattice"
        | "kagome"
        | "kagomelattice"
        | "orthorhombic"
        | "simpleorthorhombic"
        | "cubic"
        | "simplecubic"
        | "pyrochlore"
        | "tetragonal"
        | "tetragonallattice"
        | "square"
        | "squarelattice"
        | "triangular"
        | "triangularlattice"
        | "wannier90" => {
            outf!(
                o,
                "\nSorry, lattice {} is not ported to the Rust StdFace yet (issues #354-#357).\n",
                s.lattice
            );
            return exit(-1);
        }
        _ => unsupported_system(o, &s.model.clone(), &s.lattice.clone())?,
    }

    o.print("\n");
    o.print("######  Print Expert input files  ######\n");
    o.print("\n");

    print_loc_spin(o, &s)?;
    print_trans(o, &mut s)?;
    print_interactions(o, &mut s)?;
    check_mod_para(o, &mut s)?;
    print_mod_para(o, &s)?;

    if s.lGC == 0 && (s.Sz2 == 0 || s.Sz2 == NAN_I) {
        mu::print_val_i(o, "ComplexType", &mut s.ComplexType, 0);
    } else {
        mu::print_val_i(o, "ComplexType", &mut s.ComplexType, 1);
    }

    mu::generate_orb(o, &mut s)?;
    mu::proj(o, &mut s)?;
    mu::print_jastrow(o, &mut s)?;
    if s.lGC == 1 || (s.Sz2 != 0 && s.Sz2 != NAN_I) {
        print_orb_para(o, &mut s)?;
    }
    print_gutzwiller(o, &mut s)?;
    print_orb(o, &mut s)?;

    check_output_mode(o, &mut s)?;
    print_1_green(o, &s)?;
    print_2_green(o, &s)?;

    print_namelist(o, &s)?;

    o.print("\n######  Input files are generated.  ######\n\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_removes_only_the_c_set() {
        assert_eq!(trim_space_quote(b" A : \"b\";\\c\x0b\n"), b"Abc".to_vec());
        // tabs and carriage returns survive TrimSpaceQuote
        assert_eq!(trim_space_quote(b"a\t=\r1\n"), b"a\t=\r1".to_vec());
    }

    #[test]
    fn strtok_skips_leading_delimiters_and_stops_at_the_next() {
        let mut tok = Strtok { rest: b"==a==b=c" };
        assert_eq!(tok.next(b'='), Some(&b"a"[..]));
        assert_eq!(tok.next(b'='), Some(&b"b"[..]));
        assert_eq!(tok.next(b'='), Some(&b"c"[..]));
        assert_eq!(tok.next(b'='), None);
        let mut tok = Strtok { rest: b"a=" };
        assert_eq!(tok.next(b'='), Some(&b"a"[..]));
        assert_eq!(tok.next(b'='), None);
    }

    #[test]
    fn scanf_int_follows_c() {
        assert_eq!(scan_int(b"12abc"), Some(12));
        assert_eq!(scan_int(b"-7"), Some(-7));
        assert_eq!(scan_int(b"+3.9"), Some(3));
        assert_eq!(scan_int(b"abc"), None);
        assert_eq!(scan_int(b""), None);
        // glibc: strtol saturates to LONG_MAX, then the low 32 bits are stored.
        assert_eq!(scan_int(b"99999999999999999999"), Some(-1));
    }

    #[test]
    fn scanf_double_follows_c() {
        assert_eq!(scan_double(b"1e-8"), Some(1e-8));
        assert_eq!(scan_double(b"4.0"), Some(4.0));
        assert_eq!(scan_double(b".5x"), Some(0.5));
        assert_eq!(scan_double(b"-2."), Some(-2.0));
        assert_eq!(scan_double(b"3e"), Some(3.0));
        assert_eq!(scan_double(b"+1E+2"), Some(100.0));
        assert!(scan_double(b"nan").unwrap().is_nan());
        assert_eq!(scan_double(b"-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(scan_double(b"x1"), None);
        assert_eq!(scan_double(b"."), None);
    }

    #[test]
    fn fgets_splits_long_lines_at_255_bytes() {
        let mut data = vec![b'a'; 300];
        data.extend_from_slice(b"\nb=1\n");
        let chunks = fgets_chunks(&data);
        assert_eq!(
            chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            vec![255, 46, 4]
        );
        assert_eq!(chunks[2], b"b=1\n");
    }

    #[test]
    fn complex_values_split_at_the_comma() {
        let mut o = Out::new(".");
        let mut v = C64::real(f64::NAN);
        dup_c(&mut o, "t", "1.5,-2", &mut v).unwrap();
        assert_eq!((v.re, v.im), (1.5, -2.0));
        let mut v = C64::real(f64::NAN);
        dup_c(&mut o, "t", ",0.25", &mut v).unwrap();
        assert_eq!((v.re, v.im), (0.0, 0.25));
        let mut v = C64::real(f64::NAN);
        dup_c(&mut o, "t", "junk", &mut v).unwrap();
        assert_eq!((v.re, v.im), (0.0, 0.0));
        // a value that is already set is a duplicate
        assert_eq!(dup_c(&mut o, "t", "1", &mut v), Err(StdFaceError::Exit(-1)));
    }

    #[test]
    fn unsupported_and_duplicate_keywords_exit_with_the_c_messages() {
        let mut o = Out::new(".");
        let mut s = StdIntList::reset_vals();
        assert_eq!(
            store_keyword(&mut o, &mut s, "method", "lanczos"),
            Err(StdFaceError::Exit(-1))
        );
        assert_eq!(o.log, "ERROR ! Unsupported Keyword in Standard mode!\n");
        let mut o = Out::new(".");
        store_keyword(&mut o, &mut s, "u", "4").unwrap();
        assert_eq!(
            store_keyword(&mut o, &mut s, "u", "5"),
            Err(StdFaceError::Exit(-1))
        );
        assert_eq!(o.log, "ERROR !  Keyword u is duplicated ! \n");
    }
}
