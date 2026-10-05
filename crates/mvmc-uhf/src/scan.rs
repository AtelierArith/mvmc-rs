//! Minimal `sscanf`/`fscanf` conversions used by the C ComplexUHF reader.
//!
//! The C reader (`ComplexUHF/readdef.c`) parses its lines with `%s`, `%d`
//! and `%lf`. Those conversions accept numeric *prefixes* (`"1.5abc"` gives
//! `1.5`; `%d` on `"2.7"` gives `2`) and stop at the first failing
//! conversion. This scanner reproduces that behaviour for the formats that
//! occur in the reader.

/// Whitespace set of C `isspace` in the "C" locale.
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// A cursor over one line (or a whole stream) of ASCII text.
pub struct Scan<'a> {
    text: &'a [u8],
    position: usize,
}

impl<'a> Scan<'a> {
    /// Start scanning `text`. Scanning stops at an embedded NUL like a C string.
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Self {
            text: &bytes[..end],
            position: 0,
        }
    }

    fn skip_space(&mut self) {
        while self
            .text
            .get(self.position)
            .is_some_and(|&byte| is_space(byte))
        {
            self.position += 1;
        }
    }

    /// `%s`: the next whitespace-free word, or `None` at end of input.
    pub fn word(&mut self) -> Option<&'a str> {
        self.skip_space();
        let start = self.position;
        while self
            .text
            .get(self.position)
            .is_some_and(|&byte| !is_space(byte))
        {
            self.position += 1;
        }
        if self.position == start {
            None
        } else {
            std::str::from_utf8(&self.text[start..self.position]).ok()
        }
    }

    /// `%d`: an optionally signed decimal integer prefix.
    ///
    /// Values outside the `i32` range saturate (undefined behaviour in C).
    pub fn int(&mut self) -> Option<i32> {
        self.skip_space();
        let start = self.position;
        let mut cursor = self.position;
        if self
            .text
            .get(cursor)
            .is_some_and(|&byte| byte == b'+' || byte == b'-')
        {
            cursor += 1;
        }
        let digits = cursor;
        while self.text.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == digits {
            return None;
        }
        self.position = cursor;
        let parsed: i64 = std::str::from_utf8(&self.text[start..cursor])
            .ok()?
            .trim_start_matches('+')
            .parse()
            .unwrap_or(if self.text[start] == b'-' {
                i64::MIN
            } else {
                i64::MAX
            });
        Some(parsed.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// `%lf`: the longest decimal floating-point prefix (also `inf`/`nan`).
    pub fn float(&mut self) -> Option<f64> {
        self.skip_space();
        let start = self.position;
        let mut cursor = start;
        let sign_end = if self
            .text
            .get(cursor)
            .is_some_and(|&byte| byte == b'+' || byte == b'-')
        {
            cursor + 1
        } else {
            cursor
        };
        let rest = &self.text[sign_end..];
        for name in ["infinity", "inf", "nan"] {
            if rest.len() >= name.len() && rest[..name.len()].eq_ignore_ascii_case(name.as_bytes())
            {
                let end = sign_end + name.len();
                let literal = std::str::from_utf8(&self.text[start..end]).ok()?;
                self.position = end;
                return literal.parse().ok();
            }
        }
        cursor = sign_end;
        let int_start = cursor;
        while self.text.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        let mut mantissa_digits = cursor - int_start;
        if self.text.get(cursor) == Some(&b'.') {
            let after_dot = cursor + 1;
            let mut fraction_end = after_dot;
            while self.text.get(fraction_end).is_some_and(u8::is_ascii_digit) {
                fraction_end += 1;
            }
            let fraction_digits = fraction_end - after_dot;
            if mantissa_digits + fraction_digits > 0 {
                mantissa_digits += fraction_digits;
                cursor = fraction_end;
            }
        }
        if mantissa_digits == 0 {
            return None;
        }
        if self
            .text
            .get(cursor)
            .is_some_and(|&byte| byte == b'e' || byte == b'E')
        {
            let mut exponent = cursor + 1;
            if self
                .text
                .get(exponent)
                .is_some_and(|&byte| byte == b'+' || byte == b'-')
            {
                exponent += 1;
            }
            let exponent_digits = exponent;
            while self.text.get(exponent).is_some_and(u8::is_ascii_digit) {
                exponent += 1;
            }
            if exponent > exponent_digits {
                cursor = exponent;
            }
        }
        self.position = cursor;
        std::str::from_utf8(&self.text[start..cursor])
            .ok()?
            .parse()
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_takes_prefix_and_stops() {
        let mut scan = Scan::new("  12 -3 4.5 x");
        assert_eq!(scan.int(), Some(12));
        assert_eq!(scan.int(), Some(-3));
        assert_eq!(scan.int(), Some(4));
        assert_eq!(scan.int(), None);
    }

    #[test]
    fn float_takes_longest_prefix() {
        let mut scan = Scan::new("1.5e3x -.25 2e+ +7. inf");
        assert_eq!(scan.float(), Some(1500.0));
        assert_eq!(scan.word(), Some("x"));
        assert_eq!(scan.float(), Some(-0.25));
        assert_eq!(scan.float(), Some(2.0));
        assert_eq!(scan.word(), Some("e+"));
        assert_eq!(scan.float(), Some(7.0));
        assert_eq!(scan.float(), Some(f64::INFINITY));
        assert_eq!(scan.float(), None);
    }

    #[test]
    fn word_skips_whitespace() {
        let mut scan = Scan::new("\t Nsite   8\n");
        assert_eq!(scan.word(), Some("Nsite"));
        assert_eq!(scan.int(), Some(8));
        assert_eq!(scan.word(), None);
    }
}
