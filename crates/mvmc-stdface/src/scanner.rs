//! A tiny `fscanf`/`fgets` emulation over an in-memory file, enough for the Wannier90 files.
//!
//! `%d` and `%lf` skip leading white space and read the longest numeric prefix, so `1.5` read
//! with `%d` yields `1` and leaves `.5`; a conversion that fails leaves the variable unchanged
//! and ends the call (C returns the number of conversions made, or `EOF` when the input ended
//! before the first one).

/// `EOF` as returned by `fscanf`.
pub const EOF: i32 = -1;

/// A byte stream with C `FILE` read semantics.
pub struct Scanner<'a> {
    data: &'a [u8],
    pos: usize,
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

enum Item<T> {
    Value(T),
    Fail,
    Eof,
}

impl<'a> Scanner<'a> {
    /// Start at the beginning of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// `fgets(buf, 256, fp)`: consume up to 255 bytes, stopping after a newline.
    pub fn fgets(&mut self) {
        let mut n = 0;
        while self.pos < self.data.len() && n < 255 {
            let b = self.data[self.pos];
            self.pos += 1;
            n += 1;
            if b == b'\n' {
                break;
            }
        }
    }

    fn skip_space(&mut self) {
        while self.pos < self.data.len() && is_space(self.data[self.pos]) {
            self.pos += 1;
        }
    }

    fn int(&mut self) -> Item<i32> {
        self.skip_space();
        if self.pos >= self.data.len() {
            return Item::Eof;
        }
        let mut i = self.pos;
        let mut negative = false;
        if self.data[i] == b'+' || self.data[i] == b'-' {
            negative = self.data[i] == b'-';
            i += 1;
        }
        let start = i;
        let mut value: i64 = 0;
        while i < self.data.len() && self.data[i].is_ascii_digit() {
            value = value
                .saturating_mul(10)
                .saturating_add((self.data[i] - b'0') as i64);
            i += 1;
        }
        if i == start {
            return Item::Fail;
        }
        self.pos = i;
        Item::Value(if negative { -value } else { value } as i32)
    }

    fn double(&mut self) -> Item<f64> {
        self.skip_space();
        if self.pos >= self.data.len() {
            return Item::Eof;
        }
        let s = &self.data[self.pos..];
        let mut i = 0;
        let negative = s[0] == b'-';
        if s[0] == b'+' || s[0] == b'-' {
            i += 1;
        }
        let rest: Vec<u8> = s[i..]
            .iter()
            .take(8)
            .map(|b| b.to_ascii_lowercase())
            .collect();
        if rest.starts_with(b"infinity") {
            self.pos += i + 8;
            return Item::Value(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        if rest.starts_with(b"inf") {
            self.pos += i + 3;
            return Item::Value(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        if rest.starts_with(b"nan") {
            self.pos += i + 3;
            return Item::Value(if negative { -f64::NAN } else { f64::NAN });
        }
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
            return Item::Fail;
        }
        let mut end = i;
        if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
            let mut k = i + 1;
            if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
                k += 1;
            }
            let first = k;
            while k < s.len() && s[k].is_ascii_digit() {
                k += 1;
            }
            if k > first {
                end = k;
            }
        }
        let text = std::str::from_utf8(&s[..end]).unwrap_or("0");
        let value = text.trim_start_matches('+').parse::<f64>().unwrap_or(0.0);
        self.pos += end;
        Item::Value(value)
    }

    /// `fscanf(fp, "%d", &v)`: returns the conversion count (0/1) or [`EOF`].
    pub fn scan_int(&mut self, v: &mut i32) -> i32 {
        match self.int() {
            Item::Value(x) => {
                *v = x;
                1
            }
            Item::Fail => 0,
            Item::Eof => EOF,
        }
    }

    /// `fscanf(fp, "%lf%lf...", ...)` into `out`.
    pub fn scan_doubles(&mut self, out: &mut [f64]) -> i32 {
        let mut count = 0;
        for slot in out.iter_mut() {
            match self.double() {
                Item::Value(x) => {
                    *slot = x;
                    count += 1;
                }
                Item::Fail => return count,
                Item::Eof => return if count == 0 { EOF } else { count },
            }
        }
        count
    }

    /// `fscanf(fp, "%d%d%d%d%d%lf%lf", ints..., dbls...)`.
    pub fn scan_row(&mut self, ints: &mut [i32; 5], dbls: &mut [f64; 2]) -> i32 {
        let mut count = 0;
        for slot in ints.iter_mut() {
            match self.int() {
                Item::Value(x) => {
                    *slot = x;
                    count += 1;
                }
                Item::Fail => return count,
                Item::Eof => return if count == 0 { EOF } else { count },
            }
        }
        for slot in dbls.iter_mut() {
            match self.double() {
                Item::Value(x) => {
                    *slot = x;
                    count += 1;
                }
                Item::Fail => return count,
                Item::Eof => return count,
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_reads_a_prefix_and_fails_on_text() {
        let mut s = Scanner::new(b" 12.5 x");
        let mut v = 0;
        assert_eq!(s.scan_int(&mut v), 1);
        assert_eq!(v, 12);
        assert_eq!(s.scan_int(&mut v), 0); // ".5" is not an integer
        assert_eq!(v, 12);
    }

    #[test]
    fn eof_is_reported_before_the_first_conversion() {
        let mut s = Scanner::new(b"  \n");
        let mut v = [0.0; 2];
        assert_eq!(s.scan_doubles(&mut v), EOF);
        let mut s = Scanner::new(b"1.5");
        assert_eq!(s.scan_doubles(&mut v), 1);
        assert_eq!(v[0], 1.5);
    }

    #[test]
    fn fgets_stops_at_newline_or_255_bytes() {
        let mut s = Scanner::new(b"header line\n  7");
        s.fgets();
        let mut v = 0;
        assert_eq!(s.scan_int(&mut v), 1);
        assert_eq!(v, 7);
        let long = vec![b'x'; 300];
        let mut s = Scanner::new(&long);
        s.fgets();
        assert_eq!(s.pos, 255);
    }
}
