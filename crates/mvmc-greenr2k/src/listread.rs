//! Minimal emulation of Fortran list-directed input (`READ(unit,*) a, b, ...`)
//! as `greenr2k.F90` uses it.
//!
//! Items are separated by blanks or commas. Blank records are skipped while
//! items are still needed, a read continues on the following records when a
//! record holds too few items, and the remainder of the last record read is
//! discarded. Quoted character items may contain separators.
//!
//! Deliberate difference from gfortran: an unquoted character item containing
//! `/` is read as a whole. gfortran treats `/` as a terminator, which turns the
//! file name `./modpara.def` into `.`.

use std::fs;
use std::path::Path;

use crate::Error;

pub struct Records {
    name: String,
    lines: Vec<String>,
    pos: usize,
}

fn tokenize(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' || c == '\t' || c == ',' {
            i += 1;
            continue;
        }
        if c == '"' || c == '\'' {
            let quote = c;
            i += 1;
            let mut token = String::new();
            while i < chars.len() {
                if chars[i] == quote {
                    if i + 1 < chars.len() && chars[i + 1] == quote {
                        token.push(quote);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                token.push(chars[i]);
                i += 1;
            }
            tokens.push(token);
            continue;
        }
        let mut token = String::new();
        while i < chars.len() && !matches!(chars[i], ' ' | '\t' | ',') {
            token.push(chars[i]);
            i += 1;
        }
        tokens.push(token);
    }
    tokens
}

impl Records {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("cannot open '{}': {e}", path.display())))?;
        Ok(Self::from_text(&path.display().to_string(), &text))
    }

    pub fn from_text(name: &str, text: &str) -> Self {
        let lines = text
            .lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect();
        Self {
            name: name.to_string(),
            lines,
            pos: 0,
        }
    }

    /// First token of the next non-blank record (the Fortran
    /// `READ(fi,*,END=..) keyname; BACKSPACE(fi)` idiom). `None` at end of file.
    pub fn peek_key(&mut self) -> Option<String> {
        while self.pos < self.lines.len() {
            if let Some(first) = tokenize(&self.lines[self.pos]).into_iter().next() {
                return Some(first);
            }
            self.pos += 1;
        }
        None
    }

    /// `READ(fi,*) item_1, ..., item_n` returning the raw items.
    pub fn read(&mut self, n: usize) -> Result<Vec<String>, Error> {
        let mut items = Vec::with_capacity(n);
        while items.len() < n {
            if self.pos >= self.lines.len() {
                return Err(Error::Io(format!("end of file in '{}'", self.name)));
            }
            let tokens = tokenize(&self.lines[self.pos]);
            self.pos += 1;
            for token in tokens {
                if items.len() < n {
                    items.push(token);
                }
            }
        }
        Ok(items)
    }

    pub fn read_strs(&mut self, n: usize) -> Result<Vec<String>, Error> {
        self.read(n)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

pub fn parse_int(token: &str, file: &str) -> Result<i64, Error> {
    let t = token.trim_start_matches('+');
    t.parse::<i64>()
        .map_err(|_| Error::Parse(format!("bad integer '{token}' in '{file}'")))
}

/// Fortran real item: `d`/`D`/`q` exponent letters and exponents without a letter.
pub fn parse_real(token: &str, file: &str) -> Result<f64, Error> {
    let mut s = token.replace(['d', 'D', 'q', 'Q'], "e");
    if !s.contains(['e', 'E']) {
        // "1.5+3" means 1.5e+3.
        if let Some(p) = s.rfind(['+', '-']) {
            if p > 0 && s.as_bytes()[p - 1].is_ascii_digit() || p > 0 && s.as_bytes()[p - 1] == b'.'
            {
                s.insert(p, 'e');
            }
        }
    }
    s.parse::<f64>()
        .map_err(|_| Error::Parse(format!("bad real '{token}' in '{file}'")))
}
