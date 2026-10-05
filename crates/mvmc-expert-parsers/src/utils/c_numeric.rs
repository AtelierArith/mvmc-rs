//! Pure-Rust C numeric prefixes and correctly rounded hexadecimal Float64.
pub(crate) struct Scan<'a> {
    text: &'a [u8],
    position: usize,
}

impl<'a> Scan<'a> {
    pub(crate) fn new(text: &'a [u8]) -> Self {
        let end = text
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(text.len());
        Self {
            text: &text[..end],
            position: 0,
        }
    }
    fn skip_space(&mut self) {
        while self
            .text
            .get(self.position)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
        {
            self.position += 1;
        }
    }
    /// `sscanf("%s")`: skip whitespace and return the next whitespace-free word.
    pub(crate) fn token(&mut self) -> Option<&'a [u8]> {
        self.skip_space();
        let start = self.position;
        self.word().then(|| &self.text[start..self.position])
    }
    pub(crate) fn word(&mut self) -> bool {
        self.skip_space();
        let start = self.position;
        while self
            .text
            .get(self.position)
            .is_some_and(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
        {
            self.position += 1;
        }
        self.position != start
    }
    pub(crate) fn integer(&mut self) -> Result<Option<i32>, ()> {
        self.skip_space();
        let start = self.position;
        if self
            .text
            .get(self.position)
            .is_some_and(|b| matches!(b, b'+' | b'-'))
        {
            self.position += 1;
        }
        let digits = self.position;
        while self.text.get(self.position).is_some_and(u8::is_ascii_digit) {
            self.position += 1;
        }
        if self.position == digits {
            return Ok(None);
        }
        std::str::from_utf8(&self.text[start..self.position])
            .unwrap()
            .parse()
            .map(Some)
            .map_err(|_| ())
    }
    pub(crate) fn float(&mut self) -> Option<f64> {
        self.skip_space();
        let start = self.position;
        let end = float_prefix(&self.text[start..])?;
        self.position += end;
        parse_complete(std::str::from_utf8(&self.text[start..self.position]).ok()?)
    }
}

fn float_prefix(text: &[u8]) -> Option<usize> {
    let sign = usize::from(text.first().is_some_and(|b| matches!(b, b'+' | b'-')));
    let mut position = sign;
    let tail = &text[position..];
    if tail
        .get(..3)
        .is_some_and(|s| s.eq_ignore_ascii_case(b"inf"))
    {
        return Some(
            position
                + if tail
                    .get(..8)
                    .is_some_and(|s| s.eq_ignore_ascii_case(b"infinity"))
                {
                    8
                } else {
                    3
                },
        );
    }
    if tail
        .get(..3)
        .is_some_and(|s| s.eq_ignore_ascii_case(b"nan"))
    {
        position += 3;
        if text.get(position) == Some(&b'(') {
            let mut end = position + 1;
            while text
                .get(end)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
            {
                end += 1;
            }
            if text.get(end) == Some(&b')') {
                position = end + 1;
            }
        }
        return Some(position);
    }
    let hex = tail.get(..2).is_some_and(|s| s.eq_ignore_ascii_case(b"0x"));
    if hex {
        position += 2;
    }
    let digit = |byte: &u8| {
        if hex {
            byte.is_ascii_hexdigit()
        } else {
            byte.is_ascii_digit()
        }
    };
    let start = position;
    while text.get(position).is_some_and(digit) {
        position += 1;
    }
    let mut count = position - start;
    if text.get(position) == Some(&b'.') {
        position += 1;
        let start = position;
        while text.get(position).is_some_and(digit) {
            position += 1;
        }
        count += position - start;
    }
    if count == 0 {
        return hex.then_some(sign + 1);
    }
    let exponent = if hex { b'p' } else { b'e' };
    if text
        .get(position)
        .is_some_and(|b| b.to_ascii_lowercase() == exponent)
    {
        let old = position;
        position += 1;
        if text.get(position).is_some_and(|b| matches!(b, b'+' | b'-')) {
            position += 1;
        }
        let start = position;
        while text.get(position).is_some_and(u8::is_ascii_digit) {
            position += 1;
        }
        if position == start {
            position = old;
        }
    }
    Some(position)
}

pub(crate) fn parse_complete(text: &str) -> Option<f64> {
    let text = text.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']);
    if text.is_empty() || float_prefix(text.as_bytes())? != text.len() {
        return None;
    }
    let sign = u64::from(text.starts_with('-')) << 63;
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    let lower = unsigned.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity") {
        return Some(f64::from_bits(sign | 0x7ff0000000000000));
    }
    if lower.starts_with("nan") {
        let payload = lower
            .strip_prefix("nan(")
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or("");
        let bits = if let Some(digits) = payload.strip_prefix("0x") {
            u64::from_str_radix(digits, 16).ok()
        } else if payload.starts_with('0') {
            u64::from_str_radix(payload, 8).ok()
        } else {
            payload.parse::<u64>().ok()
        }
        .unwrap_or(0);
        return Some(f64::from_bits(
            sign | 0x7ff8000000000000 | (bits & 0x000fffffffffffff),
        ));
    }
    if let Some(hex) = lower.strip_prefix("0x") {
        return Some(hex_float(hex, sign));
    }
    text.parse().ok()
}

fn hex_float(text: &str, sign: u64) -> f64 {
    let (mantissa, exponent) = text.split_once('p').unwrap_or((text, "0"));
    let negative_exponent = exponent.starts_with('-');
    let mut power = 0_i64;
    for digit in exponent
        .strip_prefix(['+', '-'])
        .unwrap_or(exponent)
        .bytes()
    {
        power = power
            .saturating_mul(10)
            .saturating_add(i64::from(digit - b'0'));
    }
    if negative_exponent {
        power = power.saturating_neg();
    }
    let before = mantissa.split('.').next().unwrap().len();
    let binary: Vec<_> = mantissa
        .chars()
        .filter(|&c| c != '.')
        .flat_map(|digit| {
            let value = digit.to_digit(16).unwrap();
            (0..4).rev().map(move |bit| (value >> bit) & 1)
        })
        .collect();
    let Some(first) = binary.iter().position(|&bit| bit == 1) else {
        return f64::from_bits(sign);
    };
    let bits = &binary[first..];
    let mut exponent = power
        .saturating_add((4 * before) as i64)
        .saturating_sub(first as i64 + 1);
    if exponent > 1023 {
        return f64::from_bits(sign | 0x7ff0000000000000);
    }
    if exponent < -1075 {
        return f64::from_bits(sign);
    }
    let normal = exponent >= -1022;
    let keep = if normal {
        53
    } else {
        (exponent + 1075) as usize
    };
    let mut significand = 0_u64;
    for position in 0..keep {
        significand = (significand << 1) | u64::from(bits.get(position).copied().unwrap_or(0));
    }
    let round = bits.get(keep) == Some(&1);
    let sticky = bits.get(keep + 1..).is_some_and(|tail| tail.contains(&1));
    if round && (sticky || significand & 1 != 0) {
        significand += 1;
    }
    if !normal {
        return f64::from_bits(sign | significand);
    }
    if significand == 1 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 1023 {
        return f64::from_bits(sign | 0x7ff0000000000000);
    }
    f64::from_bits(sign | ((exponent + 1023) as u64) << 52 | (significand & 0x000fffffffffffff))
}
