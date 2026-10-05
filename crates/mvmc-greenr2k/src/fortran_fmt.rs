//! Fortran (gfortran) output formatting used by `tool/greenr2k.F90`.
//!
//! Only the edit descriptors and list-directed items that `greenr2k.F90` uses
//! are implemented: `Iw`, `Iw.m`, `I0`, `Fw.d`, `Ew.d` and list-directed
//! integer / REAL(4) / REAL(8) / character items. The behaviour is verified
//! against gfortran output in `tests/fortran_format.rs`.

/// `Iw`, or `I0` when `w == 0`. Overflow gives `w` asterisks.
pub fn fmt_i(w: usize, value: i64) -> String {
    fmt_i_min(w, 1, value)
}

/// `Iw.m` (at least `m` digits); `w == 0` is `I0` / `I0.m`.
pub fn fmt_i_min(w: usize, m: usize, value: i64) -> String {
    let digits = value.unsigned_abs().to_string();
    let mut body = if digits.len() < m {
        format!("{}{}", "0".repeat(m - digits.len()), digits)
    } else {
        digits
    };
    if value < 0 {
        body.insert(0, '-');
    }
    pad_left(w, body)
}

fn pad_left(w: usize, body: String) -> String {
    if w == 0 {
        body
    } else if body.len() > w {
        "*".repeat(w)
    } else {
        format!("{body:>w$}")
    }
}

fn non_finite(w: usize, x: f64, short_inf_ok: bool) -> Option<String> {
    let text = if x.is_nan() {
        "NaN".to_string()
    } else if x.is_infinite() {
        let neg = x < 0.0;
        let long = if neg { "-Infinity" } else { "Infinity" };
        let short = if neg { "-Inf" } else { "Inf" };
        if short_inf_ok && w >= long.len() {
            long.to_string()
        } else if w >= short.len() {
            short.to_string()
        } else {
            "*".repeat(w)
        }
    } else {
        return None;
    };
    Some(if text.starts_with('*') {
        text
    } else {
        format!("{text:>w$}")
    })
}

/// `Fw.d`.
pub fn fmt_f(w: usize, d: usize, x: f64) -> String {
    if let Some(s) = non_finite(w, x, true) {
        return s;
    }
    let negative = x.is_sign_negative();
    let body = format!("{:.*}", d, x.abs());
    let signed = if negative {
        format!("-{body}")
    } else {
        body.clone()
    };
    if signed.len() <= w {
        return format!("{signed:>w$}");
    }
    // Drop the optional leading zero of values below one when the field is tight.
    if let Some(stripped) = body.strip_prefix("0.") {
        let alt = if negative {
            format!("-.{stripped}")
        } else {
            format!(".{stripped}")
        };
        if alt.len() <= w {
            return format!("{alt:>w$}");
        }
    }
    "*".repeat(w)
}

/// Decimal digits and exponent of `|x|` rounded to `digits` significant digits:
/// `|x| ~= 0.DIGITS * 10^exp`. Zero gives `(000..., 0)`.
fn scientific(x: f64, digits: usize) -> (String, i32) {
    if x == 0.0 {
        return ("0".repeat(digits), 0);
    }
    let s = format!("{:.*e}", digits - 1, x.abs());
    let (mantissa, exp) = s.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    (mantissa.replace('.', ""), exp + 1)
}

/// `Ew.d`.
pub fn fmt_e(w: usize, d: usize, x: f64) -> String {
    if let Some(s) = non_finite(w, x, true) {
        return s;
    }
    let (digits, exp) = scientific(x, d);
    let exp_text = if exp.abs() <= 99 {
        format!("E{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else if exp.abs() <= 999 {
        format!("{}{:03}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        return "*".repeat(w);
    };
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let text = format!("{sign}0.{digits}{exp_text}");
    if text.len() <= w {
        format!("{text:>w$}")
    } else {
        // Optional leading zero is dropped when the field is exactly one short.
        let alt = format!("{sign}.{digits}{exp_text}");
        if alt.len() <= w {
            format!("{alt:>w$}")
        } else {
            "*".repeat(w)
        }
    }
}

/// Kind of a list-directed real item.
#[derive(Clone, Copy)]
pub enum RealKind {
    /// REAL(4): 9 significant digits.
    Single,
    /// REAL(8): 17 significant digits.
    Double,
}

/// One list-directed real item *without* its leading separator blank, exactly as
/// gfortran writes it (including the trailing blanks that reserve the exponent
/// field of the `G` edit descriptor).
pub fn list_real(kind: RealKind, x: f64) -> String {
    let (d, exp_width) = match kind {
        RealKind::Single => (9usize, 2usize),
        RealKind::Double => (17usize, 3usize),
    };
    let x = match kind {
        RealKind::Single => (x as f32) as f64,
        RealKind::Double => x,
    };
    let f_width = d + 3;
    let e_width = d + 1 + (exp_width + 2) + 1 + 1;
    if x.is_nan() || x.is_infinite() {
        return non_finite(e_width, x, true).expect("non-finite");
    }
    let negative = x.is_sign_negative();
    let sign = if negative { "-" } else { "" };
    let ax = x.abs();
    let (digits, exp, f_style) = if ax == 0.0 {
        ("0".repeat(d), 1, true)
    } else {
        // Significant digits of the value rounded to `d` digits.
        let (digits, exp) = match kind {
            RealKind::Single => {
                let s = format!("{:.*e}", d - 1, ax as f32);
                let (m, e) = s.split_once('e').expect("exponent");
                (
                    m.replace('.', ""),
                    e.parse::<i32>().expect("integer exponent") + 1,
                )
            }
            RealKind::Double => scientific(ax, d),
        };
        (digits, exp, exp >= 0 && exp <= d as i32)
    };
    if f_style {
        // 0.1 <= |x| < 10^d: F format with d - exp decimals (exp = 0 never occurs for
        // values >= 0.1 except through the zero special case).
        let decimals = d as i32 - exp;
        let body = if exp <= 0 {
            // 0.1 <= |x| < 1 has exp == 0 here.
            format!("0.{digits}")
        } else {
            let (int_part, frac) = digits.split_at(exp as usize);
            if decimals > 0 {
                format!("{int_part}.{frac}")
            } else {
                format!("{int_part}.")
            }
        };
        let field = format!("{sign}{body}");
        format!("{field:>f_width$}{}", " ".repeat(exp_width + 2))
    } else {
        let e = exp - 1;
        let (lead, rest) = digits.split_at(1);
        let exp_text = if e.abs() <= 99 && exp_width == 2 {
            format!("E{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
        } else {
            format!(
                "E{}{:0w$}",
                if e < 0 { '-' } else { '+' },
                e.abs(),
                w = exp_width
            )
        };
        let field = format!("{sign}{lead}.{rest}{exp_text}");
        format!("{field:>e_width$}")
    }
}

/// Builder for one list-directed record (`WRITE(unit,*) item, item, ...`).
#[derive(Default)]
pub struct ListRecord {
    text: String,
    items: usize,
    last_was_char: bool,
}

impl ListRecord {
    pub fn new() -> Self {
        Self::default()
    }

    fn separator(&mut self) {
        // gfortran: a blank precedes the first item of the record and every
        // non-character item; consecutive character items are not separated.
        self.text.push(' ');
    }

    pub fn str(&mut self, s: &str) -> &mut Self {
        if self.items == 0 {
            self.separator();
        }
        self.text.push_str(s);
        self.items += 1;
        self.last_was_char = true;
        self
    }

    pub fn int(&mut self, v: i64) -> &mut Self {
        self.separator();
        self.text.push_str(&fmt_i(11, v));
        self.items += 1;
        self.last_was_char = false;
        self
    }

    pub fn real(&mut self, kind: RealKind, v: f64) -> &mut Self {
        self.separator();
        self.text.push_str(&list_real(kind, v));
        self.items += 1;
        self.last_was_char = false;
        self
    }

    /// The record text without the terminating newline.
    pub fn finish(self) -> String {
        if self.items == 0 {
            String::new()
        } else {
            self.text
        }
    }
}
