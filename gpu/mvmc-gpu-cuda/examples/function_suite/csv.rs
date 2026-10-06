//! Unified CSV schema and timing helpers.
//!
//! Columns: family,function,variant,dtype,params,reps,median_s,min_s,max_s,dev_metric,
//! dev_value,dev_bound,dev_ratio,verdict,note

use std::io::Write;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Within the stated bound.
    Pass,
    /// Outside the bound or an independent invariant violated.
    Fail,
    /// An unexpected runtime error (counts as a failure).
    Error,
    /// This row is the oracle itself (no deviation to report).
    Oracle,
    /// Timing-only row with no numerical content.
    Info,
    /// The function or hook does not exist on this build; never filled with numbers.
    NotAvailable,
    /// Not run (memory limit or similar); the note says why.
    Skipped,
}

impl Verdict {
    fn name(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Error => "ERROR",
            Self::Oracle => "ORACLE",
            Self::Info => "INFO",
            Self::NotAvailable => "NotAvailable",
            Self::Skipped => "SKIPPED",
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Stat {
    pub reps: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Stat {
    pub fn single(s: f64) -> Self {
        Self {
            reps: 1,
            median: s,
            min: s,
            max: s,
        }
    }
    pub fn from_samples(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.total_cmp(b));
        Self {
            reps: v.len(),
            median: v[v.len() / 2],
            min: v[0],
            max: v[v.len() - 1],
        }
    }
}

/// Warm up `warmups` times, then repeat until `max_reps` or until `budget_s` is exhausted
/// (always at least one measured run). The closure returns the seconds to record.
pub fn measure<E>(
    warmups: usize,
    max_reps: usize,
    budget_s: f64,
    mut f: impl FnMut() -> Result<f64, E>,
) -> Result<Stat, E> {
    for _ in 0..warmups {
        f()?;
    }
    let start = Instant::now();
    let mut v = Vec::new();
    while v.len() < max_reps.max(1) && (v.is_empty() || start.elapsed().as_secs_f64() < budget_s) {
        v.push(f()?);
    }
    Ok(Stat::from_samples(v))
}

pub fn wall<E>(f: impl FnOnce() -> Result<(), E>) -> Result<f64, E> {
    let t = Instant::now();
    f()?;
    Ok(t.elapsed().as_secs_f64())
}

#[derive(Clone, Debug)]
pub struct Dev {
    pub metric: String,
    pub value: f64,
    pub bound: f64,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub family: String,
    pub function: String,
    pub variant: String,
    pub dtype: String,
    pub params: String,
    pub stat: Option<Stat>,
    pub dev: Option<Dev>,
    pub verdict: Verdict,
    pub note: String,
}

impl Row {
    pub fn new(family: &str, function: &str, variant: &str, dtype: &str, params: &str) -> Self {
        Self {
            family: family.into(),
            function: function.into(),
            variant: variant.into(),
            dtype: dtype.into(),
            params: params.into(),
            stat: None,
            dev: None,
            verdict: Verdict::Info,
            note: String::new(),
        }
    }
    pub fn stat(mut self, s: Stat) -> Self {
        self.stat = Some(s);
        self
    }
    /// Set the deviation and derive PASS/FAIL (`value <= bound`, finite).
    pub fn dev(mut self, metric: &str, value: f64, bound: f64) -> Self {
        self.dev = Some(Dev {
            metric: metric.into(),
            value,
            bound,
        });
        self.verdict = if value.is_finite() && value <= bound {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        self
    }
    pub fn verdict(mut self, v: Verdict, note: &str) -> Self {
        self.verdict = v;
        self.note = note.into();
        self
    }
    pub fn note(mut self, note: &str) -> Self {
        self.note = note.into();
        self
    }
    pub fn fail(self, why: &str) -> Self {
        self.verdict(Verdict::Fail, why)
    }
    pub fn error(self, why: &str) -> Self {
        self.verdict(Verdict::Error, why)
    }
}

pub struct Csv {
    file: std::fs::File,
    rows: usize,
    fails: usize,
}

const HEADER: &str = "family,function,variant,dtype,params,reps,median_s,min_s,max_s,dev_metric,dev_value,dev_bound,dev_ratio,verdict,note";

fn clean(s: &str) -> String {
    s.replace([',', '\n', '\r'], ";")
}

impl Csv {
    pub fn create(path: &str, family: &str) -> Self {
        let mut file = std::fs::File::create(path).expect("create csv");
        writeln!(file, "{HEADER}").unwrap();
        eprintln!("# csv {path} ({family})");
        Self {
            file,
            rows: 0,
            fails: 0,
        }
    }
    pub fn push(&mut self, r: Row) {
        let num = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.6e}"));
        let tm = |v: Option<f64>| num(v.filter(|x| x.is_finite()));
        let (reps, med, min, max) = r.stat.map_or((String::new(), None, None, None), |s| {
            (s.reps.to_string(), Some(s.median), Some(s.min), Some(s.max))
        });
        let (metric, value, bound, ratio) =
            r.dev
                .as_ref()
                .map_or((String::new(), None, None, None), |d| {
                    (
                        d.metric.clone(),
                        Some(d.value),
                        Some(d.bound),
                        Some(if d.bound > 0.0 {
                            d.value / d.bound
                        } else {
                            d.value
                        }),
                    )
                });
        let line = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            clean(&r.family),
            clean(&r.function),
            clean(&r.variant),
            clean(&r.dtype),
            clean(&r.params),
            reps,
            tm(med),
            tm(min),
            tm(max),
            clean(&metric),
            num(value),
            num(bound),
            num(ratio),
            r.verdict.name(),
            clean(&r.note)
        );
        println!("{line}");
        writeln!(self.file, "{line}").unwrap();
        self.file.flush().ok();
        self.rows += 1;
        if matches!(r.verdict, Verdict::Fail | Verdict::Error) {
            self.fails += 1;
        }
    }
    pub fn summary(&self) -> (usize, usize) {
        (self.fails, self.rows)
    }
}
