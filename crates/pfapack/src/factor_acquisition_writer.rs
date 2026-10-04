//! Diagnostic streaming writer: raw borrowed upper-word patches, no factor arithmetic.
use super::observer::{self, Kind};
use crate::PivotIndex1Based;
use std::cell::{Cell, RefCell};
use std::io::Write;
use std::rc::Rc;

/// Actual public pipeline results, not C-derived metadata.
pub struct Outcome {
    pub info: Option<usize>,
    pub pf: f64,
    pub pivots: Vec<PivotIndex1Based>,
    pub factor: Vec<f64>,
    pub inverse: Vec<f64>,
}
struct Stream<W> {
    writer: W,
    previous: Vec<u64>,
    events: usize,
    bytes: Rc<Cell<usize>>,
}
impl<W: Write> Stream<W> {
    fn line(&mut self, line: String) {
        assert!(line.len() <= 2 * 1024 * 1024, "stage frame cap");
        let next = self
            .bytes
            .get()
            .checked_add(line.len())
            .expect("transport count overflow");
        assert!(next <= 128 * 1024 * 1024, "transport cap");
        self.bytes.set(next);
        self.writer
            .write_all(line.as_bytes())
            .expect("stage stream write failed");
        self.writer.flush().expect("stage stream flush failed");
    }
}
/// Capture exactly one real factor call; no success end record on panic/failure.
/// `original` must be the same independent fixture buffer passed to the body.
pub fn capture<W: Write + 'static>(
    n: usize,
    original: &[f64],
    writer: W,
    total_bytes: Rc<Cell<usize>>,
    body: impl FnOnce() -> Outcome,
) -> Outcome {
    assert_eq!(
        original.len(),
        n.checked_mul(n).expect("matrix size overflow")
    );
    let state = Rc::new(RefCell::new(Stream {
        writer,
        previous: original.iter().map(|x| x.to_bits()).collect(),
        events: 0,
        bytes: total_bytes,
    }));
    let callback_state = Rc::clone(&state);
    let guard = observer::install(n, move |event, data, lda| {
        let mut s = callback_state.borrow_mut();
        match event.kind {
            Kind::BeforePivot => {
                for j in 1..n {
                    for i in 0..j {
                        assert_eq!(
                            data[i + j * lda].to_bits(),
                            s.previous[i + j * n],
                            "actual upper chronology mismatch"
                        );
                    }
                }
                s.line(format!("{{\"kind\":\"before-pivot\",\"n\":{n},\"k\":{},\"borrowedUpperMatchesPrevious\":true}}\n", event.k));
            }
            Kind::AfterSwap | Kind::AfterUpdate => {
                let kind = if event.kind == Kind::AfterSwap {
                    "after-swap"
                } else {
                    "after-update"
                };
                let mut patches = String::new();
                for j in 1..n {
                    for i in 0..j {
                        let index = i + j * n;
                        let bits = data[i + j * lda].to_bits();
                        if bits != s.previous[index] {
                            if !patches.is_empty() {
                                patches.push(',');
                            }
                            patches.push_str(&format!("[{index},\"{bits:016x}\"]"));
                            s.previous[index] = bits;
                        }
                    }
                }
                let kp = event.kp.expect("actual chosen pivot missing");
                let info = event.info.unwrap_or(0);
                s.line(format!("{{\"kind\":\"{kind}\",\"n\":{n},\"k\":{},\"kp\":{kp},\"info\":{info},\"patches\":[{patches}]}}\n",event.k));
            }
        }
        s.events += 1;
    });
    state
        .borrow_mut()
        .line(format!("{{\"kind\":\"case-start\",\"n\":{n}}}\n"));
    let result = body();
    assert_eq!(
        state.borrow().events,
        3 * (n - 1),
        "incomplete factor stream"
    );
    for (name, plane) in [("factor", &result.factor), ("inverse", &result.inverse)] {
        assert_eq!(plane.len(), n * n, "actual result cardinality");
        assert!(
            plane.iter().all(|x| x.is_finite()),
            "actual result nonfinite"
        );
        let words = plane
            .iter()
            .map(|x| format!("\"{:016x}\"", x.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        state.borrow_mut().line(format!(
            "{{\"kind\":\"result-plane\",\"n\":{n},\"name\":\"{name}\",\"words\":[{words}]}}\n"
        ));
    }
    assert!(result.pf.is_finite(), "actual Pf nonfinite");
    let pivots = result
        .pivots
        .iter()
        .map(|x| x.0.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let info = result.info.unwrap_or(0);
    state.borrow_mut().line(format!("{{\"kind\":\"case-end\",\"n\":{n},\"info\":{info},\"pfWord\":\"{:016x}\",\"pivots1based\":[{pivots}]}}\n", result.pf.to_bits()));
    drop(guard);
    result
}
