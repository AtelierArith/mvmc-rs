//! Private cfg(test)-only borrowed full-N real factor stage capture; no numerical operations.
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

/// Location of the borrowed original factor buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Before the original first-max pivot search.
    BeforePivot,
    /// After original swaps/signs, before rank2/scaling.
    AfterSwap,
    /// After original rank2/scaling, or unchanged zero-column skip.
    AfterUpdate,
}
/// Metadata from the actual factor invocation; all indices are zero-based.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    /// Borrow boundary.
    pub kind: Kind,
    /// Matrix side.
    pub n: usize,
    /// Active final column.
    pub k: usize,
    /// Actual chosen pivot, absent before search.
    pub kp: Option<usize>,
    /// Actual first zero-column INFO; does not imply singularity.
    pub info: Option<usize>,
}
type Callback = Box<dyn FnMut(Event, &[f64], usize)>;
struct State {
    n: usize,
    events: usize,
    callback: Callback,
}
thread_local! { static ACTIVE: RefCell<Option<State>> = const { RefCell::new(None) }; }
/// Thread-bound diagnostic scope. Drop clears capture during normal exit or unwind.
pub(super) struct Guard {
    _thread_bound: PhantomData<Rc<()>>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
/// Install one capture callback. Nested installation rejects without replacing the owner.
pub(super) fn install(n: usize, callback: impl FnMut(Event, &[f64], usize) + 'static) -> Guard {
    assert!(
        (2..=256).contains(&n) && n.is_multiple_of(2),
        "factor capture matrix domain"
    );
    ACTIVE.with(|slot| {
        let mut state = slot.borrow_mut();
        assert!(state.is_none(), "nested factor capture");
        *state = Some(State {
            n,
            events: 0,
            callback: Box::new(callback),
        });
    });
    Guard {
        _thread_bound: PhantomData,
    }
}
pub(crate) fn borrow(event: Event, data: &[f64], lda: usize) {
    ACTIVE.with(|slot| {
        let mut state = slot.borrow_mut();
        let Some(s) = state.as_mut() else {
            return;
        };
        assert_eq!(s.n, event.n, "factor capture owner dimension");
        assert!(event.k >= 1 && event.k < event.n && lda >= event.n);
        let required = (event.n - 1)
            .checked_mul(lda)
            .and_then(|x| x.checked_add(event.n))
            .expect("capture buffer size overflow");
        assert!(data.len() >= required, "capture buffer too short");
        assert!(s.events < 3 * (s.n - 1), "factor capture event cap");
        // Validate ONLY the authoritative borrowed upper triangle; lower padding ignored.
        for j in 1..event.n {
            for i in 0..j {
                assert!(data[i + j * lda].is_finite(), "capture nonfinite upper");
            }
        }
        s.events += 1;
        (s.callback)(event, data, lda);
    });
}
