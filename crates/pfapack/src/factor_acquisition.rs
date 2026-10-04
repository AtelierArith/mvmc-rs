//! Private test-only acquisition. No C/Julia runtime or public observer API.
#[path = "acquisition_fd.rs"]
mod acquisition_fd;
#[path = "factor_acquisition_observer.rs"]
pub(super) mod observer;
#[path = "factor_acquisition_writer.rs"]
mod writer;

use std::cell::Cell;
use std::rc::Rc;

use crate::{dsktf2, utu2inv_real, utu2pfa_real, PivotIndex1Based, SqMat};

fn input(n: usize, text: &str) -> Vec<f64> {
    assert!(text.ends_with('\n'), "input word final newline");
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(format!("# C59 independent input words; column-major; n={n}").as_str())
    );
    let data = lines
        .map(|s| {
            assert!(
                s.len() == 16
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            );
            let x = f64::from_bits(u64::from_str_radix(s, 16).expect("input word syntax"));
            assert!(x.is_finite(), "input word nonfinite");
            x
        })
        .collect::<Vec<_>>();
    assert_eq!(data.len(), n * n, "input word cardinality");
    for j in 0..n {
        assert_eq!(data[j + j * n], 0.0, "input skew diagonal");
        for i in 0..j {
            assert_eq!(data[i + j * n], -data[j + i * n], "input skew sign");
        }
    }
    data
}

#[test]
#[ignore = "CI diagnostic acquisition only: requires independently pinned Node verifier and FD3 pipe"]
fn independent_c59_four_factor_inverse_stage_stream() {
    if cfg!(any(
        not(target_os = "linux"),
        feature = "blas-backend",
        feature = "simd-backend"
    )) {
        panic!("Linux scalar default acquisition only");
    }
    assert_eq!(
        std::env::var("MVMC_FACTOR_ADMISSION_PIPE").as_deref(),
        Ok("1")
    );
    let fixtures = [
        (
            32,
            include_str!("../../../tests/fixtures/pfapack/c59_factor_admission/input_n32.words"),
        ),
        (
            64,
            include_str!("../../../tests/fixtures/pfapack/c59_factor_admission/input_n64.words"),
        ),
        (
            128,
            include_str!("../../../tests/fixtures/pfapack/c59_factor_admission/input_n128.words"),
        ),
        (
            256,
            include_str!("../../../tests/fixtures/pfapack/c59_factor_admission/input_n256.words"),
        ),
    ];
    let total = Rc::new(Cell::new(0));
    for (n, text) in fixtures {
        let original = input(n, text);
        let mut a = original.clone();
        let mut pivots = vec![PivotIndex1Based(0); n];
        // FD3 is the actual anonymous pipe installed by the reviewed CI harness.
        // It is neither a guessed result path nor an overwriteable output file.
        let pipe =
            acquisition_fd::inherited_fd3().expect("duplicate inherited FD3 acquisition pipe");
        let result = writer::capture(n, &original, pipe, Rc::clone(&total), || {
            let info = dsktf2(&mut SqMat::new(&mut a, n), &mut pivots).err();
            let factor = a.clone();
            let pf = utu2pfa_real(&SqMat::new(&mut a, n), &pivots);
            let original_pivots = pivots.clone();
            let mut vt = vec![0.; n - 1];
            let mut workspace = vec![0.; n * n];
            utu2inv_real(
                &mut SqMat::new(&mut a, n),
                &pivots,
                &mut vt,
                &mut SqMat::new(&mut workspace, n),
            );
            assert_eq!(
                pivots, original_pivots,
                "actual inverse pivot metadata unchanged"
            );
            writer::Outcome {
                info,
                pf,
                pivots: pivots.clone(),
                factor,
                inverse: a.clone(),
            }
        });
        assert_eq!(result.info, None, "actual C59 original-input factor INFO");
    }
}

#[cfg(test)]
mod controls {
    use super::*;
    use std::cell::RefCell;
    use std::io::{self, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn live_events() -> Rc<Cell<usize>> {
        Rc::new(Cell::new(0))
    }
    fn invoke() {
        let mut a = [0., -2., 2., 0.];
        let mut pivots = [PivotIndex1Based(0); 2];
        dsktf2(&mut SqMat::new(&mut a, 2), &mut pivots).unwrap();
    }
    #[test]
    fn disabled_has_no_output_or_callback() {
        invoke();
    }
    #[test]
    fn nested_scope_rejects_without_replacing_owner() {
        let count = live_events();
        let c = Rc::clone(&count);
        let guard = observer::install(2, move |_, _, _| c.set(c.get() + 1));
        assert!(catch_unwind(|| observer::install(2, |_, _, _| {})).is_err());
        invoke();
        assert_eq!(count.get(), 3);
        drop(guard);
    }
    #[test]
    fn unwind_clears_thread_local_scope() {
        assert!(catch_unwind(|| {
            let _guard = observer::install(2, |_, _, _| panic!("literal callback failure"));
            invoke();
        })
        .is_err());
        let guard = observer::install(2, |_, _, _| {});
        invoke();
        drop(guard);
    }
    #[test]
    fn callback_is_thread_local() {
        let count = live_events();
        let c = Rc::clone(&count);
        let guard = observer::install(2, move |_, _, _| c.set(c.get() + 1));
        std::thread::spawn(invoke).join().unwrap();
        assert_eq!(count.get(), 0);
        invoke();
        assert_eq!(count.get(), 3);
        drop(guard);
    }
    struct Fails {
        remaining: usize,
    }
    impl Write for Fails {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("literal write failure"));
            }
            self.remaining -= 1;
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn failed_writer(remaining: usize) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            writer::capture(
                2,
                &[0., -2., 2., 0.],
                Fails { remaining },
                Rc::new(Cell::new(0)),
                || {
                    invoke();
                    panic!("no returned outcome")
                },
            )
        }));
        assert!(result.is_err());
        let guard = observer::install(2, |_, _, _| {});
        invoke();
        drop(guard);
    }
    #[test]
    fn start_write_failure_clears_scope() {
        failed_writer(0);
    }
    #[test]
    fn runtime_write_failure_clears_scope() {
        failed_writer(1);
    }
    #[test]
    fn observer_off_on_preserves_same_call_results() {
        fn factor() -> (Vec<u64>, Vec<PivotIndex1Based>) {
            let mut a = vec![
                0., -1., -2., -4., 1., 0., -3., -1., 2., 3., 0., -2., 4., 1., 2., 0.,
            ];
            let mut p = vec![PivotIndex1Based(0); 4];
            dsktf2(&mut SqMat::new(&mut a, 4), &mut p).unwrap();
            (a.iter().map(|v| v.to_bits()).collect(), p)
        }
        let off = factor();
        let records = Rc::new(RefCell::new(Vec::new()));
        let c = Rc::clone(&records);
        let guard = observer::install(4, move |event, _, _| c.borrow_mut().push(event));
        let on = factor();
        drop(guard);
        // Same-implementation observer passivity only, not computed-FP oracle acceptance.
        assert_eq!(off, on);
        assert_eq!(records.borrow().len(), 9);
    }
}
