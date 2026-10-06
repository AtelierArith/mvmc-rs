//! Scalar math functions as the C reference calls them (issue #457).
//!
//! C mVMC calls the platform libm for `exp`, `log`, `sin`, `cos`, `hypot`, `atan2`, ... Rust's
//! `f64` methods of the same names are the platform libm too (glibc on Linux, the system libm
//! on macOS; LLVM may constant-fold or lower some to intrinsics but never changes the rounding
//! of the library call). The default path therefore uses them. Julia's own software
//! implementations (`julia_exp`, `julia_log`, `julia_trig`, `julia_hypot`) differ from glibc in
//! the last bit for some arguments; they are kept only as an explicit, process-wide opt-in
//! ([`use_julia_libm`]) for tests of archived Julia 1.11/1.13 fixtures.
//!
//! Platform caveat: glibc and the macOS libm are different implementations; a result that
//! depends on the last bit of a transcendental function can differ between Linux and macOS.
//! Linux (glibc, the platform of the C reference outputs) is the numerical reference.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::{julia_exp, julia_hypot, julia_log, julia_trig};

static JULIA_LIBM: AtomicBool = AtomicBool::new(false);
/// Serializes users of the process-wide switch (a second `use_julia_libm` waits for the first).
static JULIA_LIBM_LOCK: Mutex<()> = Mutex::new(());

/// Guard of [`use_julia_libm`]; restores the previous setting on drop.
#[derive(Debug)]
pub struct JuliaLibmGuard(bool, #[allow(dead_code)] MutexGuard<'static, ()>);

impl Drop for JuliaLibmGuard {
    fn drop(&mut self) {
        JULIA_LIBM.store(self.0, Ordering::SeqCst);
    }
}

/// Historical opt-in, **process-wide** (all threads, including worker pools): evaluate the
/// scalar math functions of this module with Julia's implementations. Only tests of archived
/// Julia fixtures use it. Holders are serialized by an internal lock (do not nest on one
/// thread); other tests running concurrently in the same process would see the switch too, so
/// run them with `cargo nextest` (one process per test), as the repository does.
pub fn use_julia_libm() -> JuliaLibmGuard {
    let lock = JULIA_LIBM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    JuliaLibmGuard(JULIA_LIBM.swap(true, Ordering::SeqCst), lock)
}

#[inline]
fn julia() -> bool {
    JULIA_LIBM.load(Ordering::Relaxed)
}

/// C `exp`.
#[inline]
pub fn exp(x: f64) -> f64 {
    if julia() {
        julia_exp::exp(x)
    } else {
        x.exp()
    }
}

/// C `log`.
#[inline]
pub fn log(x: f64) -> f64 {
    if julia() {
        julia_log::log(x)
    } else {
        x.ln()
    }
}

/// C `log1p`.
#[inline]
pub fn log1p(x: f64) -> f64 {
    if julia() {
        julia_log::log1p(x)
    } else {
        x.ln_1p()
    }
}

/// C `sin`.
#[inline]
pub fn sin(x: f64) -> f64 {
    if julia() {
        julia_trig::sin(x)
    } else {
        x.sin()
    }
}

/// C `cos`.
#[inline]
pub fn cos(x: f64) -> f64 {
    if julia() {
        julia_trig::cos(x)
    } else {
        x.cos()
    }
}

/// C `tan`.
#[inline]
pub fn tan(x: f64) -> f64 {
    if julia() {
        julia_trig::tan(x)
    } else {
        x.tan()
    }
}

/// C `atan2`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    if julia() {
        julia_trig::atan2(y, x)
    } else {
        y.atan2(x)
    }
}

/// C `sinh`.
#[inline]
pub fn sinh(x: f64) -> f64 {
    if julia() {
        julia_trig::sinh(x)
    } else {
        x.sinh()
    }
}

/// C `hypot`.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    if julia() {
        julia_hypot::hypot(x, y)
    } else {
        x.hypot(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARGS: [f64; 9] = [-3.7, -0.5, 0.0, 1e-9, 0.31, 1.0, 2.5, 17.25, 650.0];

    #[test]
    fn default_is_the_platform_libm_bit_for_bit() {
        // Take the lock so that no Julia-mode test of this process is active meanwhile.
        let _lock = JULIA_LIBM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!julia());
        for x in ARGS {
            let c = x.min(700.0);
            assert_eq!(exp(c).to_bits(), c.exp().to_bits());
            assert_eq!(sin(x).to_bits(), x.sin().to_bits());
            assert_eq!(cos(x).to_bits(), x.cos().to_bits());
            assert_eq!(tan(x).to_bits(), x.tan().to_bits());
            assert_eq!(sinh(c).to_bits(), c.sinh().to_bits());
            assert_eq!(hypot(x, 1.5).to_bits(), x.hypot(1.5).to_bits());
            assert_eq!(atan2(x, 0.7).to_bits(), x.atan2(0.7).to_bits());
            if x > 0.0 {
                assert_eq!(log(x).to_bits(), x.ln().to_bits());
            }
            if x > -1.0 {
                assert_eq!(log1p(x).to_bits(), x.ln_1p().to_bits());
            }
        }
    }

    #[test]
    fn julia_opt_in_selects_the_julia_implementations_and_restores() {
        {
            let _guard = use_julia_libm();
            assert!(julia());
            for x in ARGS {
                let c = x.min(700.0);
                assert_eq!(exp(c).to_bits(), julia_exp::exp(c).to_bits());
                assert_eq!(sin(x).to_bits(), julia_trig::sin(x).to_bits());
                assert_eq!(cos(x).to_bits(), julia_trig::cos(x).to_bits());
                assert_eq!(
                    hypot(x, 1.5).to_bits(),
                    julia_hypot::hypot(x, 1.5).to_bits()
                );
                if x > 0.0 {
                    assert_eq!(log(x).to_bits(), julia_log::log(x).to_bits());
                }
            }
        }
        let _lock = JULIA_LIBM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!julia());
    }

    /// Both implementations agree to one ulp: the switch changes at most the last bit of an
    /// ordinary argument.
    #[test]
    fn julia_and_platform_libm_agree_to_one_ulp() {
        let _guard = use_julia_libm();
        for x in ARGS {
            let c = x.min(700.0);
            for (a, b) in [
                (julia_exp::exp(c), c.exp()),
                (julia_trig::sin(x), x.sin()),
                (julia_trig::cos(x), x.cos()),
            ] {
                let ulps = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                assert!(ulps <= 1, "x = {x}: {a} vs {b} ({ulps} ulp)");
            }
        }
    }
}
