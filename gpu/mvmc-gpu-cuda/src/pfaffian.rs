//! Launch geometry of the batched Pfaffian kernels of `pfaffian_batched.cu` (issue #423).
//!
//! The kernels are compiled into the resident interval module (see [`crate::interval`]); this
//! module only exposes the matrix-size limit and the per-plane block geometry shared by
//! [`crate::interval`] and [`crate::opt`].

/// Largest supported matrix side (dynamic shared memory holds `O(n)` per block).
pub const MAX_N: usize = 1024;

/// Dynamic shared bytes and threads per block of one plane for element type
/// `elem_bytes`/`lanes` (2 for `Complex64`, 1 for the real kernels).
///
/// `pfinv_f64` (real double, even `n >= 64`) uses the specialized fast path of
/// `pfaffian_batched.cu`: `xs, ys, t1s, t2s` (4n doubles) plus a 32-entry reduction scratch, a
/// 32-entry value scratch, `piv` and `idx` (n ints each) = `40 n + 384` bytes, and it is
/// measured fastest with 256 threads for `n >= 128`. All other kernels keep the original
/// layout (`n * elem * lanes + nt * elem + nt * 4 + n * 8` bytes) and thread rule. Both are
/// returned for the f64 case and the larger is used, so the launch stays valid for either
/// kernel.
pub fn launch_geometry(n: usize, elem_bytes: usize, lanes: usize) -> (u32, u32) {
    let fast = elem_bytes == 8 && lanes == 1 && n >= 64 && n.is_multiple_of(2);
    let nt = if fast && n >= 128 {
        256
    } else {
        n.next_power_of_two().clamp(32, 256)
    };
    let generic = (n * elem_bytes * lanes + nt * elem_bytes + nt * 4 + n * 8) as u32;
    let fast_shared = if fast { (40 * n + 384) as u32 } else { 0 };
    (nt as u32, generic.max(fast_shared))
}
