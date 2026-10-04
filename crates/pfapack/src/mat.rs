//! Square column-major matrix view used by the LTL / Pfaffian routines.
//!
//! We store matrices column-major to match the upstream Julia / Fortran
//! convention (`A[i, j]` is column-contiguous). This lets the Phase-2
//! golden tests interop directly with `fimpl_dsktf2!` / `fimpl_zsktf2!`
//! (which pass `Ptr{Float64}` / `Ptr{ComplexF64}` to Fortran without a
//! layout flip).
//!
//! Indexing is **0-based on the Rust side**. The Julia source comments
//! reproduced in `ltl.rs` / `pfaffian.rs` / `utu2.rs` use 1-based indices
//! to stay readable next to the upstream code; the `(i, j)` arguments to
//! [`SqMat::get`] / [`SqMat::set`] / [`SqMat::swap_entries`] use 0-based.

use core::ops::{Index, IndexMut};

/// Mutable view over a square matrix of side `n`.
///
/// The default constructor wraps a contiguous column-major `n * n`
/// buffer. Internally the view keeps element strides so kernels can
/// eventually operate on submatrices without materializing temporary
/// contiguous buffers.
pub struct SqMat<'a, T> {
    pub(crate) data: &'a mut [T],
    pub(crate) n: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

impl<'a, T: Copy> SqMat<'a, T> {
    /// Wrap an `n*n` slice. Panics if the length mismatches.
    pub fn new(data: &'a mut [T], n: usize) -> Self {
        assert_eq!(data.len(), n * n, "SqMat::new expects n*n entries");
        Self {
            data,
            n,
            row_stride: 1,
            col_stride: n,
        }
    }

    /// Wrap a column-major matrix with an explicit leading dimension.
    ///
    /// This is mainly used by internal kernels that want a view over a
    /// submatrix or padded column-major storage. The row stride remains
    /// one element, while the column stride is `lda`.
    #[allow(dead_code)]
    pub(crate) fn from_column_major_slice_with_lda(
        data: &'a mut [T],
        n: usize,
        lda: usize,
    ) -> Self {
        assert!(lda >= n, "SqMat lda must be at least n");
        let required = n.checked_sub(1).map_or(0, |last| last * lda + n);
        assert!(
            data.len() >= required,
            "SqMat::from_column_major_slice_with_lda expects enough backing storage"
        );
        Self {
            data,
            n,
            row_stride: 1,
            col_stride: lda,
        }
    }

    /// Side length.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Linear index into the column-major buffer (0-based `i, j`).
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.row_stride + j * self.col_stride
    }

    /// Read `A[i, j]` (0-based).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[self.idx(i, j)]
    }

    /// Write `A[i, j] = v` (0-based).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        let k = self.idx(i, j);
        self.data[k] = v;
    }

    /// `(A[i1, j1], A[i2, j2]) = (A[i2, j2], A[i1, j1])` (0-based).
    #[inline]
    pub fn swap_entries(&mut self, (i1, j1): (usize, usize), (i2, j2): (usize, usize)) {
        let a = self.idx(i1, j1);
        let b = self.idx(i2, j2);
        self.data.swap(a, b);
    }

    /// Read-only iterator over `(i, j, value)` of the strict upper
    /// triangle (`i < j`). Used by debug / Pfaffian-reconstruction
    /// helpers in the test suite.
    pub fn strict_upper(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        (0..self.n).flat_map(move |j| (0..j).map(move |i| (i, j, self.get(i, j))))
    }

    /// Borrow the entire backing slice. Used by the BLAS backend to
    /// hand BLAS / LAPACK a raw pointer + leading dimension.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Mutably borrow the entire backing slice. Used by the BLAS
    /// backend.
    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Leading dimension for column-major views.
    #[inline]
    pub(crate) fn lda(&self) -> usize {
        debug_assert_eq!(self.row_stride, 1);
        self.col_stride
    }
}

// Convenience indexing in `(i, j)` form for in-module readability.
impl<T: Copy> Index<(usize, usize)> for SqMat<'_, T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        let k = self.idx(i, j);
        &self.data[k]
    }
}

impl<T: Copy> IndexMut<(usize, usize)> for SqMat<'_, T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        let k = self.idx(i, j);
        &mut self.data[k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_2x2() {
        let mut buf = vec![0.0_f64; 4];
        {
            let mut m = SqMat::new(&mut buf, 2);
            // Column-major: buf = [A[0,0], A[1,0], A[0,1], A[1,1]]
            m.set(0, 0, 11.0);
            m.set(1, 0, 21.0);
            m.set(0, 1, 12.0);
            m.set(1, 1, 22.0);
            assert_eq!(m.get(0, 0), 11.0);
            assert_eq!(m.get(1, 0), 21.0);
            assert_eq!(m.get(0, 1), 12.0);
            assert_eq!(m.get(1, 1), 22.0);
        }
        assert_eq!(buf, vec![11.0, 21.0, 12.0, 22.0]);
    }
}
