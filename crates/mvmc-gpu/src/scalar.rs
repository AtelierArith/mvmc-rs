//! Scalar abstraction over `f64` and `Complex64` for the batched backends.

use num_complex::Complex64;
use pfapack::SqMat;
use tenferro_tensor::{Tensor, TypedTensor};

use crate::cpu::PlaneWorkspace;
use crate::PlaneStatus;

pub(crate) mod private {
    use super::*;

    /// Sealed implementation surface of [`PfScalar`](super::PfScalar).
    pub trait Impl:
        Copy + Send + Sync + PartialEq + std::fmt::Debug + tenferro_tensor::TensorScalar + 'static
    {
        const ZERO: Self;
        const ONE: Self;
        const IS_COMPLEX: bool;
        /// Number of `f64` lanes per element (1 real, 2 complex).
        const LANES: usize;
        fn finite(self) -> bool;
        /// pfapack on one plane: `a` is overwritten by the inverse (or zeros on failure).
        fn pfapack_plane(
            a: &mut [Self],
            n: usize,
            ws: &mut PlaneWorkspace<Self>,
        ) -> (Self, PlaneStatus);
        fn make_tensor(shape: Vec<usize>, data: Vec<Self>) -> tenferro_tensor::Result<Tensor>;
        fn read_vec(read: &Tensor) -> tenferro_tensor::Result<Vec<Self>>;
        fn from_parts(re: f64, im: f64) -> Self;
        fn re(self) -> f64;
        fn im(self) -> f64;
        fn run_engine(
            engine: &dyn crate::BatchedEngine,
            planes: &[Self],
            n: usize,
            count: usize,
        ) -> crate::Result<crate::BatchOutput<Self>>;
    }
}

use private::Impl;

/// Element type of the batched API: `f64` or `Complex64`.
pub trait PfScalar: Impl {
    /// Reinterpret as `f64` lanes (`[re]` or `[re, im]`) for device transfers.
    fn as_f64_lanes(data: &[Self]) -> &[f64];
    /// Mutable lane view.
    fn as_f64_lanes_mut(data: &mut [Self]) -> &mut [f64];
    /// The additive identity.
    fn zero_value() -> Self;
    /// `f64` lanes per element: 1 for `f64`, 2 for `Complex64`.
    fn lanes() -> usize;
}

impl Impl for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const LANES: usize = 1;
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn pfapack_plane(a: &mut [f64], n: usize, ws: &mut PlaneWorkspace<f64>) -> (f64, PlaneStatus) {
        let info = {
            let mut sm = SqMat::new(&mut *a, n);
            pfapack::dsktf2(&mut sm, &mut ws.pivots)
        };
        if let Err(row) = info {
            a.fill(Self::ZERO);
            return (Self::ZERO, PlaneStatus::ZeroPivot { row });
        }
        let value = {
            let sm = SqMat::new(&mut *a, n);
            pfapack::utu2pfa_real(&sm, &ws.pivots)
        };
        if !value.finite() {
            a.fill(Self::ZERO);
            return (value, PlaneStatus::NonFinite);
        }
        let mut sm = SqMat::new(&mut *a, n);
        let mut m = SqMat::new(&mut ws.m, n);
        pfapack::utu2inv_real(&mut sm, &ws.pivots, &mut ws.vt, &mut m);
        (value, PlaneStatus::Ok)
    }
    fn make_tensor(shape: Vec<usize>, data: Vec<f64>) -> tenferro_tensor::Result<Tensor> {
        Ok(Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(
            shape, data,
        )?))
    }
    fn read_vec(read: &Tensor) -> tenferro_tensor::Result<Vec<f64>> {
        Ok(read.as_slice::<f64>()?.to_vec())
    }
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
    fn re(self) -> f64 {
        self
    }
    fn im(self) -> f64 {
        0.0
    }
    fn run_engine(
        engine: &dyn crate::BatchedEngine,
        planes: &[f64],
        n: usize,
        count: usize,
    ) -> crate::Result<crate::BatchOutput<f64>> {
        engine.run_f64(planes, n, count)
    }
}

impl PfScalar for f64 {
    fn zero_value() -> Self {
        0.0
    }
    fn lanes() -> usize {
        1
    }
    fn as_f64_lanes(data: &[f64]) -> &[f64] {
        data
    }
    fn as_f64_lanes_mut(data: &mut [f64]) -> &mut [f64] {
        data
    }
}

impl Impl for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const LANES: usize = 2;
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn pfapack_plane(
        a: &mut [Complex64],
        n: usize,
        ws: &mut PlaneWorkspace<Complex64>,
    ) -> (Complex64, PlaneStatus) {
        let info = {
            let mut sm = SqMat::new(&mut *a, n);
            pfapack::zsktf2(&mut sm, &mut ws.pivots)
        };
        if let Err(row) = info {
            a.fill(Self::ZERO);
            return (Self::ZERO, PlaneStatus::ZeroPivot { row });
        }
        let value = {
            let sm = SqMat::new(&mut *a, n);
            pfapack::utu2pfa_complex(&sm, &ws.pivots)
        };
        if !value.finite() {
            a.fill(Self::ZERO);
            return (value, PlaneStatus::NonFinite);
        }
        let mut sm = SqMat::new(&mut *a, n);
        let mut m = SqMat::new(&mut ws.m, n);
        pfapack::utu2inv_complex(&mut sm, &ws.pivots, &mut ws.vt, &mut m);
        (value, PlaneStatus::Ok)
    }
    fn make_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> tenferro_tensor::Result<Tensor> {
        Ok(Tensor::from_typed::<Complex64>(
            TypedTensor::from_vec_col_major(shape, data)?,
        ))
    }
    fn read_vec(read: &Tensor) -> tenferro_tensor::Result<Vec<Complex64>> {
        Ok(read.as_slice::<Complex64>()?.to_vec())
    }
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
    fn run_engine(
        engine: &dyn crate::BatchedEngine,
        planes: &[Complex64],
        n: usize,
        count: usize,
    ) -> crate::Result<crate::BatchOutput<Complex64>> {
        engine.run_c64(planes, n, count)
    }
}

impl PfScalar for Complex64 {
    fn zero_value() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn lanes() -> usize {
        2
    }
    fn as_f64_lanes(data: &[Complex64]) -> &[f64] {
        // SAFETY: Complex<f64> is `#[repr(C)]` with two f64 fields (no padding).
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f64>(), data.len() * 2) }
    }
    fn as_f64_lanes_mut(data: &mut [Complex64]) -> &mut [f64] {
        // SAFETY: same layout argument as `as_f64_lanes`.
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<f64>(), data.len() * 2) }
    }
}
