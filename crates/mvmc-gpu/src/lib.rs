//! Batched Pfaffian and inverse of skew-symmetric planes (issue #423).
//!
//! The API works on a `[n, n, NQP, B]` column-major layout: plane `p = q + NQP * b`
//! occupies `data[p*n*n .. (p+1)*n*n]` and is itself column-major (`A[i, j] =
//! data[p*n*n + i + j*n]`). For every plane the result is
//!
//! * `pf[p]`: the Pfaffian of `A` (`utu2pfa` after the skew LTL^T factorization),
//! * `inv[p]`: the true inverse `A^-1` (`utu2inv`; the production sampler stores `-A^-1`,
//!   see the sign flip in `mvmc_core::pfaffian::calc_m_all_child_real`, which is applied by
//!   the caller and not here),
//! * `status[p]`: [`PlaneStatus`], with the pfapack semantics (`dsktf2`/`zsktf2` `INFO > 0`
//!   is [`PlaneStatus::ZeroPivot`], a non-finite Pfaffian is [`PlaneStatus::NonFinite`]).
//!
//! For `status != Ok` the inverse plane is zero-filled; a zero-pivot plane reports `pf = 0`
//! (pfapack's `pfaffian_ltl` contract) and a non-finite plane reports the computed value.
//!
//! Backends ([`Backend`]):
//!
//! * [`Backend::CpuPfapack`] / [`Backend::CpuPfapackRayon`]: loop over `pfapack` per plane.
//!   This is the reference; the results are bit-identical to calling pfapack directly.
//! * [`Backend::TenferroNative`]: the same algorithm expressed as batched tenferro tensor
//!   operations on `CpuBackend` (reference for a device-portable formulation).
//! * [`Backend::TenferroExtension`]: the pfapack per-plane kernel wrapped as a tenferro
//!   `ExtensionOp`, run in a tenferro runtime on `CpuBackend` (bit-identical to pfapack).
//! * [`Backend::Engine`]: an external [`BatchedEngine`]; the CUDA engine (one thread block per
//!   plane, NVRTC kernel launched through tenferro's raw session) lives in the standalone
//!   workspace `gpu/mvmc-gpu-cuda` so this crate and `Cargo.lock` stay GPU-free.
//!
//! Nothing here is wired into the production sampler.

#![warn(missing_docs)]

use num_complex::Complex64;

mod cpu;
mod scalar;
mod tenferro_ext;
mod tenferro_native;

pub mod stages;

#[doc(hidden)]
pub mod testkit;

pub use scalar::PfScalar;

/// Per-plane outcome, mirroring pfapack and `CalcMAllError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneStatus {
    /// Factorization succeeded and the Pfaffian is finite.
    Ok,
    /// `dsktf2`/`zsktf2` found an exactly zero pivot column (`INFO > 0`, 1-based row).
    ZeroPivot {
        /// The 1-based row reported by pfapack (the first zero column met, scanning from `n`).
        row: usize,
    },
    /// The Pfaffian is not finite.
    NonFinite,
}

impl PlaneStatus {
    /// Wire encoding used by the device kernels: `0` ok, `k > 0` zero pivot at row `k`,
    /// `-1` non-finite.
    pub fn to_code(self) -> i32 {
        match self {
            Self::Ok => 0,
            Self::ZeroPivot { row } => row as i32,
            Self::NonFinite => -1,
        }
    }

    /// Inverse of [`PlaneStatus::to_code`].
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            c if c > 0 => Self::ZeroPivot { row: c as usize },
            _ => Self::NonFinite,
        }
    }
}

/// Errors of the batched API. There is no silent CPU fallback: asking for a backend that is
/// unavailable is an error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The matrix side or the buffer length is not usable.
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    /// The requested backend is not compiled in or no device is available.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    /// A backend failed at run time.
    #[error("backend failure: {0}")]
    Backend(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// An externally provided batched engine (for example the CUDA kernel).
///
/// Implementations report numerical failures per plane in the returned statuses and
/// infrastructure failures as [`Error`]; they must not fall back to another backend.
pub trait BatchedEngine: Send + Sync {
    /// Short label for reports.
    fn name(&self) -> String;
    /// `f64` planes, `n x n` each, `count` planes.
    fn run_f64(&self, planes: &[f64], n: usize, count: usize) -> Result<BatchOutput<f64>>;
    /// `Complex64` planes.
    fn run_c64(
        &self,
        planes: &[Complex64],
        n: usize,
        count: usize,
    ) -> Result<BatchOutput<Complex64>>;
}

/// Execution backend of [`pfaffian_inverse_batched`].
#[derive(Clone, Copy)]
pub enum Backend<'a> {
    /// pfapack per plane, one thread.
    CpuPfapack,
    /// pfapack per plane, rayon over planes (one workspace per rayon task).
    CpuPfapackRayon,
    /// Batched tenferro tensor operations on `CpuBackend` (faer provider).
    TenferroNative,
    /// pfapack wrapped as a tenferro `ExtensionOp` in a tenferro runtime on `CpuBackend`.
    TenferroExtension,
    /// An external engine, for example the CUDA kernel of `gpu/mvmc-gpu-cuda`.
    Engine(&'a dyn BatchedEngine),
}

impl std::fmt::Debug for Backend<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuPfapack => f.write_str("CpuPfapack"),
            Self::CpuPfapackRayon => f.write_str("CpuPfapackRayon"),
            Self::TenferroNative => f.write_str("TenferroNative"),
            Self::TenferroExtension => f.write_str("TenferroExtension"),
            Self::Engine(e) => write!(f, "Engine({})", e.name()),
        }
    }
}

/// Output of [`pfaffian_inverse_batched`]. All buffers use the plane order `p = q + NQP * b`.
#[derive(Debug, Clone)]
pub struct BatchedPfaffian<T> {
    /// Matrix side `n`.
    pub n: usize,
    /// QP planes per batch element.
    pub nqp: usize,
    /// Batch size `B`.
    pub batch: usize,
    /// Pfaffians, `[NQP, B]`.
    pub pf: Vec<T>,
    /// Inverses, `[n, n, NQP, B]` column-major.
    pub inv: Vec<T>,
    /// Per-plane status, `[NQP, B]`.
    pub status: Vec<PlaneStatus>,
}

impl<T> BatchedPfaffian<T> {
    /// Number of planes `NQP * B`.
    pub fn planes(&self) -> usize {
        self.nqp * self.batch
    }
}

/// `(pf, inv, status)` of a batch, plane order `p = q + NQP * b`.
pub type BatchOutput<T> = (Vec<T>, Vec<T>, Vec<PlaneStatus>);

fn validate(len: usize, n: usize, nqp: usize, batch: usize) -> Result<usize> {
    if n < 2 || !n.is_multiple_of(2) {
        return Err(Error::InvalidShape(format!(
            "matrix side n must be even and >= 2, got {n}"
        )));
    }
    let planes = nqp
        .checked_mul(batch)
        .ok_or_else(|| Error::InvalidShape("NQP * B overflows usize".into()))?;
    let want = n
        .checked_mul(n)
        .and_then(|nn| nn.checked_mul(planes))
        .ok_or_else(|| Error::InvalidShape("n*n*NQP*B overflows usize".into()))?;
    if len != want {
        return Err(Error::InvalidShape(format!(
            "buffer length {len} != n*n*NQP*B = {want}"
        )));
    }
    Ok(planes)
}

/// Pfaffian, inverse and status of every plane of a `[n, n, NQP, B]` skew-symmetric batch.
///
/// `planes` is column-major with plane order `p = q + NQP * b`. `n` must be even and `>= 2`.
/// Like pfapack, the factorization trusts the strict upper triangle of each plane.
///
/// # Errors
///
/// [`Error::InvalidShape`] for an unusable shape, [`Error::BackendUnavailable`] when the CUDA
/// feature or device is missing, [`Error::Backend`] for run-time failures. Numerical failures
/// of a plane are reported in [`BatchedPfaffian::status`], not as an error.
pub fn pfaffian_inverse_batched<T: PfScalar>(
    backend: &Backend<'_>,
    planes: &[T],
    n: usize,
    nqp: usize,
    batch: usize,
) -> Result<BatchedPfaffian<T>> {
    let count = validate(planes.len(), n, nqp, batch)?;
    let (pf, inv, status) = match backend {
        Backend::CpuPfapack => cpu::run(planes, n, count, false),
        Backend::CpuPfapackRayon => cpu::run(planes, n, count, true),
        Backend::TenferroNative => tenferro_native::run(planes, n, count)?,
        Backend::TenferroExtension => tenferro_ext::run(planes, n, count)?,
        Backend::Engine(engine) => T::run_engine(*engine, planes, n, count)?,
    };
    Ok(BatchedPfaffian {
        n,
        nqp,
        batch,
        pf,
        inv,
        status,
    })
}
