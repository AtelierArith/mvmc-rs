//! Stage-level backend for the Stochastic Reconfiguration (SR) tensor work (issue #421,
//! design: `docs/design/gpu-readiness.md`, sections 5.2 and 5.3).
//!
//! The SR step has four tensor-shaped stages, all dispatched through [`SrStages`]:
//!
//! 1. the Gram product `OO = O O^H` of the saved-sample store (`finalize_oo_store*`),
//! 2. the S matrix / force `g` assembly (`sr.rs`),
//! 3. the Cholesky solve of `S x = g` (`sr.rs`),
//! 4. the sampled CG matrix-vector product (`sr_cg.rs`).
//!
//! Two implementations exist:
//!
//! * [`COrderSr`] is the default and the parity oracle. It is the former code moved behind
//!   the trait without changing a single operation (LAPACK `dpotrf`/`dpotrs`, BLAS `dsyrk` and
//!   `dgemv`, the sequential complex Gram sum), so default outputs stay byte-identical.
//! * [`TenferroSr`] routes the same stages through tenferro eager ops (`dot_general`,
//!   `cholesky`, `triangular_solve`, elementwise `sub`/`mul`) on one process-wide runtime,
//!   built once and reused (eager op calls cost 26-126 us each, so the runtime and the
//!   constant operands are never rebuilt per call). It is opt-in
//!   (`MVMC_RS_SR_BACKEND=tenferro`) and is validated against [`COrderSr`] with explicit
//!   tolerances (`crates/mvmc-core/tests/sr_backend_421.rs`, `docs/NUMERICAL_COMPARISONS.md`).
//!
//! The runtime is the tenferro CPU backend with the `cpu-faer` provider; `cpu-blas` must stay
//! off because tenferro's CPU linear algebra (`cholesky`, `triangular_solve`) fails with it.
//! [`TenferroSr::with_runtime`] accepts any eager runtime, so the standalone
//! `gpu/mvmc-gpu-cuda` crate runs the identical stages on a CUDA device
//! ([`Placement::Device`] uploads and downloads explicitly; tenferro never transfers
//! implicitly).
//!
//! # Complex Gram sample order
//!
//! The complex store Gram in the C-order path is a plain sequential sum over samples for
//! every entry (`observables.rs`, `sr_backend::c_order_gram_complex`). A general GEMM may reassociate the
//! sample sum and change both the rounding and the sign of exact zeros, which changes the
//! direct SR input even for an identical RNG trajectory; the default path therefore pins the
//! order. The tenferro path uses `dot_general` (blocked GEMM) and is allowed to differ by the
//! reassociation error, bounded by `samples * eps * sum_s |O_is| |O_js|` per entry; the
//! equivalence test uses exactly that bound with a safety factor. The decision is: C order
//! stays authoritative and default, the accelerated order is opt-in and tolerance-validated.

use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

use crate::stage_backend::{centered, SrSg, StageError};

fn err(e: impl std::fmt::Display) -> StageError {
    StageError::Failed(e.to_string())
}

/// A strided view of real values: a real array or the real parts of a complex array.
#[derive(Clone, Copy)]
pub enum RealView<'a> {
    /// `f64` values.
    Real(&'a [f64]),
    /// Real parts of complex values.
    ReOfComplex(&'a [Complex64]),
}

impl RealView<'_> {
    /// Element `index` (real part for complex data).
    #[inline]
    pub fn at(&self, index: usize) -> f64 {
        match self {
            Self::Real(v) => v[index],
            Self::ReOfComplex(v) => v[index].re,
        }
    }
}

/// Inputs of the S matrix and force assembly (`stcopt.c:69`, `sr.rs`).
///
/// Component `p` of the active set sits at `oo[(p + offset) * ld + (q + offset)]` for the pair
/// `(p, q)`, at `oo[p + offset]` for the mean and at `ho[p + offset]` for the energy
/// derivative; `ho[0]` and `oo[0]` hold the energy-like constant.
pub struct SrAssembleInput<'a> {
    /// `OO` matrix (leading dimension `ld`).
    pub oo: RealView<'a>,
    /// `HO` vector.
    pub ho: RealView<'a>,
    /// Active components in S order.
    pub map: &'a [usize],
    /// Leading dimension of `oo` (`sr_opt_size` real, `2 * sr_opt_size` complex).
    pub ld: usize,
    /// Index offset of the first parameter (1 real, 2 complex).
    pub offset: usize,
    /// Diagonal regularization `DSROptStaDel`.
    pub sta_del: f64,
    /// Step size `DSROptStepDt`.
    pub step_dt: f64,
}

/// Saved-sample matrices of one CG solve, `[component, sample]` column-major.
pub struct CgSamples<'a> {
    /// Real parts.
    pub real: &'a [f64],
    /// Imaginary parts (empty for a real layout).
    pub imag: &'a [f64],
    /// Active components.
    pub components: usize,
    /// Samples.
    pub samples: usize,
    /// Operand version: equal versions mean identical sample data (constant-operand cache key).
    pub version: u64,
}

/// Stage-level SR backend.
pub trait SrStages: Send {
    /// Human-readable label for logs and result files.
    fn label(&self) -> String;

    /// Provider description recorded in the benchmark metadata.
    fn provider(&self) -> String;

    /// Profiling counters of backends that keep them (`None` otherwise).
    fn stats(&self) -> Option<TenferroStats> {
        None
    }

    /// `S = sum_s w_s dO_s dO_s^T` and `g = sum_s w_s dO_s dE_s` with `dO = O - <O>` and
    /// `dE = E - <E>` (`<.>` weighted by `w`, `sum w = 1`); `o` is column-major
    /// `nsample x npara`. This is the validation harness's composite SR stage.
    fn sr_s_g(
        &mut self,
        o: &[f64],
        nsample: usize,
        npara: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<SrSg, StageError>;

    /// Real Gram `out = O O^T` for the `[n, samples]` store; `out` is the full symmetric
    /// `n x n` column-major matrix.
    fn gram_real(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
        out: &mut [f64],
    ) -> Result<(), StageError>;

    /// Complex Gram `G[i + j n] = sum_s O[i,s] conj(O[j,s])`, column-major `n x n`.
    fn gram_complex(
        &mut self,
        store: &[Complex64],
        n: usize,
        samples: usize,
    ) -> Result<Vec<Complex64>, StageError>;

    /// S matrix (column-major) and force `g`.
    fn assemble_s_g(
        &mut self,
        input: &SrAssembleInput<'_>,
        s: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), StageError>;

    /// Solve the symmetric positive-definite system `S x = rhs` in place (`rhs` becomes `x`).
    ///
    /// `Err(())` mirrors C's `info != 0` or a nonfinite solution; `s` may be overwritten.
    #[allow(clippy::result_unit_err)] // same contract as the C-order `info != 0` result
    fn cholesky_solve(&mut self, s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()>;

    /// Local sampled CG product `z = O_r O_r^T x + O_i O_i^T x` (no mean/diagonal/MPI terms).
    fn cg_local_product(
        &mut self,
        samples: &CgSamples<'_>,
        x: &[f64],
        z: &mut [f64],
    ) -> Result<(), StageError>;
}

// ---------------------------------------------------------------------------------------
// C-order implementation (default, parity oracle)
// ---------------------------------------------------------------------------------------

/// The C-order SR implementation: the former inline code behind the trait, operation for
/// operation. Holds only the reusable CG intermediate vectors.
#[derive(Debug, Default)]
pub struct COrderSr {
    y_real: Vec<f64>,
    y_imag: Vec<f64>,
}

/// Real store Gram `O O^T` exactly as Julia `mul!(C, O, transpose(O))` selects it: SYRK for
/// `max(n, samples) >= 4`, `muladd` accumulation below (see `finalize_oo_store_real`).
pub(crate) fn c_order_gram_real(store: &[f64], n: usize, sample_size: usize, out: &mut [f64]) {
    let dim = i32::try_from(n).expect("SR Gram dimension must fit BLAS LP64");
    let samples = i32::try_from(sample_size).expect("sample count must fit BLAS LP64");
    // Julia mul!(C, O, transpose(O)) recognizes the shared operand and
    // selects SYRK. Below its max(n,samples)>=4 cutoff, generic_syrk!
    // accumulates with muladd instead. Both paths copy the upper triangle.
    if n.max(sample_size) < 4 {
        out[..n * n].fill(0.0);
        for sample in 0..sample_size {
            for j in 0..n {
                let oj = store[j + sample * n];
                for i in 0..=j {
                    let index = i + j * n;
                    out[index] = store[i + sample * n].mul_add(oj, out[index]);
                }
            }
        }
    } else {
        // SAFETY: O is [n,samples], leading n; the writable output has n*n
        // entries. SYRK reads O and overwrites the output's upper triangle.
        unsafe {
            blas::dsyrk(
                b'U',
                b'N',
                dim,
                samples,
                1.0,
                store,
                dim,
                0.0,
                &mut out[..n * n],
                dim,
            );
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            out[i + j * n] = out[j + i * n];
        }
    }
}

/// Complex store Gram with the authoritative sequential sample sum per entry.
pub(crate) fn c_order_gram_complex(
    store: &[Complex64],
    n: usize,
    samples: usize,
) -> Vec<Complex64> {
    let raw = store;
    let mut gram = vec![Complex64::new(0.0, 0.0); n * n];
    // One column is `n * samples` complex multiply-adds (about 3 ns each).
    let parallel = crate::threading::inner_parallel_work(n, 3 * n * samples);
    let observed =
        crate::threading::observe_kernel(crate::threading::ObservedWork::Entry, parallel);
    let _scope = crate::threading::profile_scope(parallel, n);
    let update = |j: usize, column: &mut [Complex64]| {
        let _entry = observed.enter_item();
        for i in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for sample in 0..samples {
                sum += raw[i + sample * n] * raw[j + sample * n].conj();
            }
            column[i] = sum;
        }
    };
    if parallel {
        use rayon::prelude::*;
        crate::threading::install_inner(|| {
            gram.par_chunks_mut(n)
                .enumerate()
                .for_each(|(j, column)| update(j, column))
        });
    } else if n > 0 {
        gram.chunks_mut(n)
            .enumerate()
            .for_each(|(j, column)| update(j, column));
    }
    gram
}

impl SrStages for COrderSr {
    fn label(&self) -> String {
        "c-order-cpu".to_string()
    }

    fn provider(&self) -> String {
        "BLAS/LAPACK (dsyrk, dgemv, dpotrf/dpotrs) + scalar loops".to_string()
    }

    fn sr_s_g(
        &mut self,
        o: &[f64],
        ns: usize,
        np: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<SrSg, StageError> {
        let (d, de) = centered(o, ns, np, e, w);
        let mut s = vec![0.0; np * np];
        let mut g = vec![0.0; np];
        for q in 0..np {
            for p in 0..np {
                let mut acc = 0.0;
                for k in 0..ns {
                    acc += w[k] * d[k + p * ns] * d[k + q * ns];
                }
                s[p + q * np] = acc;
            }
            let mut acc = 0.0;
            for k in 0..ns {
                acc += w[k] * d[k + q * ns] * de[k];
            }
            g[q] = acc;
        }
        Ok(SrSg { s, g })
    }

    fn gram_real(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
        out: &mut [f64],
    ) -> Result<(), StageError> {
        c_order_gram_real(store, n, samples, out);
        Ok(())
    }

    fn gram_complex(
        &mut self,
        store: &[Complex64],
        n: usize,
        samples: usize,
    ) -> Result<Vec<Complex64>, StageError> {
        Ok(c_order_gram_complex(store, n, samples))
    }

    fn assemble_s_g(
        &mut self,
        input: &SrAssembleInput<'_>,
        s: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), StageError> {
        let n_smat = input.map.len();
        let ratio_diag = 1.0 + input.sta_del;
        // C stcopt.c:69 `omp parallel for` over the S entries; every entry has one
        // producer, so the columns are filled independently (column `sj` of S).
        crate::threading::for_each_chunk_mut(s, n_smat, 3 * n_smat, |sj, column| {
            let pj = input.map[sj];
            for (si, &pi) in input.map.iter().enumerate() {
                let tmp = input.oo.at(pi + input.offset);
                let oo_idx = (pi + input.offset) * input.ld + (pj + input.offset);
                column[si] = input.oo.at(oo_idx) - tmp * input.oo.at(pj + input.offset);
                if si == sj {
                    column[si] *= ratio_diag;
                }
            }
        });
        let ho_0 = input.ho.at(0);
        crate::threading::for_each_mut(g, 3, |si, value| {
            let pi = input.map[si];
            let v = input.ho.at(pi + input.offset) - ho_0 * input.oo.at(pi + input.offset);
            *value = -2.0 * input.step_dt * v;
        });
        Ok(())
    }

    fn cholesky_solve(&mut self, s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
        crate::sr::cholesky_solve(s, rhs, n)
    }

    fn cg_local_product(
        &mut self,
        m: &CgSamples<'_>,
        x: &[f64],
        z: &mut [f64],
    ) -> Result<(), StageError> {
        let complex = !m.imag.is_empty();
        if m.samples == 0 {
            z.fill(0.0);
            return Ok(());
        }
        self.y_real.resize(m.samples, 0.0);
        if complex {
            self.y_imag.resize(m.samples, 0.0);
        }
        crate::serial_blas::initialize();
        let rows = i32::try_from(m.components).expect("CG component count must fit BLAS LP64");
        let cols = i32::try_from(m.samples).expect("CG sample count must fit BLAS LP64");
        // SAFETY: matrix buffers have exactly rows*cols entries, leading
        // dimensions are rows, and every input/output vector has the
        // required length. Mutable outputs never alias matrix/input views.
        unsafe {
            blas::dgemv(
                b'T',
                rows,
                cols,
                1.0,
                m.real,
                rows,
                x,
                1,
                0.0,
                &mut self.y_real,
                1,
            );
            if complex {
                blas::dgemv(
                    b'T',
                    rows,
                    cols,
                    1.0,
                    m.imag,
                    rows,
                    x,
                    1,
                    0.0,
                    &mut self.y_imag,
                    1,
                );
            }
            blas::dgemv(
                b'N',
                rows,
                cols,
                1.0,
                m.real,
                rows,
                &self.y_real,
                1,
                0.0,
                z,
                1,
            );
            if complex {
                blas::dgemv(
                    b'N',
                    rows,
                    cols,
                    1.0,
                    m.imag,
                    rows,
                    &self.y_imag,
                    1,
                    1.0,
                    z,
                    1,
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// tenferro implementation (opt-in)
// ---------------------------------------------------------------------------------------

/// Where the eager runtime keeps its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Host memory (`CpuBackend`): tensors are created directly.
    Host,
    /// Device memory (CUDA): explicit upload and download, never implicit transfers.
    Device,
}

/// Cached constant CG operand (`[components, samples (+ samples)]`), keyed by version.
struct CgOperand {
    version: u64,
    components: usize,
    samples: usize,
    complex: bool,
    tensor: EagerTensor,
}

/// Counters for profiling and tests (calls per stage, uploads of cached operands).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TenferroStats {
    /// `cg_local_product` calls.
    pub cg_products: u64,
    /// Constant CG operand uploads (cache misses).
    pub cg_uploads: u64,
}

/// SR stages through tenferro eager ops.
pub struct TenferroSr {
    ctx: Arc<EagerRuntime>,
    placement: Placement,
    label: String,
    cg: Option<CgOperand>,
    stats: TenferroStats,
}

impl std::fmt::Debug for TenferroSr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenferroSr")
            .field("label", &self.label)
            .field("placement", &self.placement)
            .field("stats", &self.stats)
            .finish()
    }
}

impl TenferroSr {
    /// CPU runtime with the `cpu-faer` provider (built once; see [`acquire`]).
    pub fn new_cpu() -> Result<Self, StageError> {
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).map_err(err)?;
        Ok(Self::with_runtime(
            ctx,
            Placement::Host,
            "tenferro cpu-faer".to_string(),
        ))
    }

    /// Any eager runtime (for example CUDA, built by `gpu/mvmc-gpu-cuda`).
    pub fn with_runtime(ctx: Arc<EagerRuntime>, placement: Placement, label: String) -> Self {
        Self {
            ctx,
            placement,
            label,
            cg: None,
            stats: TenferroStats::default(),
        }
    }

    /// Profiling counters.
    pub fn stats(&self) -> TenferroStats {
        self.stats
    }

    fn upload(&self, host: Tensor) -> Result<EagerTensor, StageError> {
        let tensor = match self.placement {
            Placement::Host => host,
            Placement::Device => self
                .ctx
                .with_execution_session(|s| s.upload_host_tensor(TensorRead::from_tensor(&host)))
                .map_err(err)?
                .map_err(err)?,
        };
        EagerTensor::from_tensor_in(tensor, self.ctx.clone()).map_err(err)
    }

    fn upload_real(&self, shape: Vec<usize>, data: Vec<f64>) -> Result<EagerTensor, StageError> {
        // Column-major on purpose: every buffer in this module is column-major.
        self.upload(Tensor::from_vec_col_major(shape, data).map_err(err)?)
    }

    fn download(&self, t: &EagerTensor) -> Result<Tensor, StageError> {
        self.ctx.synchronize().map_err(err)?;
        let dev = t.to_tensor().map_err(err)?;
        match self.placement {
            Placement::Host => Ok(dev),
            Placement::Device => self
                .ctx
                .with_execution_session(|s| s.download_to_host(TensorRead::from_tensor(&dev)))
                .map_err(err)?
                .map_err(err),
        }
    }

    fn download_f64(&self, t: &EagerTensor) -> Result<Vec<f64>, StageError> {
        let host = self.download(t)?;
        let typed = host
            .into_typed::<f64>()
            .map_err(|_| err("expected an F64 tensor"))?;
        Ok(typed.host_data().map_err(err)?.to_vec())
    }
}

fn dot(contract_l: usize, contract_r: usize) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![contract_l],
        rhs_contracting_dims: vec![contract_r],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

impl SrStages for TenferroSr {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn provider(&self) -> String {
        match self.placement {
            Placement::Host => "tenferro-ad EagerRuntime + CpuBackend (cpu-faer)".to_string(),
            Placement::Device => {
                "tenferro-ad EagerRuntime + tenferro-gpu CudaBackend (cuBLAS/cuSOLVER)".to_string()
            }
        }
    }

    fn stats(&self) -> Option<TenferroStats> {
        Some(self.stats)
    }

    fn sr_s_g(
        &mut self,
        o: &[f64],
        ns: usize,
        np: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<SrSg, StageError> {
        let (d, de) = centered(o, ns, np, e, w);
        let mut dw = d.clone();
        for p in 0..np {
            for k in 0..ns {
                dw[k + p * ns] *= w[k];
            }
        }
        let dt = self.upload_real(vec![ns, np], d)?;
        let dwt = self.upload_real(vec![ns, np], dw)?;
        let det = self.upload_real(vec![ns], de)?;
        let st = dwt.dot_general(&dt, dot(0, 0)).map_err(err)?;
        let gt = dwt.dot_general(&det, dot(0, 0)).map_err(err)?;
        Ok(SrSg {
            s: self.download_f64(&st)?,
            g: self.download_f64(&gt)?,
        })
    }

    fn gram_real(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
        out: &mut [f64],
    ) -> Result<(), StageError> {
        let o = self.upload_real(vec![n, samples], store.to_vec())?;
        // OO[i,j] = sum_s O[i,s] O[j,s]: contract the sample axis of both operands.
        let gram = o.dot_general(&o, dot(1, 1)).map_err(err)?;
        let host = self.download_f64(&gram)?;
        out[..n * n].copy_from_slice(&host);
        // GEMM does not guarantee exact symmetry; the C-order SYRK copies one triangle.
        // Symmetrize the same way so downstream S is exactly symmetric.
        for j in 0..n {
            for i in j + 1..n {
                out[i + j * n] = out[j + i * n];
            }
        }
        Ok(())
    }

    fn gram_complex(
        &mut self,
        store: &[Complex64],
        n: usize,
        samples: usize,
    ) -> Result<Vec<Complex64>, StageError> {
        let host = Tensor::from_vec_col_major(vec![n, samples], store.to_vec()).map_err(err)?;
        let o = self.upload(host)?;
        // G[i,j] = sum_s O[i,s] conj(O[j,s]).
        let gram = o
            .dot_general_with_conj(&o, dot(1, 1), false, true)
            .map_err(err)?;
        let host = self.download(&gram)?;
        let typed = host
            .into_typed::<Complex64>()
            .map_err(|_| err("expected a C64 tensor"))?;
        Ok(typed.host_data().map_err(err)?.to_vec())
    }

    fn assemble_s_g(
        &mut self,
        input: &SrAssembleInput<'_>,
        s: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), StageError> {
        let n = input.map.len();
        // Gather the active block, the means and the energy derivatives (index tables are
        // small host data; the gather is the only non-tensor step).
        let mut oo_sub = vec![0.0; n * n];
        let mut mean = vec![0.0; n];
        let mut ho_sub = vec![0.0; n];
        for (sj, &pj) in input.map.iter().enumerate() {
            for (si, &pi) in input.map.iter().enumerate() {
                oo_sub[si + sj * n] = input
                    .oo
                    .at((pi + input.offset) * input.ld + pj + input.offset);
            }
            mean[sj] = input.oo.at(pj + input.offset);
            ho_sub[sj] = input.ho.at(pj + input.offset);
        }
        // Diagonal regularization as a weight matrix W = 1 + delta I (off-diagonal exactly 1,
        // diagonal `1.0 + sta_del` as in C).
        let mut weight = vec![1.0; n * n];
        for i in 0..n {
            weight[i + i * n] = 1.0 + input.sta_del;
        }
        let oo = self.upload_real(vec![n, n], oo_sub)?;
        let w = self.upload_real(vec![n, n], weight)?;
        // m m^T as [n,1] x [1,n] contracting the unit axis: one exact product per entry.
        let m_col = self.upload_real(vec![n, 1], mean.clone())?;
        let m_row = self.upload_real(vec![1, n], mean.clone())?;
        let outer = m_col.dot_general(&m_row, dot(1, 0)).map_err(err)?;
        let s_t = oo.sub(&outer).map_err(err)?.mul(&w).map_err(err)?;
        // g = -2 dt (ho - ho_0 m); each elementwise step is one correctly rounded IEEE
        // operation in the same order as the C expression. Scalars are uploaded as constant
        // vectors: tenferro never creates device tensors implicitly, and a host scalar
        // operand fails on CUDA.
        let m = self.upload_real(vec![n], mean)?;
        let ho = self.upload_real(vec![n], ho_sub)?;
        let ho0 = self.upload_real(vec![n], vec![input.ho.at(0); n])?;
        let coef = self.upload_real(vec![n], vec![-2.0 * input.step_dt; n])?;
        let g_t = ho
            .sub(&m.mul(&ho0).map_err(err)?)
            .map_err(err)?
            .mul(&coef)
            .map_err(err)?;
        s[..n * n].copy_from_slice(&self.download_f64(&s_t)?);
        g[..n].copy_from_slice(&self.download_f64(&g_t)?);
        Ok(())
    }

    fn cholesky_solve(&mut self, s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
        if n == 0 {
            return Ok(());
        }
        let run = || -> Result<Vec<f64>, StageError> {
            let a = self.upload_real(vec![n, n], s[..n * n].to_vec())?;
            let b = self.upload_real(vec![n, 1], rhs[..n].to_vec())?;
            // S = L L^T with lower L; solve L y = b, then L^T x = y.
            let l = a.cholesky().map_err(err)?;
            let y = l
                .triangular_solve(&b, true, true, false, false)
                .map_err(err)?;
            let x = l
                .triangular_solve(&y, true, true, true, false)
                .map_err(err)?;
            self.download_f64(&x)
        };
        match run() {
            Ok(x) if x.iter().all(|v| v.is_finite()) => {
                rhs[..n].copy_from_slice(&x);
                Ok(())
            }
            Ok(_) => Err(()),
            Err(e) => {
                tracing::debug!("{e}");
                Err(())
            }
        }
    }

    fn cg_local_product(
        &mut self,
        m: &CgSamples<'_>,
        x: &[f64],
        z: &mut [f64],
    ) -> Result<(), StageError> {
        self.stats.cg_products += 1;
        if m.samples == 0 || m.components == 0 {
            z.fill(0.0);
            return Ok(());
        }
        let complex = !m.imag.is_empty();
        let stale = self.cg.as_ref().is_none_or(|c| {
            c.version != m.version
                || c.components != m.components
                || c.samples != m.samples
                || c.complex != complex
        });
        if stale {
            // Constant operand: [real | imag] side by side, so one pair of products gives
            // O_r O_r^T x + O_i O_i^T x. Uploaded once per solve (version counter).
            let mut data = m.real.to_vec();
            data.extend_from_slice(m.imag);
            let cols = if complex { 2 * m.samples } else { m.samples };
            let tensor = self.upload_real(vec![m.components, cols], data)?;
            self.cg = Some(CgOperand {
                version: m.version,
                components: m.components,
                samples: m.samples,
                complex,
                tensor,
            });
            self.stats.cg_uploads += 1;
        }
        let xt = self.upload_real(vec![m.components, 1], x.to_vec())?;
        let op = self.cg.as_ref().expect("operand just cached");
        // y = O^T x  -> [cols, 1];  z = O y -> [components, 1].
        let y = op.tensor.dot_general(&xt, dot(0, 0)).map_err(err)?;
        let zt = op.tensor.dot_general(&y, dot(1, 0)).map_err(err)?;
        z.copy_from_slice(&self.download_f64(&zt)?);
        Ok(())
    }
}
