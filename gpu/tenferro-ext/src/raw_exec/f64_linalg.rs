//! FP64 BLAS and resident direct solves on the raw scope's stream.
use super::{backend, cublas, DevPtr, Op, RawExec};
use cudarc::cusolver::{result as solver, sys};
use std::cell::RefCell;
use tenferro_gpu::cuda::CudaRuntimeIdentity;

thread_local! {
    // Like RawExec's BLAS cache: a thread/runtime-owned handle. Avoid destruction
    // after CUDA context teardown at thread exit.
    static SOLVERS: RefCell<Vec<(CudaRuntimeIdentity, usize)>> = const { RefCell::new(Vec::new()) };
}

impl RawExec<'_, '_> {
    /// FP64 column-major GEMM, with the same operand contract as `gemm`.
    /// # Safety
    /// Every operand address/leading dimension must be in range and remain alive
    /// until the raw scope synchronizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dgemm(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: DevPtr,
        lda: usize,
        b: DevPtr,
        ldb: usize,
        beta: f64,
        c: DevPtr,
        ldc: usize,
    ) -> tenferro_tensor::Result<()> {
        if [m, n, k, lda, ldb, ldc]
            .iter()
            .any(|&v| v > i32::MAX as usize)
        {
            return Err(backend(
                "raw_exec::dgemm",
                "dimension exceeds CUDA integer capacity",
            ));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        unsafe {
            cublas::dgemm(
                self.blas,
                op_a.sys(),
                op_b.sys(),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a as *const f64,
                lda as i32,
                b as *const f64,
                ldb as i32,
                &beta,
                c as *mut f64,
                ldc as i32,
            )
        }
        .map_err(|e| backend("raw_exec::dgemm", e.to_string()))
    }

    fn solver(&self) -> tenferro_tensor::Result<sys::cusolverDnHandle_t> {
        let identity = self.raw.runtime_identity();
        let cached = SOLVERS.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|(id, _)| *id == identity)
                .map(|(_, h)| *h)
        });
        let handle = match cached {
            Some(handle) => handle as sys::cusolverDnHandle_t,
            None => {
                let handle = solver::dn_create()
                    .map_err(|e| backend("raw_exec::solver", format!("{e:?}")))?;
                SOLVERS.with(|cache| cache.borrow_mut().push((identity, handle as usize)));
                handle
            }
        };
        unsafe { solver::dn_set_stream(handle, self.stream as sys::cudaStream_t) }
            .map_err(|e| backend("raw_exec::solver", format!("{e:?}")))?;
        Ok(handle)
    }

    /// Required FP64 workspace for an upper Cholesky factorization.
    /// # Safety
    /// `matrix` addresses an `n*n` FP64 device matrix on this scope's context.
    pub unsafe fn dpotrf_workspace(
        &self,
        matrix: DevPtr,
        n: usize,
    ) -> tenferro_tensor::Result<usize> {
        if n == 0 || n > i32::MAX as usize {
            return Err(backend(
                "raw_exec::dpotrf_workspace",
                "invalid matrix dimension",
            ));
        }
        let mut count = 0;
        unsafe {
            sys::cusolverDnDpotrf_bufferSize(
                self.solver()?,
                sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                n as i32,
                matrix as *mut f64,
                n as i32,
                &mut count,
            )
            .result()
        }
        .map_err(|e| backend("raw_exec::dpotrf_workspace", format!("{e:?}")))?;
        usize::try_from(count)
            .map_err(|_| backend("raw_exec::dpotrf_workspace", "negative workspace size"))
    }

    /// Factor `matrix` and solve one RHS in place. Both status words stay on the
    /// device; the consumer must inspect them before applying the update.
    /// # Safety
    /// `matrix` is `n*n`, `rhs` is `n`, `work` has the queried FP64 workspace size,
    /// and `info` holds two device i32 words. All buffers outlive the raw scope.
    pub unsafe fn dpotrf_solve(
        &self,
        matrix: DevPtr,
        rhs: DevPtr,
        n: usize,
        work: DevPtr,
        work_len: usize,
        info: DevPtr,
    ) -> tenferro_tensor::Result<()> {
        if n == 0 || n > i32::MAX as usize || work_len > i32::MAX as usize {
            return Err(backend(
                "raw_exec::dpotrf_solve",
                "invalid matrix/workspace dimension",
            ));
        }
        let handle = self.solver()?;
        let uplo = sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
        unsafe {
            sys::cusolverDnDpotrf(
                handle,
                uplo,
                n as i32,
                matrix as *mut f64,
                n as i32,
                work as *mut f64,
                work_len as i32,
                info as *mut i32,
            )
            .result()
            .map_err(|e| backend("raw_exec::dpotrf", format!("{e:?}")))?;
            sys::cusolverDnDpotrs(
                handle,
                uplo,
                n as i32,
                1,
                matrix as *const f64,
                n as i32,
                rhs as *mut f64,
                n as i32,
                (info + 4) as *mut i32,
            )
            .result()
            .map_err(|e| backend("raw_exec::dpotrs", format!("{e:?}")))?;
        }
        Ok(())
    }
}
