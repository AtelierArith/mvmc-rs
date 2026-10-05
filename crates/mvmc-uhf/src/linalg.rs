//! BLAS/LAPACK wrappers matching `ComplexUHF/matrixlapack.c`.
//!
//! The C tool calls LAPACK `zheev_` (`jobz='V'`, `uplo='U'`, `lwork=4*N`) and
//! BLAS `zgemm_`. The same routines are called here through the workspace
//! `lapack`/`blas` crates; the provider is whatever the final link supplies
//! (OpenBLAS on Linux, Accelerate or Homebrew OpenBLAS on macOS).

use num_complex::Complex64;

use crate::UhfError;

/// `ZHEEVall`: all eigenpairs of the Hermitian matrix whose upper triangle is
/// given by the row-major `a` (`a[i*n + j] = A[i][j]`).
///
/// Returns ascending eigenvalues and eigenvectors **as rows**:
/// `vectors[k*n + l]` is component `l` of eigenvector `k` (the C `vec[k][l]`).
/// A non-zero LAPACK `info` is an error (C silently keeps stale zeros).
pub fn zheev_all(n: usize, a: &[Complex64]) -> Result<(Vec<f64>, Vec<Complex64>), UhfError> {
    assert_eq!(a.len(), n * n);
    let n_i = n as i32;
    let lwork = 4 * n_i;
    let mut column_major = vec![Complex64::new(0.0, 0.0); n * n];
    let mut k = 0;
    for j in 0..n {
        for i in 0..n {
            column_major[k] = a[i * n + j];
            k += 1;
        }
    }
    let mut w = vec![0.0_f64; n];
    let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0_f64; lwork as usize];
    let mut info = 0_i32;
    // SAFETY: all buffers are sized as LAPACK requires (n*n, n, 4n, 4n >= 3n-2).
    unsafe {
        lapack::zheev(
            b'V',
            b'U',
            n_i,
            &mut column_major,
            n_i,
            &mut w,
            &mut work,
            lwork,
            &mut rwork,
            &mut info,
        );
    }
    if info != 0 {
        return Err(UhfError::Lapack(format!("zheev returned info={info}")));
    }
    Ok((w, column_major))
}

/// `cmp_MMProd`: `C = A * B` with `A` of shape `ns x ne`, `B` of shape
/// `ne x ns` (row-major in and out), through `zgemm('N','N')`.
pub fn zgemm_nn(ns: usize, ne: usize, a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), ns * ne);
    assert_eq!(b.len(), ne * ns);
    let zero = Complex64::new(0.0, 0.0);
    let mut a_col = vec![zero; ns * ne];
    let mut k = 0;
    for j in 0..ne {
        for i in 0..ns {
            a_col[k] = a[i * ne + j];
            k += 1;
        }
    }
    let mut b_col = vec![zero; ne * ns];
    k = 0;
    for j in 0..ns {
        for i in 0..ne {
            b_col[k] = b[i * ns + j];
            k += 1;
        }
    }
    let mut c_col = vec![zero; ns * ns];
    // SAFETY: leading dimensions and lengths follow the column-major layout above.
    unsafe {
        blas::zgemm(
            b'N',
            b'N',
            ns as i32,
            ns as i32,
            ne as i32,
            Complex64::new(1.0, 0.0),
            &a_col,
            ns as i32,
            &b_col,
            ne as i32,
            Complex64::new(0.0, 0.0),
            &mut c_col,
            ns as i32,
        );
    }
    let mut c = vec![zero; ns * ns];
    k = 0;
    for j in 0..ns {
        for i in 0..ns {
            c[i * ns + j] = c_col[k];
            k += 1;
        }
    }
    c
}
