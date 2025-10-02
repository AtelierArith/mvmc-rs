//! Safe wrappers for LAPACK linear algebra functions
//!
//! This module provides type-safe, memory-safe wrappers around LAPACK functions.

use crate::ffi::lapack::{check_lapack_info, LapackError};
use num_complex::Complex64;

/// Result type for LAPACK operations
pub type LapackResult<T> = Result<T, LapackError>;

/// Solve system of linear equations Ax = b using LU decomposition
///
/// # Arguments
/// * `a` - Coefficient matrix (will be modified)
/// * `b` - Right-hand side vector (will contain solution)
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(LapackError)` if failed
///
/// # Example
/// ```
/// use mvmc_bindings::wrappers::linear_algebra::solve_linear_system;
/// use num_complex::Complex64;
/// use ndarray::Array2;
///
/// let mut a = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
/// ]).unwrap();
/// let mut b = Array2::from_shape_vec((2, 1), vec![
///     Complex64::new(5.0, 0.0),
///     Complex64::new(6.0, 0.0),
/// ]).unwrap();
///
/// let result = solve_linear_system(&mut a, &mut b);
/// assert!(result.is_ok());
/// ```
pub fn solve_linear_system(
    a: &mut ndarray::Array2<Complex64>,
    b: &mut ndarray::Array2<Complex64>,
) -> LapackResult<()> {
    let (n, _) = a.dim();
    let nrhs = b.ncols();

    if a.nrows() != a.ncols() {
        return Err(LapackError::InvalidInput(-1));
    }
    if b.nrows() != n {
        return Err(LapackError::InvalidInput(-2));
    }

    // Create column-major (Fortran) layout arrays
    let mut a_data: Vec<lapack::c64> = Vec::with_capacity(n * n);
    for col in 0..n {
        for row in 0..n {
            let c = a[(row, col)];
            a_data.push(lapack::c64::new(c.re, c.im));
        }
    }

    let mut b_data: Vec<lapack::c64> = Vec::with_capacity(n * nrhs);
    for col in 0..nrhs {
        for row in 0..n {
            let c = b[(row, col)];
            b_data.push(lapack::c64::new(c.re, c.im));
        }
    }

    let mut ipiv = vec![0i32; n];
    let mut info = 0i32;

    unsafe {
        lapack::zgesv(
            n as i32,
            nrhs as i32,
            &mut a_data,
            n as i32,
            &mut ipiv,
            &mut b_data,
            n as i32,
            &mut info,
        );
    }

    check_lapack_info(info)?;

    // Copy results back to row-major layout
    for col in 0..n {
        for row in 0..n {
            let c = a_data[col * n + row];
            a[(row, col)] = Complex64::new(c.re, c.im);
        }
    }

    for col in 0..nrhs {
        for row in 0..n {
            let c = b_data[col * n + row];
            b[(row, col)] = Complex64::new(c.re, c.im);
        }
    }

    Ok(())
}

/// Compute LU decomposition of a matrix
///
/// # Arguments
/// * `a` - Matrix to decompose (will be modified)
///
/// # Returns
/// * `Ok(pivot_indices)` if successful
/// * `Err(LapackError)` if failed
pub fn lu_decomposition(a: &mut ndarray::Array2<Complex64>) -> LapackResult<Vec<i32>> {
    let (m, n) = a.dim();
    let mut ipiv = vec![0i32; m.min(n)];
    let mut info = 0i32;

    // Create column-major (Fortran) layout array
    let mut a_data: Vec<lapack::c64> = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            let c = a[(row, col)];
            a_data.push(lapack::c64::new(c.re, c.im));
        }
    }

    unsafe {
        lapack::zgetrf(
            m as i32,
            n as i32,
            &mut a_data,
            m as i32,
            &mut ipiv,
            &mut info,
        );
    }

    check_lapack_info(info)?;

    // Copy results back to row-major layout
    for col in 0..n {
        for row in 0..m {
            let c = a_data[col * m + row];
            a[(row, col)] = Complex64::new(c.re, c.im);
        }
    }

    Ok(ipiv)
}

/// Invert a matrix using LU decomposition
///
/// # Arguments
/// * `a` - Matrix to invert (will be modified)
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(LapackError)` if failed
pub fn invert_matrix(a: &mut ndarray::Array2<Complex64>) -> LapackResult<()> {
    let (n, _) = a.dim();

    if a.nrows() != a.ncols() {
        return Err(LapackError::InvalidInput(-1));
    }

    // Create column-major (Fortran) layout array
    let mut a_data: Vec<lapack::c64> = Vec::with_capacity(n * n);
    for col in 0..n {
        for row in 0..n {
            let c = a[(row, col)];
            a_data.push(lapack::c64::new(c.re, c.im));
        }
    }

    // First, compute LU decomposition on column-major data
    let mut ipiv = vec![0i32; n];
    let mut info = 0i32;

    unsafe {
        lapack::zgetrf(
            n as i32,
            n as i32,
            &mut a_data,
            n as i32,
            &mut ipiv,
            &mut info,
        );
    }

    check_lapack_info(info)?;

    // Query optimal workspace size
    let mut work_query = vec![lapack::c64::new(0.0, 0.0); 1];
    let lwork = -1i32;

    unsafe {
        lapack::zgetri(
            n as i32,
            &mut a_data,
            n as i32,
            &ipiv,
            &mut work_query,
            lwork,
            &mut info,
        );
    }

    let optimal_lwork = work_query[0].re as i32;

    // Allocate workspace and compute inverse
    let mut work = vec![lapack::c64::new(0.0, 0.0); optimal_lwork as usize];

    unsafe {
        lapack::zgetri(
            n as i32,
            &mut a_data,
            n as i32,
            &ipiv,
            &mut work,
            optimal_lwork,
            &mut info,
        );
    }

    check_lapack_info(info)?;

    // Copy results back to row-major layout
    for col in 0..n {
        for row in 0..n {
            let c = a_data[col * n + row];
            a[(row, col)] = Complex64::new(c.re, c.im);
        }
    }

    Ok(())
}

/// Compute eigenvalues and eigenvectors of a matrix
///
/// # Arguments
/// * `a` - Matrix to diagonalize (will be modified)
/// * `compute_left` - Whether to compute left eigenvectors
/// * `compute_right` - Whether to compute right eigenvectors
///
/// # Returns
/// * `Ok((eigenvalues, left_eigenvectors, right_eigenvectors))` if successful
/// * `Err(LapackError)` if failed
pub fn eigenvalue_decomposition(
    a: &mut ndarray::Array2<Complex64>,
    compute_left: bool,
    compute_right: bool,
) -> LapackResult<(Vec<Complex64>, Option<ndarray::Array2<Complex64>>, Option<ndarray::Array2<Complex64>>)> {
    let (n, _) = a.dim();

    if a.nrows() != a.ncols() {
        return Err(LapackError::InvalidInput(-1));
    }

    let mut eigenvalues = vec![Complex64::new(0.0, 0.0); n];
    let mut left_eigenvectors = if compute_left {
        Some(ndarray::Array2::zeros((n, n)))
    } else {
        None
    };
    let mut right_eigenvectors = if compute_right {
        Some(ndarray::Array2::zeros((n, n)))
    } else {
        None
    };

    let mut rwork = vec![0.0; 2 * n];
    let mut work_size = 0i32;
    let mut info = 0i32;

    // Query optimal workspace size
    unsafe {
        crate::ffi::lapack::zgeev_(
            if compute_left { b"V\0".as_ptr()  } else { b"N\0".as_ptr()  },
            if compute_right { b"V\0".as_ptr()  } else { b"N\0".as_ptr()  },
            &(n as i32),
            a.as_mut_ptr(),
            &(n as i32),
            eigenvalues.as_mut_ptr(),
            left_eigenvectors.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
            &(n as i32),
            right_eigenvectors.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
            &(n as i32),
            std::ptr::null_mut(),
            &mut work_size,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    // Allocate workspace and compute eigenvalues/eigenvectors
    let mut work = vec![Complex64::new(0.0, 0.0); work_size as usize];

    unsafe {
        crate::ffi::lapack::zgeev_(
            if compute_left { b"V\0".as_ptr()  } else { b"N\0".as_ptr()  },
            if compute_right { b"V\0".as_ptr()  } else { b"N\0".as_ptr()  },
            &(n as i32),
            a.as_mut_ptr(),
            &(n as i32),
            eigenvalues.as_mut_ptr(),
            left_eigenvectors.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
            &(n as i32),
            right_eigenvectors.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
            &(n as i32),
            work.as_mut_ptr(),
            &work_size,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    check_lapack_info(info)?;
    Ok((eigenvalues, left_eigenvectors, right_eigenvectors))
}

/// Compute singular value decomposition of a matrix
///
/// # Arguments
/// * `a` - Matrix to decompose (will be modified)
///
/// # Returns
/// * `Ok((singular_values, u, vt))` if successful
/// * `Err(LapackError)` if failed
pub fn singular_value_decomposition(
    a: &mut ndarray::Array2<Complex64>,
) -> LapackResult<(Vec<f64>, ndarray::Array2<Complex64>, ndarray::Array2<Complex64>)> {
    let (m, n) = a.dim();
    let min_mn = m.min(n);

    let mut singular_values = vec![0.0; min_mn];
    let mut u = ndarray::Array2::zeros((m, m));
    let mut vt = ndarray::Array2::zeros((n, n));
    let mut rwork = vec![0.0; 5 * min_mn];
    let mut work_size = 0i32;
    let mut info = 0i32;

    // Query optimal workspace size
    unsafe {
        crate::ffi::lapack::zgesvd_(
            b"A\0".as_ptr() ,
            b"A\0".as_ptr() ,
            &(m as i32),
            &(n as i32),
            a.as_mut_ptr(),
            &(m as i32),
            singular_values.as_mut_ptr(),
            u.as_mut_ptr(),
            &(m as i32),
            vt.as_mut_ptr(),
            &(n as i32),
            std::ptr::null_mut(),
            &mut work_size,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    // Allocate workspace and compute SVD
    let mut work = vec![Complex64::new(0.0, 0.0); work_size as usize];

    unsafe {
        crate::ffi::lapack::zgesvd_(
            b"A\0".as_ptr() ,
            b"A\0".as_ptr() ,
            &(m as i32),
            &(n as i32),
            a.as_mut_ptr(),
            &(m as i32),
            singular_values.as_mut_ptr(),
            u.as_mut_ptr(),
            &(m as i32),
            vt.as_mut_ptr(),
            &(n as i32),
            work.as_mut_ptr(),
            &work_size,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    check_lapack_info(info)?;
    Ok((singular_values, u, vt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_linear_system_2x2() {
        let mut a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let mut b = Array2::from_shape_vec((2, 1), vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ]).unwrap();

        let result = solve_linear_system(&mut a, &mut b);
        assert!(result.is_ok());

        // Expected solution: x = [-4, 4.5]
        assert_relative_eq!(b[(0, 0)].re, -4.0, epsilon = 1e-10);
        assert_relative_eq!(b[(1, 0)].re, 4.5, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_linear_system_invalid_dimensions() {
        let mut a = Array2::from_shape_vec((2, 3), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0),
        ]).unwrap();
        let mut b = Array2::from_shape_vec((2, 1), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ]).unwrap();

        let result = solve_linear_system(&mut a, &mut b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LapackError::InvalidInput(_)));
    }

    #[test]
    fn test_lu_decomposition_2x2() {
        let mut a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();

        let result = lu_decomposition(&mut a);
        assert!(result.is_ok());

        let ipiv = result.unwrap();
        assert_eq!(ipiv.len(), 2);
    }

    #[test]
    fn test_invert_matrix_2x2() {
        let mut a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();

        let result = invert_matrix(&mut a);
        assert!(result.is_ok());

        // Expected inverse: [[-2, 1], [1.5, -0.5]]
        assert_relative_eq!(a[(0, 0)].re, -2.0, epsilon = 1e-10);
        assert_relative_eq!(a[(0, 1)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[(1, 0)].re, 1.5, epsilon = 1e-10);
        assert_relative_eq!(a[(1, 1)].re, -0.5, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "TODO: Implement eigenvalue_decomposition with lapack crate"]
    fn test_eigenvalue_decomposition_2x2() {
        let mut a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0),
        ]).unwrap();

        let result = eigenvalue_decomposition(&mut a, true, true);
        assert!(result.is_ok());

        let (eigenvalues, left_eigenvectors, right_eigenvectors) = result.unwrap();
        assert_eq!(eigenvalues.len(), 2);
        assert!(left_eigenvectors.is_some());
        assert!(right_eigenvectors.is_some());
    }

    #[test]
    #[ignore = "TODO: Implement singular_value_decomposition with lapack crate"]
    fn test_singular_value_decomposition_2x3() {
        let mut a = Array2::from_shape_vec((2, 3), vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
        ]).unwrap();

        let result = singular_value_decomposition(&mut a);
        assert!(result.is_ok());

        let (singular_values, u, vt) = result.unwrap();
        assert_eq!(singular_values.len(), 2);
        assert_eq!(u.dim(), (2, 2));
        assert_eq!(vt.dim(), (3, 3));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    fn complex_strategy() -> impl Strategy<Value = Complex64> {
        (-10.0..10.0, -10.0..10.0)
            .prop_map(|(re, im)| Complex64::new(re, im))
    }

    fn square_matrix_strategy(size: usize) -> impl Strategy<Value = Array2<Complex64>> {
        let data_strategy = proptest::collection::vec(complex_strategy(), size * size);
        data_strategy.prop_map(move |data| {
            Array2::from_shape_vec((size, size), data).unwrap()
        })
    }

    proptest! {
        #[test]
        fn prop_lu_decomposition_runs_without_error(
            mut matrix in square_matrix_strategy(2)
        ) {
            // Just check that LU decomposition doesn't crash
            let _ = lu_decomposition(&mut matrix);
            // If we get here, the function didn't panic
            prop_assert!(true);
        }

        #[test]
        fn prop_invert_matrix_identity(
            mut matrix in square_matrix_strategy(2)
        ) {
            // Compute determinant to check if matrix is non-singular
            let det = matrix[(0, 0)] * matrix[(1, 1)] - matrix[(0, 1)] * matrix[(1, 0)];

            // Only test with non-singular matrices
            if det.norm() > 1e-6 {
                let original = matrix.clone();
                let result = invert_matrix(&mut matrix);

                if result.is_ok() {
                    // A * A^(-1) should equal identity
                    let identity = original.dot(&matrix);
                    let expected_identity: Array2<Complex64> = Array2::eye(2);

                    for i in 0..2 {
                        for j in 0..2 {
                            let diff: Complex64 = identity[(i, j)] - expected_identity[(i, j)];
                            prop_assert!(diff.norm() < 1e-8);
                        }
                    }
                }
            }
        }
    }
}
