//! Linear algebra operations for quantum physics calculations
//!
//! This module provides utilities for matrix operations commonly used
//! in variational Monte Carlo calculations.

use ndarray::Array2;
use num_complex::Complex64;

/// A complex matrix wrapper with additional functionality
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexMatrix {
    data: Array2<Complex64>,
}

impl ComplexMatrix {
    /// Create a new complex matrix from a 2D array
    pub fn new(data: Array2<Complex64>) -> Self {
        Self { data }
    }

    /// Create a zero matrix of the given dimensions
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::zeros((rows, cols)),
        }
    }

    /// Create an identity matrix of the given size
    pub fn identity(size: usize) -> Self {
        let mut data = Array2::zeros((size, size));
        for i in 0..size {
            data[(i, i)] = Complex64::new(1.0, 0.0);
        }
        Self { data }
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Get the shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &Array2<Complex64> {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut Array2<Complex64> {
        &mut self.data
    }

    /// Get the element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[(row, col)]
    }

    /// Set the element at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[(row, col)] = value;
    }

    /// Compute the trace of the matrix
    pub fn trace(&self) -> Complex64 {
        if self.rows() != self.cols() {
            panic!("Trace is only defined for square matrices");
        }
        (0..self.rows()).map(|i| self.get(i, i)).sum()
    }

    /// Compute the determinant of the matrix
    pub fn det(&self) -> Complex64 {
        if self.rows() != self.cols() {
            panic!("Determinant is only defined for square matrices");
        }
        if self.rows() == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if self.rows() == 1 {
            return self.get(0, 0);
        }
        if self.rows() == 2 {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }
        // For larger matrices, use LU decomposition
        self.det_lu()
    }

    /// Compute determinant using LU decomposition
    fn det_lu(&self) -> Complex64 {
        let (_l, u, p) = self.lu_decomposition();
        let mut det = Complex64::new(1.0, 0.0);

        // Determinant of L is 1 (unit triangular)
        // Determinant of U is product of diagonal elements
        for i in 0..u.rows() {
            det = det * u.get(i, i);
        }

        // Determinant of P is (-1)^(number of swaps)
        let swaps = p.iter().enumerate().filter(|(i, j)| *i != **j).count();
        if swaps % 2 == 1 {
            det = -det;
        }

        det
    }

    /// Perform LU decomposition: A = P * L * U
    /// Returns (L, U, P) where P is a permutation vector
    pub fn lu_decomposition(&self) -> (ComplexMatrix, ComplexMatrix, Vec<usize>) {
        if self.rows() != self.cols() {
            panic!("LU decomposition is only defined for square matrices");
        }

        let n = self.rows();
        let mut l = ComplexMatrix::identity(n);
        let mut u = self.clone();
        let mut p: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            let mut max_val = u.get(k, k).norm();

            for i in (k + 1)..n {
                let val = u.get(i, k).norm();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows if necessary
            if max_row != k {
                u.swap_rows(k, max_row);
                if k > 0 {
                    l.swap_rows(k, max_row);
                }
                p.swap(k, max_row);
            }

            // Perform elimination
            let pivot = u.get(k, k);
            if pivot.norm() < 1e-12 {
                panic!("Matrix is singular");
            }

            for i in (k + 1)..n {
                let factor = u.get(i, k) / pivot;
                l.set(i, k, factor);

                for j in k..n {
                    let new_val = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, new_val);
                }
            }
        }

        (l, u, p)
    }

    /// Swap two rows of the matrix
    fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.cols() {
            let temp = self.get(i, k);
            self.set(i, k, self.get(j, k));
            self.set(j, k, temp);
        }
    }

    /// Compute the matrix transpose
    pub fn transpose(&self) -> ComplexMatrix {
        let mut result = ComplexMatrix::zeros(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Compute the conjugate transpose (Hermitian transpose)
    pub fn hermitian_transpose(&self) -> ComplexMatrix {
        let mut result = ComplexMatrix::zeros(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(j, i, self.get(i, j).conj());
            }
        }
        result
    }

    /// Check if the matrix is Hermitian (A = A†)
    pub fn is_hermitian(&self) -> bool {
        if self.rows() != self.cols() {
            return false;
        }
        let hermitian = self.hermitian_transpose();
        self.approx_eq(&hermitian, 1e-10)
    }

    /// Check if two matrices are approximately equal
    pub fn approx_eq(&self, other: &ComplexMatrix, epsilon: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let diff = self.get(i, j) - other.get(i, j);
                if diff.norm() > epsilon {
                    return false;
                }
            }
        }
        true
    }
}

impl std::ops::Add for ComplexMatrix {
    type Output = ComplexMatrix;

    fn add(self, other: ComplexMatrix) -> ComplexMatrix {
        if self.shape() != other.shape() {
            panic!("Matrix dimensions must match for addition");
        }
        ComplexMatrix::new(self.data + other.data)
    }
}

impl std::ops::Sub for ComplexMatrix {
    type Output = ComplexMatrix;

    fn sub(self, other: ComplexMatrix) -> ComplexMatrix {
        if self.shape() != other.shape() {
            panic!("Matrix dimensions must match for subtraction");
        }
        ComplexMatrix::new(self.data - other.data)
    }
}

impl std::ops::Mul for ComplexMatrix {
    type Output = ComplexMatrix;

    fn mul(self, other: ComplexMatrix) -> ComplexMatrix {
        if self.cols() != other.rows() {
            panic!("Matrix dimensions must be compatible for multiplication");
        }
        ComplexMatrix::new(self.data.dot(&other.data))
    }
}

impl std::ops::Mul<Complex64> for ComplexMatrix {
    type Output = ComplexMatrix;

    fn mul(self, scalar: Complex64) -> ComplexMatrix {
        ComplexMatrix::new(self.data * scalar)
    }
}

impl std::ops::Mul<ComplexMatrix> for Complex64 {
    type Output = ComplexMatrix;

    fn mul(self, matrix: ComplexMatrix) -> ComplexMatrix {
        matrix * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_matrix_creation() {
        let data = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.shape(), (2, 2));
    }

    #[test]
    fn test_zeros_matrix() {
        let matrix = ComplexMatrix::zeros(3, 4);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(matrix.get(i, j), Complex64::new(0.0, 0.0));
            }
        }
    }

    #[test]
    fn test_identity_matrix() {
        let matrix = ComplexMatrix::identity(3);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(matrix.get(i, j), Complex64::new(1.0, 0.0));
                } else {
                    assert_eq!(matrix.get(i, j), Complex64::new(0.0, 0.0));
                }
            }
        }
    }

    #[test]
    fn test_trace() {
        let data = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);

        let trace = matrix.trace();
        assert_relative_eq!(trace.re, 5.0, epsilon = 1e-10);
        assert_relative_eq!(trace.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_2x2() {
        let data = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);

        let det = matrix.det();
        assert_relative_eq!(det.re, -2.0, epsilon = 1e-10);
        assert_relative_eq!(det.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_1x1() {
        let data = Array2::from_shape_vec((1, 1), vec![Complex64::new(3.0, 4.0)]).unwrap();
        let matrix = ComplexMatrix::new(data);

        let det = matrix.det();
        assert_relative_eq!(det.re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(det.im, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transpose() {
        let data = Array2::from_shape_vec((2, 3), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0), Complex64::new(3.0, 0.0),
            Complex64::new(4.0, -1.0), Complex64::new(5.0, 0.0), Complex64::new(6.0, 1.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);
        let transposed = matrix.transpose();

        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(matrix.get(i, j), transposed.get(j, i));
            }
        }
    }

    #[test]
    fn test_hermitian_transpose() {
        let data = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);
        let hermitian = matrix.hermitian_transpose();

        assert_eq!(hermitian.rows(), 2);
        assert_eq!(hermitian.cols(), 2);

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(hermitian.get(i, j), matrix.get(j, i).conj());
            }
        }
    }

    #[test]
    fn test_matrix_addition() {
        let data1 = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let data2 = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0),
        ]).unwrap();

        let matrix1 = ComplexMatrix::new(data1);
        let matrix2 = ComplexMatrix::new(data2);
        let result = matrix1 + matrix2;

        // Expected result: [[6, 8], [10, 12]]
        assert_eq!(result.get(0, 0), Complex64::new(6.0, 0.0));
        assert_eq!(result.get(0, 1), Complex64::new(8.0, 0.0));
        assert_eq!(result.get(1, 0), Complex64::new(10.0, 0.0));
        assert_eq!(result.get(1, 1), Complex64::new(12.0, 0.0));
    }

    #[test]
    fn test_matrix_multiplication() {
        let data1 = Array2::from_shape_vec((2, 3), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0),
        ]).unwrap();
        let data2 = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0),
            Complex64::new(9.0, 0.0), Complex64::new(10.0, 0.0),
            Complex64::new(11.0, 0.0), Complex64::new(12.0, 0.0),
        ]).unwrap();

        let matrix1 = ComplexMatrix::new(data1);
        let matrix2 = ComplexMatrix::new(data2);
        let result = matrix1 * matrix2;

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);

        // Expected result: [[58, 64], [139, 154]]
        assert_relative_eq!(result.get(0, 0).re, 58.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).re, 64.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).re, 139.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 1).re, 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scalar_multiplication() {
        let data = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);
        let scalar = Complex64::new(2.0, 1.0);
        let result = matrix.clone() * scalar;

        for i in 0..2 {
            for j in 0..2 {
                let expected = matrix.get(i, j) * scalar;
                assert_eq!(result.get(i, j), expected);
            }
        }
    }

    #[test]
    fn test_lu_decomposition() {
        let data = Array2::from_shape_vec((3, 3), vec![
            Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
        ]).unwrap();
        let matrix = ComplexMatrix::new(data);
        let (l, u, p) = matrix.lu_decomposition();

        // Reconstruct the matrix: P * L * U should equal original
        let mut reconstructed = ComplexMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    sum = sum + l.get(i, k) * u.get(k, j);
                }
                reconstructed.set(p[i], j, sum);
            }
        }

        assert!(matrix.approx_eq(&reconstructed, 1e-10));
    }

    #[test]
    fn test_is_hermitian() {
        // Hermitian matrix
        let data1 = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(1.0, 1.0),
            Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0),
        ]).unwrap();
        let matrix1 = ComplexMatrix::new(data1);
        assert!(matrix1.is_hermitian());

        // Non-Hermitian matrix
        let data2 = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let matrix2 = ComplexMatrix::new(data2);
        assert!(!matrix2.is_hermitian());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn complex_strategy() -> impl Strategy<Value = Complex64> {
        (-10.0..10.0, -10.0..10.0)
            .prop_map(|(re, im)| Complex64::new(re, im))
    }

    fn square_matrix_strategy(size: usize) -> impl Strategy<Value = ComplexMatrix> {
        let data_strategy = proptest::collection::vec(complex_strategy(), size * size);
        data_strategy.prop_map(move |data| {
            let array = Array2::from_shape_vec((size, size), data).unwrap();
            ComplexMatrix::new(array)
        })
    }

    proptest! {
        #[test]
        fn prop_transpose_transpose_identity(
            matrix in square_matrix_strategy(3)
        ) {
            let double_transpose = matrix.transpose().transpose();
            prop_assert!(matrix.approx_eq(&double_transpose, 1e-10));
        }

        #[test]
        fn prop_hermitian_transpose_transpose_identity(
            matrix in square_matrix_strategy(3)
        ) {
            let double_hermitian = matrix.hermitian_transpose().hermitian_transpose();
            prop_assert!(matrix.approx_eq(&double_hermitian, 1e-10));
        }

        #[test]
        fn prop_matrix_addition_commutative(
            a in square_matrix_strategy(2),
            b in square_matrix_strategy(2)
        ) {
            let result1 = a.clone() + b.clone();
            let result2 = b + a;
            prop_assert!(result1.approx_eq(&result2, 1e-10));
        }

        #[test]
        fn prop_matrix_addition_associative(
            a in square_matrix_strategy(2),
            b in square_matrix_strategy(2),
            c in square_matrix_strategy(2)
        ) {
            let result1 = (a.clone() + b.clone()) + c.clone();
            let result2 = a + (b + c);
            prop_assert!(result1.approx_eq(&result2, 1e-10));
        }

        #[test]
        fn prop_scalar_multiplication_distributive(
            matrix in square_matrix_strategy(2),
            scalar1 in complex_strategy(),
            scalar2 in complex_strategy()
        ) {
            let result1 = (scalar1 + scalar2) * matrix.clone();
            let result2 = scalar1 * matrix.clone() + scalar2 * matrix;
            prop_assert!(result1.approx_eq(&result2, 1e-10));
        }

        #[test]
        fn prop_trace_linear(
            a in square_matrix_strategy(3),
            b in square_matrix_strategy(3),
            scalar in complex_strategy()
        ) {
            let trace_a = a.trace();
            let trace_b = b.trace();
            let trace_sum = (a.clone() + b).trace();
            let trace_scaled = (scalar * a).trace();

            prop_assert!((trace_sum - (trace_a + trace_b)).norm() < 1e-8);
            prop_assert!((trace_scaled - scalar * trace_a).norm() < 1e-8);
        }
    }
}
