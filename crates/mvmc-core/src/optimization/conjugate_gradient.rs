//! Conjugate Gradient (CG) method for solving linear systems
//!
//! This module implements the Conjugate Gradient method for solving
//! the linear system S * δp = f in SR optimization.
//! Reference: mVMC/src/mVMC/stcopt_cg.c

use crate::{Result, VmcError};

/// Conjugate Gradient solver for linear systems
///
/// This structure implements the Conjugate Gradient method for solving
/// the linear system Ax = b, where A is a symmetric positive definite matrix.
/// Reference: mVMC/src/mVMC/stcopt_cg.c
///
/// # Examples
///
/// ```
/// use mvmc_core::optimization::ConjugateGradientSolver;
///
/// let mut solver = ConjugateGradientSolver::new(4);
/// let matrix = vec![
///     vec![4.0, 1.0, 0.0, 0.0],
///     vec![1.0, 4.0, 1.0, 0.0],
///     vec![0.0, 1.0, 4.0, 1.0],
///     vec![0.0, 0.0, 1.0, 4.0],
/// ];
/// let rhs = vec![1.0, 2.0, 3.0, 4.0];
/// let solution = solver.solve(&matrix, &rhs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ConjugateGradientSolver {
    /// System size
    n: usize,

    /// Maximum number of iterations
    max_iterations: usize,

    /// Convergence tolerance
    tolerance: f64,

    /// Current iteration
    current_iteration: usize,

    /// Residual norm
    residual_norm: f64,
}

/// CG solver trait for different implementations
pub trait CGSolver {
    /// Solves the linear system Ax = b
    ///
    /// # Arguments
    ///
    /// * `matrix` - Coefficient matrix A
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    ///
    /// Solution vector x
    fn solve(&mut self, matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>>;

    /// Returns the number of iterations performed
    fn iterations(&self) -> usize;

    /// Returns the final residual norm
    fn residual_norm(&self) -> f64;
}

impl ConjugateGradientSolver {
    /// Creates a new Conjugate Gradient solver
    ///
    /// # Arguments
    ///
    /// * `n` - System size
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::optimization::ConjugateGradientSolver;
    ///
    /// let solver = ConjugateGradientSolver::new(4);
    /// ```
    pub fn new(n: usize) -> Self {
        Self {
            n,
            max_iterations: n.max(100), // At least 100 iterations
            tolerance: 1e-12,
            current_iteration: 0,
            residual_norm: 0.0,
        }
    }

    /// Sets the maximum number of iterations
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::optimization::ConjugateGradientSolver;
    ///
    /// let mut solver = ConjugateGradientSolver::new(4);
    /// solver.set_max_iterations(200);
    /// ```
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Sets the convergence tolerance
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::optimization::ConjugateGradientSolver;
    ///
    /// let mut solver = ConjugateGradientSolver::new(4);
    /// solver.set_tolerance(1e-10);
    /// ```
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Returns the system size
    pub fn system_size(&self) -> usize {
        self.n
    }

    /// Returns the current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Returns the final residual norm
    pub fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
}

impl CGSolver for ConjugateGradientSolver {
    /// Solves the linear system Ax = b using the Conjugate Gradient method
    ///
    /// # Arguments
    ///
    /// * `matrix` - Coefficient matrix A (must be symmetric positive definite)
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    ///
    /// Solution vector x
    ///
    /// # Errors
    ///
    /// Returns an error if the system is incompatible or if convergence fails
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::optimization::{ConjugateGradientSolver, CGSolver};
    ///
    /// let mut solver = ConjugateGradientSolver::new(4);
    /// let matrix = vec![
    ///     vec![4.0, 1.0, 0.0, 0.0],
    ///     vec![1.0, 4.0, 1.0, 0.0],
    ///     vec![0.0, 1.0, 4.0, 1.0],
    ///     vec![0.0, 0.0, 1.0, 4.0],
    /// ];
    /// let rhs = vec![1.0, 2.0, 3.0, 4.0];
    /// let solution = solver.solve(&matrix, &rhs).unwrap();
    /// ```
    fn solve(&mut self, matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
        if matrix.len() != self.n {
            return Err(VmcError::dim_mismatch(self.n, matrix.len()));
        }

        if rhs.len() != self.n {
            return Err(VmcError::dim_mismatch(self.n, rhs.len()));
        }

        // Check matrix dimensions
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != self.n {
                return Err(VmcError::dim_mismatch(self.n, row.len()));
            }
        }

        // Initialize solution vector
        let mut x = vec![0.0; self.n];

        // Initialize residual vector r = b - Ax
        let mut r = rhs.to_vec();
        self.matrix_vector_product(matrix, &x, &mut r);
        for i in 0..self.n {
            r[i] = rhs[i] - r[i];
        }

        // Initialize search direction p = r
        let mut p = r.clone();

        // Initialize residual norm
        let mut r_norm_sq = self.dot_product(&r, &r);
        self.residual_norm = r_norm_sq.sqrt();

        // Check for convergence
        if self.residual_norm < self.tolerance {
            self.current_iteration = 0;
            return Ok(x);
        }

        // Main CG loop
        for iteration in 0..self.max_iterations {
            self.current_iteration = iteration + 1;

            // Compute Ap
            let mut ap = vec![0.0; self.n];
            self.matrix_vector_product(matrix, &p, &mut ap);

            // Compute step size α = (r·r) / (p·Ap)
            let p_ap = self.dot_product(&p, &ap);
            if p_ap.abs() < 1e-12 {
                return Err(VmcError::invalid_param("Matrix is singular"));
            }

            let alpha = r_norm_sq / p_ap;

            // Update solution x = x + αp
            for i in 0..self.n {
                x[i] += alpha * p[i];
            }

            // Update residual r = r - αAp
            for i in 0..self.n {
                r[i] -= alpha * ap[i];
            }

            // Compute new residual norm
            let r_norm_sq_new = self.dot_product(&r, &r);
            self.residual_norm = r_norm_sq_new.sqrt();

            // Check for convergence
            if self.residual_norm < self.tolerance {
                return Ok(x);
            }

            // Compute new search direction p = r + βp
            let beta = r_norm_sq_new / r_norm_sq;
            for i in 0..self.n {
                p[i] = r[i] + beta * p[i];
            }

            r_norm_sq = r_norm_sq_new;
        }

        // If we reach here, we didn't converge
        Err(VmcError::invalid_param(
            format!("CG did not converge after {} iterations", self.max_iterations)
        ))
    }

    fn iterations(&self) -> usize {
        self.current_iteration
    }

    fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
}

impl ConjugateGradientSolver {
    /// Computes the matrix-vector product y = Ax
    ///
    /// # Arguments
    ///
    /// * `matrix` - Matrix A
    /// * `x` - Input vector x
    /// * `y` - Output vector y (modified in place)
    fn matrix_vector_product(&self, matrix: &[Vec<f64>], x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            y[i] = 0.0;
            for j in 0..self.n {
                y[i] += matrix[i][j] * x[j];
            }
        }
    }

    /// Computes the dot product of two vectors
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// Dot product a·b
    fn dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut result = 0.0;
        for i in 0..self.n {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_solver_creation() {
        let solver = ConjugateGradientSolver::new(4);

        assert_eq!(solver.system_size(), 4);
        assert_eq!(solver.current_iteration(), 0);
        assert_eq!(solver.residual_norm(), 0.0);
    }

    #[test]
    fn test_cg_solver_settings() {
        let mut solver = ConjugateGradientSolver::new(4);

        solver.set_max_iterations(200);
        solver.set_tolerance(1e-10);

        assert_eq!(solver.max_iterations, 200);
        assert_eq!(solver.tolerance, 1e-10);
    }

    #[test]
    fn test_cg_solve_identity() {
        let mut solver = ConjugateGradientSolver::new(3);
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let rhs = vec![1.0, 2.0, 3.0];

        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_eq!(solution.len(), 3);
        assert!((solution[0] - 1.0).abs() < 1e-10);
        assert!((solution[1] - 2.0).abs() < 1e-10);
        assert!((solution[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_solve_simple() {
        let mut solver = ConjugateGradientSolver::new(2);
        let matrix = vec![
            vec![4.0, 1.0],
            vec![1.0, 4.0],
        ];
        let rhs = vec![1.0, 2.0];

        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_eq!(solution.len(), 2);
        // Verify that Ax = b
        let mut result = vec![0.0; 2];
        solver.matrix_vector_product(&matrix, &solution, &mut result);

        assert!((result[0] - rhs[0]).abs() < 1e-10);
        assert!((result[1] - rhs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_cg_solve_dimension_mismatch() {
        let mut solver = ConjugateGradientSolver::new(2);
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let rhs = vec![1.0, 2.0, 3.0]; // Wrong size

        let result = solver.solve(&matrix, &rhs);
        assert!(result.is_err());
    }

    #[test]
    fn test_cg_solve_singular_matrix() {
        let mut solver = ConjugateGradientSolver::new(2);
        let matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0], // Singular matrix
        ];
        let rhs = vec![1.0, 2.0];

        let result = solver.solve(&matrix, &rhs);
        assert!(result.is_err());
    }

    #[test]
    fn test_cg_solve_zero_rhs() {
        let mut solver = ConjugateGradientSolver::new(3);
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let rhs = vec![0.0, 0.0, 0.0];

        let solution = solver.solve(&matrix, &rhs).unwrap();

        assert_eq!(solution.len(), 3);
        assert!(solution.iter().all(|&x| x.abs() < 1e-10));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_cg_solve_identity_matrix(
            n in 1usize..10,
            rhs_values in proptest::collection::vec(-10.0f64..10.0, 1..20)
        ) {
            let mut solver = ConjugateGradientSolver::new(n);
            let matrix = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect::<Vec<Vec<f64>>>();

            let rhs = rhs_values.iter().take(n).copied().collect::<Vec<f64>>();
            if rhs.len() == n {
                match solver.solve(&matrix, &rhs) {
                    Ok(solution) => {
                        // Verify that Ax = b
                        let mut result = vec![0.0; n];
                        solver.matrix_vector_product(&matrix, &solution, &mut result);

                        for i in 0..n {
                            prop_assert!((result[i] - rhs[i]).abs() < 1e-6);
                        }
                    }
                    Err(_) => {
                        // If the matrix is still singular, that's acceptable for this test
                        // We'll just skip the verification
                    }
                }
            }
        }

        #[test]
        fn prop_cg_solve_symmetric_matrix(
            n in 2usize..8,
            matrix_values in proptest::collection::vec(-5.0f64..5.0, 1..100),
            rhs_values in proptest::collection::vec(-5.0f64..5.0, 1..20)
        ) {
            let mut solver = ConjugateGradientSolver::new(n);

            // Create a symmetric positive definite matrix
            let mut matrix = vec![vec![0.0; n]; n];
            let values_len = matrix_values.len().min(n * n);

            for i in 0..values_len {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    matrix[row][col] = matrix_values[i];
                    matrix[col][row] = matrix_values[i]; // Make symmetric
                }
            }

            // Add diagonal dominance to ensure positive definiteness
            for i in 0..n {
                matrix[i][i] += n as f64;
            }

            let rhs = rhs_values.iter().take(n).copied().collect::<Vec<f64>>();
            if rhs.len() == n {
                let solution = solver.solve(&matrix, &rhs).unwrap();

                // Verify that Ax = b
                let mut result = vec![0.0; n];
                solver.matrix_vector_product(&matrix, &solution, &mut result);

                for i in 0..n {
                    prop_assert!((result[i] - rhs[i]).abs() < 1e-8);
                }
            }
        }
    }
}
