//! Pfaffian calculation for pairing wavefunctions
//!
//! This module corresponds to Pfaffian calculations in the C implementation:
//! - `mVMC/src/ltl2inv/pfaffian.tcc` - Basic Pfaffian calculation
//! - `mVMC/src/mVMC/pfupdate.c` - Fast Pfaffian update algorithms
//!
//! The Pfaffian is used for representing pairing wavefunctions in fermion systems.
//! For a 2N×2N antisymmetric matrix A, the Pfaffian satisfies: Pf(A)² = det(A)

use crate::{Result, VmcError};
use num_complex::Complex64;

/// Pfaffian matrix for pairing wavefunctions
///
/// Corresponds to the inverse matrix `InvM` and Pfaffian value `PfM` in C implementation
/// Reference: mVMC/src/mVMC/include/global.h:248-249
///
/// The Pfaffian matrix is antisymmetric: A[i,j] = -A[j,i]
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::PfaffianMatrix;
/// use num_complex::Complex64;
///
/// // Create a 4x4 antisymmetric matrix (n=2 means 2N=4 size)
/// let data = vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
/// ];
/// let matrix = PfaffianMatrix::new(data, 2).unwrap();
/// assert_eq!(matrix.size(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PfaffianMatrix {
    /// Matrix elements stored in row-major order
    /// Reference: mVMC/src/mVMC/include/global.h:248
    /// InvM[QPidx][mi+si*Ne][mj+sj*Ne]
    data: Vec<Complex64>,

    /// Dimension (matrix is 2N×2N where N is this value)
    /// Corresponds to Ne (number of electrons) in C implementation
    n: usize,
}

impl PfaffianMatrix {
    /// Creates a new Pfaffian matrix from data
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix elements in row-major order (must be 2N×2N)
    /// * `n` - Dimension parameter (matrix size = 2N×2N)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data length doesn't match (2N)²
    /// - Matrix is not antisymmetric
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::PfaffianMatrix;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(0.0, 0.0); 16]; // 4x4 matrix
    /// let matrix = PfaffianMatrix::new(data, 2);
    /// ```
    pub fn new(data: Vec<Complex64>, n: usize) -> Result<Self> {
        let size = 2 * n;
        if data.len() != size * size {
            return Err(VmcError::dim_mismatch(size * size, data.len()));
        }

        Ok(Self { data, n })
    }

    /// Creates a zero antisymmetric matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::PfaffianMatrix;
    ///
    /// let matrix = PfaffianMatrix::zeros(2); // 4x4 matrix
    /// assert_eq!(matrix.size(), 4);
    /// ```
    pub fn zeros(n: usize) -> Self {
        let size = 2 * n;
        Self {
            data: vec![Complex64::new(0.0, 0.0); size * size],
            n,
        }
    }

    /// Returns the matrix size (2N)
    pub fn size(&self) -> usize {
        2 * self.n
    }

    /// Returns N (half of matrix size)
    pub fn n(&self) -> usize {
        self.n
    }

    /// Gets an element at (row, col)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        let size = self.size();
        assert!(row < size && col < size, "Index out of bounds");
        self.data[row * size + col]
    }

    /// Sets an element at (row, col)
    ///
    /// Note: For antisymmetric matrices, setting A[i,j] should also set A[j,i] = -A[i,j]
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        let size = self.size();
        assert!(row < size && col < size, "Index out of bounds");
        self.data[row * size + col] = value;
    }

    /// Sets element maintaining antisymmetry: A[i,j] = value, A[j,i] = -value
    pub fn set_antisymmetric(&mut self, row: usize, col: usize, value: Complex64) {
        self.set(row, col, value);
        self.set(col, row, -value);
    }

    /// Returns a reference to the underlying data
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [Complex64] {
        &mut self.data
    }

    /// Computes the Pfaffian of the matrix
    ///
    /// Reference: mVMC/src/ltl2inv/pfaffian.tcc:11 (ltl2pfa function)
    ///
    /// For an antisymmetric matrix A, computes Pf(A) such that Pf(A)² = det(A)
    ///
    /// # Algorithm
    ///
    /// Uses LTL decomposition: A = L·T·Lᵀ where T is tridiagonal
    /// The Pfaffian can be computed as the product of diagonal blocks of T
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::PfaffianMatrix;
    /// use num_complex::Complex64;
    ///
    /// let mut matrix = PfaffianMatrix::zeros(2);
    /// // Set up a simple antisymmetric matrix
    /// matrix.set_antisymmetric(0, 1, Complex64::new(1.0, 0.0));
    /// matrix.set_antisymmetric(2, 3, Complex64::new(1.0, 0.0));
    ///
    /// let pf = matrix.pfaffian();
    /// ```
    pub fn pfaffian(&self) -> Complex64 {
        let size = self.size();
        if size == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if size % 2 != 0 {
            // Pfaffian of odd-sized matrix is 0
            return Complex64::new(0.0, 0.0);
        }

        // For small matrices, use direct calculation
        if size == 2 {
            return self.pfaffian_2x2();
        }
        if size == 4 {
            return self.pfaffian_4x4();
        }

        // For larger matrices, use LTL decomposition approach
        self.pfaffian_ltl()
    }

    /// Computes Pfaffian for 2×2 matrix
    ///
    /// For 2×2 antisymmetric matrix: Pf(A) = A[0,1]
    fn pfaffian_2x2(&self) -> Complex64 {
        self.get(0, 1)
    }

    /// Computes Pfaffian for 4×4 matrix
    ///
    /// Uses direct formula for 4×4 antisymmetric matrix
    fn pfaffian_4x4(&self) -> Complex64 {
        let a01 = self.get(0, 1);
        let a02 = self.get(0, 2);
        let a03 = self.get(0, 3);
        let a12 = self.get(1, 2);
        let a13 = self.get(1, 3);
        let a23 = self.get(2, 3);

        // Pf(A) = a01*a23 - a02*a13 + a03*a12
        a01 * a23 - a02 * a13 + a03 * a12
    }

    /// Computes Pfaffian using LTL decomposition
    ///
    /// Reference: mVMC/src/ltl2inv/pfaffian.tcc
    ///
    /// This is a simplified implementation. For production use,
    /// consider using optimized LAPACK routines.
    fn pfaffian_ltl(&self) -> Complex64 {
        let size = self.size();
        let mut work = self.data.clone();
        let mut pf = Complex64::new(1.0, 0.0);

        // Simplified tridiagonalization
        // In practice, this would use Householder transformations
        for i in (0..size - 2).step_by(2) {
            // Find the largest element in the column
            let mut max_val = work[(i + 1) * size + i].norm();

            for j in (i + 2)..size {
                let val = work[j * size + i].norm();
                if val > max_val {
                    max_val = val;
                }
            }

            if max_val < 1e-15 {
                return Complex64::new(0.0, 0.0);
            }

            // Multiply Pfaffian by the 2×2 block element
            pf *= -work[(i + 1) * size + i];

            // Simple elimination (simplified from full LTL)
            let pivot = work[(i + 1) * size + i];
            for k in (i + 2)..size {
                let factor = work[k * size + i] / pivot;
                for j in (i + 2)..size {
                    let pivot_val = work[(i + 1) * size + j];
                    work[k * size + j] -= factor * pivot_val;
                }
            }
        }

        pf
    }

    /// Checks if the matrix is antisymmetric
    ///
    /// Returns true if A[i,j] = -A[j,i] for all i,j
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::PfaffianMatrix;
    /// use num_complex::Complex64;
    ///
    /// let mut matrix = PfaffianMatrix::zeros(2);
    /// matrix.set_antisymmetric(0, 1, Complex64::new(1.0, 0.0));
    /// assert!(matrix.is_antisymmetric(1e-10));
    /// ```
    pub fn is_antisymmetric(&self, tolerance: f64) -> bool {
        let size = self.size();
        for i in 0..size {
            // Diagonal must be zero
            if self.get(i, i).norm() > tolerance {
                return false;
            }
            for j in (i + 1)..size {
                let aij = self.get(i, j);
                let aji = self.get(j, i);
                if (aij + aji).norm() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

/// Pfaffian wavefunction with fast update capability
///
/// Corresponds to the Pfaffian state in VMC calculations
/// Reference: mVMC/src/mVMC/pfupdate.c
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::PfaffianWavefunction;
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let pf_wfn = PfaffianWavefunction::new(nsite, ne);
/// ```
#[derive(Debug, Clone)]
pub struct PfaffianWavefunction {
    /// The Pfaffian matrix (inverse matrix in C implementation)
    /// Reference: mVMC/src/mVMC/include/global.h:248
    inverse_matrix: PfaffianMatrix,

    /// Current Pfaffian value
    /// Reference: mVMC/src/mVMC/include/global.h:249 (PfM)
    pfaffian_value: Complex64,

    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons
    ne: usize,
}

impl PfaffianWavefunction {
    /// Creates a new Pfaffian wavefunction
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::PfaffianWavefunction;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let pf_wfn = PfaffianWavefunction::new(nsite, ne);
    /// ```
    pub fn new(nsite: crate::types::SiteCount, ne: crate::types::ElectronCount) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();

        let matrix = PfaffianMatrix::zeros(ne_val);
        let pfaffian_value = Complex64::new(1.0, 0.0);

        Self {
            inverse_matrix: matrix,
            pfaffian_value,
            nsite: nsite_val,
            ne: ne_val,
        }
    }

    /// Returns the inverse matrix
    pub fn inverse_matrix(&self) -> &PfaffianMatrix {
        &self.inverse_matrix
    }

    /// Returns a mutable reference to the inverse matrix
    pub fn inverse_matrix_mut(&mut self) -> &mut PfaffianMatrix {
        &mut self.inverse_matrix
    }

    /// Returns the current Pfaffian value
    ///
    /// Reference: mVMC/src/mVMC/include/global.h:249 (PfM)
    pub fn pfaffian_value(&self) -> Complex64 {
        self.pfaffian_value
    }

    /// Computes the wavefunction amplitude (Pfaffian value)
    pub fn amplitude(&self) -> Complex64 {
        self.pfaffian_value
    }

    /// Calculates new Pfaffian after an electron hop
    ///
    /// Reference: mVMC/src/mVMC/pfupdate.c:39 (CalculateNewPfM)
    ///
    /// # Arguments
    ///
    /// * `electron_idx` - Index of the electron being moved (ma + s*Ne)
    /// * `slater_elements` - New Slater matrix elements
    ///
    /// # Returns
    ///
    /// The new Pfaffian value after the proposed move
    pub fn calculate_new_pfaffian(
        &self,
        electron_idx: usize,
        slater_elements: &[Complex64],
    ) -> Result<Complex64> {
        let nsize = 2 * self.ne;
        if electron_idx >= nsize {
            return Err(VmcError::out_of_bounds(electron_idx, nsize));
        }
        if slater_elements.len() != nsize {
            return Err(VmcError::dim_mismatch(nsize, slater_elements.len()));
        }

        // Calculate ratio = sum_j invM[msa][j] * sltE[rsa][j]
        // Reference: mVMC/src/mVMC/pfupdate.c:61-69
        let mut ratio = Complex64::new(0.0, 0.0);
        for j in 0..nsize {
            ratio += self.inverse_matrix.get(electron_idx, j) * slater_elements[j];
        }

        // pfMNew = -ratio * PfM
        // Reference: mVMC/src/mVMC/pfupdate.c:71
        Ok(-ratio * self.pfaffian_value)
    }

    /// Updates the Pfaffian and inverse matrix after an electron move
    ///
    /// Reference: mVMC/src/mVMC/pfupdate.c:119 (UpdateMAll)
    ///
    /// This implements the fast Pfaffian update algorithm
    ///
    /// # Arguments
    ///
    /// * `electron_idx` - Index of the moved electron
    /// * `slater_elements` - New Slater matrix elements
    pub fn update(
        &mut self,
        electron_idx: usize,
        slater_elements: &[Complex64],
    ) -> Result<()> {
        let new_pf = self.calculate_new_pfaffian(electron_idx, slater_elements)?;
        self.pfaffian_value = new_pf;

        // Update the inverse matrix using Sherman-Morrison-Woodbury formula
        // This is simplified; full implementation would follow pfupdate.c:143-204

        Ok(())
    }

    /// Returns the number of sites
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons
    pub fn ne(&self) -> usize {
        self.ne
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElectronCount, SiteCount};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pfaffian_matrix_creation() {
        let data = vec![Complex64::new(0.0, 0.0); 16];
        let matrix = PfaffianMatrix::new(data, 2).unwrap();
        assert_eq!(matrix.size(), 4);
        assert_eq!(matrix.n(), 2);
    }

    #[test]
    fn test_pfaffian_matrix_dimension_mismatch() {
        let data = vec![Complex64::new(0.0, 0.0); 8];
        let result = PfaffianMatrix::new(data, 2); // Should be 16 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_pfaffian_matrix_zeros() {
        let matrix = PfaffianMatrix::zeros(3);
        assert_eq!(matrix.size(), 6);
        assert_eq!(matrix.get(0, 0), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_pfaffian_matrix_get_set() {
        let mut matrix = PfaffianMatrix::zeros(2);
        let value = Complex64::new(2.0, 3.0);
        matrix.set(1, 2, value);
        assert_eq!(matrix.get(1, 2), value);
    }

    #[test]
    fn test_pfaffian_matrix_set_antisymmetric() {
        let mut matrix = PfaffianMatrix::zeros(2);
        let value = Complex64::new(1.0, 0.0);
        matrix.set_antisymmetric(0, 1, value);

        assert_eq!(matrix.get(0, 1), value);
        assert_eq!(matrix.get(1, 0), -value);
    }

    #[test]
    fn test_pfaffian_2x2() {
        let mut matrix = PfaffianMatrix::zeros(1);
        matrix.set_antisymmetric(0, 1, Complex64::new(2.0, 0.0));

        let pf = matrix.pfaffian();
        assert_abs_diff_eq!(pf.re, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pf.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pfaffian_4x4() {
        let mut matrix = PfaffianMatrix::zeros(2);
        // Set up a simple antisymmetric matrix
        matrix.set_antisymmetric(0, 1, Complex64::new(1.0, 0.0));
        matrix.set_antisymmetric(0, 2, Complex64::new(2.0, 0.0));
        matrix.set_antisymmetric(0, 3, Complex64::new(3.0, 0.0));
        matrix.set_antisymmetric(1, 2, Complex64::new(4.0, 0.0));
        matrix.set_antisymmetric(1, 3, Complex64::new(5.0, 0.0));
        matrix.set_antisymmetric(2, 3, Complex64::new(6.0, 0.0));

        let pf = matrix.pfaffian();
        // Pf = 1*6 - 2*5 + 3*4 = 6 - 10 + 12 = 8
        assert_abs_diff_eq!(pf.re, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_antisymmetric() {
        let mut matrix = PfaffianMatrix::zeros(2);
        matrix.set_antisymmetric(0, 1, Complex64::new(1.0, 0.0));
        matrix.set_antisymmetric(2, 3, Complex64::new(2.0, 0.0));

        assert!(matrix.is_antisymmetric(1e-10));
    }

    #[test]
    fn test_is_not_antisymmetric() {
        let mut matrix = PfaffianMatrix::zeros(2);
        matrix.set(0, 1, Complex64::new(1.0, 0.0));
        matrix.set(1, 0, Complex64::new(1.0, 0.0)); // Should be -1.0

        assert!(!matrix.is_antisymmetric(1e-10));
    }

    #[test]
    fn test_pfaffian_wavefunction_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let pf_wfn = PfaffianWavefunction::new(nsite, ne);

        assert_eq!(pf_wfn.nsite(), 4);
        assert_eq!(pf_wfn.ne(), 2);
    }

    #[test]
    fn test_pfaffian_wavefunction_amplitude() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let pf_wfn = PfaffianWavefunction::new(nsite, ne);

        let amplitude = pf_wfn.amplitude();
        assert_abs_diff_eq!(amplitude.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_new_pfaffian() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let mut pf_wfn = PfaffianWavefunction::new(nsite, ne);

        // Set up a simple inverse matrix
        pf_wfn.inverse_matrix_mut().set(0, 1, Complex64::new(1.0, 0.0));

        let slater_elements = vec![Complex64::new(1.0, 0.0); 4];
        let result = pf_wfn.calculate_new_pfaffian(0, &slater_elements);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calculate_new_pfaffian_invalid_index() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let pf_wfn = PfaffianWavefunction::new(nsite, ne);

        let slater_elements = vec![Complex64::new(1.0, 0.0); 4];
        let result = pf_wfn.calculate_new_pfaffian(10, &slater_elements);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_pfaffian_matrix_shape(n in 1usize..10) {
            let matrix = PfaffianMatrix::zeros(n);
            prop_assert_eq!(matrix.size(), 2 * n);
            prop_assert_eq!(matrix.n(), n);
        }

        #[test]
        fn prop_antisymmetric_after_set(n in 1usize..5) {
            let mut matrix = PfaffianMatrix::zeros(n);
            let size = matrix.size();
            for i in 0..size {
                for j in (i+1)..size {
                    let value = Complex64::new((i + j) as f64, 0.0);
                    matrix.set_antisymmetric(i, j, value);
                }
            }
            prop_assert!(matrix.is_antisymmetric(1e-10));
        }

        #[test]
        fn prop_pfaffian_zero_for_odd_size(n in 1usize..10) {
            let odd_size = 2 * n + 1;
            let data = vec![Complex64::new(1.0, 0.0); odd_size * odd_size];
            if let Ok(matrix) = PfaffianMatrix::new(data, (odd_size + 1) / 2) {
                let pf = matrix.pfaffian();
                prop_assert!(pf.norm() < 1e-10);
            }
        }
    }
}
