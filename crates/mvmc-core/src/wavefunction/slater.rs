//! Slater determinant implementation for fermion systems
//!
//! This module corresponds to `mVMC/src/mVMC/slater.c` in the C implementation.
//!
//! The Slater determinant is the fundamental wavefunction for fermion systems,
//! represented as the determinant of a matrix of single-particle orbitals.

use crate::{Result, VmcError};
use num_complex::Complex64;

/// Slater matrix representing single-particle orbitals
///
/// Corresponds to `SlaterElm` in C implementation (global.h:247)
/// Reference: mVMC/src/mVMC/include/global.h:247
///
/// # Structure
///
/// The Slater matrix is a complex matrix where:
/// - Rows correspond to electron positions
/// - Columns correspond to orbital indices
/// - Each element represents the single-particle orbital value
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::SlaterMatrix;
/// use num_complex::Complex64;
///
/// let orbitals = vec![
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
/// ];
/// let matrix = SlaterMatrix::new(orbitals, 2, 2).unwrap();
/// assert_eq!(matrix.rows(), 2);
/// assert_eq!(matrix.cols(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SlaterMatrix {
    /// Matrix elements stored in row-major order
    /// Reference: mVMC/src/mVMC/include/global.h:247
    /// SlaterElm[QPidx][ri+si*Nsite][rj+sj*Nsite]
    data: Vec<Complex64>,

    /// Number of rows (electron positions)
    rows: usize,

    /// Number of columns (orbital indices)
    cols: usize,
}

impl SlaterMatrix {
    /// Creates a new Slater matrix from data
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix elements in row-major order
    /// * `rows` - Number of rows (electron positions)
    /// * `cols` - Number of columns (orbital indices)
    ///
    /// # Errors
    ///
    /// Returns an error if the data length doesn't match rows * cols
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterMatrix;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 0.0); 4];
    /// let matrix = SlaterMatrix::new(data, 2, 2).unwrap();
    /// ```
    pub fn new(data: Vec<Complex64>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(VmcError::dim_mismatch(rows * cols, data.len()));
        }

        Ok(Self { data, rows, cols })
    }

    /// Creates a zero matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterMatrix;
    ///
    /// let matrix = SlaterMatrix::zeros(3, 3);
    /// assert_eq!(matrix.rows(), 3);
    /// assert_eq!(matrix.cols(), 3);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterMatrix;
    /// use num_complex::Complex64;
    ///
    /// let matrix = SlaterMatrix::identity(2);
    /// assert_eq!(matrix.get(0, 0), Complex64::new(1.0, 0.0));
    /// assert_eq!(matrix.get(0, 1), Complex64::new(0.0, 0.0));
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); size * size];
        for i in 0..size {
            data[i * size + i] = Complex64::new(1.0, 0.0);
        }
        Self {
            data,
            rows: size,
            cols: size,
        }
    }

    /// Returns the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Gets an element at (row, col)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Sets an element at (row, col)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Returns a reference to the underlying data
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [Complex64] {
        &mut self.data
    }

    /// Computes the determinant of the matrix
    ///
    /// Reference: mVMC/src/mVMC/slater.c uses this for wavefunction calculation
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterMatrix;
    /// use num_complex::Complex64;
    ///
    /// let matrix = SlaterMatrix::identity(2);
    /// let det = matrix.determinant();
    /// assert!((det - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    /// ```
    pub fn determinant(&self) -> Complex64 {
        assert_eq!(
            self.rows, self.cols,
            "Determinant only defined for square matrices"
        );

        match self.rows {
            0 => Complex64::new(1.0, 0.0),
            1 => self.get(0, 0),
            2 => self.det_2x2(),
            3 => self.det_3x3(),
            _ => self.det_lu(),
        }
    }

    /// Computes determinant for 2x2 matrix
    fn det_2x2(&self) -> Complex64 {
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    /// Computes determinant for 3x3 matrix using Sarrus' rule
    fn det_3x3(&self) -> Complex64 {
        let a00 = self.get(0, 0);
        let a01 = self.get(0, 1);
        let a02 = self.get(0, 2);
        let a10 = self.get(1, 0);
        let a11 = self.get(1, 1);
        let a12 = self.get(1, 2);
        let a20 = self.get(2, 0);
        let a21 = self.get(2, 1);
        let a22 = self.get(2, 2);

        a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21
            - a02 * a11 * a20
            - a01 * a10 * a22
            - a00 * a12 * a21
    }

    /// Computes determinant using LU decomposition for larger matrices
    ///
    /// This is a simplified implementation. For production use,
    /// consider using optimized LAPACK routines.
    fn det_lu(&self) -> Complex64 {
        let n = self.rows;
        let mut lu = self.data.clone();
        let mut det = Complex64::new(1.0, 0.0);
        let mut sign = 1;

        // LU decomposition with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = lu[k * n + k].norm();
            for i in (k + 1)..n {
                let val = lu[i * n + k].norm();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if necessary
            if max_idx != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_idx * n + j);
                }
                sign *= -1;
            }

            let pivot = lu[k * n + k];
            if pivot.norm() < 1e-15 {
                return Complex64::new(0.0, 0.0);
            }

            det *= pivot;

            // Eliminate column
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / pivot;
                lu[i * n + k] = Complex64::new(0.0, 0.0);
                for j in (k + 1)..n {
                    let k_val = lu[k * n + j];
                    lu[i * n + j] -= factor * k_val;
                }
            }
        }

        det * Complex64::new(sign as f64, 0.0)
    }

    /// Updates a single matrix element
    ///
    /// Reference: mVMC/src/mVMC/slater.c:37 (UpdateSlaterElm_fcmp)
    ///
    /// This corresponds to updating the Slater matrix when an electron
    /// moves from one site to another in Monte Carlo sampling.
    pub fn update_element(&mut self, row: usize, col: usize, value: Complex64) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(VmcError::out_of_bounds(row, self.rows));
        }
        self.set(row, col, value);
        Ok(())
    }
}

/// Slater determinant wavefunction
///
/// Represents the complete Slater determinant wavefunction for a fermion system,
/// including both spin-up and spin-down components.
///
/// Reference: mVMC/src/mVMC/slater.c
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::SlaterDeterminant;
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let slater = SlaterDeterminant::new(nsite, ne);
/// ```
#[derive(Debug, Clone)]
pub struct SlaterDeterminant {
    /// Slater matrix for the system
    /// Reference: mVMC/src/mVMC/include/global.h:247
    matrix: SlaterMatrix,

    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons
    ne: usize,
}

impl SlaterDeterminant {
    /// Creates a new Slater determinant
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterDeterminant;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let slater = SlaterDeterminant::new(nsite, ne);
    /// ```
    pub fn new(nsite: crate::types::SiteCount, ne: crate::types::ElectronCount) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();

        // Matrix size is 2*Ne (spin up + spin down)
        // Reference: mVMC/src/mVMC/slater.c:73-74 (rsi0, rsi1)
        let size = 2 * ne_val;
        let matrix = SlaterMatrix::zeros(size, size);

        Self {
            matrix,
            nsite: nsite_val,
            ne: ne_val,
        }
    }

    /// Returns the Slater matrix
    pub fn matrix(&self) -> &SlaterMatrix {
        &self.matrix
    }

    /// Returns a mutable reference to the Slater matrix
    pub fn matrix_mut(&mut self) -> &mut SlaterMatrix {
        &mut self.matrix
    }

    /// Computes the wavefunction amplitude (determinant)
    ///
    /// Reference: mVMC/src/mVMC/slater.c
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::SlaterDeterminant;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(2);
    /// let ne = ElectronCount::new(1);
    /// let mut slater = SlaterDeterminant::new(nsite, ne);
    /// let amplitude = slater.amplitude();
    /// ```
    pub fn amplitude(&self) -> Complex64 {
        self.matrix.determinant()
    }

    /// Updates the Slater matrix for a Monte Carlo move
    ///
    /// Reference: mVMC/src/mVMC/slater.c:37 (UpdateSlaterElm_fcmp)
    ///
    /// # Arguments
    ///
    /// * `electron_idx` - Index of the electron being moved
    /// * `new_site` - New site index for the electron
    /// * `orbital_values` - New orbital values for this configuration
    pub fn update(
        &mut self,
        electron_idx: usize,
        new_site: usize,
        orbital_values: &[Complex64],
    ) -> Result<()> {
        if electron_idx >= self.ne * 2 {
            return Err(VmcError::out_of_bounds(electron_idx, self.ne * 2));
        }
        if new_site >= self.nsite {
            return Err(VmcError::out_of_bounds(new_site, self.nsite));
        }
        if orbital_values.len() != self.ne * 2 {
            return Err(VmcError::dim_mismatch(
                self.ne * 2,
                orbital_values.len(),
            ));
        }

        // Update the row corresponding to the moved electron
        // Reference: mVMC/src/mVMC/slater.c:85-91
        for (col, &value) in orbital_values.iter().enumerate() {
            self.matrix.set(electron_idx, col, value);
        }

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
    fn test_slater_matrix_creation() {
        let data = vec![Complex64::new(1.0, 0.0); 4];
        let matrix = SlaterMatrix::new(data, 2, 2).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
    }

    #[test]
    fn test_slater_matrix_dimension_mismatch() {
        let data = vec![Complex64::new(1.0, 0.0); 4];
        let result = SlaterMatrix::new(data, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_slater_matrix_zeros() {
        let matrix = SlaterMatrix::zeros(3, 3);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.get(0, 0), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_slater_matrix_identity() {
        let matrix = SlaterMatrix::identity(3);
        assert_eq!(matrix.get(0, 0), Complex64::new(1.0, 0.0));
        assert_eq!(matrix.get(0, 1), Complex64::new(0.0, 0.0));
        assert_eq!(matrix.get(1, 1), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_slater_matrix_get_set() {
        let mut matrix = SlaterMatrix::zeros(2, 2);
        let value = Complex64::new(2.0, 3.0);
        matrix.set(1, 1, value);
        assert_eq!(matrix.get(1, 1), value);
    }

    #[test]
    fn test_determinant_identity() {
        let matrix = SlaterMatrix::identity(3);
        let det = matrix.determinant();
        assert_abs_diff_eq!(det.re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(det.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_2x2() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let matrix = SlaterMatrix::new(data, 2, 2).unwrap();
        let det = matrix.determinant();
        // det = 1*4 - 2*3 = -2
        assert_abs_diff_eq!(det.re, -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(det.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let matrix = SlaterMatrix::new(data, 3, 3).unwrap();
        let det = matrix.determinant();
        // det = 1*1*0 + 2*4*5 + 3*0*6 - 3*1*5 - 2*0*0 - 1*4*6
        //     = 0 + 40 + 0 - 15 - 0 - 24 = 1
        assert_abs_diff_eq!(det.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_complex() {
        let data = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
        ];
        let matrix = SlaterMatrix::new(data, 2, 2).unwrap();
        let det = matrix.determinant();
        // det = (1+i)(1-i) = 1 - i^2 = 1 + 1 = 2
        assert_abs_diff_eq!(det.re, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(det.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_slater_determinant_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let slater = SlaterDeterminant::new(nsite, ne);
        assert_eq!(slater.nsite(), 4);
        assert_eq!(slater.ne(), 2);
        assert_eq!(slater.matrix().rows(), 4); // 2*ne
        assert_eq!(slater.matrix().cols(), 4);
    }

    #[test]
    fn test_slater_determinant_amplitude() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut slater = SlaterDeterminant::new(nsite, ne);

        // Set up an identity-like matrix
        slater.matrix_mut().set(0, 0, Complex64::new(1.0, 0.0));
        slater.matrix_mut().set(1, 1, Complex64::new(1.0, 0.0));

        let amplitude = slater.amplitude();
        assert_abs_diff_eq!(amplitude.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_slater_determinant_update() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let mut slater = SlaterDeterminant::new(nsite, ne);

        let orbital_values = vec![Complex64::new(1.0, 0.0); 4];
        let result = slater.update(0, 1, &orbital_values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_slater_determinant_update_invalid_electron() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let mut slater = SlaterDeterminant::new(nsite, ne);

        let orbital_values = vec![Complex64::new(1.0, 0.0); 4];
        let result = slater.update(10, 1, &orbital_values); // Invalid electron index
        assert!(result.is_err());
    }

    #[test]
    fn test_slater_determinant_update_invalid_site() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let mut slater = SlaterDeterminant::new(nsite, ne);

        let orbital_values = vec![Complex64::new(1.0, 0.0); 4];
        let result = slater.update(0, 10, &orbital_values); // Invalid site index
        assert!(result.is_err());
    }

    #[test]
    fn test_update_element() {
        let mut matrix = SlaterMatrix::zeros(3, 3);
        let value = Complex64::new(5.0, 2.0);
        let result = matrix.update_element(1, 2, value);
        assert!(result.is_ok());
        assert_eq!(matrix.get(1, 2), value);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn complex_strategy() -> impl Strategy<Value = Complex64> {
        (-10.0..10.0, -10.0..10.0).prop_map(|(re, im)| Complex64::new(re, im))
    }

    proptest! {
        #[test]
        fn prop_determinant_identity_is_one(size in 1usize..10) {
            let matrix = SlaterMatrix::identity(size);
            let det = matrix.determinant();
            prop_assert!((det.re - 1.0).abs() < 1e-10);
            prop_assert!(det.im.abs() < 1e-10);
        }

        #[test]
        fn prop_determinant_zero_row(size in 2usize..8) {
            let mut matrix = SlaterMatrix::identity(size);
            // Set first row to zeros
            for col in 0..size {
                matrix.set(0, col, Complex64::new(0.0, 0.0));
            }
            let det = matrix.determinant();
            prop_assert!(det.norm() < 1e-10);
        }

        #[test]
        fn prop_slater_matrix_shape(rows in 1usize..10, cols in 1usize..10) {
            let matrix = SlaterMatrix::zeros(rows, cols);
            prop_assert_eq!(matrix.rows(), rows);
            prop_assert_eq!(matrix.cols(), cols);
            prop_assert_eq!(matrix.shape(), (rows, cols));
        }

        #[test]
        fn prop_get_set_roundtrip(
            value in complex_strategy(),
            row in 0usize..5,
            col in 0usize..5
        ) {
            let mut matrix = SlaterMatrix::zeros(5, 5);
            matrix.set(row, col, value);
            let retrieved = matrix.get(row, col);
            prop_assert!((retrieved - value).norm() < 1e-10);
        }
    }
}
