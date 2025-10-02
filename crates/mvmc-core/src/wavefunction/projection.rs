//! Projection operators for spin constraints and correlation factors
//!
//! This module corresponds to `mVMC/src/mVMC/projection.c` in the C implementation.
//! Reference: mVMC/src/mVMC/projection.c
//!
//! The projection operators include:
//! - Gutzwiller factor: exp(g * n_i^up * n_i^down)
//! - Jastrow factor: exp(sum_{i<j} v_ij * (n_i-1) * (n_j-1))
//! - Doublon-Holon correlation factors: 2-site and 4-site correlations
//! - BackFlow factor: advanced correlation factors

use crate::{Result, VmcError};
use num_complex::Complex64;

/// Projection operator for variational Monte Carlo calculations
///
/// This structure manages various projection factors used in VMC:
/// - Gutzwiller factor for local double occupancy suppression
/// - Jastrow factor for two-body correlations
/// - Doublon-Holon correlation factors
///
/// Reference: mVMC/src/mVMC/projection.c
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::ProjectionOperator;
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let proj = ProjectionOperator::new(nsite, ne);
/// ```
#[derive(Debug, Clone)]
pub struct ProjectionOperator {
    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons (for future use)
    #[allow(dead_code)]
    ne: usize,

    /// Projection parameters (Gutzwiller, Jastrow, etc.)
    /// Reference: mVMC/src/mVMC/include/global.h (Proj array)
    parameters: Vec<Complex64>,

    /// Number of projection parameters
    n_proj: usize,

    /// Gutzwiller indices for each site
    /// Reference: mVMC/src/mVMC/include/global.h (GutzwillerIdx)
    gutzwiller_indices: Vec<usize>,

    /// Jastrow indices for site pairs (for future use)
    /// Reference: mVMC/src/mVMC/include/global.h (JastrowIdx)
    #[allow(dead_code)]
    jastrow_indices: Vec<Vec<usize>>,
}

/// Projection count for a given electron configuration
///
/// This structure stores the counts for various projection factors
/// and is used to efficiently compute projection ratios.
///
/// Reference: mVMC/src/mVMC/projection.c (projCnt array)
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionCount {
    /// Counts for each projection parameter
    counts: Vec<i32>,

    /// Number of projection parameters
    n_proj: usize,
}

impl ProjectionOperator {
    /// Creates a new projection operator
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::ProjectionOperator;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let proj = ProjectionOperator::new(nsite, ne);
    /// ```
    pub fn new(nsite: crate::types::SiteCount, ne: crate::types::ElectronCount) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();

        // Initialize with basic Gutzwiller factor only
        let n_gutzwiller = nsite_val;
        let n_jastrow = 0; // No Jastrow factor by default
        let n_dh_2site = 0; // No 2-site DH factor by default
        let n_dh_4site = 0; // No 4-site DH factor by default

        let n_proj = n_gutzwiller + n_jastrow + 6 * n_dh_2site + 6 * n_dh_4site;

        // Initialize parameters (all zeros for now)
        let parameters = vec![Complex64::new(0.0, 0.0); n_proj];

        // Initialize Gutzwiller indices (one-to-one mapping)
        let gutzwiller_indices = (0..nsite_val).collect();

        // Initialize Jastrow indices (empty for now)
        let jastrow_indices = vec![vec![0; nsite_val]; nsite_val];

        Self {
            nsite: nsite_val,
            ne: ne_val,
            parameters,
            n_proj,
            gutzwiller_indices,
            jastrow_indices,
        }
    }

    /// Returns the number of projection parameters
    pub fn n_proj(&self) -> usize {
        self.n_proj
    }

    /// Returns the number of lattice sites
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Sets a projection parameter
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the parameter
    /// * `value` - Parameter value
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds
    pub fn set_parameter(&mut self, index: usize, value: Complex64) -> Result<()> {
        if index >= self.n_proj {
            return Err(VmcError::out_of_bounds(index, self.n_proj));
        }
        self.parameters[index] = value;
        Ok(())
    }

    /// Gets a projection parameter
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the parameter
    ///
    /// # Returns
    ///
    /// The parameter value, or an error if the index is out of bounds
    pub fn get_parameter(&self, index: usize) -> Result<Complex64> {
        if index >= self.n_proj {
            return Err(VmcError::out_of_bounds(index, self.n_proj));
        }
        Ok(self.parameters[index])
    }
}

impl ProjectionCount {
    /// Creates a new projection count
    ///
    /// # Arguments
    ///
    /// * `n_proj` - Number of projection parameters
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::ProjectionCount;
    ///
    /// let proj_cnt = ProjectionCount::new(10);
    /// assert_eq!(proj_cnt.n_proj(), 10);
    /// ```
    pub fn new(n_proj: usize) -> Self {
        Self {
            counts: vec![0; n_proj],
            n_proj,
        }
    }

    /// Returns the number of projection parameters
    pub fn n_proj(&self) -> usize {
        self.n_proj
    }

    /// Gets a count value
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the count
    ///
    /// # Returns
    ///
    /// The count value, or an error if the index is out of bounds
    pub fn get(&self, index: usize) -> Result<i32> {
        if index >= self.n_proj {
            return Err(VmcError::out_of_bounds(index, self.n_proj));
        }
        Ok(self.counts[index])
    }

    /// Sets a count value
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the count
    /// * `value` - Count value
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds
    pub fn set(&mut self, index: usize, value: i32) -> Result<()> {
        if index >= self.n_proj {
            return Err(VmcError::out_of_bounds(index, self.n_proj));
        }
        self.counts[index] = value;
        Ok(())
    }

    /// Adds to a count value
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the count
    /// * `value` - Value to add
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds
    pub fn add(&mut self, index: usize, value: i32) -> Result<()> {
        if index >= self.n_proj {
            return Err(VmcError::out_of_bounds(index, self.n_proj));
        }
        self.counts[index] += value;
        Ok(())
    }
}

/// Computes the logarithm of the projection value
///
/// This corresponds to `LogProjVal` in the C implementation.
/// Reference: mVMC/src/mVMC/projection.c:32-39
///
/// # Arguments
///
/// * `proj_op` - The projection operator
/// * `proj_cnt` - The projection count
///
/// # Returns
///
/// The logarithm of the projection value
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::{ProjectionOperator, ProjectionCount, log_projection_value};
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let proj_op = ProjectionOperator::new(nsite, ne);
/// let proj_cnt = ProjectionCount::new(proj_op.n_proj());
///
/// let log_val = log_projection_value(&proj_op, &proj_cnt);
/// ```
pub fn log_projection_value(proj_op: &ProjectionOperator, proj_cnt: &ProjectionCount) -> f64 {
    let mut z = 0.0;

    for idx in 0..proj_op.n_proj() {
        if let Ok(param) = proj_op.get_parameter(idx) {
            if let Ok(count) = proj_cnt.get(idx) {
                z += param.re * count as f64;
            }
        }
    }

    z
}

/// Computes the logarithm of the projection ratio
///
/// This corresponds to `LogProjRatio` in the C implementation.
/// Reference: mVMC/src/mVMC/projection.c:41-48
///
/// # Arguments
///
/// * `proj_op` - The projection operator
/// * `proj_cnt_new` - New projection count
/// * `proj_cnt_old` - Old projection count
///
/// # Returns
///
/// The logarithm of the projection ratio
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::{ProjectionOperator, ProjectionCount, log_projection_ratio};
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let proj_op = ProjectionOperator::new(nsite, ne);
/// let proj_cnt_old = ProjectionCount::new(proj_op.n_proj());
/// let proj_cnt_new = ProjectionCount::new(proj_op.n_proj());
///
/// let log_ratio = log_projection_ratio(&proj_op, &proj_cnt_new, &proj_cnt_old);
/// ```
pub fn log_projection_ratio(
    proj_op: &ProjectionOperator,
    proj_cnt_new: &ProjectionCount,
    proj_cnt_old: &ProjectionCount,
) -> f64 {
    let mut z = 0.0;

    for idx in 0..proj_op.n_proj() {
        if let (Ok(param), Ok(count_new), Ok(count_old)) = (
            proj_op.get_parameter(idx),
            proj_cnt_new.get(idx),
            proj_cnt_old.get(idx),
        ) {
            z += param.re * (count_new - count_old) as f64;
        }
    }

    z
}

/// Computes the projection ratio
///
/// This corresponds to `ProjRatio` in the C implementation.
/// Reference: mVMC/src/mVMC/projection.c:50-57
///
/// # Arguments
///
/// * `proj_op` - The projection operator
/// * `proj_cnt_new` - New projection count
/// * `proj_cnt_old` - Old projection count
///
/// # Returns
///
/// The projection ratio (exp of log ratio)
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::{ProjectionOperator, ProjectionCount, projection_ratio};
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let proj_op = ProjectionOperator::new(nsite, ne);
/// let proj_cnt_old = ProjectionCount::new(proj_op.n_proj());
/// let proj_cnt_new = ProjectionCount::new(proj_op.n_proj());
///
/// let ratio = projection_ratio(&proj_op, &proj_cnt_new, &proj_cnt_old);
/// ```
pub fn projection_ratio(
    proj_op: &ProjectionOperator,
    proj_cnt_new: &ProjectionCount,
    proj_cnt_old: &ProjectionCount,
) -> f64 {
    let log_ratio = log_projection_ratio(proj_op, proj_cnt_new, proj_cnt_old);
    log_ratio.exp()
}

/// Makes projection count for a given electron configuration
///
/// This corresponds to `MakeProjCnt` in the C implementation.
/// Reference: mVMC/src/mVMC/projection.c:58-152
///
/// # Arguments
///
/// * `proj_op` - The projection operator
/// * `proj_cnt` - The projection count to fill
/// * `ele_num` - Electron configuration (n_up, n_down for each site)
///
/// # Errors
///
/// Returns an error if the configuration is invalid
///
/// # Examples
///
/// ```
/// use mvmc_core::wavefunction::{ProjectionOperator, ProjectionCount, make_projection_count};
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let proj_op = ProjectionOperator::new(nsite, ne);
/// let mut proj_cnt = ProjectionCount::new(proj_op.n_proj());
///
/// // Electron configuration: [n_up, n_down] for each site
/// let ele_num = vec![1, 0, 1, 0, 0, 1, 0, 1]; // 4 sites, 2 electrons
/// make_projection_count(&proj_op, &mut proj_cnt, &ele_num).unwrap();
/// ```
pub fn make_projection_count(
    proj_op: &ProjectionOperator,
    proj_cnt: &mut ProjectionCount,
    ele_num: &[i32],
) -> Result<()> {
    let nsite = proj_op.nsite();

    // Check that we have the right number of elements
    if ele_num.len() != 2 * nsite {
        return Err(VmcError::dim_mismatch(2 * nsite, ele_num.len()));
    }

    // Split into up and down spin arrays
    let n_up = &ele_num[0..nsite];
    let n_down = &ele_num[nsite..2 * nsite];

    // Initialize all counts to zero
    for idx in 0..proj_op.n_proj() {
        proj_cnt.set(idx, 0)?;
    }

    // Gutzwiller factor: exp(g * n_i^up * n_i^down)
    // Reference: mVMC/src/mVMC/projection.c:73-78
    for ri in 0..nsite {
        let idx = proj_op.gutzwiller_indices[ri];
        let double_occupancy = n_up[ri] * n_down[ri];
        proj_cnt.add(idx, double_occupancy)?;
    }

    // TODO: Implement Jastrow factor
    // TODO: Implement Doublon-Holon correlation factors

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount};

    #[test]
    fn test_projection_operator_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let proj = ProjectionOperator::new(nsite, ne);

        assert_eq!(proj.nsite(), 4);
        assert_eq!(proj.ne(), 2);
        assert_eq!(proj.n_proj(), 4); // Only Gutzwiller factor
    }

    #[test]
    fn test_projection_count_creation() {
        let proj_cnt = ProjectionCount::new(10);
        assert_eq!(proj_cnt.n_proj(), 10);
    }

    #[test]
    fn test_projection_count_operations() {
        let mut proj_cnt = ProjectionCount::new(5);

        // Test set and get
        proj_cnt.set(0, 42).unwrap();
        assert_eq!(proj_cnt.get(0).unwrap(), 42);

        // Test add
        proj_cnt.add(0, 8).unwrap();
        assert_eq!(proj_cnt.get(0).unwrap(), 50);

        // Test error handling
        assert!(proj_cnt.get(10).is_err());
        assert!(proj_cnt.set(10, 0).is_err());
    }

    #[test]
    fn test_log_projection_value() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut proj_op = ProjectionOperator::new(nsite, ne);

        // Set some parameters
        proj_op.set_parameter(0, Complex64::new(1.0, 0.0)).unwrap();
        proj_op.set_parameter(1, Complex64::new(2.0, 0.0)).unwrap();

        let mut proj_cnt = ProjectionCount::new(proj_op.n_proj());
        proj_cnt.set(0, 3).unwrap();
        proj_cnt.set(1, 4).unwrap();

        let log_val = log_projection_value(&proj_op, &proj_cnt);
        assert_eq!(log_val, 1.0 * 3.0 + 2.0 * 4.0); // 11.0
    }

    #[test]
    fn test_log_projection_ratio() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut proj_op = ProjectionOperator::new(nsite, ne);

        proj_op.set_parameter(0, Complex64::new(1.0, 0.0)).unwrap();

        let mut proj_cnt_old = ProjectionCount::new(proj_op.n_proj());
        proj_cnt_old.set(0, 2).unwrap();

        let mut proj_cnt_new = ProjectionCount::new(proj_op.n_proj());
        proj_cnt_new.set(0, 5).unwrap();

        let log_ratio = log_projection_ratio(&proj_op, &proj_cnt_new, &proj_cnt_old);
        assert_eq!(log_ratio, 1.0 * (5 - 2) as f64); // 3.0
    }

    #[test]
    fn test_projection_ratio() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut proj_op = ProjectionOperator::new(nsite, ne);

        proj_op.set_parameter(0, Complex64::new(1.0, 0.0)).unwrap();

        let mut proj_cnt_old = ProjectionCount::new(proj_op.n_proj());
        proj_cnt_old.set(0, 1).unwrap();

        let mut proj_cnt_new = ProjectionCount::new(proj_op.n_proj());
        proj_cnt_new.set(0, 3).unwrap();

        let ratio = projection_ratio(&proj_op, &proj_cnt_new, &proj_cnt_old);
        let expected = (1.0 * (3 - 1) as f64).exp(); // exp(2.0)
        assert!((ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn test_make_projection_count() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(2);
        let proj_op = ProjectionOperator::new(nsite, ne);
        let mut proj_cnt = ProjectionCount::new(proj_op.n_proj());

        // Electron configuration: site 0 has both spins, site 1 is empty
        let ele_num = vec![1, 0, 1, 0]; // [n_up[0], n_up[1], n_down[0], n_down[1]]

        make_projection_count(&proj_op, &mut proj_cnt, &ele_num).unwrap();

        // Site 0 should have double occupancy (1*1=1), site 1 should have none (0*0=0)
        assert_eq!(proj_cnt.get(0).unwrap(), 1); // Gutzwiller for site 0
        assert_eq!(proj_cnt.get(1).unwrap(), 0); // Gutzwiller for site 1
    }

    #[test]
    fn test_make_projection_count_invalid_config() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(2);
        let proj_op = ProjectionOperator::new(nsite, ne);
        let mut proj_cnt = ProjectionCount::new(proj_op.n_proj());

        // Wrong number of elements
        let ele_num = vec![1, 0, 1]; // Should be 4 elements

        let result = make_projection_count(&proj_op, &mut proj_cnt, &ele_num);
        assert!(result.is_err());
    }

    #[test]
    fn test_parameter_operations() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut proj_op = ProjectionOperator::new(nsite, ne);

        // Test setting and getting parameters
        let param = Complex64::new(1.5, 2.5);
        proj_op.set_parameter(0, param).unwrap();

        let retrieved = proj_op.get_parameter(0).unwrap();
        assert_eq!(retrieved, param);

        // Test error handling
        assert!(proj_op.get_parameter(10).is_err());
        assert!(proj_op.set_parameter(10, param).is_err());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_projection_count_roundtrip(
            n_proj in 1usize..100,
            index in 0usize..100,
            value in -100i32..100
        ) {
            if index < n_proj {
                let mut proj_cnt = ProjectionCount::new(n_proj);
                proj_cnt.set(index, value).unwrap();
                prop_assert_eq!(proj_cnt.get(index).unwrap(), value);
            }
        }

        #[test]
        fn prop_log_projection_value_linearity(
            nsite in 1usize..10,
            counts in prop::collection::vec(-10i32..10, 1..20)
        ) {
            if counts.len() >= nsite {
                let nsite_val = SiteCount::new(nsite);
                let ne = ElectronCount::new(2);
                let mut proj_op = ProjectionOperator::new(nsite_val, ne);

                // Set random parameters (only for the number of sites)
                for i in 0..nsite {
                    let param = Complex64::new((i as f64) * 0.1, 0.0);
                    proj_op.set_parameter(i, param).unwrap();
                }

                let mut proj_cnt = ProjectionCount::new(nsite);
                for i in 0..nsite {
                    proj_cnt.set(i, counts[i]).unwrap();
                }

                let log_val = log_projection_value(&proj_op, &proj_cnt);

                // Check linearity: should be sum of param[i] * count[i]
                let expected: f64 = (0..nsite)
                    .map(|i| (i as f64) * 0.1 * counts[i] as f64)
                    .sum();

                prop_assert!((log_val - expected).abs() < 1e-10);
            }
        }
    }
}
