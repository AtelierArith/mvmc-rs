//! Basic type definitions for mVMC
//!
//! This module provides newtype wrappers for fundamental quantities in VMC calculations,
//! ensuring type safety and preventing mixing of incompatible values.

use std::fmt;

/// Index for a lattice site
///
/// Valid range: [0, Nsite)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SiteIndex(usize);

impl SiteIndex {
    /// Creates a new site index
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::SiteIndex;
    ///
    /// let site = SiteIndex::new(0);
    /// assert_eq!(site.get(), 0);
    /// ```
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Gets the underlying index value
    pub fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for SiteIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Site({})", self.0)
    }
}

/// Number of lattice sites
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SiteCount(usize);

impl SiteCount {
    /// Creates a new site count
    ///
    /// # Panics
    ///
    /// Panics if count is zero
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::SiteCount;
    ///
    /// let nsite = SiteCount::new(4);
    /// assert_eq!(nsite.get(), 4);
    /// ```
    pub fn new(count: usize) -> Self {
        assert!(count > 0, "Site count must be positive");
        Self(count)
    }

    /// Gets the underlying count value
    pub fn get(self) -> usize {
        self.0
    }
}

/// Number of electrons
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElectronCount(usize);

impl ElectronCount {
    /// Creates a new electron count
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::ElectronCount;
    ///
    /// let ne = ElectronCount::new(8);
    /// assert_eq!(ne.get(), 8);
    /// ```
    pub fn new(count: usize) -> Self {
        Self(count)
    }

    /// Gets the underlying count value
    pub fn get(self) -> usize {
        self.0
    }

    /// Returns twice the electron count (for total number including spin)
    pub fn twice(self) -> usize {
        self.0 * 2
    }
}

/// Spin quantum number (2*Sz)
///
/// Stored as an integer to avoid floating-point arithmetic.
/// Actual Sz = TwoSz / 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TwoSz(i32);

impl TwoSz {
    /// Creates a new 2*Sz value
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::TwoSz;
    ///
    /// let sz = TwoSz::new(0); // Sz = 0
    /// assert_eq!(sz.get(), 0);
    ///
    /// let sz_up = TwoSz::new(1); // Sz = 1/2
    /// assert_eq!(sz_up.get(), 1);
    /// ```
    pub fn new(two_sz: i32) -> Self {
        Self(two_sz)
    }

    /// Gets the underlying 2*Sz value
    pub fn get(self) -> i32 {
        self.0
    }

    /// Returns Sz as a floating-point value
    pub fn as_f64(self) -> f64 {
        self.0 as f64 / 2.0
    }
}

/// Calculation mode for VMC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalcMode {
    /// Optimization of variational parameters
    Optimization,
    /// Calculation of expectation values
    Expectation,
}

impl CalcMode {
    /// Converts to the integer representation used in C implementation
    pub fn to_int(self) -> i32 {
        match self {
            CalcMode::Optimization => 0,
            CalcMode::Expectation => 1,
        }
    }

    /// Creates from integer representation
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::CalcMode;
    ///
    /// assert_eq!(CalcMode::from_int(0), Some(CalcMode::Optimization));
    /// assert_eq!(CalcMode::from_int(1), Some(CalcMode::Expectation));
    /// assert_eq!(CalcMode::from_int(2), None);
    /// ```
    pub fn from_int(value: i32) -> Option<Self> {
        match value {
            0 => Some(CalcMode::Optimization),
            1 => Some(CalcMode::Expectation),
            _ => None,
        }
    }
}

/// Lanczos mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanczosMode {
    /// No Lanczos step
    None,
    /// Only energy calculation
    Energy,
    /// Green function calculation
    GreenFunction,
}

impl LanczosMode {
    /// Converts to integer representation
    pub fn to_int(self) -> i32 {
        match self {
            LanczosMode::None => 0,
            LanczosMode::Energy => 1,
            LanczosMode::GreenFunction => 2,
        }
    }

    /// Creates from integer representation
    pub fn from_int(value: i32) -> Option<Self> {
        match value {
            0 => Some(LanczosMode::None),
            1 => Some(LanczosMode::Energy),
            2 => Some(LanczosMode::GreenFunction),
            _ => None,
        }
    }
}

/// Random seed for pseudorandom number generator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomSeed(u64);

impl RandomSeed {
    /// Creates a new random seed
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::types::RandomSeed;
    ///
    /// let seed = RandomSeed::new(12345);
    /// assert_eq!(seed.get(), 12345);
    /// ```
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Gets the underlying seed value
    pub fn get(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_site_index_creation() {
        let site = SiteIndex::new(5);
        assert_eq!(site.get(), 5);
    }

    #[test]
    fn test_site_index_ordering() {
        let s1 = SiteIndex::new(1);
        let s2 = SiteIndex::new(2);
        assert!(s1 < s2);
        assert!(s2 > s1);
        assert_eq!(s1, s1);
    }

    #[test]
    fn test_site_index_display() {
        let site = SiteIndex::new(42);
        assert_eq!(format!("{}", site), "Site(42)");
    }

    #[test]
    fn test_site_count_creation() {
        let nsite = SiteCount::new(10);
        assert_eq!(nsite.get(), 10);
    }

    #[test]
    #[should_panic(expected = "Site count must be positive")]
    fn test_site_count_zero_panics() {
        SiteCount::new(0);
    }

    #[test]
    fn test_electron_count() {
        let ne = ElectronCount::new(8);
        assert_eq!(ne.get(), 8);
        assert_eq!(ne.twice(), 16);
    }

    #[test]
    fn test_electron_count_zero() {
        // Zero electrons is valid (vacuum state)
        let ne = ElectronCount::new(0);
        assert_eq!(ne.get(), 0);
        assert_eq!(ne.twice(), 0);
    }

    #[test]
    fn test_two_sz_zero() {
        let sz = TwoSz::new(0);
        assert_eq!(sz.get(), 0);
        assert_eq!(sz.as_f64(), 0.0);
    }

    #[test]
    fn test_two_sz_half() {
        let sz = TwoSz::new(1);
        assert_eq!(sz.get(), 1);
        assert_eq!(sz.as_f64(), 0.5);
    }

    #[test]
    fn test_two_sz_negative() {
        let sz = TwoSz::new(-3);
        assert_eq!(sz.get(), -3);
        assert_eq!(sz.as_f64(), -1.5);
    }

    #[test]
    fn test_calc_mode_conversion() {
        assert_eq!(CalcMode::Optimization.to_int(), 0);
        assert_eq!(CalcMode::Expectation.to_int(), 1);

        assert_eq!(CalcMode::from_int(0), Some(CalcMode::Optimization));
        assert_eq!(CalcMode::from_int(1), Some(CalcMode::Expectation));
        assert_eq!(CalcMode::from_int(2), None);
        assert_eq!(CalcMode::from_int(-1), None);
    }

    #[test]
    fn test_lanczos_mode_conversion() {
        assert_eq!(LanczosMode::None.to_int(), 0);
        assert_eq!(LanczosMode::Energy.to_int(), 1);
        assert_eq!(LanczosMode::GreenFunction.to_int(), 2);

        assert_eq!(LanczosMode::from_int(0), Some(LanczosMode::None));
        assert_eq!(LanczosMode::from_int(1), Some(LanczosMode::Energy));
        assert_eq!(LanczosMode::from_int(2), Some(LanczosMode::GreenFunction));
        assert_eq!(LanczosMode::from_int(3), None);
    }

    #[test]
    fn test_random_seed() {
        let seed = RandomSeed::new(12345);
        assert_eq!(seed.get(), 12345);

        let seed2 = RandomSeed::new(54321);
        assert_ne!(seed, seed2);
    }

    #[test]
    fn test_site_index_as_array_index() {
        // Demonstrate using SiteIndex to index into arrays
        let sites = vec![10, 20, 30, 40];
        let idx = SiteIndex::new(2);
        assert_eq!(sites[idx.get()], 30);
    }

    #[test]
    fn test_electron_count_arithmetic() {
        let ne1 = ElectronCount::new(4);
        let ne2 = ElectronCount::new(4);
        assert_eq!(ne1, ne2);

        let ne3 = ElectronCount::new(8);
        assert!(ne3 > ne1);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_site_index_roundtrip(idx in 0usize..10000) {
            let site = SiteIndex::new(idx);
            prop_assert_eq!(site.get(), idx);
        }

        #[test]
        fn prop_site_count_positive(count in 1usize..10000) {
            let nsite = SiteCount::new(count);
            prop_assert_eq!(nsite.get(), count);
            prop_assert!(nsite.get() > 0);
        }

        #[test]
        fn prop_electron_count_twice(count in 0usize..1000) {
            let ne = ElectronCount::new(count);
            prop_assert_eq!(ne.twice(), count * 2);
        }

        #[test]
        fn prop_two_sz_conversion(two_sz in -100i32..100) {
            let sz = TwoSz::new(two_sz);
            prop_assert_eq!(sz.get(), two_sz);
            prop_assert_eq!(sz.as_f64(), two_sz as f64 / 2.0);
        }

        #[test]
        fn prop_calc_mode_roundtrip(mode_int in 0i32..2) {
            let mode = CalcMode::from_int(mode_int).unwrap();
            prop_assert_eq!(mode.to_int(), mode_int);
        }

        #[test]
        fn prop_lanczos_mode_roundtrip(mode_int in 0i32..3) {
            let mode = LanczosMode::from_int(mode_int).unwrap();
            prop_assert_eq!(mode.to_int(), mode_int);
        }

        #[test]
        fn prop_site_index_ordering(i in 0usize..100, j in 0usize..100) {
            let si = SiteIndex::new(i);
            let sj = SiteIndex::new(j);

            if i < j {
                prop_assert!(si < sj);
            } else if i > j {
                prop_assert!(si > sj);
            } else {
                prop_assert_eq!(si, sj);
            }
        }
    }
}
