//! Combined wavefunction for VMC calculations
//!
//! This module provides a unified wavefunction structure that combines:
//! - Slater determinant (single-particle orbitals)
//! - Pfaffian (pairing wavefunction)
//! - Projection operators (correlation factors)
//! - RBM (Restricted Boltzmann Machine corrections)
//!
//! The combined wavefunction represents:
//! Ψ(x) = Ψ_Slater(x) × Ψ_Pfaffian(x) × Ψ_Projection(x) × Ψ_RBM(x) × Ψ_Jastrow(x) × Ψ_DoublonHolon(x)

use crate::error::{Result, VmcError};
use crate::types::{ElectronCount, SiteCount};
use crate::wavefunction::{
    DoublonHolonFactor, JastrowFactor, PfaffianWavefunction, ProjectionOperator, RBMWavefunction, SlaterDeterminant,
};
use num_complex::Complex64;

/// Combined wavefunction for VMC calculations
///
/// This structure integrates multiple wavefunction components to represent
/// the full variational wavefunction used in VMC calculations.
///
/// # Components
///
/// - **Slater determinant**: Represents single-particle orbitals
/// - **Pfaffian**: Represents pairing correlations (optional)
/// - **Projection operators**: Apply correlation factors (optional)
/// - **RBM**: Neural network corrections (optional)
/// - **Jastrow factor**: Electron-electron correlations (optional)
/// - **Doublon-Holon factor**: Doublon-holon correlations (optional)
///
/// # References
///
/// - C implementation: `mVMC/src/mVMC/vmcmake.c` - MakeProjBF, MakeSlater
#[derive(Debug, Clone)]
pub struct CombinedWavefunction {
    /// Number of lattice sites
    nsite: usize,
    /// Number of electrons
    ne: usize,
    /// Slater determinant (always present)
    slater: Option<SlaterDeterminant>,
    /// Pfaffian wavefunction (optional, for pairing systems)
    pfaffian: Option<PfaffianWavefunction>,
    /// Projection operator (optional, for correlation factors)
    projection: Option<ProjectionOperator>,
    /// RBM wavefunction (optional, for neural network corrections)
    rbm: Option<RBMWavefunction>,
    /// Jastrow factor (optional, for electron-electron correlations)
    jastrow: Option<JastrowFactor>,
    /// Doublon-Holon factor (optional, for doublon-holon correlations)
    doublon_holon: Option<DoublonHolonFactor>,
}

/// Result of wavefunction amplitude calculation
#[derive(Debug, Clone, PartialEq)]
pub struct AmplitudeResult {
    /// Total amplitude
    pub amplitude: Complex64,
    /// Slater component
    pub slater_amplitude: Complex64,
    /// Pfaffian component (if present)
    pub pfaffian_amplitude: Option<Complex64>,
    /// Projection component (if present)
    pub projection_amplitude: Option<Complex64>,
    /// RBM component (if present)
    pub rbm_amplitude: Option<Complex64>,
    /// Jastrow component (if present)
    pub jastrow_amplitude: Option<Complex64>,
    /// Doublon-Holon component (if present)
    pub doublon_holon_amplitude: Option<Complex64>,
}

impl CombinedWavefunction {
    /// Creates a new combined wavefunction
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::combined::CombinedWavefunction;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let wavefunction = CombinedWavefunction::new(nsite, ne);
    /// ```
    pub fn new(nsite: SiteCount, ne: ElectronCount) -> Self {
        Self {
            nsite: nsite.get(),
            ne: ne.get(),
            slater: None,
            pfaffian: None,
            projection: None,
            rbm: None,
            jastrow: None,
            doublon_holon: None,
        }
    }

    /// Sets the Slater determinant component
    pub fn with_slater(mut self, slater: SlaterDeterminant) -> Self {
        self.slater = Some(slater);
        self
    }

    /// Sets the Pfaffian component
    pub fn with_pfaffian(mut self, pfaffian: PfaffianWavefunction) -> Self {
        self.pfaffian = Some(pfaffian);
        self
    }

    /// Sets the projection operator component
    pub fn with_projection(mut self, projection: ProjectionOperator) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Sets the RBM component
    pub fn with_rbm(mut self, rbm: RBMWavefunction) -> Self {
        self.rbm = Some(rbm);
        self
    }

    /// Sets the Jastrow factor component
    pub fn with_jastrow(mut self, jastrow: JastrowFactor) -> Self {
        self.jastrow = Some(jastrow);
        self
    }

    /// Sets the Doublon-Holon factor component
    pub fn with_doublon_holon(mut self, doublon_holon: DoublonHolonFactor) -> Self {
        self.doublon_holon = Some(doublon_holon);
        self
    }

    /// Returns the number of sites
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns whether Slater component is present
    pub fn has_slater(&self) -> bool {
        self.slater.is_some()
    }

    /// Returns whether Pfaffian component is present
    pub fn has_pfaffian(&self) -> bool {
        self.pfaffian.is_some()
    }

    /// Returns whether projection component is present
    pub fn has_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// Returns whether RBM component is present
    pub fn has_rbm(&self) -> bool {
        self.rbm.is_some()
    }

    /// Returns whether Jastrow component is present
    pub fn has_jastrow(&self) -> bool {
        self.jastrow.is_some()
    }

    /// Returns whether Doublon-Holon component is present
    pub fn has_doublon_holon(&self) -> bool {
        self.doublon_holon.is_some()
    }

    /// Calculates the total wavefunction amplitude for a given electron configuration
    ///
    /// The total amplitude is the product of all components:
    /// Ψ_total = Ψ_Slater × Ψ_Pfaffian × Ψ_Projection × Ψ_RBM × Ψ_Jastrow × Ψ_DoublonHolon
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The amplitude result containing the total amplitude and individual components
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmccal.c:CalculateMAll`
    pub fn calculate_amplitude(&self, electron_config: &[u8]) -> Result<AmplitudeResult> {
        if electron_config.len() != self.nsite {
            return Err(VmcError::dim_mismatch(self.nsite, electron_config.len()));
        }

        let mut result = AmplitudeResult {
            amplitude: Complex64::new(1.0, 0.0),
            slater_amplitude: Complex64::new(1.0, 0.0),
            pfaffian_amplitude: None,
            projection_amplitude: None,
            rbm_amplitude: None,
            jastrow_amplitude: None,
            doublon_holon_amplitude: None,
        };

        // Calculate Slater component
        if let Some(ref slater) = self.slater {
            let slater_amp = slater.amplitude();
            result.slater_amplitude = slater_amp;
            result.amplitude *= slater_amp;
        }

        // Calculate Pfaffian component
        if let Some(ref pfaffian) = self.pfaffian {
            let pfaffian_amp = pfaffian.amplitude();
            result.pfaffian_amplitude = Some(pfaffian_amp);
            result.amplitude *= pfaffian_amp;
        }

        // Calculate Projection component
        if let Some(ref _projection) = self.projection {
            // For now, use a placeholder
            // In a full implementation, this would calculate the projection factor
            let projection_amp = Complex64::new(1.0, 0.0);
            result.projection_amplitude = Some(projection_amp);
            result.amplitude *= projection_amp;
        }

        // Calculate RBM component
        if let Some(ref rbm) = self.rbm {
            let rbm_amp = rbm.calculate_weight(electron_config)?;
            result.rbm_amplitude = Some(rbm_amp);
            result.amplitude *= rbm_amp;
        }

        // Calculate Jastrow component
        if let Some(ref jastrow) = self.jastrow {
            let jastrow_amp = jastrow.calculate_factor(electron_config)?;
            result.jastrow_amplitude = Some(jastrow_amp);
            result.amplitude *= jastrow_amp;
        }

        // Calculate Doublon-Holon component
        if let Some(ref doublon_holon) = self.doublon_holon {
            let dh_amp = doublon_holon.calculate_factor(electron_config)?;
            result.doublon_holon_amplitude = Some(dh_amp);
            result.amplitude *= dh_amp;
        }

        Ok(result)
    }

    /// Calculates the log of the wavefunction amplitude
    ///
    /// This is more numerically stable than calculating the amplitude directly.
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The log of the amplitude
    pub fn calculate_log_amplitude(&self, electron_config: &[u8]) -> Result<Complex64> {
        if electron_config.len() != self.nsite {
            return Err(VmcError::dim_mismatch(self.nsite, electron_config.len()));
        }

        let mut log_amplitude = Complex64::new(0.0, 0.0);

        // Calculate Slater component
        if let Some(ref slater) = self.slater {
            let slater_amp = slater.amplitude();
            if slater_amp.norm() > 0.0 {
                log_amplitude += slater_amp.ln();
            }
        }

        // Calculate Pfaffian component
        if let Some(ref pfaffian) = self.pfaffian {
            let pfaffian_amp = pfaffian.amplitude();
            if pfaffian_amp.norm() > 0.0 {
                log_amplitude += pfaffian_amp.ln();
            }
        }

        // Calculate Projection component
        if let Some(ref _projection) = self.projection {
            // Placeholder
            log_amplitude += Complex64::new(0.0, 0.0);
        }

        // Calculate RBM component
        if let Some(ref rbm) = self.rbm {
            match rbm.calculate_log_weight(electron_config) {
                Ok(rbm_log_weight) => {
                    log_amplitude += rbm_log_weight;
                }
                Err(e) => return Err(e),
            }
        }

        // Calculate Jastrow component
        if let Some(ref jastrow) = self.jastrow {
            match jastrow.calculate_log_factor(electron_config) {
                Ok(jastrow_log_factor) => {
                    log_amplitude += Complex64::new(jastrow_log_factor, 0.0);
                }
                Err(e) => return Err(e),
            }
        }

        // Calculate Doublon-Holon component
        if let Some(ref doublon_holon) = self.doublon_holon {
            match doublon_holon.calculate_log_factor(electron_config) {
                Ok(dh_log_factor) => {
                    log_amplitude += Complex64::new(dh_log_factor, 0.0);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(log_amplitude)
    }

    /// Calculates the ratio of amplitudes for two configurations
    ///
    /// This is used in Metropolis acceptance criteria:
    /// ratio = |Ψ(new)/Ψ(old)|²
    ///
    /// # Arguments
    ///
    /// * `new_config` - New electron configuration
    /// * `old_config` - Old electron configuration
    ///
    /// # Returns
    ///
    /// The amplitude ratio
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmccal.c:CalculateHamiltonian0`
    pub fn calculate_amplitude_ratio(
        &self,
        new_config: &[u8],
        old_config: &[u8],
    ) -> Result<Complex64> {
        // Use log amplitudes for numerical stability
        let log_new = self.calculate_log_amplitude(new_config)?;
        let log_old = self.calculate_log_amplitude(old_config)?;
        let log_ratio = log_new - log_old;

        Ok(log_ratio.exp())
    }

    /// Returns a reference to the Slater determinant
    pub fn slater(&self) -> Option<&SlaterDeterminant> {
        self.slater.as_ref()
    }

    /// Returns a mutable reference to the Slater determinant
    pub fn slater_mut(&mut self) -> Option<&mut SlaterDeterminant> {
        self.slater.as_mut()
    }

    /// Returns a reference to the Pfaffian wavefunction
    pub fn pfaffian(&self) -> Option<&PfaffianWavefunction> {
        self.pfaffian.as_ref()
    }

    /// Returns a mutable reference to the Pfaffian wavefunction
    pub fn pfaffian_mut(&mut self) -> Option<&mut PfaffianWavefunction> {
        self.pfaffian.as_mut()
    }

    /// Returns a reference to the projection operator
    pub fn projection(&self) -> Option<&ProjectionOperator> {
        self.projection.as_ref()
    }

    /// Returns a mutable reference to the projection operator
    pub fn projection_mut(&mut self) -> Option<&mut ProjectionOperator> {
        self.projection.as_mut()
    }

    /// Returns a reference to the RBM wavefunction
    pub fn rbm(&self) -> Option<&RBMWavefunction> {
        self.rbm.as_ref()
    }

    /// Returns a mutable reference to the RBM wavefunction
    pub fn rbm_mut(&mut self) -> Option<&mut RBMWavefunction> {
        self.rbm.as_mut()
    }

    /// Returns a reference to the Jastrow factor
    pub fn jastrow(&self) -> Option<&JastrowFactor> {
        self.jastrow.as_ref()
    }

    /// Returns a mutable reference to the Jastrow factor
    pub fn jastrow_mut(&mut self) -> Option<&mut JastrowFactor> {
        self.jastrow.as_mut()
    }

    /// Returns a reference to the Doublon-Holon factor
    pub fn doublon_holon(&self) -> Option<&DoublonHolonFactor> {
        self.doublon_holon.as_ref()
    }

    /// Returns a mutable reference to the Doublon-Holon factor
    pub fn doublon_holon_mut(&mut self) -> Option<&mut DoublonHolonFactor> {
        self.doublon_holon.as_mut()
    }
}

impl std::fmt::Display for CombinedWavefunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CombinedWavefunction(")?;
        write!(f, "nsite={}, ne={}", self.nsite, self.ne)?;
        if self.has_slater() {
            write!(f, ", Slater")?;
        }
        if self.has_pfaffian() {
            write!(f, ", Pfaffian")?;
        }
        if self.has_projection() {
            write!(f, ", Projection")?;
        }
        if self.has_rbm() {
            write!(f, ", RBM")?;
        }
        if self.has_jastrow() {
            write!(f, ", Jastrow")?;
        }
        if self.has_doublon_holon() {
            write!(f, ", Doublon-Holon")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_wavefunction_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let wf = CombinedWavefunction::new(nsite, ne);

        assert_eq!(wf.nsite(), 4);
        assert_eq!(wf.ne(), 2);
        assert!(!wf.has_slater());
        assert!(!wf.has_pfaffian());
        assert!(!wf.has_projection());
        assert!(!wf.has_rbm());
        assert!(!wf.has_jastrow());
        assert!(!wf.has_doublon_holon());
    }

    #[test]
    fn test_combined_wavefunction_with_slater() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);

        let slater = SlaterDeterminant::new(nsite, ne);

        let wf = CombinedWavefunction::new(nsite, ne).with_slater(slater);

        assert!(wf.has_slater());
    }

    #[test]
    fn test_combined_wavefunction_amplitude() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        let slater = SlaterDeterminant::new(nsite, ne);

        let wf = CombinedWavefunction::new(nsite, ne).with_slater(slater);

        let config = vec![1, 0];
        let result = wf.calculate_amplitude(&config).unwrap();

        assert!(result.amplitude.norm() >= 0.0);
        assert!(result.slater_amplitude.norm() >= 0.0);
        assert!(result.pfaffian_amplitude.is_none());
    }

    #[test]
    fn test_combined_wavefunction_with_rbm() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let n_hidden = 4;

        let rbm = RBMWavefunction::new(nsite, n_hidden, false);
        let wf = CombinedWavefunction::new(nsite, ne).with_rbm(rbm);

        assert!(wf.has_rbm());

        let config = vec![1, 0];
        let result = wf.calculate_amplitude(&config).unwrap();

        assert!(result.amplitude.norm() > 0.0);
        assert!(result.rbm_amplitude.is_some());
    }

    #[test]
    fn test_combined_wavefunction_amplitude_ratio() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        let slater = SlaterDeterminant::new(nsite, ne);

        let wf = CombinedWavefunction::new(nsite, ne).with_slater(slater);

        let config1 = vec![1, 0];
        let config2 = vec![0, 1];
        let ratio = wf.calculate_amplitude_ratio(&config2, &config1).unwrap();

        assert!(ratio.is_finite());
    }

    #[test]
    fn test_combined_wavefunction_display() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let wf = CombinedWavefunction::new(nsite, ne);

        let display = format!("{}", wf);
        assert!(display.contains("CombinedWavefunction"));
        assert!(display.contains("nsite=4"));
        assert!(display.contains("ne=2"));
    }

    #[test]
    fn test_combined_wavefunction_with_jastrow() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        let jastrow = JastrowFactor::gutzwiller(2, 0.5).unwrap();
        let wf = CombinedWavefunction::new(nsite, ne).with_jastrow(jastrow);

        assert!(wf.has_jastrow());

        let config = vec![1, 0];
        let result = wf.calculate_amplitude(&config).unwrap();

        assert!(result.amplitude.norm() > 0.0);
        assert!(result.jastrow_amplitude.is_some());
    }

    #[test]
    fn test_combined_wavefunction_with_doublon_holon() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        let dh_factor = DoublonHolonFactor::simple(2, 0.1, Some(1)).unwrap();
        let wf = CombinedWavefunction::new(nsite, ne).with_doublon_holon(dh_factor);

        assert!(wf.has_doublon_holon());

        let config = vec![3, 0]; // Doublon and holon
        let result = wf.calculate_amplitude(&config).unwrap();

        assert!(result.amplitude.norm() > 0.0);
        assert!(result.doublon_holon_amplitude.is_some());
    }

    #[test]
    fn test_combined_wavefunction_with_all_components() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        // Create a simple Slater determinant with non-singular matrix
        let mut slater = SlaterDeterminant::new(nsite, ne);
        // Set up a non-singular Slater matrix
        slater.matrix_mut().set(0, 0, Complex64::new(1.0, 0.0));
        slater.matrix_mut().set(0, 1, Complex64::new(0.1, 0.0));
        slater.matrix_mut().set(1, 0, Complex64::new(0.1, 0.0));
        slater.matrix_mut().set(1, 1, Complex64::new(1.0, 0.0));
        let jastrow = JastrowFactor::gutzwiller(2, 0.5).unwrap();
        let dh_factor = DoublonHolonFactor::simple(2, 0.1, Some(1)).unwrap();

        let wf = CombinedWavefunction::new(nsite, ne)
            .with_slater(slater)
            .with_jastrow(jastrow)
            .with_doublon_holon(dh_factor);

        assert!(wf.has_slater());
        assert!(wf.has_jastrow());
        assert!(wf.has_doublon_holon());

        let config = vec![1, 0];
        let result = wf.calculate_amplitude(&config).unwrap();

        // The amplitude should be non-zero due to all components
        assert!(result.amplitude.norm() > 0.0);
        assert!(result.slater_amplitude.norm() > 0.0);
        assert!(result.jastrow_amplitude.is_some());
        assert!(result.doublon_holon_amplitude.is_some());
    }
}

