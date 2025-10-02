//! Output module for mVMC calculation results.
//!
//! This module provides functionality to write VMC calculation results
//! to various file formats.
//!
//! # Modules
//!
//! - `data`: Energy and observable data output
//! - `parameters`: Optimized variational parameter output
//!
//! # Reference
//!
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/vmcmain.c`: `outputData()` function

pub mod data;
pub mod parameters;

// Re-export commonly used types
pub use data::{EnergyData, ObservableData, OutputFormat, VariationalData};
pub use parameters::OptimizedParameters;

use crate::Result;
use std::path::{Path, PathBuf};

/// Output manager for organizing VMC calculation output files.
///
/// Manages the creation and organization of output files with a consistent
/// directory structure and naming convention.
///
/// # Example
///
/// ```
/// use mvmc_io::output::OutputManager;
/// use num_complex::Complex64;
///
/// let manager = OutputManager::new("output", "my_calculation");
/// manager.ensure_output_dir().unwrap();
///
/// // Output files will be created in:
/// // output/my_calculation_out.dat
/// // output/my_calculation_var.dat
/// // output/my_calculation_opt.dat
/// ```
#[derive(Debug, Clone)]
pub struct OutputManager {
    /// Base output directory
    pub output_dir: PathBuf,
    /// Prefix for output files
    pub prefix: String,
}

impl OutputManager {
    /// Creates a new output manager.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Base directory for output files
    /// * `prefix` - Prefix for output file names
    pub fn new<P: AsRef<Path>>(output_dir: P, prefix: &str) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            prefix: prefix.to_string(),
        }
    }

    /// Ensures the output directory exists.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn ensure_output_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }

    /// Gets the path for the energy output file (`zvo_out.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c` - zvo_out.dat
    pub fn energy_output_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_out.dat", self.prefix))
    }

    /// Gets the path for the variational data file (`zvo_var.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c` - zvo_var.dat
    pub fn variational_output_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_var.dat", self.prefix))
    }

    /// Gets the path for the optimized parameters file (`zqp_opt.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c` - zqp_opt output
    pub fn optimized_params_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_opt.dat", self.prefix))
    }

    /// Gets the path for an observable file.
    ///
    /// # Arguments
    ///
    /// * `observable_name` - Name of the observable (e.g., "correlation", "green")
    pub fn observable_path(&self, observable_name: &str) -> PathBuf {
        self.output_dir
            .join(format!("{}_{}.dat", self.prefix, observable_name))
    }

    /// Writes energy data to the standard output file.
    pub fn write_energy(&self, energy: &EnergyData, format: OutputFormat) -> Result<()> {
        self.ensure_output_dir()?;
        energy.write(self.energy_output_path(), format)
    }

    /// Writes variational data to the standard output file.
    pub fn write_variational(&self, var_data: &VariationalData, format: OutputFormat) -> Result<()> {
        self.ensure_output_dir()?;
        var_data.write(self.variational_output_path(), format)
    }

    /// Appends variational data to the variational output file.
    ///
    /// Useful for writing optimization history.
    pub fn append_variational(&self, var_data: &VariationalData) -> Result<()> {
        self.ensure_output_dir()?;
        var_data.append_text(self.variational_output_path())
    }

    /// Writes optimized parameters to the standard output file.
    pub fn write_optimized_params(&self, params: &OptimizedParameters) -> Result<()> {
        self.ensure_output_dir()?;
        params.write_text(self.optimized_params_path())
    }

    /// Writes observable data to a file.
    pub fn write_observable(
        &self,
        observable: &ObservableData,
        format: OutputFormat,
    ) -> Result<()> {
        self.ensure_output_dir()?;
        observable.write(self.observable_path(&observable.name), format)
    }

    /// Cleans up old output files.
    ///
    /// Removes all files with the current prefix from the output directory.
    pub fn clean(&self) -> Result<()> {
        if !self.output_dir.exists() {
            return Ok(());
        }

        let pattern = format!("{}_", self.prefix);
        for entry in std::fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                if filename.to_string_lossy().starts_with(&pattern) {
                    std::fs::remove_file(path)?;
                }
            }
        }

        Ok(())
    }
}

/// Builder for OutputManager with convenient defaults.
pub struct OutputManagerBuilder {
    output_dir: PathBuf,
    prefix: String,
}

impl OutputManagerBuilder {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("output"),
            prefix: "zvo".to_string(),
        }
    }

    /// Sets the output directory.
    pub fn output_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.output_dir = dir.as_ref().to_path_buf();
        self
    }

    /// Sets the file prefix.
    pub fn prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Builds the OutputManager.
    pub fn build(self) -> OutputManager {
        OutputManager::new(self.output_dir, &self.prefix)
    }
}

impl Default for OutputManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_output_manager_new() {
        let manager = OutputManager::new("test_output", "test");

        assert_eq!(manager.output_dir, PathBuf::from("test_output"));
        assert_eq!(manager.prefix, "test");
    }

    #[test]
    fn test_output_manager_paths() {
        let manager = OutputManager::new("output", "zvo");

        assert_eq!(manager.energy_output_path(), PathBuf::from("output/zvo_out.dat"));
        assert_eq!(
            manager.variational_output_path(),
            PathBuf::from("output/zvo_var.dat")
        );
        assert_eq!(
            manager.optimized_params_path(),
            PathBuf::from("output/zvo_opt.dat")
        );
        assert_eq!(
            manager.observable_path("correlation"),
            PathBuf::from("output/zvo_correlation.dat")
        );
    }

    #[test]
    fn test_output_manager_ensure_dir() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_output_manager");

        let manager = OutputManager::new(&test_dir, "test");
        manager.ensure_output_dir().unwrap();

        assert!(test_dir.exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_write_energy() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_energy_write");

        let manager = OutputManager::new(&test_dir, "test");

        let energy = EnergyData::new(
            Complex64::new(-2.5, 0.0),
            Complex64::new(6.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );

        manager
            .write_energy(&energy, OutputFormat::Text)
            .unwrap();

        assert!(manager.energy_output_path().exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_write_variational() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_var_write");

        let manager = OutputManager::new(&test_dir, "test");

        let energy = EnergyData::new(
            Complex64::new(-2.5, 0.0),
            Complex64::new(6.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let params = vec![Complex64::new(0.5, 0.1)];
        let var_data = VariationalData::new(energy, params);

        manager
            .write_variational(&var_data, OutputFormat::Text)
            .unwrap();

        assert!(manager.variational_output_path().exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_append_variational() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_var_append");

        let manager = OutputManager::new(&test_dir, "test");

        let energy1 = EnergyData::new(
            Complex64::new(-2.5, 0.0),
            Complex64::new(6.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let var_data1 = VariationalData::new(energy1, vec![Complex64::new(0.5, 0.1)]);

        manager.append_variational(&var_data1).unwrap();

        let energy2 = EnergyData::new(
            Complex64::new(-2.6, 0.0),
            Complex64::new(6.8, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let var_data2 = VariationalData::new(energy2, vec![Complex64::new(0.6, 0.2)]);

        manager.append_variational(&var_data2).unwrap();

        // Verify file has two lines
        let content = std::fs::read_to_string(manager.variational_output_path()).unwrap();
        assert_eq!(content.lines().count(), 2);

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_clean() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_clean");

        let manager = OutputManager::new(&test_dir, "test");
        manager.ensure_output_dir().unwrap();

        // Create some test files
        std::fs::write(test_dir.join("test_out.dat"), "data").unwrap();
        std::fs::write(test_dir.join("test_var.dat"), "data").unwrap();
        std::fs::write(test_dir.join("other_file.txt"), "data").unwrap();

        manager.clean().unwrap();

        // test_* files should be removed
        assert!(!test_dir.join("test_out.dat").exists());
        assert!(!test_dir.join("test_var.dat").exists());

        // other files should remain
        assert!(test_dir.join("other_file.txt").exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_builder() {
        let manager = OutputManagerBuilder::new()
            .output_dir("custom_output")
            .prefix("custom")
            .build();

        assert_eq!(manager.output_dir, PathBuf::from("custom_output"));
        assert_eq!(manager.prefix, "custom");
    }

    #[test]
    fn test_output_manager_builder_default() {
        let manager = OutputManagerBuilder::default().build();

        assert_eq!(manager.output_dir, PathBuf::from("output"));
        assert_eq!(manager.prefix, "zvo");
    }

    #[test]
    fn test_output_manager_write_optimized_params() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_opt_params");

        let manager = OutputManager::new(&test_dir, "test");

        let params = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let opt_params = OptimizedParameters::new(params);

        manager.write_optimized_params(&opt_params).unwrap();

        assert!(manager.optimized_params_path().exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_write_observable() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_observable");

        let manager = OutputManager::new(&test_dir, "test");

        let values = vec![Complex64::new(1.0, 0.0)];
        let obs = ObservableData::new("correlation".to_string(), values);

        manager
            .write_observable(&obs, OutputFormat::Text)
            .unwrap();

        assert!(manager.observable_path("correlation").exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
}
