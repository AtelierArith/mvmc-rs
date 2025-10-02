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
/// directory structure and naming convention compatible with C implementation.
///
/// # Example
///
/// ```
/// use mvmc_io::output::OutputManager;
/// use num_complex::Complex64;
///
/// let manager = OutputManager::new("output", "zvo", 1);
/// manager.ensure_output_dir().unwrap();
///
/// // Output files will be created in:
/// // output/zvo_out_001.dat
/// // output/zvo_var_001.dat
/// // output/zvo_opt.dat
/// ```
#[derive(Debug, Clone)]
pub struct OutputManager {
    /// Base output directory
    pub output_dir: PathBuf,
    /// File header (CDataFileHead in C implementation)
    pub file_head: String,
    /// Data index start (NDataIdxStart in C implementation)
    pub data_idx_start: usize,
}

impl OutputManager {
    /// Creates a new output manager.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Base directory for output files
    /// * `file_head` - File header (CDataFileHead in C implementation)
    /// * `data_idx_start` - Data index start (NDataIdxStart in C implementation)
    pub fn new<P: AsRef<Path>>(output_dir: P, file_head: &str, data_idx_start: usize) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            file_head: file_head.to_string(),
            data_idx_start,
        }
    }

    /// Ensures the output directory exists.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn ensure_output_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }

    /// Gets the path for the energy output file (`zvo_out_001.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/initfile.c` - sprintf(fileName, "%s_out_%03d.dat", CDataFileHead, idx)
    pub fn energy_output_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_out_{:03}.dat", self.file_head, self.data_idx_start))
    }

    /// Gets the path for the variational data file (`zvo_var_001.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/initfile.c` - sprintf(fileName, "%s_var_%03d.dat", CDataFileHead, idx)
    pub fn variational_output_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_var_{:03}.dat", self.file_head, self.data_idx_start))
    }

    /// Gets the path for the time data file (`zvo_time_001.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/initfile.c` - sprintf(fileName, "%s_time_%03d.dat", CDataFileHead, NDataIdxStart)
    pub fn time_output_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_time_{:03}.dat", self.file_head, self.data_idx_start))
    }

    /// Gets the path for the optimized parameters file (`zqp_opt.dat`).
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c` - zqp_opt output
    pub fn optimized_params_path(&self) -> PathBuf {
        self.output_dir.join(format!("{}_opt.dat", self.file_head))
    }

    /// Gets the path for an observable file.
    ///
    /// # Arguments
    ///
    /// * `observable_name` - Name of the observable (e.g., "correlation", "green")
    pub fn observable_path(&self, observable_name: &str) -> PathBuf {
        self.output_dir
            .join(format!("{}_{}.dat", self.file_head, observable_name))
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
    /// Removes all files with the current file head from the output directory.
    pub fn clean(&self) -> Result<()> {
        if !self.output_dir.exists() {
            return Ok(());
        }

        let pattern = format!("{}_", self.file_head);
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
    file_head: String,
    data_idx_start: usize,
}

impl OutputManagerBuilder {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("output"),
            file_head: "zvo".to_string(),
            data_idx_start: 1,
        }
    }

    /// Sets the output directory.
    pub fn output_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.output_dir = dir.as_ref().to_path_buf();
        self
    }

    /// Sets the file head (CDataFileHead).
    pub fn file_head(mut self, file_head: &str) -> Self {
        self.file_head = file_head.to_string();
        self
    }

    /// Sets the data index start (NDataIdxStart).
    pub fn data_idx_start(mut self, data_idx_start: usize) -> Self {
        self.data_idx_start = data_idx_start;
        self
    }

    /// Builds the OutputManager.
    pub fn build(self) -> OutputManager {
        OutputManager::new(self.output_dir, &self.file_head, self.data_idx_start)
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
        let manager = OutputManager::new("test_output", "test", 1);

        assert_eq!(manager.output_dir, PathBuf::from("test_output"));
        assert_eq!(manager.file_head, "test");
        assert_eq!(manager.data_idx_start, 1);
    }

    #[test]
    fn test_output_manager_paths() {
        let manager = OutputManager::new("output", "zvo", 1);

        assert_eq!(manager.energy_output_path(), PathBuf::from("output/zvo_out_001.dat"));
        assert_eq!(
            manager.variational_output_path(),
            PathBuf::from("output/zvo_var_001.dat")
        );
        assert_eq!(
            manager.time_output_path(),
            PathBuf::from("output/zvo_time_001.dat")
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

        let manager = OutputManager::new(&test_dir, "test", 1);
        manager.ensure_output_dir().unwrap();

        assert!(test_dir.exists());

        // Clean up
        std::fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn test_output_manager_write_energy() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_energy_write");

        let manager = OutputManager::new(&test_dir, "test", 1);

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

        let manager = OutputManager::new(&test_dir, "test", 1);

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

        let manager = OutputManager::new(&test_dir, "test", 1);

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

        let manager = OutputManager::new(&test_dir, "test", 1);
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
            .file_head("custom")
            .data_idx_start(2)
            .build();

        assert_eq!(manager.output_dir, PathBuf::from("custom_output"));
        assert_eq!(manager.file_head, "custom");
        assert_eq!(manager.data_idx_start, 2);
    }

    #[test]
    fn test_output_manager_builder_default() {
        let manager = OutputManagerBuilder::default().build();

        assert_eq!(manager.output_dir, PathBuf::from("output"));
        assert_eq!(manager.file_head, "zvo");
        assert_eq!(manager.data_idx_start, 1);
    }

    #[test]
    fn test_output_manager_write_optimized_params() {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join("test_opt_params");

        let manager = OutputManager::new(&test_dir, "test", 1);

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

        let manager = OutputManager::new(&test_dir, "test", 1);

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
