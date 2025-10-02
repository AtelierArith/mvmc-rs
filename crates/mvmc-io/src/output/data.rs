//! Data output module for mVMC calculations.
//!
//! This module provides functionality to write calculation results to files
//! in various formats (text, binary).
//!
//! # Reference
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/vmcmain.c`: `outputData()` function
//! - Output files: `zvo_out.dat`, `zvo_var.dat`

use crate::Result;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Format for data output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Formatted text output (default).
    Text,
    /// Binary output for large datasets.
    Binary,
}

/// Energy statistics from VMC calculation.
///
/// Corresponds to the energy output in C implementation.
/// Reference: `mVMC/src/mVMC/vmcmain.c:outputData()`
#[derive(Debug, Clone, Copy)]
pub struct EnergyData {
    /// Total energy ⟨E⟩
    pub energy: Complex64,
    /// Energy squared ⟨E²⟩
    pub energy_squared: Complex64,
    /// Variance of energy (E²-E²)/E²
    pub variance: Complex64,
    /// Total Sz ⟨Sz⟩
    pub sz_total: Complex64,
    /// Sz squared ⟨Sz²⟩
    pub sz_squared: Complex64,
}

impl EnergyData {
    /// Creates a new energy data instance.
    pub fn new(
        energy: Complex64,
        energy_squared: Complex64,
        sz_total: Complex64,
        sz_squared: Complex64,
    ) -> Self {
        let variance = (energy_squared - energy * energy) / (energy * energy);
        Self {
            energy,
            energy_squared,
            variance,
            sz_total,
            sz_squared,
        }
    }

    /// Writes energy data to a file in text format.
    ///
    /// Format: `Re(E) Im(E) Re(E²) Re(Var) Re(Sz) Re(Sz²)`
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c:outputData()` line ~1040
    pub fn write_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "{:.18e} {:.18e}  {:.18e} {:.18e} {:.18e} {:.18e}",
            self.energy.re,
            self.energy.im,
            self.energy_squared.re,
            self.variance.re,
            self.sz_total.re,
            self.sz_squared.re
        )?;

        Ok(())
    }

    /// Writes energy data to a file in binary format.
    pub fn write_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write as double-precision floating point numbers
        writer.write_all(&self.energy.re.to_le_bytes())?;
        writer.write_all(&self.energy.im.to_le_bytes())?;
        writer.write_all(&self.energy_squared.re.to_le_bytes())?;
        writer.write_all(&self.variance.re.to_le_bytes())?;
        writer.write_all(&self.sz_total.re.to_le_bytes())?;
        writer.write_all(&self.sz_squared.re.to_le_bytes())?;

        Ok(())
    }

    /// Writes energy data to a file with specified format.
    pub fn write<P: AsRef<Path>>(&self, path: P, format: OutputFormat) -> Result<()> {
        match format {
            OutputFormat::Text => self.write_text(path),
            OutputFormat::Binary => self.write_binary(path),
        }
    }
}

/// Variational parameter data.
///
/// Contains optimized variational parameters from VMC calculation.
/// Reference: `mVMC/src/mVMC/vmcmain.c:outputData()` - zvo_var.dat output
#[derive(Debug, Clone)]
pub struct VariationalData {
    /// Energy statistics
    pub energy: EnergyData,
    /// Variational parameters
    pub parameters: Vec<Complex64>,
}

impl VariationalData {
    /// Creates a new variational data instance.
    pub fn new(energy: EnergyData, parameters: Vec<Complex64>) -> Self {
        Self { energy, parameters }
    }

    /// Writes variational data to a file in text format.
    ///
    /// Format: `Re(E) Im(E) 0.0 Re(E²) Im(E²) 0.0 [Re(p_i) Im(p_i) 0.0]...`
    ///
    /// Reference: `mVMC/src/mVMC/vmcmain.c:outputData()` line ~1043-1046
    pub fn write_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write energy (real, imag, 0.0) and energy squared (real, imag, 0.0)
        write!(
            writer,
            "{:.18e} {:.18e} 0.0 {:.18e} {:.18e} 0.0 ",
            self.energy.energy.re,
            self.energy.energy.im,
            self.energy.energy_squared.re,
            self.energy.energy_squared.im
        )?;

        // Write parameters (real, imag, 0.0 for each)
        for param in &self.parameters {
            write!(writer, "{:.18e} {:.18e} 0.0 ", param.re, param.im)?;
        }

        writeln!(writer)?;

        Ok(())
    }

    /// Writes variational data to a file in binary format.
    pub fn write_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write energy
        writer.write_all(&self.energy.energy.re.to_le_bytes())?;
        writer.write_all(&self.energy.energy.im.to_le_bytes())?;
        writer.write_all(&self.energy.energy_squared.re.to_le_bytes())?;
        writer.write_all(&self.energy.energy_squared.im.to_le_bytes())?;

        // Write parameters
        for param in &self.parameters {
            writer.write_all(&param.re.to_le_bytes())?;
            writer.write_all(&param.im.to_le_bytes())?;
        }

        Ok(())
    }

    /// Writes variational data to a file with specified format.
    pub fn write<P: AsRef<Path>>(&self, path: P, format: OutputFormat) -> Result<()> {
        match format {
            OutputFormat::Text => self.write_text(path),
            OutputFormat::Binary => self.write_binary(path),
        }
    }

    /// Appends variational data to an existing file in text format.
    ///
    /// Useful for writing multiple optimization steps.
    pub fn append_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        write!(
            writer,
            "{:.18e} {:.18e} 0.0 {:.18e} {:.18e} 0.0 ",
            self.energy.energy.re,
            self.energy.energy.im,
            self.energy.energy_squared.re,
            self.energy.energy_squared.im
        )?;

        for param in &self.parameters {
            write!(writer, "{:.18e} {:.18e} 0.0 ", param.re, param.im)?;
        }

        writeln!(writer)?;

        Ok(())
    }
}

/// Observable data for various physical quantities.
///
/// Contains calculated observables such as correlations, Green's functions, etc.
#[derive(Debug, Clone)]
pub struct ObservableData {
    /// Name of the observable
    pub name: String,
    /// Values of the observable (site/momentum dependent)
    pub values: Vec<Complex64>,
}

impl ObservableData {
    /// Creates a new observable data instance.
    pub fn new(name: String, values: Vec<Complex64>) -> Self {
        Self { name, values }
    }

    /// Writes observable data to a file in text format.
    ///
    /// Format: One line per value with index and complex number
    pub fn write_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# {}", self.name)?;
        for (i, value) in self.values.iter().enumerate() {
            writeln!(writer, "{} {:.18e} {:.18e}", i, value.re, value.im)?;
        }

        Ok(())
    }

    /// Writes observable data to a file in binary format.
    pub fn write_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write number of values
        writer.write_all(&(self.values.len() as u64).to_le_bytes())?;

        // Write values
        for value in &self.values {
            writer.write_all(&value.re.to_le_bytes())?;
            writer.write_all(&value.im.to_le_bytes())?;
        }

        Ok(())
    }

    /// Writes observable data to a file with specified format.
    pub fn write<P: AsRef<Path>>(&self, path: P, format: OutputFormat) -> Result<()> {
        match format {
            OutputFormat::Text => self.write_text(path),
            OutputFormat::Binary => self.write_binary(path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::io::Read;

    #[test]
    fn test_energy_data_new() {
        let energy = Complex64::new(1.0, 0.1);
        let energy_squared = Complex64::new(1.5, 0.2);
        let sz_total = Complex64::new(0.5, 0.0);
        let sz_squared = Complex64::new(0.25, 0.0);

        let data = EnergyData::new(energy, energy_squared, sz_total, sz_squared);

        assert_eq!(data.energy, energy);
        assert_eq!(data.energy_squared, energy_squared);
        assert_eq!(data.sz_total, sz_total);
        assert_eq!(data.sz_squared, sz_squared);
    }

    #[test]
    fn test_energy_data_write_text() {
        let energy = Complex64::new(-2.5, 0.0);
        let energy_squared = Complex64::new(6.5, 0.0);
        let sz_total = Complex64::new(1.0, 0.0);
        let sz_squared = Complex64::new(1.0, 0.0);

        let data = EnergyData::new(energy, energy_squared, sz_total, sz_squared);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_energy_out.dat");

        data.write_text(&file_path).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("-2.500000000000000000e0"));
        assert!(content.contains("0.000000000000000000e0")); // imaginary part

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_energy_data_write_binary() {
        let energy = Complex64::new(-2.5, 0.0);
        let energy_squared = Complex64::new(6.5, 0.0);
        let sz_total = Complex64::new(1.0, 0.0);
        let sz_squared = Complex64::new(1.0, 0.0);

        let data = EnergyData::new(energy, energy_squared, sz_total, sz_squared);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_energy_out.bin");

        data.write_binary(&file_path).unwrap();

        // Verify file exists and has correct size (6 f64 values)
        let metadata = std::fs::metadata(&file_path).unwrap();
        assert_eq!(metadata.len(), 6 * 8); // 6 double values

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_variational_data_write_text() {
        let energy = Complex64::new(-2.5, 0.0);
        let energy_squared = Complex64::new(6.5, 0.0);
        let sz_total = Complex64::new(1.0, 0.0);
        let sz_squared = Complex64::new(1.0, 0.0);
        let energy_data = EnergyData::new(energy, energy_squared, sz_total, sz_squared);

        let parameters = vec![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.1, 0.0),
        ];

        let var_data = VariationalData::new(energy_data, parameters);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_var_out.dat");

        var_data.write_text(&file_path).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("-2.500000000000000000e0"));
        assert!(content.contains("5.000000000000000000e-1")); // first parameter real

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_variational_data_append() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_var_append_unique.dat");

        // Remove old file if exists
        let _ = std::fs::remove_file(&file_path);

        // First write
        let energy1 = EnergyData::new(
            Complex64::new(-2.5, 0.0),
            Complex64::new(6.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let var_data1 = VariationalData::new(energy1, vec![Complex64::new(0.5, 0.1)]);
        var_data1.append_text(&file_path).unwrap();

        // Second write
        let energy2 = EnergyData::new(
            Complex64::new(-2.6, 0.0),
            Complex64::new(6.8, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let var_data2 = VariationalData::new(energy2, vec![Complex64::new(0.6, 0.2)]);
        var_data2.append_text(&file_path).unwrap();

        // Verify both lines exist
        let content = std::fs::read_to_string(&file_path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("-2.5"));
        assert!(lines[1].contains("-2.6"));

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_observable_data_write_text() {
        let values = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.5, 0.25),
            Complex64::new(0.25, 0.125),
        ];

        let obs_data = ObservableData::new("correlation".to_string(), values);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_observable.dat");

        obs_data.write_text(&file_path).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("# correlation"));
        assert!(content.contains("0 1.000000000000000000e0"));

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_output_format_enum() {
        assert_eq!(OutputFormat::Text, OutputFormat::Text);
        assert_ne!(OutputFormat::Text, OutputFormat::Binary);
    }

    #[test]
    fn test_energy_variance_calculation() {
        let energy = Complex64::new(2.0, 0.0);
        let energy_squared = Complex64::new(5.0, 0.0);
        let sz_total = Complex64::new(0.0, 0.0);
        let sz_squared = Complex64::new(0.0, 0.0);

        let data = EnergyData::new(energy, energy_squared, sz_total, sz_squared);

        // Variance = (E² - E²) / E² = (5 - 4) / 4 = 0.25
        let expected_variance = 0.25;
        assert_relative_eq!(data.variance.re, expected_variance, epsilon = 1e-10);
    }

    #[test]
    fn test_observable_data_binary_roundtrip() {
        let values = vec![
            Complex64::new(1.5, -0.5),
            Complex64::new(2.5, 1.5),
        ];

        let obs_data = ObservableData::new("test".to_string(), values.clone());

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_observable.bin");

        obs_data.write_binary(&file_path).unwrap();

        // Read back manually
        let mut file = File::open(&file_path).unwrap();
        let mut buffer = [0u8; 8];

        // Read length
        file.read_exact(&mut buffer).unwrap();
        let length = u64::from_le_bytes(buffer);
        assert_eq!(length, 2);

        // Read first complex number
        file.read_exact(&mut buffer).unwrap();
        let re1 = f64::from_le_bytes(buffer);
        file.read_exact(&mut buffer).unwrap();
        let im1 = f64::from_le_bytes(buffer);
        assert_relative_eq!(re1, 1.5, epsilon = 1e-10);
        assert_relative_eq!(im1, -0.5, epsilon = 1e-10);

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }
}
