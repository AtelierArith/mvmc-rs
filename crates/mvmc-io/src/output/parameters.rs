//! Parameter output module for saving optimized variational parameters.
//!
//! This module provides functionality to save and load optimized variational
//! parameters for reproducibility and restart capabilities.
//!
//! # Reference
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/vmcmain.c`: Parameter saving in `outputData()`
//! - Output files: `zqp_opt.dat` (optimized parameters)

use crate::{IoError, Result};
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Optimized variational parameters.
///
/// Contains the complete set of variational parameters after optimization.
/// Reference: `mVMC/src/mVMC/vmcmain.c` - zqp_opt output
#[derive(Debug, Clone)]
pub struct OptimizedParameters {
    /// Number of parameters
    pub n_params: usize,
    /// Parameter values
    pub parameters: Vec<Complex64>,
    /// Optional parameter names/descriptions
    pub names: Option<Vec<String>>,
}

impl OptimizedParameters {
    /// Creates a new optimized parameters instance.
    pub fn new(parameters: Vec<Complex64>) -> Self {
        let n_params = parameters.len();
        Self {
            n_params,
            parameters,
            names: None,
        }
    }

    /// Creates a new optimized parameters instance with names.
    pub fn with_names(parameters: Vec<Complex64>, names: Vec<String>) -> Result<Self> {
        if parameters.len() != names.len() {
            return Err(IoError::ValidationError(
                "Number of parameters and names must match".to_string(),
            ));
        }

        let n_params = parameters.len();
        Ok(Self {
            n_params,
            parameters,
            names: Some(names),
        })
    }

    /// Writes optimized parameters to a file in text format.
    ///
    /// Format: One parameter per line as `Re(p) Im(p)`
    ///
    /// Reference: C implementation outputs complex parameters as pairs
    pub fn write_text<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header with number of parameters
        writeln!(writer, "# Optimized variational parameters")?;
        writeln!(writer, "# Number of parameters: {}", self.n_params)?;

        // Write parameters
        for (i, param) in self.parameters.iter().enumerate() {
            if let Some(ref names) = self.names {
                writeln!(
                    writer,
                    "{} {:.18e} {:.18e}  # {}",
                    i, param.re, param.im, names[i]
                )?;
            } else {
                writeln!(writer, "{} {:.18e} {:.18e}", i, param.re, param.im)?;
            }
        }

        Ok(())
    }

    /// Reads optimized parameters from a text file.
    ///
    /// Parses files in the format produced by `write_text`.
    pub fn read_text<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut parameters = Vec::new();
        let mut names = Vec::new();
        let mut has_names = false;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments (unless they contain parameter info)
            if line.is_empty() || (line.starts_with('#') && !line.contains("Number")) {
                continue;
            }

            // Skip header lines
            if line.starts_with('#') {
                continue;
            }

            // Parse parameter line: "index real imag [# name]"
            let parts: Vec<&str> = line.split('#').collect();
            let values_part = parts[0].trim();
            let name_part = if parts.len() > 1 {
                has_names = true;
                Some(parts[1].trim().to_string())
            } else {
                None
            };

            let values: Vec<&str> = values_part.split_whitespace().collect();
            if values.len() >= 3 {
                let re: f64 = values[1]
                    .parse()
                    .map_err(|e| IoError::ParseError(format!("Failed to parse real part: {}", e)))?;
                let im: f64 = values[2]
                    .parse()
                    .map_err(|e| IoError::ParseError(format!("Failed to parse imag part: {}", e)))?;

                parameters.push(Complex64::new(re, im));

                if let Some(name) = name_part {
                    names.push(name);
                }
            }
        }

        if parameters.is_empty() {
            return Err(IoError::ParseError(
                "No parameters found in file".to_string(),
            ));
        }

        if has_names {
            Self::with_names(parameters, names)
        } else {
            Ok(Self::new(parameters))
        }
    }

    /// Writes parameters in a compact format suitable for restart.
    ///
    /// Format: All parameters on one line, space-separated
    pub fn write_compact<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "{}", self.n_params)?;

        for param in &self.parameters {
            write!(writer, "{:.18e} {:.18e} ", param.re, param.im)?;
        }
        writeln!(writer)?;

        Ok(())
    }

    /// Reads parameters from compact format.
    pub fn read_compact<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;

        let n_params: usize = first_line
            .trim()
            .parse()
            .map_err(|e| IoError::ParseError(format!("Failed to parse n_params: {}", e)))?;

        let mut second_line = String::new();
        reader.read_line(&mut second_line)?;

        let values: Vec<f64> = second_line
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|e| IoError::ParseError(format!("Failed to parse value: {}", e)))
            })
            .collect::<Result<Vec<f64>>>()?;

        if values.len() != n_params * 2 {
            return Err(IoError::ParseError(format!(
                "Expected {} values, found {}",
                n_params * 2,
                values.len()
            )));
        }

        let parameters: Vec<Complex64> = values
            .chunks(2)
            .map(|chunk| Complex64::new(chunk[0], chunk[1]))
            .collect();

        Ok(Self::new(parameters))
    }

    /// Writes parameters in binary format for efficient storage.
    pub fn write_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write number of parameters
        writer.write_all(&(self.n_params as u64).to_le_bytes())?;

        // Write parameters
        for param in &self.parameters {
            writer.write_all(&param.re.to_le_bytes())?;
            writer.write_all(&param.im.to_le_bytes())?;
        }

        Ok(())
    }

    /// Reads parameters from binary format.
    pub fn read_binary<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buffer = [0u8; 8];

        // Read number of parameters
        file.read_exact(&mut buffer)?;
        let n_params = u64::from_le_bytes(buffer) as usize;

        // Read parameters
        let mut parameters = Vec::with_capacity(n_params);
        for _ in 0..n_params {
            file.read_exact(&mut buffer)?;
            let re = f64::from_le_bytes(buffer);
            file.read_exact(&mut buffer)?;
            let im = f64::from_le_bytes(buffer);
            parameters.push(Complex64::new(re, im));
        }

        Ok(Self::new(parameters))
    }

    /// Gets a parameter by index.
    pub fn get(&self, index: usize) -> Option<Complex64> {
        self.parameters.get(index).copied()
    }

    /// Sets a parameter by index.
    pub fn set(&mut self, index: usize, value: Complex64) -> Result<()> {
        if index >= self.n_params {
            return Err(IoError::ValidationError(format!(
                "Parameter index {} out of range (0-{})",
                index,
                self.n_params - 1
            )));
        }
        self.parameters[index] = value;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_optimized_parameters_new() {
        let params = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(3.0, 0.0),
        ];

        let opt_params = OptimizedParameters::new(params.clone());

        assert_eq!(opt_params.n_params, 3);
        assert_eq!(opt_params.parameters, params);
        assert!(opt_params.names.is_none());
    }

    #[test]
    fn test_optimized_parameters_with_names() {
        let params = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let names = vec!["param1".to_string(), "param2".to_string()];

        let opt_params = OptimizedParameters::with_names(params.clone(), names.clone()).unwrap();

        assert_eq!(opt_params.n_params, 2);
        assert_eq!(opt_params.names, Some(names));
    }

    #[test]
    fn test_optimized_parameters_with_names_mismatch() {
        let params = vec![Complex64::new(1.0, 0.0)];
        let names = vec!["param1".to_string(), "param2".to_string()];

        let result = OptimizedParameters::with_names(params, names);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_read_text() {
        let params = vec![
            Complex64::new(1.5, -0.5),
            Complex64::new(2.5, 1.5),
            Complex64::new(3.5, -1.5),
        ];

        let opt_params = OptimizedParameters::new(params.clone());

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_params.txt");

        opt_params.write_text(&file_path).unwrap();

        let loaded = OptimizedParameters::read_text(&file_path).unwrap();

        assert_eq!(loaded.n_params, 3);
        for (i, param) in loaded.parameters.iter().enumerate() {
            assert_relative_eq!(param.re, params[i].re, epsilon = 1e-15);
            assert_relative_eq!(param.im, params[i].im, epsilon = 1e-15);
        }

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_write_read_text_with_names() {
        let params = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let names = vec!["alpha".to_string(), "beta".to_string()];

        let opt_params = OptimizedParameters::with_names(params.clone(), names.clone()).unwrap();

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_params_named.txt");

        opt_params.write_text(&file_path).unwrap();

        let loaded = OptimizedParameters::read_text(&file_path).unwrap();

        assert_eq!(loaded.n_params, 2);
        assert_eq!(loaded.names, Some(names));

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_write_read_compact() {
        let params = vec![
            Complex64::new(1.5, -0.5),
            Complex64::new(2.5, 1.5),
        ];

        let opt_params = OptimizedParameters::new(params.clone());

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_params_compact.txt");

        opt_params.write_compact(&file_path).unwrap();

        let loaded = OptimizedParameters::read_compact(&file_path).unwrap();

        assert_eq!(loaded.n_params, 2);
        for (i, param) in loaded.parameters.iter().enumerate() {
            assert_relative_eq!(param.re, params[i].re, epsilon = 1e-15);
            assert_relative_eq!(param.im, params[i].im, epsilon = 1e-15);
        }

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_write_read_binary() {
        let params = vec![
            Complex64::new(1.5, -0.5),
            Complex64::new(2.5, 1.5),
            Complex64::new(3.5, -1.5),
        ];

        let opt_params = OptimizedParameters::new(params.clone());

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_params.bin");

        opt_params.write_binary(&file_path).unwrap();

        let loaded = OptimizedParameters::read_binary(&file_path).unwrap();

        assert_eq!(loaded.n_params, 3);
        for (i, param) in loaded.parameters.iter().enumerate() {
            assert_relative_eq!(param.re, params[i].re, epsilon = 1e-15);
            assert_relative_eq!(param.im, params[i].im, epsilon = 1e-15);
        }

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_get_set_parameter() {
        let params = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let mut opt_params = OptimizedParameters::new(params);

        // Test get
        assert_eq!(opt_params.get(0), Some(Complex64::new(1.0, 0.0)));
        assert_eq!(opt_params.get(1), Some(Complex64::new(2.0, 0.0)));
        assert_eq!(opt_params.get(2), None);

        // Test set
        opt_params.set(0, Complex64::new(3.0, 0.5)).unwrap();
        assert_eq!(opt_params.get(0), Some(Complex64::new(3.0, 0.5)));

        // Test set out of range
        let result = opt_params.set(5, Complex64::new(0.0, 0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_empty_file() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_empty.txt");

        std::fs::write(&file_path, "# Only comments\n# No parameters\n").unwrap();

        let result = OptimizedParameters::read_text(&file_path);
        assert!(result.is_err());

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }

    #[test]
    fn test_binary_file_size() {
        let params = vec![Complex64::new(1.0, 0.0); 10];
        let opt_params = OptimizedParameters::new(params);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_params_size.bin");

        opt_params.write_binary(&file_path).unwrap();

        let metadata = std::fs::metadata(&file_path).unwrap();
        // Size should be: 8 bytes (n_params) + 10 * 2 * 8 bytes (10 complex numbers)
        assert_eq!(metadata.len(), 8 + 10 * 2 * 8);

        // Clean up
        std::fs::remove_file(&file_path).unwrap();
    }
}
