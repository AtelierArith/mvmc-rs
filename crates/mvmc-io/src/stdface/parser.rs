//! StdFace format parser implementation.

use super::{StdFaceConfig, StdFaceError};
use super::error::Result;
use std::collections::HashMap;

/// StdFace format parser.
///
/// This parser handles the StdFace format used by mVMC for input configuration.
/// The format is a simple key-value format with some special handling for
/// different parameter types.
#[derive(Debug, Clone)]
pub struct StdFaceParser {
    /// Whether to be strict about parameter validation
    strict_mode: bool,
}

impl StdFaceParser {
    /// Creates a new StdFace parser.
    pub fn new() -> Self {
        Self { strict_mode: false }
    }

    /// Creates a new StdFace parser with strict mode enabled.
    pub fn strict() -> Self {
        Self { strict_mode: true }
    }

    /// Parses a StdFace configuration from a file.
    pub fn parse_file(&self, path: &str) -> Result<StdFaceConfig> {
        let content = std::fs::read_to_string(path)?;
        self.parse_str(&content)
    }

    /// Parses a StdFace configuration from a string.
    pub fn parse_str(&self, input: &str) -> Result<StdFaceConfig> {
        let mut config = StdFaceConfig::new();
        let mut additional = HashMap::new();

        let lines: Vec<&str> = input.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
                continue;
            }

            // Parse key-value pairs
            if let Some((key, value)) = self.parse_key_value(line) {
                match self.process_parameter(&key, &value, &mut config, &mut additional) {
                    Ok(()) => {}
                    Err(e) => {
                        if self.strict_mode {
                            return Err(StdFaceError::ParseError {
                                line: line_num + 1,
                                message: format!("Error processing parameter '{}': {}", key, e),
                            });
                        }
                        // In non-strict mode, add to additional parameters
                        additional.insert(key, value);
                    }
                }
            } else {
                if self.strict_mode {
                    return Err(StdFaceError::ParseError {
                        line: line_num + 1,
                        message: format!("Invalid line format: {}", line),
                    });
                }
            }
        }

        config.additional = additional;

        // Set default particle count based on model type
        config.set_default_particle_count();

        // Validate the configuration
        if self.strict_mode {
            config.validate()?;
        }

        Ok(config)
    }

    /// Parses a key-value pair from a line.
    fn parse_key_value(&self, line: &str) -> Option<(String, String)> {
        // Handle different separators: =, space, tab
        let separators = [" = ", "=", " ", "\t"];

        for sep in &separators {
            if let Some(pos) = line.find(sep) {
                let key = line[..pos].trim().to_string();
                let value = line[pos + sep.len()..].trim().to_string();

                if !key.is_empty() && !value.is_empty() {
                    return Some((key, value));
                }
            }
        }

        None
    }

    /// Processes a parameter and updates the configuration.
    fn process_parameter(
        &self,
        key: &str,
        value: &str,
        config: &mut StdFaceConfig,
        additional: &mut HashMap<String, String>,
    ) -> Result<()> {
        match key {
            // Lattice parameters
            "L" => {
                let dim = self.parse_positive_integer(value)?;
                if config.lattice.dimensions.is_empty() {
                    config.lattice.dimensions = vec![dim];
                } else if config.lattice.dimensions.len() == 1 {
                    // If we already have one dimension, this is the second dimension
                    config.lattice.dimensions.push(dim);
                } else {
                    // If we have two dimensions, update the second one
                    config.lattice.dimensions[1] = dim;
                }
            }
            "W" => {
                let dim = self.parse_positive_integer(value)?;
                if config.lattice.dimensions.is_empty() {
                    config.lattice.dimensions = vec![dim, 0];
                } else {
                    config.lattice.dimensions[0] = dim;
                }
            }
            "Lsub" => {
                let dim = self.parse_positive_integer(value)?;
                if config.lattice.sub_dimensions.is_empty() {
                    config.lattice.sub_dimensions = vec![dim];
                } else if config.lattice.sub_dimensions.len() == 1 {
                    config.lattice.sub_dimensions.push(dim);
                } else {
                    config.lattice.sub_dimensions[1] = dim;
                }
            }
            "Wsub" => {
                let dim = self.parse_positive_integer(value)?;
                if config.lattice.sub_dimensions.is_empty() {
                    config.lattice.sub_dimensions = vec![dim, 0];
                } else {
                    config.lattice.sub_dimensions[0] = dim;
                }
            }
            "lattice" => {
                config.lattice.lattice_type = self.strip_quotes(value).to_string();
            }

            // Model parameters
            "model" => {
                config.model.model_type = self.strip_quotes(value).to_string();
            }
            "t" => {
                let val = self.parse_float(value)?;
                config.model.parameters.insert("t".to_string(), val);
            }
            "U" => {
                let val = self.parse_float(value)?;
                config.model.parameters.insert("U".to_string(), val);
            }
            "J" => {
                let val = self.parse_float(value)?;
                config.model.parameters.insert("J".to_string(), val);
            }
            "h" => {
                let val = self.parse_float(value)?;
                config.model.parameters.insert("h".to_string(), val);
            }
            "mu" => {
                let val = self.parse_float(value)?;
                config.model.parameters.insert("mu".to_string(), val);
            }

            // Calculation parameters
            "Ncond" | "ncond" => {
                let val = self.parse_positive_integer(value)?;
                config.calculation.n_particles = Some(val);
            }
            "2Sz" => {
                let val = self.parse_integer(value)?;
                config.calculation.total_sz = Some(val);
            }
            "RndSeed" => {
                let val = self.parse_positive_integer(value)? as u64;
                config.calculation.random_seed = Some(val);
            }

            // Optimization parameters
            "NSROptItrStep" => {
                let val = self.parse_positive_integer(value)?;
                config.optimization.sr_steps = Some(val);
            }
            "DSROptRedCut" => {
                let val = self.parse_float(value)?;
                config.optimization.sr_reduction_cutoff = Some(val);
            }
            "DSROptStaDel" => {
                let val = self.parse_float(value)?;
                config.optimization.sr_stabilization_delta = Some(val);
            }
            "DSROptStepDt" => {
                let val = self.parse_float(value)?;
                config.optimization.sr_step_delta = Some(val);
            }
            "NSRCG" => {
                let val = self.parse_positive_integer(value)?;
                config.optimization.sr_cg = Some(val);
            }
            "NSROptCGMaxIter" => {
                let val = self.parse_positive_integer(value)?;
                config.optimization.sr_cg_max_iter = Some(val);
            }
            "DSROptCGTol" => {
                let val = self.parse_float(value)?;
                config.optimization.sr_cg_tol = Some(val);
            }

            // Monte Carlo parameters
            "NVMCSample" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.vmc_samples = Some(val);
            }
            "NVMCWarmUp" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.vmc_warmup_steps = Some(val);
            }
            "NVMCInterval" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.vmc_sampling_interval = Some(val);
            }
            "NVMCCalMode" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.vmc_calculation_mode = Some(val);
            }
            "NExUpdatePath" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.ex_update_path = Some(val);
            }
            "NBlockUpdateSize" => {
                let val = self.parse_positive_integer(value)?;
                config.monte_carlo.block_update_size = Some(val);
            }
            "NExUpdateRatio" => {
                let val = self.parse_float(value)?;
                config.monte_carlo.ex_update_ratio = Some(val);
            }

            // Unknown parameters go to additional
            _ => {
                additional.insert(key.to_string(), value.to_string());
            }
        }

        Ok(())
    }

    /// Parses a positive integer from a string.
    fn parse_positive_integer(&self, value: &str) -> Result<usize> {
        let val = value.parse::<usize>()
            .map_err(|_| StdFaceError::InvalidParameterValue {
                parameter: "integer".to_string(),
                value: value.to_string(),
            })?;

        if val == 0 {
            return Err(StdFaceError::InvalidParameterValue {
                parameter: "positive integer".to_string(),
                value: value.to_string(),
            });
        }

        Ok(val)
    }

    /// Parses an integer from a string.
    fn parse_integer(&self, value: &str) -> Result<i32> {
        value.parse::<i32>()
            .map_err(|_| StdFaceError::InvalidParameterValue {
                parameter: "integer".to_string(),
                value: value.to_string(),
            })
    }

    /// Parses a float from a string.
    fn parse_float(&self, value: &str) -> Result<f64> {
        value.parse::<f64>()
            .map_err(|_| StdFaceError::InvalidParameterValue {
                parameter: "float".to_string(),
                value: value.to_string(),
            })
    }

    /// Strips quotes from a string value.
    fn strip_quotes<'a>(&self, value: &'a str) -> &'a str {
        let value = value.trim();
        if (value.starts_with('"') && value.ends_with('"')) ||
           (value.starts_with('\'') && value.ends_with('\'')) {
            &value[1..value.len()-1]
        } else {
            value
        }
    }
}

impl Default for StdFaceParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hubbard_chain() {
        let input = r#"
L = 6
Lsub = 2
model = "Hubbard"
lattice = "chain"
U = 4.0
t = 1.0
Ncond = 6
NSROptItrStep = 500
NVMCSample = 100
2Sz = 0
DSROptRedCut = 1e-8
DSROptStaDel = 1e-2
DSROptStepDt = 3e-3
RndSeed = 1
"#;

        let parser = StdFaceParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.lattice.lattice_type, "chain");
        assert_eq!(config.lattice.dimensions, vec![6]);
        assert_eq!(config.lattice.sub_dimensions, vec![2]);
        assert_eq!(config.model.model_type, "Hubbard");
        assert_eq!(config.model.parameters.get("t"), Some(&1.0));
        assert_eq!(config.model.parameters.get("U"), Some(&4.0));
        assert_eq!(config.calculation.n_particles, Some(6));
        assert_eq!(config.calculation.total_sz, Some(0));
        assert_eq!(config.calculation.random_seed, Some(1));
    }

    #[test]
    fn test_parse_heisenberg_chain() {
        let input = r#"
L = 6
Lsub = 2
model = "Spin"
lattice = "chain"
J = 1.0
NSROptItrStep = 1000
NVMCSample = 100
2Sz = 0
DSROptRedCut = 1e-10
DSROptStaDel = 1e-5
DSROptStepDt = 1e-2
RndSeed = 1
"#;

        let parser = StdFaceParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.model.model_type, "Spin");
        assert_eq!(config.model.parameters.get("J"), Some(&1.0));
        assert_eq!(config.optimization.sr_steps, Some(1000));
    }

    #[test]
    fn test_parse_2d_lattice() {
        let input = r#"
W = 4
L = 2
Wsub = 2
Lsub = 2
model = "FermionHubbard"
lattice = "Tetragonal"
t = 1.0
U = 4.0
ncond = 8
"#;

        let parser = StdFaceParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.lattice.dimensions, vec![4, 2]);
        assert_eq!(config.lattice.sub_dimensions, vec![2, 2]);
        assert_eq!(config.lattice.lattice_type, "Tetragonal");
        assert_eq!(config.model.model_type, "FermionHubbard");
        assert_eq!(config.calculation.n_particles, Some(8));
    }

    #[test]
    fn test_parse_with_comments() {
        let input = r#"
L = 6
// This is a comment
model = "Hubbard"
# Another comment
t = 1.0
U = 4.0
"#;

        let parser = StdFaceParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.lattice.dimensions, vec![6]);
        assert_eq!(config.model.model_type, "Hubbard");
        assert_eq!(config.model.parameters.get("t"), Some(&1.0));
        assert_eq!(config.model.parameters.get("U"), Some(&4.0));
    }

    #[test]
    fn test_parse_invalid_integer() {
        let input = "L = abc";

        let parser = StdFaceParser::strict();
        let result = parser.parse_str(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_float() {
        let input = "t = abc";

        let parser = StdFaceParser::strict();
        let result = parser.parse_str(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_lines() {
        let input = r#"
L = 6

model = "Hubbard"

t = 1.0
"#;

        let parser = StdFaceParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.lattice.dimensions, vec![6]);
        assert_eq!(config.model.model_type, "Hubbard");
        assert_eq!(config.model.parameters.get("t"), Some(&1.0));
    }
}
