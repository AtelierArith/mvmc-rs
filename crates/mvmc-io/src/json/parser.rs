//! JSON format parser implementation.

use super::{JsonConfig, JsonError};
use super::error::Result;
use crate::{ConfigParser, IoError};

/// JSON format parser.
///
/// This parser handles JSON configuration files for mVMC calculations.
#[derive(Debug, Clone)]
pub struct JsonParser;

impl JsonParser {
    /// Creates a new JSON parser.
    pub fn new() -> Self {
        Self
    }
}

impl ConfigParser<JsonConfig> for JsonParser {
    fn parse_str(&self, input: &str) -> crate::Result<JsonConfig> {
        let config: JsonConfig = serde_json::from_str(input)
            .map_err(|e| IoError::ParseError(format!("JSON parse error: {}", e)))?;

        // Basic validation
        if config.lattice.dimensions.is_empty() {
            return Err(IoError::ValidationError("Lattice dimensions cannot be empty".to_string()));
        }

        for &dim in &config.lattice.dimensions {
            if dim == 0 {
                return Err(IoError::ValidationError("Lattice dimensions must be positive".to_string()));
            }
        }

        Ok(config)
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json() {
        let input = r#"{
            "lattice": {
                "lattice_type": "chain",
                "dimensions": [6],
                "sub_dimensions": [2]
            },
            "model": {
                "model_type": "Hubbard",
                "parameters": {
                    "t": 1.0,
                    "U": 4.0
                }
            },
            "calculation": {
                "n_particles": 6,
                "total_sz": 0,
                "random_seed": 1
            },
            "optimization": {
                "sr_steps": 500,
                "sr_reduction_cutoff": 1e-8,
                "sr_stabilization_delta": 1e-2,
                "sr_step_delta": 3e-3
            },
            "monte_carlo": {
                "vmc_samples": 100,
                "vmc_calculation_mode": 0
            }
        }"#;

        let parser = JsonParser::new();
        let config = parser.parse_str(input).unwrap();

        assert_eq!(config.lattice.lattice_type, "chain");
        assert_eq!(config.lattice.dimensions, vec![6]);
        assert_eq!(config.model.model_type, "Hubbard");
        assert_eq!(config.model.parameters.get("t"), Some(&1.0));
        assert_eq!(config.model.parameters.get("U"), Some(&4.0));
    }
}
