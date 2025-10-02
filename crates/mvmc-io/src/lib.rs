//! mvmc-io: Input/output handling for mVMC calculations.
//!
//! This crate provides parsers and writers for various input/output formats
//! used in mVMC calculations, including StdFace format, TOML, and JSON.
//!
//! It also provides output functionality for VMC calculation results.

pub mod stdface;
pub mod toml;
pub mod json;
pub mod output;

// Re-export commonly used types
pub use stdface::{StdFaceParser, StdFaceError, StdFaceConfig};
pub use toml::{TomlParser, TomlError, TomlConfig};
pub use json::{JsonParser, JsonError, JsonConfig};
pub use output::{
    EnergyData, ObservableData, OptimizedParameters, OutputFormat, OutputManager,
    VariationalData,
};

/// Common error types for I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("File I/O error: {0}")]
    FileError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// Result type for I/O operations.
pub type Result<T> = std::result::Result<T, IoError>;

/// Trait for configuration parsers.
pub trait ConfigParser<T> {
    /// Parses configuration from a string.
    fn parse_str(&self, input: &str) -> Result<T>;

    /// Parses configuration from a file.
    fn parse_file(&self, path: &str) -> Result<T> {
        let content = std::fs::read_to_string(path)?;
        self.parse_str(&content)
    }
}

/// Trait for configuration writers.
pub trait ConfigWriter<T> {
    /// Writes configuration to a string.
    fn write_str(&self, config: &T) -> Result<String>;

    /// Writes configuration to a file.
    fn write_file(&self, config: &T, path: &str) -> Result<()> {
        let content = self.write_str(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Utility functions for I/O operations.
pub mod utils {

    /// Detects the format of a configuration file based on its extension.
    pub fn detect_format(filename: &str) -> Option<&'static str> {
        let ext = filename.split('.').last()?.to_lowercase();
        match ext.as_str() {
            "def" => Some("stdface"),
            "toml" => Some("toml"),
            "json" => Some("json"),
            _ => None,
        }
    }

    /// Normalizes a file path for cross-platform compatibility.
    pub fn normalize_path(path: &str) -> String {
        path.replace('\\', "/")
    }

    /// Extracts the base name of a file without extension.
    pub fn base_name(filename: &str) -> String {
        filename
            .split('/')
            .last()
            .unwrap_or(filename)
            .split('.')
            .next()
            .unwrap_or(filename)
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::utils;

    #[test]
    fn test_detect_format() {
        assert_eq!(utils::detect_format("config.def"), Some("stdface"));
        assert_eq!(utils::detect_format("config.toml"), Some("toml"));
        assert_eq!(utils::detect_format("config.json"), Some("json"));
        assert_eq!(utils::detect_format("config.txt"), None);
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(utils::normalize_path("path\\to\\file"), "path/to/file");
        assert_eq!(utils::normalize_path("path/to/file"), "path/to/file");
    }

    #[test]
    fn test_base_name() {
        assert_eq!(utils::base_name("path/to/file.def"), "file");
        assert_eq!(utils::base_name("file.def"), "file");
        assert_eq!(utils::base_name("file"), "file");
    }
}