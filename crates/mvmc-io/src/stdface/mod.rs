//! StdFace format parser implementation.
//!
//! This module provides parsing functionality for the StdFace format,
//! which is the standard input format for mVMC calculations.

pub mod parser;
pub mod config;
pub mod error;

pub use parser::StdFaceParser;
pub use config::StdFaceConfig;
pub use error::StdFaceError;
