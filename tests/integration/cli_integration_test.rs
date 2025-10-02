//! Integration tests for CLI commands
//!
//! This module contains integration tests that verify the CLI commands
//! work correctly with the VMC engine.

use std::fs;
use tempfile::TempDir;

/// Test CLI run command with StdFace configuration
#[test]
fn test_cli_run_stdface() {
    // Create a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple StdFace configuration file
    let config_content = r#"
L = 2
W = 1
model = "FermionHubbard"
lattice = "Chain"
t = 1.0
U = 4.0
Ncond = 2
2Sz = 0
RndSeed = 12345
NVMCCalMode = 0
"#;

    let config_file = temp_path.join("test.def");
    fs::write(&config_file, config_content).unwrap();

    // Create output directory
    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test the run command
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false, // text format
        Some(1), // single thread
    );

    // The command should succeed
    assert!(result.is_ok(), "CLI run command failed: {:?}", result.err());
}

/// Test CLI run command with TOML configuration
#[test]
fn test_cli_run_toml() {
    // Create a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple TOML configuration file
    let config_content = r#"
[lattice]
Lx = 2
Ly = 1
lattice_type = "Chain"

[model]
model_type = "FermionHubbard"
Ncond = 2
two_sz = 0
RndSeed = 12345
"#;

    let config_file = temp_path.join("test.toml");
    fs::write(&config_file, config_content).unwrap();

    // Create output directory
    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test the run command
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false, // text format
        Some(1), // single thread
    );

    // The command should succeed
    assert!(result.is_ok(), "CLI run command failed: {:?}", result.err());
}

/// Test CLI run command with JSON configuration
#[test]
fn test_cli_run_json() {
    // Create a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple JSON configuration file
    let config_content = r#"
{
    "lattice": {
        "Lx": 2,
        "Ly": 1,
        "lattice_type": "Chain"
    },
    "model": {
        "model_type": "FermionHubbard",
        "Ncond": 2,
        "two_sz": 0,
        "RndSeed": 12345
    }
}
"#;

    let config_file = temp_path.join("test.json");
    fs::write(&config_file, config_content).unwrap();

    // Create output directory
    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test the run command
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false, // text format
        Some(1), // single thread
    );

    // The command should succeed
    assert!(result.is_ok(), "CLI run command failed: {:?}", result.err());
}

/// Test CLI run command with binary output
#[test]
fn test_cli_run_binary_output() {
    // Create a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple configuration file
    let config_content = r#"
L = 2
W = 1
model = "FermionHubbard"
lattice = "Chain"
t = 1.0
U = 4.0
Ncond = 2
2Sz = 0
RndSeed = 12345
NVMCCalMode = 0
"#;

    let config_file = temp_path.join("test.def");
    fs::write(&config_file, config_content).unwrap();

    // Create output directory
    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test the run command with binary output
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        true, // binary format
        Some(1), // single thread
    );

    // The command should succeed
    assert!(result.is_ok(), "CLI run command failed: {:?}", result.err());
}

/// Test CLI run command with expectation mode
#[test]
fn test_cli_run_expectation_mode() {
    // Create a temporary directory
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a configuration file with expectation mode
    let config_content = r#"
L = 2
W = 1
model = "FermionHubbard"
lattice = "Chain"
t = 1.0
U = 4.0
Ncond = 2
2Sz = 0
RndSeed = 12345
NVMCCalMode = 1
"#;

    let config_file = temp_path.join("test.def");
    fs::write(&config_file, config_content).unwrap();

    // Create output directory
    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test the run command
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false, // text format
        Some(1), // single thread
    );

    // The command should succeed
    assert!(result.is_ok(), "CLI run command failed: {:?}", result.err());
}

/// Test CLI run command with nonexistent file
#[test]
fn test_cli_run_nonexistent_file() {
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    let config_file = temp_path.join("nonexistent.def");
    let output_dir = temp_path.join("output");

    // Test the run command with nonexistent file
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false,
        Some(1),
    );

    // The command should fail
    assert!(result.is_err(), "CLI run command should fail with nonexistent file");
}

/// Test CLI run command with invalid file format
#[test]
fn test_cli_run_invalid_format() {
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a file with invalid extension
    let config_file = temp_path.join("test.xyz");
    fs::write(&config_file, "dummy content").unwrap();

    let output_dir = temp_path.join("output");

    // Test the run command with invalid format
    let result = mvmc_cli::commands::run::execute(
        config_file,
        output_dir,
        false,
        Some(1),
    );

    // The command should fail
    assert!(result.is_err(), "CLI run command should fail with invalid format");
}

/// Test CLI run command with different thread counts
#[test]
fn test_cli_run_different_threads() {
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple configuration file
    let config_content = r#"
L = 2
W = 1
model = "FermionHubbard"
lattice = "Chain"
t = 1.0
U = 4.0
Ncond = 2
2Sz = 0
RndSeed = 12345
NVMCCalMode = 0
"#;

    let config_file = temp_path.join("test.def");
    fs::write(&config_file, config_content).unwrap();

    let output_dir = temp_path.join("output");
    fs::create_dir_all(&output_dir).unwrap();

    // Test with different thread counts
    let thread_counts = vec![None, Some(1), Some(2), Some(4)];

    for thread_count in thread_counts {
        let result = mvmc_cli::commands::run::execute(
            config_file.clone(),
            output_dir.clone(),
            false,
            thread_count,
        );

        // All should succeed
        assert!(result.is_ok(), "CLI run command failed with thread count {:?}: {:?}", thread_count, result.err());
    }
}
