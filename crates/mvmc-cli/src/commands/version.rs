//! Version command - Display version information.

use crate::error::CliResult;
use colored::Colorize;

/// Displays version information.
pub fn execute() -> CliResult<()> {
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("  {}", "mVMC - Many-variable Variational Monte Carlo".cyan().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    println!("Version:        {}", env!("CARGO_PKG_VERSION"));
    println!("Rust edition:   {}", "2024");
    println!();

    println!("{}", "Components:".bold());
    println!("  mvmc-core:     v{}", env!("CARGO_PKG_VERSION"));
    println!("  mvmc-math:     v{}", env!("CARGO_PKG_VERSION"));
    println!("  mvmc-physics:  v{}", env!("CARGO_PKG_VERSION"));
    println!("  mvmc-io:       v{}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("{}", "Supported models:".bold());
    println!("  • Hubbard model");
    println!("  • Heisenberg model");
    println!("  • Kondo lattice model");
    println!("  • Multi-orbital Hubbard model");
    println!();

    println!("{}", "License:".bold());
    println!("  GPL-3.0");
    println!();

    println!("{}", "Repository:".bold());
    println!("  https://github.com/issp-center-dev/mVMC");
    println!();

    println!("{}", "═══════════════════════════════════════════".cyan());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_execute() {
        // Should not panic or return error
        let result = execute();
        assert!(result.is_ok());
    }
}
