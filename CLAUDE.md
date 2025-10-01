# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mvmc-rs** is a Rust port of mVMC (many-variable Variational Monte Carlo method), a numerical solver for quantum lattice models. The original C implementation is included as a git submodule for reference.

### What is mVMC?

mVMC performs highly-accurate variational Monte Carlo calculations for strongly correlated electron systems with a simple user interface and large-scale parallelization support.

**Target Models:**
- Hubbard model
- Heisenberg model
- Kondo lattice model
- Multi-orbital Hubbard model

**Available Physical Quantities:**
- Ground-state energy
- Spin/charge structure factors
- Superconducting correlations

### Current State

- **Language**: Rust (edition 2024)
- **Package**: `mvmc` v0.1.0
- **Status**: Initial stage - placeholder "Hello, world!" program
- **Planning**: See `PLAN.md` for comprehensive migration strategy (Japanese)

### Submodules

- **mVMC** (`mVMC/`) - Original C implementation
  - Source: https://github.com/issp-center-dev/mVMC.git
  - Build system: CMake
  - Key directories:
    - `src/mVMC/` - Core VMC engine (~50 C files)
    - `src/ComplexUHF/` - Complex unrestricted Hartree-Fock
    - `src/ltl2inv/` - Linear algebra (LTL decomposition, C++)
    - `src/pfupdates/` - Pfaffian update algorithms (C++)
    - `src/sfmt/` - Fast random number generator (SIMD-oriented)
    - `src/common/` - Shared utilities
    - `src/StdFace/` - Input file generator
    - `src/pfapack/` - Pfaffian calculation library
    - `src/blis/` - BLAS-like library

- **mVMC-tutorial** (`mVMC-tutorial/`) - Tutorial materials
  - Source: https://github.com/issp-center-dev/mVMC-tutorial.git

## Development Commands

### Rust Development

```bash
# Build the project
cargo build

# Build with optimizations
cargo build --release

# Run the project
cargo run

# Run tests
cargo test

# Check without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy
```

### Original mVMC (C Implementation)

The C implementation uses CMake. To build:

```bash
cd mVMC
mkdir build && cd build
cmake .. [options]
make
```

**Important CMake Options:**
- `-DUSE_SCALAPACK=ON` - Enable ScaLAPACK for distributed linear algebra
- `-DPFAFFIAN_BLOCKED=ON` - Use blocked-update Pfaffian (faster)
- `-DUSE_GEMMT=ON` - Use GEMMT (recommended, default ON)
- `-DCONFIG=<file>` - Load config from `config/<file>.cmake`

**Dependencies (C version):**
- C/C++ compiler (Intel, GNU, Fujitsu, etc.)
- Fortran compiler (for LAPACK/BLAS interfaces)
- MPI library
- LAPACK/ScaLAPACK (Intel MKL, ATLAS, etc.)
- OpenMP support

## Architecture Notes

### Original C Implementation Structure

The C codebase uses a modular structure with separate compilation units:

**Core Components:**
- Wave function representation (Slater matrices, Pfaffians)
- Optimization algorithms (conjugate gradient, stochastic reconfiguration, Lanczos)
- Monte Carlo sampling (Metropolis algorithm)
- Observable calculations (energy, Green's functions, correlations)

**Parallelization:**
- MPI for distributed-memory parallelism
- OpenMP for shared-memory parallelism
- ScaLAPACK for distributed linear algebra (optional)

**Numerical Libraries:**
- BLAS/LAPACK for linear algebra
- Custom Pfaffian libraries (pfapack, ltl2inv)
- SFMT for high-quality random numbers
- BLIS for optimized BLAS operations (when not using MKL)

### Migration Strategy

According to `PLAN.md`, the Rust port will be structured as a workspace with multiple crates:

**Planned Crates:**
- `mvmc-core` - Core VMC calculation engine
- `mvmc-math` - Numerical computation (linear algebra, random numbers)
- `mvmc-physics` - Physical models and observables
- `mvmc-io` - Input/output handling
- `mvmc-parallel` - Parallelization (MPI, threading)
- `mvmc-cli` - Command-line interface
- `mvmc-bindings` - FFI bindings to C libraries (LAPACK, etc.)

**Migration Phases:**
1. Foundation (mvmc-math with FFI bindings)
2. Core functionality (wave functions, optimization)
3. Physical models
4. Integration and optimization

### Input File Formats

**StdFace format (`.def`)** - Simple key-value format:
```
W = 4
L = 2
model = "FermionHubbard"
lattice = "Tetragonal"
t = 1.0
U = 4.0
nelec = 8
```

**Future formats:** TOML, JSON (planned for Rust version)

## References

- Original mVMC paper: [Computer Physics Communications, 235, 447-462 (2019)](https://www.sciencedirect.com/science/article/pii/S0010465518303102)
- Based on mVMC-mini (BSD 3-Clause License)
- Main license: GPL v3