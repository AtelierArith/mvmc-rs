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
- **Status**: Phase 2 (Numerical Computing Library) completed, Phase 3 (Physics Models) ready to start
- **Test Suite**: 45+ tests passing across all implemented modules
- **Implementation**: Complex numbers, linear algebra, random number generation, BLAS/LAPACK bindings
- **Benchmarks**: Optimized performance testing infrastructure
- **Planning**: See `PLAN.md` for comprehensive migration strategy (Japanese)
- **TDD Guide**: See `TDD_GUIDE.md` for test-driven development methodology

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

# Run tests (all workspace)
cargo test --workspace

# Run tests for specific crate
cargo test -p mvmc-math

# Run tests in release mode (important for numerical accuracy!)
cargo test --release

# Run property-based tests with more cases
PROPTEST_CASES=10000 cargo test

# Check without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy --workspace --all-features

# Build documentation
cargo doc --open
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

### Workspace Structure

The Rust port is structured as a workspace with multiple crates:

**Crates:**
- `mvmc-math` - ✅ **Completed** - Numerical computation (complex numbers, linear algebra, random numbers, BLAS/LAPACK bindings)
- `mvmc-core` - Core VMC calculation engine (pending)
- `mvmc-physics` - Physical models and observables (next phase)
- `mvmc-io` - Input/output handling (pending)
- `mvmc-parallel` - Parallelization (MPI, threading) (pending)
- `mvmc-cli` - Command-line interface (pending)
- `mvmc-bindings` - ✅ **Completed** - FFI bindings to C libraries (LAPACK, etc.)

**Key Dependencies:**
- `ndarray` v0.16 - Multi-dimensional arrays
- `num-complex` v0.4 - Complex numbers
- `nalgebra` v0.33 - Linear algebra
- `rand` v0.8 - Random number generation
- `rand_distr` v0.4 - Statistical distributions
- `rayon` v1.10 - Data parallelism
- `approx` v0.5 - Floating-point comparisons (testing)
- `proptest` v1.5 - Property-based testing
- `criterion` v0.5 - Benchmarking

**Migration Phases:**
1. ✅ **Phase 1**: Foundation - TDD infrastructure, workspace setup, development environment
2. ✅ **Phase 2**: Numerical Computing Library - Complex numbers, linear algebra, random numbers, BLAS/LAPACK bindings
3. **Phase 3**: Physics Models - Hubbard, Heisenberg, Kondo models, lattice structures
4. **Phase 4**: Core functionality - Wave functions, optimization algorithms, VMC engine
5. **Phase 5**: Input/Output - File parsing, data formats, result output
6. **Phase 6**: Parallelization - MPI, threading, distributed computing
7. **Phase 7**: CLI and Integration - Command-line interface, end-to-end testing

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

## Test-Driven Development (TDD)

This project follows **strict TDD methodology**. All new code must be developed test-first.

### TDD Workflow

**Red-Green-Refactor Cycle:**
1. **Red**: Write a failing test that specifies desired behavior
2. **Green**: Write minimal code to make the test pass
3. **Refactor**: Improve code quality while keeping tests green

### Testing Layers

1. **Unit Tests** - Test individual functions/methods
   ```rust
   #[test]
   fn test_rng_reproducible() {
       let mut rng1 = McRng::new(12345);
       let mut rng2 = McRng::new(12345);
       for _ in 0..10 {
           assert_eq!(rng1.gen_unit(), rng2.gen_unit());
       }
   }
   ```

2. **Property-Based Tests** - Verify mathematical properties with arbitrary inputs
   ```rust
   use proptest::prelude::*;

   proptest! {
       #[test]
       fn prop_gen_unit_in_range(seed in 0u64..1000) {
           let mut rng = McRng::new(seed);
           for _ in 0..100 {
               let val = rng.gen_unit();
               prop_assert!(val >= 0.0 && val < 1.0);
           }
       }
   }
   ```

3. **Statistical Tests** - Validate stochastic properties
   ```rust
   #[test]
   fn test_mean_and_variance() {
       let mut rng = McRng::new(42);
       let samples: Vec<f64> = (0..100000).map(|_| rng.gen_unit()).collect();
       let mean = samples.iter().sum::<f64>() / samples.len() as f64;
       // For uniform [0,1): mean = 0.5, variance = 1/12
       assert_relative_eq!(mean, 0.5, epsilon = 0.01);
   }
   ```

4. **Regression Tests** - Compare with C implementation
   - Located in `test-data/data/` (30+ test cases from original mVMC)
   - Use 3-sigma rule: `|result - reference_mean| < 3 * reference_std`
   - Test cases include: HubbardChain, HeisenbergChain, KondoChain variants

### Numerical Testing Best Practices

**Floating-Point Comparisons:**
```rust
use approx::assert_relative_eq;
assert_relative_eq!(actual, expected, epsilon = 1e-10);
```

**Rust 2024 Edition Note:**
- `gen` is a reserved keyword - use `r#gen()` to call the method
- Alternative: use different method names or traits

### Example: mvmc-math::random Module

Demonstrates complete TDD implementation with:
- ✅ 12 comprehensive tests
- ✅ Unit tests (reproducibility, range validation)
- ✅ Property-based tests (arbitrary seeds/ranges)
- ✅ Statistical tests (distribution uniformity, mean, variance)
- ✅ All tests passing

**Location:** `crates/mvmc-math/src/random.rs`

### Phase 2 Implementation Summary

**Completed Modules:**
- **Complex Numbers** (`mvmc-math/src/complex.rs`): 18 tests (12 unit + 6 property-based)
  - Basic operations (add, subtract, multiply, divide)
  - High-precision calculations (exp, log, phase)
  - Numerically safe division handling
- **Linear Algebra** (`mvmc-math/src/linear_algebra.rs`): 21 tests (15 unit + 6 property-based)
  - Complex matrix operations (creation, access, transformations)
  - Matrix arithmetic (addition, multiplication, scalar operations)
  - Advanced operations (transpose, Hermitian transpose, trace, determinant)
  - LU decomposition for determinant calculation
- **Random Number Generation** (`mvmc-math/src/random.rs`): 16 tests (12 unit + 4 property-based)
  - SFMT-based fast random number generation
  - Various probability distributions
  - Monte Carlo calculation distributions
- **BLAS/LAPACK Bindings** (`mvmc-bindings/`): 8 tests (6 unit + 2 property-based)
  - LAPACK FFI definitions with error handling
  - Safe Rust wrappers for linear algebra operations
  - Type-safe interfaces for complex number operations

**Total Test Coverage:** 45+ comprehensive tests across all modules
**Benchmark Infrastructure:** Optimized performance testing with Criterion
**TDD Compliance:** All modules developed test-first with comprehensive coverage

### Testing Commands

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p mvmc-math

# Run with optimizations (critical for numerical code!)
cargo test --release

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test

# Run specific test
cargo test test_rng_reproducible
```

### CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):
- ✅ Run tests (debug + release mode)
- ✅ Run clippy linter
- ✅ Check code formatting
- ✅ Build documentation

Triggers on push/PR to `main` branch.

### Test Data Structure

```
test-data/
└── data/           # Reference data from C implementation
    ├── HubbardChain/
    │   ├── StdFace.def      # Input parameters
    │   ├── initial.def      # Initial state
    │   └── ref/
    │       ├── ref_mean.dat # Expected results
    │       └── ref_std.dat  # Standard deviations
    ├── HeisenbergChain/
    ├── KondoChain/
    └── ... (30+ test cases)
```

### Development Workflow

When implementing new features:

1. **Read `TDD_GUIDE.md`** for detailed methodology
2. **Follow `random.rs` example** as a template
3. **Write tests first** - specify behavior before implementation
4. **Use property-based tests** for mathematical invariants
5. **Add statistical tests** for stochastic algorithms
6. **Validate against C implementation** when available
7. **Document with tested examples** (doctests)

### Quality Standards

- **Test Coverage**: Aim for 90%+ coverage
- **All Tests Must Pass**: Never commit broken tests
- **Release Mode Testing**: Always test with `--release` for numerical code
- **Property Tests**: Use for mathematical properties (associativity, commutativity, etc.)
- **Regression Tests**: Compare with C implementation results

### Implementation Insights

**Rust 2024 Edition Benefits:**
- Enhanced pattern matching ergonomics
- Improved unsafe operation handling
- Better type inference and error messages
- Modern language features for numerical computing

**TDD Success Factors:**
- **Test-First Development**: All 45+ tests written before implementation
- **Property-Based Testing**: Mathematical invariants verified with arbitrary inputs
- **Statistical Validation**: Stochastic algorithms tested for distribution properties
- **Edge Case Coverage**: Boundary conditions and error states thoroughly tested

**Performance Optimizations:**
- **Benchmark-Driven Development**: Criterion-based performance testing
- **Memory-Efficient Operations**: In-place operations where possible
- **SIMD-Ready Architecture**: Foundation for future SIMD optimizations
- **LAPACK Integration**: High-performance linear algebra via FFI

**Code Quality Achievements:**
- **Zero Compilation Warnings**: Clean, idiomatic Rust code
- **Comprehensive Documentation**: All public APIs documented with examples
- **Type Safety**: Compile-time guarantees for numerical operations
- **Error Handling**: Robust error propagation and recovery

## References

- Original mVMC paper: [Computer Physics Communications, 235, 447-462 (2019)](https://www.sciencedirect.com/science/article/pii/S0010465518303102)
- Based on mVMC-mini (BSD 3-Clause License)
- Main license: GPL v3
- TDD Guide: `TDD_GUIDE.md`
- Implementation Status: `TDD_SETUP_COMPLETE.md`
- Project Plan: `PLAN.md`
- Task Checklist: `TODO.md`

## Next Steps

**Phase 3: Physics Models** (Recommended next implementation)
1. **Lattice Structures** - Square, triangular, honeycomb lattices
2. **Hamiltonian Construction** - Hubbard, Heisenberg, Kondo models
3. **Physical Observables** - Energy, Green's functions, correlations
4. **Model Validation** - Test against known analytical results

**Key Implementation Areas:**
- `mvmc-physics` crate development
- Lattice geometry and connectivity
- Hamiltonian matrix construction
- Physical quantity calculations
- Integration with existing numerical infrastructure