# mvmc-rs

A Rust implementation of mVMC (many-variable Variational Monte Carlo method), a numerical solver for quantum lattice models.

[![CI](https://github.com/atelierarith/mvmc-rs/workflows/CI/badge.svg)](https://github.com/atelierarith/mvmc-rs/actions)

## Overview

mVMC performs highly-accurate variational Monte Carlo calculations for strongly correlated electron systems including:

- **Hubbard model**
- **Heisenberg model**
- **Kondo lattice model**
- **Multi-orbital Hubbard model**

This Rust port aims to provide the same functionality with improved safety, maintainability, and performance.

## Project Status

🚀 **Phase 2-3, 5, and 7 Complete - Core Foundation Ready**

- ✅ **Phase 1**: TDD infrastructure and workspace setup
- ✅ **Phase 2**: Numerical computing library (`mvmc-math`) - 45 tests
- ✅ **Phase 3**: Input/output processing (`mvmc-io`) - 50 tests
- ✅ **Phase 5**: Physics models (`mvmc-physics`) - 99 tests
- ✅ **Phase 7**: Command-line interface (`mvmc-cli`) - Basic implementation
- 🔄 **Phase 4**: Core VMC engine (next)

**Total Test Suite**: 194+ tests passing across all implemented modules

### Implemented Features

#### Numerical Computing (`mvmc-math`)
- Complex number operations with high precision
- Linear algebra with BLAS/LAPACK bindings
- SFMT-based random number generation
- Matrix operations and decompositions

#### Input/Output (`mvmc-io`)
- StdFace format parser (mVMC standard)
- TOML and JSON configuration support
- Comprehensive validation and error handling
- Output data management (text/binary formats)
- Optimized parameter saving/loading
- **Definition file generation** - C実装と同等の13種類の`.def`ファイル生成

#### Physics Models (`mvmc-physics`)
- Lattice structures (1D chain, 2D square)
- Hamiltonians (Hubbard, Heisenberg models)
- Physical observables (energy, magnetization, correlations)

#### Command-Line Interface (`mvmc-cli`)
- Four operational commands: `run`, `info`, `validate`, `version`
- Multiple input format support (StdFace, TOML, JSON)
- Text and binary output options
- Thread pool configuration for parallel computing

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/atelierarith/mvmc-rs.git
cd mvmc-rs

# Build the CLI
cargo build --release

# Run VMC calculation
./target/release/mvmc run examples/hubbard_chain.def output/

# Run all tests
cargo test --workspace

# Run tests with optimizations (recommended for numerical code)
cargo test --workspace --release

# Build documentation
cargo doc --open
```

## CLI Usage

### Build and Install

```bash
# Build in release mode (recommended for performance)
cargo build --release -p mvmc-cli

# Install to cargo bin directory
cargo install --path crates/mvmc-cli

# Or run directly
cargo run --release -p mvmc-cli -- [COMMAND]
```

### Commands

#### Run VMC Calculation

```bash
# Run with StdFace format input
mvmc run examples/hubbard_chain.def output/

# Run with TOML format
mvmc run config.toml output/

# Run with JSON format
mvmc run config.json output/

# Use binary output format (faster I/O)
mvmc run config.def output/ --binary

# Specify number of threads
mvmc run config.def output/ --threads 8

# Enable verbose logging
mvmc run config.def output/ -v

# Quiet mode (errors only)
mvmc run config.def output/ -q
```

**Output files:**
- `{output_dir}/output_out.dat` - Energy data per iteration
- `{output_dir}/output_var.dat` - Variational data
- `{output_dir}/output_opt.dat` - Optimized parameters

#### Show Configuration Info

```bash
# Display configuration details
mvmc info examples/hubbard_chain.def

# Works with all formats
mvmc info config.toml
mvmc info config.json
```

**Output:**
```
Configuration Information
Model: Hubbard
Lattice: Chain
Lattice Size: L=10
Physical parameters:
  - t (hopping): 1.0
  - U (interaction): 4.0
  - mu (chemical potential): 0.0
```

#### Validate Configuration

```bash
# Check configuration file for errors
mvmc validate examples/hubbard_chain.def

# Validates all supported formats
mvmc validate config.toml
mvmc validate config.json
```

**Output:**
```
✓ Configuration is valid
Model: Hubbard
Lattice: Chain
```

#### Show Version

```bash
# Display version information
mvmc version
```

**Output:**
```
mvmc 0.1.0
Commit: <git-hash>
```

### Input File Formats

#### StdFace Format (`.def`)

```
L = 10
model = "Hubbard"
lattice = "chain"
t = 1.0
U = 4.0
mu = 0.0
Ncond = 10
```

#### TOML Format (`.toml`)

```toml
[model]
type = "Hubbard"

[lattice]
type = "chain"
L = 10

[parameters]
t = 1.0
U = 4.0
mu = 0.0
Ncond = 10
```

#### JSON Format (`.json`)

```json
{
  "model": {
    "type": "Hubbard"
  },
  "lattice": {
    "type": "chain",
    "L": 10
  },
  "parameters": {
    "t": 1.0,
    "U": 4.0,
    "mu": 0.0,
    "Ncond": 10
  }
}
```

### Global Options

```bash
# Verbose mode - show detailed logs
mvmc [COMMAND] -v

# Quiet mode - show only errors
mvmc [COMMAND] -q

# Show help
mvmc --help
mvmc [COMMAND] --help
```

### Testing Examples

```bash
# Test specific crate
cargo test -p mvmc-math
cargo test -p mvmc-physics
cargo test -p mvmc-io

# Test specific module
cargo test -p mvmc-math --lib complex
cargo test -p mvmc-physics --lib lattice

# Run with more property test cases
PROPTEST_CASES=10000 cargo test --workspace

# Run specific test
cargo test -p mvmc-math test_rng_reproducible
cargo test -p mvmc-physics test_hubbard_hamiltonian
```

## Development Approach

This project follows **strict Test-Driven Development (TDD)**:

1. ✅ Write tests first
2. ✅ Implement minimal code to pass
3. ✅ Refactor while keeping tests green

See [`TDD_GUIDE.md`](TDD_GUIDE.md) for detailed methodology.

### Testing Layers

- **Unit Tests** - Function-level validation
- **Property-Based Tests** - Mathematical invariants with `proptest`
- **Statistical Tests** - Distribution properties for stochastic algorithms
- **Regression Tests** - Validation against C implementation (3-sigma rule)

### Example: Physics Models

```rust
use mvmc_physics::lattice::ChainLattice;
use mvmc_physics::hamiltonian::{HubbardHamiltonian, Spin};
use mvmc_physics::observables::EnergyCalculator;

// Create a 1D chain lattice
let lattice = ChainLattice::new(6, true).unwrap();

// Create Hubbard Hamiltonian
let hamiltonian = HubbardHamiltonian::new(
    lattice,
    1.0,  // hopping parameter
    4.0,  // interaction parameter
    0.5   // chemical potential
).unwrap();

// Create energy calculator
let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

// Calculate energy for a spin configuration
let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Up];
let energy = energy_calc.calculate(&config);
println!("Energy: {}", energy);
```

### Example: Input Parsing

```rust
use mvmc_io::stdface::StdFaceParser;

// Parse StdFace configuration
let parser = StdFaceParser::new();
let config = parser.parse_str(r#"
L = 6
model = "Hubbard"
lattice = "chain"
t = 1.0
U = 4.0
Ncond = 6
"#).unwrap();

println!("Lattice dimensions: {:?}", config.lattice.dimensions);
println!("Model type: {}", config.model.model_type);
```

**Comprehensive testing** with 164+ tests including:
- Unit tests for individual functions
- Property-based tests for mathematical invariants
- Statistical tests for stochastic algorithms
- Integration tests for end-to-end workflows

## Project Structure

```
mvmc-rs/
├── crates/               # Workspace crates
│   ├── mvmc-math/       # ✅ Numerical computing (45 tests)
│   ├── mvmc-core/       # 🔄 Core VMC engine (next phase)
│   ├── mvmc-physics/    # ✅ Physics models (99 tests)
│   ├── mvmc-io/         # ✅ Input/output (50 tests)
│   ├── mvmc-parallel/   # 🔄 Parallelization (future)
│   ├── mvmc-cli/        # ✅ Command-line interface (basic)
│   └── mvmc-bindings/   # ✅ FFI bindings (8 tests)
├── tests/               # Integration tests
├── benches/             # Benchmarks
├── test-data/           # Reference data (30+ test cases)
└── mVMC/                # Original C implementation (submodule)
```

## Dependencies

### Numerical Computing
- `ndarray` - Multi-dimensional arrays
- `nalgebra` - Linear algebra
- `num-complex` - Complex numbers
- `rand` - Random number generation

### Testing
- `approx` - Floating-point comparisons
- `proptest` - Property-based testing
- `criterion` - Benchmarking

## Testing Guide

### Basic Testing Commands

```bash
# Run all tests in the workspace
cargo test --workspace

# Run tests with optimizations (recommended for numerical code)
cargo test --workspace --release

# Run tests for a specific crate
cargo test -p mvmc-math
cargo test -p mvmc-physics
cargo test -p mvmc-io
cargo test -p mvmc-bindings
```

### Advanced Testing

```bash
# Test specific modules within a crate
cargo test -p mvmc-math --lib complex
cargo test -p mvmc-math --lib linear_algebra
cargo test -p mvmc-physics --lib lattice
cargo test -p mvmc-physics --lib hamiltonian
cargo test -p mvmc-physics --lib observables

# Run specific test functions
cargo test -p mvmc-math test_rng_reproducible
cargo test -p mvmc-physics test_hubbard_hamiltonian
cargo test -p mvmc-io test_parse_hubbard_chain

# Run property-based tests with more cases
PROPTEST_CASES=10000 cargo test --workspace

# Run tests with verbose output
cargo test --workspace -- --nocapture

# Run tests in a specific directory
cargo test --test integration_tests
```

### Test Categories

```bash
# Run only unit tests
cargo test --workspace --lib

# Run only integration tests
cargo test --workspace --test '*'

# Run only property-based tests
cargo test --workspace -- --test-threads=1

# Run tests with specific patterns
cargo test --workspace -- test_parse
cargo test --workspace -- hubbard
cargo test --workspace -- lattice
```

### Development Commands

```bash
# Build and check
cargo build                    # Build debug
cargo build --release         # Build optimized
cargo check --workspace       # Check without building

# Code quality
cargo clippy --workspace      # Lint
cargo fmt                     # Format code
cargo fmt --check             # Check formatting

# Documentation
cargo doc --open              # Build and view docs
cargo doc --workspace --open  # Build docs for all crates
```

### Benchmarking

```bash
# Run benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench -p mvmc-math --bench complex_operations
cargo bench -p mvmc-physics --bench hamiltonian_operations
```

## Original mVMC

The C implementation is included as a git submodule:

```bash
cd mVMC
mkdir build && cd build
cmake ..
make
```

See `CLAUDE.md` for detailed build options.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project guide for Claude Code
- **[TDD_GUIDE.md](TDD_GUIDE.md)** - Test-driven development methodology
- **[TDD_SETUP_COMPLETE.md](TDD_SETUP_COMPLETE.md)** - Current implementation status
- **[PLAN.md](PLAN.md)** - Comprehensive migration strategy (Japanese)

## Contributing

This project follows TDD. When adding features:

1. Read `TDD_GUIDE.md`
2. Follow the `random.rs` example
3. Write tests first
4. Use property-based tests for mathematical properties
5. Validate against C implementation when available

## License

GPL v3 (same as original mVMC)

Based on [mVMC](https://github.com/issp-center-dev/mVMC) which is based on mVMC-mini (BSD 3-Clause).

## References

- mVMC paper: [Computer Physics Communications, 235, 447-462 (2019)](https://www.sciencedirect.com/science/article/pii/S0010465518303102)
- Original repository: https://github.com/issp-center-dev/mVMC
- Tutorial: https://github.com/issp-center-dev/mVMC-tutorial

## Roadmap

### Phase 1: Foundation ✅
- [x] TDD infrastructure
- [x] Workspace setup
- [x] Development environment

### Phase 2: Numerical Computing ✅
- [x] Complex number operations
- [x] Linear algebra with BLAS/LAPACK bindings
- [x] Random number generation (SFMT)
- [x] Matrix operations and decompositions

### Phase 3: Input/Output ✅
- [x] StdFace format parser
- [x] TOML and JSON support
- [x] Configuration validation
- [x] Output data management (text/binary formats)
- [x] Optimized parameter saving/loading

### Phase 4: Core VMC Engine 🔄
- [ ] Wave function representations (Slater matrices, Pfaffians)
- [ ] Optimization algorithms (conjugate gradient, SR)
- [ ] Monte Carlo sampling
- [ ] VMC calculation engine

### Phase 5: Physics Models ✅
- [x] Lattice structures (1D chain, 2D square)
- [x] Hamiltonians (Hubbard, Heisenberg)
- [x] Physical observables (energy, magnetization, correlations)

### Phase 6: Parallelization
- [ ] Thread-based parallelization (rayon)
- [ ] MPI support for distributed computing
- [ ] Performance optimization

### Phase 7: CLI and Integration ✅ (Basic)
- [x] Command-line interface (run, info, validate, version commands)
- [x] Multiple input format support (StdFace, TOML, JSON)
- [x] Output file management (text/binary)
- [x] Thread pool configuration
- [ ] Full regression testing
- [ ] Production readiness

---

**Status**: Phases 1, 2, 3, 5, and 7 (basic) complete. Ready for Phase 4 (Core VMC Engine) implementation.
