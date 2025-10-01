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

🚧 **Early Development - Phase 1 Complete**

- ✅ TDD infrastructure established
- ✅ Workspace structure created (7 crates)
- ✅ `mvmc-math::random` module implemented with comprehensive tests
- ✅ CI/CD pipeline configured
- ✅ Reference test data imported (30+ test cases)
- 🔄 Linear algebra operations (next)

**Test Suite**: 17 tests passing (12 comprehensive tests in mvmc-math)

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/atelierarith/mvmc-rs.git
cd mvmc-rs

# Run tests
cargo test --workspace

# Run specific crate tests
cargo test -p mvmc-math

# Build documentation
cargo doc --open
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

### Example: Random Number Generator

```rust
use mvmc_math::random::McRng;

// Create seeded RNG for reproducibility
let mut rng = McRng::new(12345);

// Generate random values
let x = rng.gen_unit();        // [0, 1)
let y = rng.gen_range(-1.0, 1.0);  // [-1, 1)
let idx = rng.gen_index(10);   // [0, 10)
```

**Fully tested** with 12 comprehensive tests including:
- Reproducibility (same seed → same sequence)
- Range validation
- Statistical properties (mean, variance, uniformity)
- Property-based tests (arbitrary inputs)

## Project Structure

```
mvmc-rs/
├── crates/               # Workspace crates
│   ├── mvmc-math/       # ✅ Numerical primitives (in progress)
│   ├── mvmc-core/       # Core VMC engine
│   ├── mvmc-physics/    # Physical models
│   ├── mvmc-io/         # Input/output
│   ├── mvmc-parallel/   # Parallelization
│   ├── mvmc-cli/        # Command-line interface
│   └── mvmc-bindings/   # FFI bindings
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

## Commands

```bash
# Development
cargo build                    # Build debug
cargo build --release         # Build optimized
cargo test --workspace        # Run all tests
cargo test --release          # Test with optimizations (important!)
cargo clippy --workspace      # Lint
cargo fmt                     # Format code

# Testing
cargo test -p mvmc-math       # Test specific crate
PROPTEST_CASES=10000 cargo test  # More property test cases
cargo test test_rng_reproducible # Run specific test

# Documentation
cargo doc --open              # Build and view docs
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
- [x] Random number generation

### Phase 2: Mathematical Primitives
- [ ] Matrix operations (transpose, multiplication)
- [ ] Linear algebra (LU decomposition, inversion)
- [ ] Complex number operations
- [ ] Pfaffian calculations

### Phase 3: Core VMC
- [ ] Wave function representations
- [ ] Optimization algorithms
- [ ] Monte Carlo sampling

### Phase 4: Physical Models
- [ ] Hubbard model
- [ ] Heisenberg model
- [ ] Kondo lattice model

### Phase 5: Integration
- [ ] Full regression testing
- [ ] Performance benchmarking
- [ ] Production readiness

---

**Status**: Phase 1 complete. Ready for Phase 2 implementation following established TDD patterns.
