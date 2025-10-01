# Test-Driven Development Guide for mvmc-rs

This document describes the test-driven development (TDD) approach for the mvmc-rs project.

## Overview

mvmc-rs follows a strict TDD methodology to ensure correctness and reliability of this numerical software. Our testing strategy includes:

1. **Unit Tests** - Test individual functions and methods
2. **Property-Based Tests** - Verify mathematical properties hold for arbitrary inputs
3. **Integration Tests** - Test module interactions
4. **Regression Tests** - Validate against C implementation results
5. **Benchmarks** - Track performance

## Project Structure

```
mvmc-rs/
├── crates/           # Workspace crates
│   ├── mvmc-math/   # Numerical computing primitives (START HERE)
│   ├── mvmc-core/   # Core VMC engine
│   ├── mvmc-physics/# Physical models
│   ├── mvmc-io/     # Input/output
│   ├── mvmc-parallel/ # Parallelization
│   ├── mvmc-cli/    # Command-line interface
│   └── mvmc-bindings/# FFI bindings
├── tests/           # Integration tests
│   ├── integration/ # Cross-crate tests
│   ├── regression/  # Regression tests
│   └── fixtures/    # Test data
├── benches/         # Benchmarks
└── test-data/       # Reference data from mVMC
```

## TDD Workflow

### Red-Green-Refactor Cycle

1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass the test
3. **Refactor**: Improve code quality while keeping tests green

### Example: Random Number Generator (mvmc-math/src/random.rs)

#### 1. Write Tests First

```rust
#[test]
fn test_rng_reproducible() {
    let mut rng1 = McRng::new(12345);
    let mut rng2 = McRng::new(12345);

    for _ in 0..10 {
        assert_eq!(rng1.gen_unit(), rng2.gen_unit());
    }
}

#[test]
fn test_gen_unit_range() {
    let mut rng = McRng::new(42);

    for _ in 0..1000 {
        let val = rng.gen_unit();
        assert!(val >= 0.0 && val < 1.0);
    }
}
```

#### 2. Implement Minimal Functionality

```rust
pub struct McRng {
    rng: rand::rngs::StdRng,
}

impl McRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    pub fn gen_unit(&mut self) -> f64 {
        self.rng.r#gen()  // Note: r# escapes reserved keyword in Rust 2024
    }
}
```

#### 3. Add Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_gen_unit_in_range(seed in 0u64..1000) {
        let mut rng = McRng::new(seed);
        for _ in 0..100 {
            let val = rng.gen_unit();
            prop_assert!(val >= 0.0);
            prop_assert!(val < 1.0);
        }
    }
}
```

#### 4. Statistical Validation

```rust
#[test]
fn test_mean_and_variance() {
    let mut rng = McRng::new(42);
    let samples: Vec<f64> = (0..100000).map(|_| rng.gen_unit()).collect();

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / samples.len() as f64;

    // For uniform [0,1): mean = 0.5, variance = 1/12
    assert_relative_eq!(mean, 0.5, epsilon = 0.01);
    assert_relative_eq!(variance, 1.0/12.0, epsilon = 0.01);
}
```

## Testing Numerical Code

### Floating-Point Comparisons

Use the `approx` crate for floating-point assertions:

```rust
use approx::assert_relative_eq;

assert_relative_eq!(actual, expected, epsilon = 1e-10);
```

### Statistical Testing

For Monte Carlo simulations, use statistical properties:

```rust
// Check distribution uniformity
let tolerance = 0.1;  // 10% tolerance
let relative_error = (observed - expected).abs() / expected;
assert!(relative_error < tolerance);
```

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test -p mvmc-math

# Run with optimizations (important for numerical accuracy!)
cargo test --release

# Run property-based tests with more cases
PROPTEST_CASES=10000 cargo test

# Run specific test
cargo test test_rng_reproducible
```

## Test Categories

### Unit Tests (in module files)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Test implementation
    }
}
```

### Property-Based Tests

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_some_property(input in strategy()) {
            // Property verification
        }
    }
}
```

### Integration Tests (tests/ directory)

```rust
// tests/integration/test_workflow.rs
use mvmc_math::random::McRng;
use mvmc_core::wavefunction::Slater;

#[test]
fn test_complete_workflow() {
    // Test cross-crate functionality
}
```

### Regression Tests

Compare outputs with reference data from C implementation:

```rust
#[test]
fn test_hubbard_chain_regression() {
    let result = calculate_energy("test-data/data/HubbardChain/StdFace.def");
    let reference = load_reference("test-data/data/HubbardChain/ref/ref_mean.dat");
    let std_dev = load_reference("test-data/data/HubbardChain/ref/ref_std.dat");

    // 3-sigma rule from original mVMC tests
    assert!((result - reference).abs() < 3.0 * std_dev);
}
```

## Continuous Integration

CI runs automatically on push/PR (`.github/workflows/ci.yml`):

- ✅ Unit tests
- ✅ Property tests
- ✅ Integration tests
- ✅ Clippy linting
- ✅ Code formatting
- ✅ Documentation build

## Best Practices

### 1. Test Before Implementing

Always write tests first. This ensures:
- Clear specification
- Testable design
- Complete coverage

### 2. Test One Thing at a Time

Each test should verify a single behavior:

```rust
#[test]
fn test_addition() { /* ... */ }

#[test]
fn test_multiplication() { /* ... */ }
```

### 3. Use Descriptive Names

```rust
#[test]
fn test_rng_produces_different_sequences_with_different_seeds() {
    // Clear from name what is being tested
}
```

### 4. Test Edge Cases

```rust
#[test]
fn test_matrix_multiplication_with_empty_matrix() { /* ... */ }

#[test]
fn test_division_by_small_number() { /* ... */ }
```

### 5. Document Complex Tests

```rust
#[test]
fn test_pfaffian_update() {
    // Pfaffian update algorithm from Abe (2002)
    // Tests that Pf(A') = Pf(A) * det(1 + S)
    // where S is the Sherman-Morrison update matrix
    // ...
}
```

## Phase 1: mvmc-math Implementation

Current status: ✅ Random number generator with full TDD

Next steps:
1. Matrix operations (transpose, multiplication, addition)
2. Linear algebra (LU decomposition, matrix inversion)
3. Complex number operations
4. FFT routines
5. Pfaffian calculations

Each feature should follow the TDD cycle demonstrated with the RNG.

## Resources

- [Rust Book: Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Documentation](https://altsysrq.github.io/rustdoc/proptest/)
- [approx crate](https://docs.rs/approx/)
- [criterion benchmarking](https://bheisler.github.io/criterion.rs/book/)

## Reference Test Data

The `test-data/data/` directory contains reference test cases from the original mVMC C implementation:

- **HubbardChain** - 1D Hubbard model test cases
- **HeisenbergChain** - 1D Heisenberg model test cases
- **KondoChain** - Kondo lattice model test cases
- Various MPI and momentum projection variants

Each test case includes:
- `StdFace.def` - Input parameters
- `initial.def` - Initial state
- `ref/ref_mean.dat` - Expected mean values
- `ref/ref_std.dat` - Standard deviations

Use these for regression testing to ensure Rust implementation matches C results.
