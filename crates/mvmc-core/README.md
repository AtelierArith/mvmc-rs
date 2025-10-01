# mvmc-core

Core type definitions and error handling for mVMC (many-variable Variational Monte Carlo method).

## Overview

This crate provides:

- **Type-safe wrappers** for physical quantities (sites, electrons, spin)
- **Error handling** with descriptive error types
- **Zero-cost abstractions** for better type safety without runtime overhead

## Features

### Type Definitions (`types.rs`)

Type-safe wrappers prevent accidentally mixing incompatible values:

```rust
use mvmc_core::types::{SiteIndex, SiteCount, ElectronCount, TwoSz};

// Create site count (number of lattice sites)
let nsite = SiteCount::new(4);  // 4 sites

// Create electron count
let ne = ElectronCount::new(2);  // 2 electrons

// Create site index
let site = SiteIndex::new(0);  // First site

// Create spin quantum number (2*Sz)
let sz = TwoSz::new(0);  // Sz = 0 (singlet)
assert_eq!(sz.as_f64(), 0.0);

let sz_up = TwoSz::new(1);  // Sz = 1/2 (spin up)
assert_eq!(sz_up.as_f64(), 0.5);
```

**Available Types:**

- `SiteIndex` - Index for a lattice site [0, Nsite)
- `SiteCount` - Number of lattice sites (must be > 0)
- `ElectronCount` - Number of electrons
- `TwoSz` - Spin quantum number (stored as 2*Sz to avoid floats)
- `CalcMode` - Optimization or expectation value calculation
- `LanczosMode` - Lanczos method configuration
- `RandomSeed` - Seed for pseudorandom number generator

### Calculation Modes

```rust
use mvmc_core::types::{CalcMode, LanczosMode};

// Calculation mode
let mode = CalcMode::Optimization;
assert_eq!(mode.to_int(), 0);  // Compatible with C implementation

// Lanczos mode
let lanczos = LanczosMode::GreenFunction;
assert_eq!(lanczos.to_int(), 2);
```

### Error Handling (`error.rs`)

Comprehensive error types with helpful messages:

```rust
use mvmc_core::{VmcError, Result};

fn check_site_index(idx: usize, nsite: usize) -> Result<usize> {
    if idx >= nsite {
        return Err(VmcError::out_of_bounds(idx, nsite));
    }
    Ok(idx)
}

// Error: "Index out of bounds: 10 >= 5"
let err = check_site_index(10, 5);
assert!(err.is_err());
```

**Error Types:**

- `InvalidParameter` - Invalid parameter value
- `IndexOutOfBounds` - Array index out of range
- `InvalidConfiguration` - Invalid system configuration
- `DimensionMismatch` - Mismatched array dimensions
- `SingularMatrix` - Singular matrix in linear algebra
- `ConvergenceFailure` - Algorithm failed to converge
- `NumericalError` - Numerical issues (overflow, NaN, etc.)
- `Io` - File I/O errors

## Testing

Comprehensive test coverage with:

- **36 unit tests** - Basic functionality
- **Property-based tests** - Using `proptest` for arbitrary inputs
- **Doctests** - All examples in documentation are tested

```bash
# Run all tests
cargo test -p mvmc-core

# Run with more property test cases
PROPTEST_CASES=10000 cargo test -p mvmc-core
```

**Test Results:**
```
running 36 tests
test result: ok. 36 passed; 0 failed; 0 ignored

running 7 doctests
test result: ok. 7 passed; 0 failed; 0 ignored
```

## Examples

### Type Safety

```rust
use mvmc_core::types::{SiteIndex, ElectronCount};

let site_idx = SiteIndex::new(0);
let electron_count = ElectronCount::new(2);

// This would NOT compile (type safety):
// assert_eq!(site_idx, electron_count);  // Error: mismatched types

// Must explicitly convert to compare
assert_eq!(site_idx.get(), 0);
assert_eq!(electron_count.get(), 2);
```

### Ordering and Comparisons

```rust
use mvmc_core::types::SiteIndex;

let s1 = SiteIndex::new(1);
let s2 = SiteIndex::new(2);

assert!(s1 < s2);
assert!(s2 > s1);
assert_eq!(s1, s1);
```

### Using with Arrays

```rust
use mvmc_core::types::SiteIndex;

let energies = vec![1.0, 2.0, 3.0, 4.0];
let site = SiteIndex::new(2);

let energy = energies[site.get()];
assert_eq!(energy, 3.0);
```

## Design Principles

1. **Type Safety** - Newtype pattern prevents mixing incompatible values
2. **Zero Cost** - All wrappers compile to same code as raw integers
3. **Tested** - Comprehensive test coverage including property-based tests
4. **Documented** - All public APIs have documentation with examples
5. **Compatible** - Integer conversion methods for C FFI compatibility

## Integration

This crate is designed to be used with:

- `mvmc-math` - Numerical computing primitives
- `mvmc-physics` - Physical models (Hubbard, Heisenberg, etc.)
- `mvmc-io` - Input/output handling

## License

GPL v3 (same as original mVMC)

## References

Based on the C implementation:
- Repository: https://github.com/issp-center-dev/mVMC
- Paper: [Computer Physics Communications, 235, 447-462 (2019)](https://www.sciencedirect.com/science/article/pii/S0010465518303102)
