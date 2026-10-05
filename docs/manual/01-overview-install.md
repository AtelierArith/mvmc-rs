# 1. Overview and installation

[Back to contents](README.md) · Next: [2. Theory I](02-theory-vmc-hamiltonian.md)

## 1.1 What this package is

`mvmc-rs` is a pure-Rust port of **mVMC**, the many-variable variational Monte
Carlo solver for lattice fermion and spin models. Like mVMC it

1. reads an *Expert-mode* input set (`namelist.def` plus the definition files it
   lists),
2. optimizes the parameters of a Pfaffian pair-product wave function with
   correlation factors and quantum-number projections by stochastic
   reconfiguration (SR), `NVMCCalMode = 0`, and
3. evaluates physical quantities (Green functions, optionally the single-step
   Lanczos correction) for a fixed parameter set, `NVMCCalMode = 1`.

The Rust workspace follows a layered authority rule that is used throughout this
manual (see [AGENTS.md](../../AGENTS.md)):

| Question | Authority |
|----------|-----------|
| Physics, algorithms, input contract, output formats, parameter layout, RNG draw order, arithmetic order and signs | the C implementation `extern/mVMC-1.3.0` |
| Public Rust API shape, runner structure, lifecycle, test organization | the Julia port `extern/Julia-mVMC` |
| Floating-point comparison | explicit absolute/relative tolerances; never bitwise (see [11.4](11-compatibility.md#114-numerical-comparison-policy)) |

The project is a *pure Rust* implementation: it links BLAS/LAPACK (OpenBLAS) but
has no C or Fortran foreign-function dependency on mVMC or PFAPACK, and Rust
builds and tests do not need `c_toolbox/`.

## 1.2 Workspace layout

| Crate | Role |
|-------|------|
| `crates/sfmt19937` | SFMT-19937 random number generator with C-compatible seeding, 32-bit draws and `genrand_real2` conversion (`Sfmt19937Rng`, `crates/sfmt19937/src/lib.rs:74`) |
| `crates/pfapack` | Pfaffian, skew-symmetric LTL factorization and inverse kernels (`pfaffian_ltl_real`, `zsktf2_c_compat`, `utu2inv_complex`) with scalar and BLAS backends |
| `crates/mvmc-expert-parsers` | Expert-mode file parsers, `ExpertModeData`, parameter initialization, quantum-projection weights |
| `crates/mvmc-core` | The VMC engine: sampling, observables, SR, Lanczos, output, MPI reducers, validation |
| `crates/mvmc-cli` | The `mvmc` binary and four example programs |
| `crates/mvmc-greenr2k` | The `greenr2k` binary: Fourier transform of the Green functions (port of `tool/greenr2k.F90`, [9.9](09-output-files.md#99-post-processing-greenr2k-fourier-transform-of-the-green-functions)) |
| `xtask` | Benchmark and regression automation |

## 1.3 What is supported

| Capability | Status in `mvmc-rs` |
|------------|---------------------|
| Parameter optimization (`NVMCCalMode = 0`) with direct SR (`NSRCG = 0`) or CG SR (`NSRCG = 1`) | supported |
| Fixed-parameter physical quantities (`NVMCCalMode = 1`): `OneBodyG`, `TwoBodyG`, `TwoBodyGEx` | supported |
| Single-step Lanczos (`NLanczosMode = 1, 2`) | supported for the sz-conserved path (any `NSplitSize`), no `InterAll`, no spin-changing `Trans` (see [7.5](07-input-files.md#75-supported-and-rejected-inputs)) |
| Real and complex wave functions | supported (decided by the input declarations, see [3.3](03-theory-wavefunction.md#33-real-and-complex-modes)) |
| `Orbital`/`OrbitalAntiParallel`, `OrbitalParallel`, `OrbitalGeneral` (FSZ) | supported |
| Gutzwiller, Jastrow, 2-/4-site doublon-holon, charge/spin/general RBM, `OptTrans` | supported |
| `InterAll` Hamiltonian terms | parsed and evaluated for optimization; ownership of this family is separate from this manual |
| BackFlow (`BF`, `BFRange`), `SpinJastrow`, `NSRCG >= 2`, `useDiagScale`, `RescaleSmat` | rejected |
| MPI, including grouped execution (`NSplitSize > 1`) | supported with the `mpi` feature, with restrictions ([8.4](08-running.md#84-mpi-and-grouped-execution)) |
| Standard mode (`-s`, StdFace) for the 3D and Wannier90 lattices not yet ported, multi-definition mode (`-m`), binary output (`-b`) of the C driver | not provided ([7.6](07-input-files.md#76-standard-mode-stdface)) |

## 1.4 Requirements

- A recent stable Rust toolchain. `rust-toolchain.toml` selects the `stable`
  channel with `rustfmt` and `clippy`; there is no pinned MSRV.
- **OpenBLAS and LAPACK (LP64 interface)**. `crates/mvmc-core/build.rs` emits
  `cargo:rustc-link-lib=dylib=openblas`. On Linux install the system package
  (`libopenblas-dev liblapack-dev` on Debian/Ubuntu). On macOS install
  `brew install openblas`; the formula is keg-only and the build script adds the
  Homebrew library path (`/opt/homebrew/opt/openblas` on Apple Silicon,
  `/usr/local/opt/openblas` on Intel).
- For the `mpi` feature: an MPI implementation with a C compiler wrapper and
  `libclang` (the `mpi` crate generates bindings). The CI setup action installs
  `libopenblas-dev liblapack-dev libclang-dev libmpich-dev mpich pkg-config` on
  Ubuntu and `openblas mpich` on macOS
  (`.github/actions/setup-rust-ci/action.yml`).
- Linux x86_64 is the numerical reference platform; a Dev Container is provided
  ([docs/DEV_CONTAINER.md](../DEV_CONTAINER.md)) and macOS is used for
  portability checks ([docs/NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md)).

## 1.5 Cargo features

| Feature | Where | Meaning |
|---------|-------|---------|
| `blas-backend` | `pfapack` | Use BLAS/LAPACK routines (`dger`, `zgeru`, `dtrtri`, `dtrmm`, `dscal`, ...) for the Pfaffian kernels. **`mvmc-core` always enables it** (`pfapack = { features = ["blas-backend"] }` in `crates/mvmc-core/Cargo.toml`), so the optimizer always uses the BLAS/LAPACK backend. The standalone `pfapack` crate defaults to its scalar reference backend. |
| `simd-backend` | `pfapack` | Enable SIMD (`pulp`) kernels in the standalone `pfapack` crate. It is **not forwarded** by `mvmc-core` or `mvmc-cli` (`crates/mvmc-cli/Cargo.toml` defines only `mpi`). Use it for the PfaPack tests and benchmarks: `cargo nextest run -p pfapack --features 'simd-backend blas-backend'`. |
| `mpi` | `mvmc-core`, `mvmc-cli` | Compile MPI support (`MpiContext`, `MpiGroupContext`). Without it a launcher-detected multi-rank run is rejected with an error asking to rebuild with `--features mpi`. |

## 1.6 Build and smoke test

```bash
# type-check everything
cargo check --workspace

# build the optimized CLI (serial)
cargo build --release -p mvmc-cli
# ... with MPI support
cargo build --release -p mvmc-cli --features mpi

# the binary is target/release/mvmc
target/release/mvmc --help
```

Development checks from [AGENTS.md](../../AGENTS.md):

```bash
cargo nextest run --workspace --cargo-profile test-fast   # all unit/integration tests
cargo test --workspace --doc                              # doc tests (not run by nextest)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

A working end-to-end run is shown in [chapter 10](10-tutorial.md). The
`cargo build --release -p mvmc-cli` build used for this manual finished in about
6 minutes from a cold target directory on Linux x86_64 **(observed)**. The
`--features mpi` build was **not** exercised while writing this manual
**(unverified)**.

## 1.7 Where to go next

- To understand what the program computes: [chapters 2–6](02-theory-vmc-hamiltonian.md).
- To run a calculation: [chapter 7](07-input-files.md), [8](08-running.md),
  [9](09-output-files.md) and the [tutorial](10-tutorial.md).
- Developer-oriented documents: [docs/DEVELOPMENT.md](../DEVELOPMENT.md),
  [docs/NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md),
  [docs/PORTING_PLAN.md](../PORTING_PLAN.md).
