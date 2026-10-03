# Development builds and tests

For a Linux x86_64 environment with Rust, nextest, kache, uv, BLAS/LAPACK and
MPI already configured, open the repository's [Dev Container](DEV_CONTAINER.md).
Its Cargo targets and caches are isolated from host macOS artifacts.

Use `cargo nextest run` for Rust unit and integration tests. Start with a
targeted test while changing code, then run the workspace regressions. Run
documentation tests separately:

```sh
cargo nextest run -p mvmc-expert-parsers --test c_jastrow_contracts
cargo nextest run --workspace --locked --cargo-profile test-fast \
  --no-fail-fast --retries 0
cargo test --workspace --locked --doc
```

## Build profiles

Normal `cargo build`, `cargo check` and `cargo nextest run` use development
settings with `debug = 0` and `strip = "symbols"`. Tests retain optimization
level 0, debug assertions, integer overflow checks and incremental compilation.
This keeps the short edit/build/test cycle unoptimized while reducing debug
information and executable size.

Use the optimized `test-fast` profile for full-workspace runs and long numerical
regressions. Reserve the normal profile for short targeted TDD checks:

```sh
cargo nextest run --workspace --locked --cargo-profile test-fast \
  --no-fail-fast --retries 0
```

This profile uses optimization level 2, disables LTO and retains debug
assertions, overflow checks and incremental compilation. Its initial dependency
build costs more time; subsequent test execution benefits from optimized
numerical kernels. Cargo stores it in `target/test-fast`, separately from
normal development artifacts. The production `release` and `bench` profiles
keep their existing optimization and LTO settings.

GitHub Actions uses `ci`, derived from `test-fast`: optimization level 2,
no LTO, debug assertions and overflow checks, `debug = 0`, stripped symbols,
and `incremental = false`. Following tenferro-rs, hosted builds omit incremental
state and debug artifacts; this workspace retains optimization for its long
numerical regressions. Artifacts live in `target/ci`.

```sh
cargo nextest run --workspace --locked --cargo-profile ci --no-fail-fast --retries 0
cargo nextest run --workspace --all-features --locked --cargo-profile ci --no-fail-fast --retries 0
cargo test --workspace --all-features --locked --doc --profile ci
```

The [CI workflow](../.github/workflows/ci.yml) runs these commands on Ubuntu
24.04 x86_64 and macOS 15 ARM64 in six independent jobs: four test jobs
(default features and all features on each platform), one lint job (rustfmt,
all-feature Clippy) and one documentation job (doctests and API
documentation), with lint and documentation checks running on Linux x86_64.
These job families have no dependencies on one another.
The shared [setup action](../.github/actions/setup-rust-ci/action.yml) installs
numerical libraries and MPI; cache keys distinguish each check configuration.
Explicitly ignored reference/MPI developer gates retain
their documented opt-in selectors and launch requirements.

Stripped development executables omit debugger symbols and source-level
backtraces. Enable full symbols for a debugging session with environment
overrides:

```sh
CARGO_PROFILE_DEV_DEBUG=2 CARGO_PROFILE_DEV_STRIP=none cargo build
CARGO_PROFILE_TEST_DEBUG=2 CARGO_PROFILE_TEST_STRIP=none \
  cargo nextest run -p mvmc-core --lib
CARGO_PROFILE_TEST_FAST_DEBUG=2 CARGO_PROFILE_TEST_FAST_STRIP=none \
  cargo nextest run -p mvmc-core --lib --cargo-profile test-fast
```

The settings follow the approach in the
[build-artifact article](https://zenn.dev/terasakisatoshi/articles/9725e1c93fc05c)
and the [Cargo profile reference](https://doc.rust-lang.org/cargo/reference/profiles.html).

## Compiler cache and checkout isolation

Use the repository's [kache skill](../skills/kache/SKILL.md) to configure and
verify the compiler cache. Keep machine-specific binary paths and cache
configuration out of commits. Each checkout/worktree should have its own Cargo
target directory; share compiled dependencies through kache. Do not clean the
user's target directory to measure builds.

## Measurement procedure

The profile comparison uses the existing deterministic Hubbard direct-SR test,
which checks parameter updates, samples, energies and the RNG trajectory at
multiple step counts. It does not alter fixtures or tolerances. These are
measurements of development tests, rather than production VMC benchmarks.

To compare profiles on the same source revision, record the compiler, nextest
version and host, keep the same feature set, warm each profile once and then
time the same test sequentially. Limit BLAS threading identically in each run:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
cargo nextest run -p mvmc-core --lib --locked --cargo-profile test-fast \
  -E 'test(=run::callback_tests::hubbard_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng)'
/usr/bin/time -p cargo nextest run -p mvmc-core --lib --locked \
  --cargo-profile test-fast \
  -E 'test(=run::callback_tests::hubbard_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng)'
```

Repeat without `--cargo-profile test-fast` for the normal test profile. To
reproduce the former debug-symbol baseline, set `CARGO_PROFILE_TEST_DEBUG=2`
and `CARGO_PROFILE_TEST_STRIP=none` for both its warm-up and timed run. Record
the Cargo compilation time, nextest test execution time and complete command
time separately; discovery and startup can dominate a short filtered run.

### Local results (2026-10-02)

Source: `497d68d446c7ec59680836464d6da48656849fad`, with only profile/documentation
edits. Host: macOS 15.8.1, x86_64; Rust 1.99.0
(`b940084d7`, 2026-09-28), cargo-nextest 0.9.146, kache 0.28.1. The comparison
used default features, `-p mvmc-core --lib`, the exact filter above and all four
BLAS thread environment variables set to 1. Each profile ran once to warm its
artifacts, followed by one measured run, sequentially with no competing build.

| Profile | Cargo compile time, warm | Test execution | Complete command | Core unit-test executable |
| --- | ---: | ---: | ---: | ---: |
| Former default: opt 0, debug 2, unstripped | 0.13 s | 5.283 s | 7.20 s | 227.32 MiB |
| New default: opt 0, debug 0, stripped | 0.13 s | 3.940 s | 4.39 s | 127.55 MiB |
| `test-fast`: opt 2, debug 0, stripped, no LTO | 0.12 s | 0.820 s | 1.26 s | 37.32 MiB |

The new default reduced this executable by about 44%. The optimized test took
about 4.8 times less execution time than the new default in this single local
comparison. These timings include the existing numerical and RNG assertions;
fixtures and tolerances stayed identical. Executable sizes come from the same
145-test core library harness, rather than accumulated target-directory sizes.

Initial workspace profile builds observed 2m 37s of Cargo compilation for the
unoptimized stripped candidate and 6m 29s for `test-fast`. Cache and scheduling
conditions differed between those initial builds; these record the build cost
of selecting an optimized profile. Rebuilding the core-library-only optimized
feature selection also took 4m 56s before its first test run. Choose `test-fast`
when repeated numerical execution outweighs that initial compilation cost.

During the comparison and verification in a separate target directory, kache's
reported local hits increased from 8 to 207. This confirms compiler cache reuse
even when Cargo artifacts are isolated between checkouts.

Final verification used a separate target directory, `--workspace --all-features
--locked --offline`, single-threaded BLAS, zero retries and no fail-fast. Both
profiles passed all 419 tests, with the same 8 ignored tests. The already
completed comparison recorded 660.239 s of nextest execution for the normal
profile and 84.498 s for `test-fast`, excluding compilation and discovery. Use
`test-fast` for subsequent full runs. All-target, all-feature Clippy with
`-D warnings`, the workspace doctest command and `cargo fmt --all --check` also
passed; the workspace currently contains no runnable doctests.

## Optional inner-kernel threading

The independent QP loop in the Pfaffian setup can be enabled without changing
the sequential sampling/RNG contract. The default is one worker. Set
`MVMC_RS_INNER_THREADS` to the requested worker count and
`MVMC_RS_INNER_THRESHOLD` to the minimum QP range length; each worker owns a
scratch workspace and results are copied back in QP order.

For example, the deterministic `spin_chain_lanczos` parity gate was measured
on 2026-10-03 with the same BLAS settings and warm Cargo target:

```sh
env -u MVMC_RS_INNER_THREADS -u MVMC_RS_INNER_THRESHOLD \
  MVMC_RS_LANCZOS_PHYSICAL=1 MVMC_RS_LANCZOS_MODEL=spin_chain_lanczos \
  cargo nextest run -p mvmc-core --test lanczos_transfer_physcal
env MVMC_RS_INNER_THREADS=2 MVMC_RS_INNER_THRESHOLD=1 \
  MVMC_RS_LANCZOS_PHYSICAL=1 MVMC_RS_LANCZOS_MODEL=spin_chain_lanczos \
  cargo nextest run -p mvmc-core --test lanczos_transfer_physcal
```

The sequential and two-worker runs both passed the full Julia fixture gate;
wall time was 189.10 s and 185.92 s respectively on the recorded host. The
small difference is workload and host dependent, so this control should be
benchmarked with the target model before enabling it by default.

## MPI build

MPI support is an opt-in `mvmc-core/mpi` feature. It owns the `mpi::Universe`
inside `MpiContext`, exposes rank/size and root checks, and implements the
workspace `Reducer` allreduces without calling `MPI_Finalize` from library
code. Build and run MPI tests on a host with `mpicc`/`mpirun` available:

```sh
cargo check -p mvmc-core --features mpi
mpirun -n 2 cargo nextest run -p mvmc-core --features mpi
```

The ordinary build does not enable this feature. On hosts without an MPI
installation, the feature check fails during `mpi-sys` discovery; the default
single-process build and tests remain independent of that system dependency.


## Portable numerical comparisons

After #186, computed floating-point parity uses explicit absolute and relative
bounds on native Linux and macOS. C remains the numerical algorithm authority,
Julia the design reference; Linux is the reference-generation environment.
RNG words, draw counts, proposals, acceptance and saved configurations remain
exact gates. See [NUMERICAL_COMPARISONS.md](NUMERICAL_COMPARISONS.md) for the
comparison inventory, operation budgets, residual checks and nonfinite rules.
Use `cargo nextest run --workspace --locked --cargo-profile test-fast` for
full checks; include `--all-features --run-ignored all` for optional coverage.
