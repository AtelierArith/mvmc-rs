# Linux x86_64 development container

The repository's `.devcontainer/devcontainer.json` builds an Ubuntu 24.04
`linux/amd64` environment. Open the repository in VS Code with the **Dev
Containers** extension and select **Dev Containers: Reopen in Container**.
Use **Rebuild Container** after changing the Dockerfile. Docker must be running;
on an ARM host Docker uses x86_64 emulation, so execution can be slower.

Until issue #186 is resolved, numerical work prioritizes native macOS C parity.
Afterward Linux becomes the numerical reference environment, and non-Linux
hosts should use this Dev Container for numerical work. Keep each platform's
provenance and distinguish their verification results. Rust and Julia BLAS
providers differ: evidence-based absolute/relative bounds may be used for
BLAS-dependent Julia comparisons after verifying exact RNG, controls and
configurations. Reproducible C kernel contracts remain strict.

The image includes Rust stable with Clippy and rustfmt, cargo-nextest 0.9.146,
kache 0.28.1, uv 0.12.21, Clang 18/libclang, GCC/Fortran, OpenBLAS/LAPACK and
MPICH. Rust follows the repository's `rust-toolchain.toml`; rebuilding can
install a newer stable compiler. The Ubuntu package repository and base image
also receive updates. Record `rustc -Vv`, `clang-18 --version`, `mpichversion`,
`pkg-config --modversion openblas`, `ldd --version` and the image ID alongside
numerical verification. The environment fixes the OS/architecture and tooling
choices, rather than asserting that future package updates are byte identical.

The shell runs as `vscode`. Setup repairs ownership of newly created named
volumes, initializes the pinned Julia reference data submodule used by existing
Rust integration tests, and prints tool/backend versions. It is safe to rerun:

```sh
bash .devcontainer/setup.sh
```

Cargo registry data and Linux build/cache data use named volumes unique to this
Dev Container. Binaries live in the image under `/opt/rust/bin` and
`/opt/devtools/bin`. `CARGO_TARGET_DIR` points to
`/home/vscode/.cache/mvmc/target/linux-x86_64`; kache's local store and uv's cache
are under the same Linux-only cache volume. A read-only bind mount places
`.devcontainer/cargo-config.toml` over the workspace's `.cargo/config.toml`
inside the container, preserving the host file. This prevents host settings
such as a forced absolute `KACHE_CONFIG` from overriding the container setup.
The container uses its own Rust wrapper and target directory. `KACHE_CONFIG` points to
`/opt/devcontainer/kache.toml`, and `RUSTC_WRAPPER` selects the installed kache.
BLAS/OpenMP thread counts default to one for deterministic comparisons.

## Commands inside the container

```sh
cargo nextest run -p mvmc-core -E 'test(<name>)' --locked
cargo nextest run --workspace --locked --cargo-profile test-fast \
  --no-fail-fast --retries 0
cargo nextest run --workspace --all-features --locked --cargo-profile test-fast \
  --run-ignored all --no-fail-fast --retries 0
cargo test --workspace --all-features --locked --doc
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --all --check
kache stats --json
uv run --no-project python scripts/check_complex_division_c_parity.py
```

Use `cargo nextest` for development tests and `test-fast` for full verification.
The all-feature run uses system MPICH via `MPICC=/usr/bin/mpicc`; bindgen locates
libclang through `LIBCLANG_PATH=/usr/lib/llvm-18/lib`. A normal nextest invocation
is single-process execution, not proof of a multi-rank MPI workload. Run and
record a separate MPI workload when verifying distributed behavior.

The last Python command is an optional C oracle check and requires the
reference submodule. Ordinary Cargo tests read checked-in fixtures and do not
require C or Julia oracle execution. To obtain reference sources explicitly:

```sh
git submodule update --init --recursive
```

Historical macOS fixtures retain their provenance. Platform-specific expected
values must come from independent C/Julia oracle execution with the compiler,
libm and BLAS versions recorded. Linux failures must be investigated at the
first numerical or RNG divergence; changing OS is not permission to relax
Rust-to-C comparisons, skip failures or derive expectations from Rust output.

Julia is optional for ordinary Rust development. The previous unpinned Julia
feature is replaced by a pinned, checksum-verified installation command:

```sh
bash .devcontainer/install-julia.sh
export JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot
/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia \
  --project=extern/Julia-mVMC -e 'using Pkg; Pkg.instantiate(); Pkg.build("PfaPack"); Pkg.build("SFMT")'
/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia \
  --project=extern/Julia-mVMC path/to/reference-script.jl
```

This invokes Julia 1.13.1 directly, equivalent to selecting `julia +1.13.1`
through juliaup, and uses the checked-in `Manifest-v1.13.toml`. The persistent
Linux cache volume stores both Julia and its depot. Record Julia and BLAS versions and confirm matching RNG
state, draw order/count and configurations before attributing differences to
numerical calculation. Do not present historical Julia 1.11 fixtures as fresh
Julia 1.13 results.

Linux GNU fixture selection, the pure-Rust GNU quotient and exact historical
Julia runner overlays are described in [Linux numerical contracts](LINUX_NUMERICAL_CONTRACTS.md).

Existing Julia, Python/Jupyter, Git and C/CMake editor extensions and forwarded
ports 1234, 8000, 8080 and 8888 are preserved. Python tools use uv rather than
global pip installations; for example `uv tool run --from jupyterlab jupyter-lab` starts an optional
notebook server and `uv tool run ruff check path/to/code` runs its linter.

## Command-line Dev Container startup

The official Dev Container CLI can exercise the same configuration as VS Code:

```sh
npx --yes @devcontainers/cli up --workspace-folder "$PWD"
npx --yes @devcontainers/cli exec --workspace-folder "$PWD" \
  cargo nextest run --workspace --locked --cargo-profile test-fast \
  --no-fail-fast --retries 0
```

When an execution sandbox makes the default npm or buildx cache unwritable,
set `npm_config_cache` and `BUILDX_CONFIG` to writable task-specific directories.
These overrides concern the host CLI only; do not change the user's existing
Docker or shell configuration.

## Verified environment (2026-10-03)

Actual `@devcontainers/cli` 0.89.0 `up` and `exec` ran on Docker Desktop 4.93.0,
Engine 29.8.1 (Linux amd64). The built image ID was
`sha256:9d1e684e69aaa8bfd963da33c27939c08053d4449f3c2c7f9d9ba1ff28289f84`;
Ubuntu base digest was
`sha256:d94c97dd9cacf183d0a6fd12a8e87b526e9e928307674ae9c94139139c0c6eae`.

| Component | Observed version/settings |
| --- | --- |
| Rust / Cargo | 1.99.0, Rust commit `b940084d7`, LLVM 23.1.1 |
| nextest / kache / uv | 0.9.146 / 0.28.1 / 0.12.21 |
| Clang | 18.1.3, Ubuntu `1ubuntu1` |
| glibc | 2.39, Ubuntu `2.39-0ubuntu8.6` |
| Rust BLAS backend | OpenBLAS 0.3.26 (`0.3.26+ds-1ubuntu0.1`), LP64, pthread |
| LAPACK | Ubuntu `3.12.0-3build1.1` |
| MPI | MPICH 4.2.0, `ch4:ucx` |
| BLAS settings | `OPENBLAS_NUM_THREADS=1`, OpenMP/MKL/BLIS threads 1; `OPENBLAS_CORETYPE` unset (auto) |
| Optional Julia | 1.13.1, official SHA-256 checked before extraction |

A dependency-free library compiled twice with incremental compilation disabled
and distinct `/tmp` target directories. In a fresh dedicated kache store,
statistics changed from zero hits/zero misses to zero hits/one miss, then one
local hit/one miss. The actual configured wrapper also served an equivalent
smoke build, and `uv run --no-project --no-python-downloads python` reported
`Linux x86_64`. These prove environment/cache execution; workspace correctness
is established separately by the full test commands above.


The final Linux workspace passed 486 default tests (8 optional-feature tests
ignored), and 494 tests with all features and `--run-ignored all` (zero skipped).
Documentation tests, formatting, and all-target/all-feature Clippy with
`-D warnings` passed. Native GNU quotient/Green fixtures and independently
generated Linux Julia references retain their exact gates. This observed
agreement does not require all future BLAS providers to produce identical bits.
The reference Julia used bundled OpenBLAS 0.3.30 ILP64; its native inverse and
Rust linked system OpenBLAS 0.3.26 LP64, single-thread Haswell dispatch.

The actual CLI was also recreated with a host-local Cargo configuration forcing
`KACHE_CONFIG` to a macOS absolute path. The read-only configuration mount made
workspace `cargo check` succeed and a pair of fresh-target smoke builds produced
one actual kache local hit; the host configuration's SHA-256 stayed unchanged.


The shell orchestrator also completed an independent full Linux Julia overlay
check: 29 jobs, prefixes 1/2/3/50, 871 generated files. Every one of the 856
recorded reference hashes reproduced exactly, with zero overlay mismatches and
zero staged fixture changes. The 15 additional unconsumed loaded/history files
are recorded in the external staging manifest. Historical macOS archives were
restored in staging as well as preserved in the source checkout.


Native macOS verification of the same main-plus-#187 source passed all 496
all-feature tests with ignored tests enabled (zero skipped), using Apple Clang
17, Rust 1.99.0 and local MPICH 4.2.3. The archived macOS numerical checks and
callback arithmetic therefore remain verified separately from Linux.
The previous Julia editor project setting `JULIA_PROJECT=@.` is preserved;
Julia threads default to one for deterministic reference work.

## Explicit MPI gates (#392)

The `#[ignore]`d MPI tests are invisible to ordinary CI. Run all of them (2 and 4
ranks, every parameter cell they need, plus the `mvmc-cli` MPI tests) inside the
container with `scripts/run_explicit_mpi_gates.sh /tmp/new-scratch-dir`; every line
must end in `rc=0`. Run it after changing validation, initialization, run-log output
files or the grouped (`NSplitSize`) paths.
