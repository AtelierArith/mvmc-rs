# Generate numerical references

Run the independent C and Julia reference generators through one shell command:

```sh
scripts/generate-numerical-references.sh --list
scripts/generate-numerical-references.sh --suite c-projection
```

C is the numerical authority. Julia provides port/design comparisons and
historical references. The script never runs Rust to produce expected values,
and normal Cargo tests never invoke the script or a reference runtime.

By default the script runs the listed C suites. Select individual suites with
repeatable `--suite NAME`, or select `--reference julia` for the two small FSZ
setup/move suites. Sampling subsets require explicit suite selection; no full
50-step workload runs automatically. Use `--help` for all options.

## Prerequisites

Install uv, a C compiler and Git, then initialize reference data explicitly:

```sh
git submodule update --init --recursive -- extern/mVMC-1.3.0 extern/Julia-mVMC
```

Python runs exclusively through `uv run --no-project python`. `--compiler`
selects one executable, such as `gcc`, `clang-18` or an absolute path; it defaults
to `$CC` or `cc`. Each generator retains its own real compiler options. The
shell wrapper records the actual compiler executable and arguments, including
those options. Compiler/runtime differences can change exact numerical output.
A generator error stops the run and produces a failed manifest; it does not
skip the suite or automatically switch compilers.

For Julia, use exactly Julia 1.13.1 and the pinned
`extern/Julia-mVMC/Manifest-v1.13.toml`. Pass a direct binary with `--julia` or
`JULIA_BINARY`; otherwise the script selects `julia +1.13.1` when juliaup is
available, or verifies the `julia` binary on PATH. A different version fails
before generation. Package and native reference library setup can run in the
staging copy with `--instantiate`; this calls `Pkg.instantiate()` and builds
both `PfaPack` and `SFMT`. It refuses generation if setup changes the pinned
manifest. Use a writable, platform-specific Julia depot and uv cache.

```sh
JULIA_DEPOT_PATH=/path/to/linux-julia-depot \
  scripts/generate-numerical-references.sh --reference julia \
  --suite julia-fsz-moves --instantiate --julia /path/to/julia-1.13.1/bin/julia
```

The script fixes BLAS/OpenMP/MKL/BLIS and Julia thread counts to one and records
`OPENBLAS_CORETYPE` without choosing a CPU backend silently. Ordinary Rust
BLAS/LAPACK bindings use LP64; Julia's backend is recorded separately and may
use ILP64. Record compiler, libm, BLAS and package versions when interpreting
numeric differences.

## Stage, inspect and explicitly apply

Generation copies scripts, C toolbox, fixtures and available reference source
into a fresh staging tree. It does not use writable source symlinks or change
upstream sources, checked-in macOS expectations, or the running test checkout.
The default is a persistent temporary directory; provide a new or empty
`--output` directory to choose its location:

```sh
scripts/generate-numerical-references.sh --suite c-projection \
  --output /tmp/mvmc-reference-review
cat /tmp/mvmc-reference-review/changes.json
cat /tmp/mvmc-reference-review/manifest.json
diff -ru tests/fixtures /tmp/mvmc-reference-review/workspace/tests/fixtures
```

The staging directory contains per-suite logs, `commands.txt`,
`compiler-commands.log`, `baseline.json`, `changes.json` and `manifest.json`.
The manifest records actual OS/architecture, C compiler, libc, BLAS discovery,
thread variables, Julia version/backend when selected, source checkout and
submodule revisions, input SHA-256 hashes, exact generator/compiler commands,
output hashes and success/failure status. Julia package builds can create native
libraries in the staging reference copy; these are recorded as supporting
changes and are never applied to upstream source.

Some existing generators retain historical Apple/Julia source headers. Those
legacy headers are preserved by this orchestration script; they are not proof
that a new Linux file was generated on macOS. The staging manifest and compiler
log identify the actual generation environment. Review and document provenance
before deciding which new platform fixture belongs in a test contract. Never
claim native C executable/MPI/sampling coverage from a standalone kernel probe.

After reviewing a successful run, apply its fixture/toolbox changes explicitly:

```sh
scripts/generate-numerical-references.sh --apply /tmp/mvmc-reference-review
git diff -- tests/fixtures c_toolbox
cargo nextest run --workspace --locked --cargo-profile test-fast \
  --no-fail-fast --retries 0
```

Apply validates every input hash against the full staged snapshot before
writing anything. It refuses stale scripts, reference source/manifests, input
fixtures or toolbox files, altered staged outputs, failed generation and
symlink destinations. New outputs must still be absent in the destination.
Unrelated Rust implementation changes are allowed. Only changed files beneath
`tests/fixtures/` and `c_toolbox/` are copied; upstream source is never applied.
Each destination file is replaced atomically after all preflight checks pass.
This is a review step, not permission to weaken Rust-to-C comparisons, replace
macOS fixtures blindly, reseed or conceal RNG/configuration drift.

`--check` invokes the existing generators without their `--write` flag in the
copy. It verifies their selected fixture contracts and records failures:

```sh
scripts/generate-numerical-references.sh --suite c-projection --check
```

To use generators from another checkout, including a reviewed branch containing
newer suites, select its source tree explicitly. Existing files in the caller's
checkout are not overwritten:

```sh
/path/to/generate-numerical-references.sh --workspace /path/to/reviewed-checkout \
  --suite c-projection --output /tmp/another-reference-review
```

## Linux verification observed on 2026-10-03

In a real Linux x86_64 Ubuntu 24.04 container with Rust 1.99.0, Clang 18.1.3,
glibc 2.39 and system OpenBLAS 0.3.26 LP64, the `c-projection` generation and
`--check` each passed all 10 cases without source changes. A direct Julia 1.13.1
FSZ move subset passed 8 move cases and 1 bilinear reduction test, and generated
one changed fixture only in staging. Its loaded backend was
`LBTConfig([ILP64] libopenblas64_.so)` with one BLAS thread. The runtime library
reported `OpenBLAS 0.3.30 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell`;
the manifest also records system OpenBLAS's runtime configuration/CPU backend
separately from Julia's bundled ILP64 library.

An initial full-suite Clang run exposed the RBM-counter probe's unsupported
legacy `CMPLX` macro use on glibc. A guarded `__builtin_complex` fallback fixes
only probe input construction, preserving signed zero and leaving upstream
kernels unchanged. Both Clang 18 and explicit `--compiler gcc` then passed all
19 C suites, producing six changed fixture files in staging only. The initial
failure was recorded as failed and could not be applied.

The Julia default setup/move suite also ran successfully with `--instantiate`,
including both native package builds: 96 setup matrix checks, 3 failure checks,
10 initial-sample/retry checks, 8 move checks and 1 bilinear check. It produced
two changed fixtures in staging. Apply mechanics were verified in an isolated
temporary Git repository: changed oracle source and altered staged outputs
were rejected before writes, while unrelated Rust edits were allowed and file
permissions were preserved. These tool checks do not prove full Rust workspace
tests are green or that numerical contracts are identical across platforms.


For independent historical Linux Julia overlays, use an explicit suite:

```sh
scripts/generate-numerical-references.sh --suite julia-linux-cg-fsz --steps 1 --check
scripts/generate-numerical-references.sh --suite julia-linux-all --output /tmp/linux-julia-review
```

These commands require Linux x86_64 and Julia 1.13.1 with the pinned manifest.
The full suite includes setup, sampling, DH histories and all eight FSZ runner
families, using prefixes 1,2,3,50. It is longer than the default two small Julia
suites. `--list` lists per-case choices and `--steps` restricts runner prefixes.
The staged generators record every independent output hash, preserve the
historical fixture tree, and write changed consumed values to the Linux overlay.
`--check` compares the generated values with that overlay or an identical
historical fallback. The source checkout remains unchanged. Review the stage's
`regeneration.json` in the overlay (a generation run) and `manifest.json` before applying.


The `julia-fsz-*` suites retain their raw generator layout for small probes;
select `julia-linux-*` when preparing platform overlays for application.
