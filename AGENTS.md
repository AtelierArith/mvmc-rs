# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust port of mVMC organized as a Cargo workspace. Core crates live under `crates/`: `sfmt19937`, `pfapack`, `mvmc-expert-parsers`, `mvmc-core`, and `mvmc-cli`. Workspace automation is in `xtask/`. Cross-crate fixtures and reference data live in `tests/fixtures/`, while crate-local integration tests live in each crate's `tests/` directory. Benchmark code is under `benchmark/pfapack_compare/`; generated benchmark output belongs in `benchmark/pfapack_compare/results/`. Upstream/reference implementations are kept in `extern/` and `reference/`; avoid editing vendored sources unless the task explicitly requires it.

## Build, Test, and Development Commands

- `cargo check --workspace`: type-check the full workspace.
- `cargo nextest run -p <crate> -E 'test(<name>)'`: run short targeted development tests.
- `cargo nextest run --workspace --cargo-profile test-fast`: run all Rust unit and integration tests with optimized kernels and development checks.
- `cargo nextest run -p pfapack --features 'simd-backend blas-backend'`: test optimized PfaPack backends.
- `cargo test --workspace --doc`: run documentation tests separately.
- `cargo clippy --workspace --all-targets -- -D warnings`: lint all workspace targets.
- `cargo fmt --all --check`: verify formatting.
- `cargo run -p mvmc-cli -- <namelist.def>`: run the CLI on an Expert-mode input.
- `cargo run -p xtask -- bench-julia --steps 50 --reps 5 --warmups 1 --threads 1`: compare Rust and Julia workloads.
- `scripts/run_all.sh`: run the PfaPack comparison suite and generate a report.

## Coding Style & Naming Conventions

Rust uses edition 2021 with `rustfmt.toml` enforcing `max_width = 100`, field init shorthand, and `?` shorthand. Use four-space indentation and idiomatic Rust naming: `snake_case` for functions/modules, `PascalCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Keep numerical kernels explicit and localized; preserve scalar reference paths when adding feature-gated optimized implementations.

## Development and Backward Compatibility

This Rust crate and its workspace are under active development. Backward compatibility may be completely ignored during development, including compatibility with existing Rust APIs. Treat the C implementation in `extern/mVMC-1.3.0/` as the authoritative source of truth. Use `extern/Julia-mVMC/` as a secondary port/reference and test aid; when their behavior differs, inspect C and make Rust follow its supported input contract and numerical behavior. Do not copy Julia-only repairs or extensions as C parity. Change or remove existing APIs and update their callers and tests whenever needed to match that implementation; do not retain compatibility wrappers solely to preserve the old Rust API.

The historical implementation was written in C and later reimplemented in Julia. Rust should follow the Julia design for public APIs, runner structure, lifecycle, and test organization, while following C for numerical authority: parameter layout and offsets, initialization and draw order, arithmetic and operation order, signs, formatting, and floating-point results. When the two references differ, keep the Julia architecture and adopt the defined C numerical contract. Record the distinction in focused tests and preserve the authoritative algorithm and input contracts when applying the numerical-comparison policy below.

The project target is a pure-Rust implementation. When an implementation choice is unclear, port the C algorithm and numerical operation order into Rust rather than adding a C FFI or runtime dependency; use C only as the reference for behavior and validation.

## Testing Guidelines

Use Rust's built-in test framework plus crate-local integration tests. Golden and parity tests compare against C/reference fixtures, using Julia fixtures where they agree with C; keep tolerances explicit near assertions. Name tests by behavior, for example `pfaffian_matches_julia_fixture` or `rejects_invalid_header`. For performance-sensitive changes, run both correctness tests and the relevant benchmark variant.

Use `cargo nextest run` for development unit and integration test runs. Use `--cargo-profile test-fast` for full-workspace verification and long numerical regressions; reserve the normal profile for short targeted TDD checks instead of running the full suite unoptimized. Preserve requested features, profiles and lock-file constraints; use `--no-fail-fast --retries 0` when collecting all failures. Run documentation tests separately with `cargo test --workspace --doc`, because nextest does not run doctests. Compiler caching remains configured through kache.

All tests on `main` must pass. Before merging a milestone, fix known workspace test failures, including pre-existing regressions, and verify the final changes. Do not merge with failing tests or hide failures by skipping tests, arbitrarily increasing numerical tolerances, or replacing independent expectations with Rust-generated results. Distinguish historical Julia fixtures from C-compatible references explicitly when their contracts differ.

The normal development/test profiles omit debug information and strip symbols. Use `test-fast` for long numerical regressions; it retains debug assertions and overflow checks while avoiding release LTO. Keep each checkout's target directory separate and use kache for cross-checkout cache reuse. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for measurements and environment overrides when debugger symbols are needed.

### C Reference Toolbox

Accumulate reusable C comparison programs and useful extracted C source in `c_toolbox/`. Keep their upstream origin, source SHA-256 hashes, extraction boundaries, compiler options and reproduction commands documented. Preserve the authoritative vendored source in `extern/`; use the toolbox for probes and supporting code. Distinguish standalone kernel checks from full C executable/MPI/sampling validation.

Rust builds and tests must remain independent of `c_toolbox/`. Do not compile, invoke or read toolbox programs from Cargo build scripts or Rust tests. Generate C-derived expected values separately and check them into `tests/fixtures/` with provenance. Normal Rust tests must run using those fixtures without `c_toolbox/`, rebuilding the C reference, or invoking C/Julia oracle programs; preserve ordinary Rust compiler/linker and BLAS/LAPACK requirements. Toolbox checks and fixture regeneration are explicit, optional developer commands.

### Deterministic Reference Parity

Issue #186 was resolved by PR #191 with native macOS and Linux checks. Use Linux as the numerical reference environment and the Dev Container for reference numerical work on non-Linux hosts; verify test portability on native Linux and macOS. Keep platform references and results explicitly labelled, and record the architecture, compiler, Rust/Julia versions and actual BLAS backends with generated results.

Do not compare computed floating-point results bitwise, including Rust-to-C comparisons. Follow the portable numerical policy tracked by #190 and documented in [docs/NUMERICAL_COMPARISONS.md](docs/NUMERICAL_COMPARISONS.md). Use explicit absolute and relative tolerances justified by the first numerical divergence, operation, problem scale, conditioning and solver residuals; Rust and Julia may use different BLAS providers. Test both Linux and macOS. Preserve exact RNG initialization, state, draw order/count and conversion behavior independently for a fixed control path. General mathematical-function roundoff and addition-order differences may change acceptance decisions and subsequent long-run configurations/final RNG state across implementations; this is permitted after locating the numerical divergence. A tolerance must not conceal an algorithm or RNG defect.

When using Julia for parity comparisons, use Julia 1.13.1 and the reference workspace's `extern/Julia-mVMC/Manifest-v1.13.toml`. Run reference scripts with `julia +1.13.1 --project=extern/Julia-mVMC`. Record the Julia and BLAS versions when generating numerical fixtures; historical Julia 1.11 fixtures must not be presented as newly verified Julia 1.13 results.

The Rust and Julia implementations already use matching random-number algorithms and seeds. Port the authoritative C behavior faithfully, using Julia comparisons where their behavior agrees: preserve RNG initialization, draw order and count, integer/float conversion, and RNG state throughout parameter initialization, burn-in, move proposals, acceptance/rejection, and sampling. This includes draws on rejected moves and conditional branches. Refactoring and optimization must preserve RNG behavior on identical control paths. Small numerical changes may alter acceptance branches; same-implementation runs with the same inputs and seed must remain reproducible.

Do not accept discrepancies as Monte Carlo noise or statistical fluctuations. Compare RNG states, proposed moves, acceptance decisions, saved configurations, and intermediate numerical results to locate the first divergence. Preserve numerical operation order where required for parity. Rust must follow the authoritative C numerical algorithm, operation order and signs, using the comparison policy above for computed floating-point results. Do not increase tolerances without numerical evidence, reseed, or average repeated runs to conceal a mismatch; floating-point tolerances must never excuse an RNG defect; subsequent trajectory divergence caused by a demonstrated numerical acceptance difference is permitted. Statistical reference checks supplement deterministic reference parity checks and do not replace them.

Julia-side verification may accept numerical calculation error with explicit, justified absolute/relative tolerances only after verifying exact RNG initialization, state, draw order and draw count. Locate and explain the numerical divergence before choosing a tolerance. After #186, justified numerical tolerances also apply to Rust-to-C checks under #190; they do not relax the algorithm/input or independent RNG contracts. Use 20-step long-run repeatability tests rather than forcing historical reference trajectories after a numerical acceptance divergence.

## Commit & Pull Request Guidelines

Create an issue describing the problem, scope and acceptance criteria before starting new implementation work. If the related issue is already closed but work remains, create a follow-up issue and link the earlier issue. Implement and validate against that open issue, then reference it in the pull request.

Recent history uses short imperative commit subjects such as `Optimize utu2 inverse slice access` and `Add SIMD plus BLAS benchmark variant`. Keep commits focused and avoid mixing generated benchmark artifacts with code changes unless the report is requested. Pull requests should describe the numerical behavior changed, list commands run, mention enabled features such as `simd-backend` or `blas-backend`, and link related issues. Every pull request body must explicitly identify the related issue with `Closes #...` when it completes the issue or `Related to #...` when the issue remains open. Include benchmark report paths when performance claims are made.

For the issue #56 implementation work, commit each validated implementation milestone, create a pull request, and merge it into `main` before starting the next milestone on a new branch. This workflow is authorized by the user; do not ask for confirmation at each commit, pull request, or merge. Keep issues open until their full acceptance criteria have been implemented and verified.

Julia reference defects should be fixed on a dedicated fork branch with focused regression tests. Submit all Julia-side patches to a single upstream pull request titled `julia-patch` at `https://github.com/tmisawa/Julia-mVMC`; append further fixes to that same PR instead of creating a separate PR for each defect. The current Julia patch PR is `https://github.com/tmisawa/Julia-mVMC/pull/54`. This Julia aggregation rule is separate from the Rust milestone commit/PR/merge workflow.

## Agent-Specific Instructions

Do not revert unrelated user changes. Prefer `rg` for repository searches. Treat `extern/` as reference material unless directed otherwise, and keep generated `target*` directories out of commits.

Use the Linux x86_64 environment in `.devcontainer/` for container development. Keep container Cargo targets and caches in its named volumes, separate from macOS artifacts. Follow [docs/DEV_CONTAINER.md](docs/DEV_CONTAINER.md) for opening, verification and reference-generation commands; ordinary Rust tests must remain independent of reference runtimes.

## Local Agent Skills

- Use [uv-python](skills/uv-python/SKILL.md) for Python execution, scripts, modules, dependency management and environments. Run Python through `uv`; use the project's environment and lock where applicable, or `uv run --no-project` for independent utilities.
- Use [kache](skills/kache/SKILL.md) when installing, updating, configuring or verifying the Rust compiler cache. Verify the configured binary and an actual cache hit.
