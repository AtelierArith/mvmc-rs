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

## Testing Guidelines

Use Rust's built-in test framework plus crate-local integration tests. Golden and parity tests compare against C/reference fixtures, using Julia fixtures where they agree with C; keep tolerances explicit near assertions. Name tests by behavior, for example `pfaffian_matches_julia_fixture` or `rejects_invalid_header`. For performance-sensitive changes, run both correctness tests and the relevant benchmark variant.

Use `cargo nextest run` for development unit and integration test runs. Use `--cargo-profile test-fast` for full-workspace verification and long numerical regressions; reserve the normal profile for short targeted TDD checks instead of running the full suite unoptimized. Preserve requested features, profiles and lock-file constraints; use `--no-fail-fast --retries 0` when collecting all failures. Run documentation tests separately with `cargo test --workspace --doc`, because nextest does not run doctests. Compiler caching remains configured through kache.

The normal development/test profiles omit debug information and strip symbols. Use `test-fast` for long numerical regressions; it retains debug assertions and overflow checks while avoiding release LTO. Keep each checkout's target directory separate and use kache for cross-checkout cache reuse. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for measurements and environment overrides when debugger symbols are needed.

### C Reference Toolbox

Accumulate reusable C comparison programs and useful extracted C source in `c_toolbox/`. Keep their upstream origin, source SHA-256 hashes, extraction boundaries, compiler options and reproduction commands documented. Preserve the authoritative vendored source in `extern/`; use the toolbox for probes and supporting code. Distinguish standalone kernel checks from full C executable/MPI/sampling validation.

Rust builds and tests must remain independent of `c_toolbox/`. Do not compile, invoke or read toolbox programs from Cargo build scripts or Rust tests. Generate C-derived expected values separately and check them into `tests/fixtures/` with provenance. Normal Rust tests must run using those fixtures without `c_toolbox/`, rebuilding the C reference, or invoking C/Julia oracle programs; preserve ordinary Rust compiler/linker and BLAS/LAPACK requirements. Toolbox checks and fixture regeneration are explicit, optional developer commands.

### Deterministic Reference Parity

When using Julia for parity comparisons, use Julia 1.13.1 and the reference workspace's `extern/Julia-mVMC/Manifest-v1.13.toml`. Run reference scripts with `julia +1.13.1 --project=extern/Julia-mVMC`. Record the Julia and BLAS versions when generating numerical fixtures; historical Julia 1.11 fixtures must not be presented as newly verified Julia 1.13 results.

The Rust and Julia implementations already use matching random-number algorithms and seeds. Port the authoritative C behavior faithfully, using Julia comparisons where their behavior agrees: preserve RNG initialization, draw order and count, integer/float conversion, and RNG state throughout parameter initialization, burn-in, move proposals, acceptance/rejection, and sampling. This includes draws on rejected moves and conditional branches. Refactoring and optimization must preserve the same deterministic trajectory for the same inputs and seed.

Do not accept discrepancies as Monte Carlo noise or statistical fluctuations. Compare RNG states, proposed moves, acceptance decisions, saved configurations, and intermediate numerical results to locate the first divergence. Preserve numerical operation order where required for parity. Do not loosen tolerances, reseed, or average repeated runs to conceal a mismatch; floating-point tolerances must never excuse RNG or sampling-trajectory drift. Statistical reference checks supplement deterministic reference parity checks and do not replace them.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `Optimize utu2 inverse slice access` and `Add SIMD plus BLAS benchmark variant`. Keep commits focused and avoid mixing generated benchmark artifacts with code changes unless the report is requested. Pull requests should describe the numerical behavior changed, list commands run, mention enabled features such as `simd-backend` or `blas-backend`, and link related issues. Include benchmark report paths when performance claims are made.

For the issue #56 implementation work, commit each validated implementation milestone, create a pull request, and merge it into `main` before starting the next milestone on a new branch. This workflow is authorized by the user; do not ask for confirmation at each commit, pull request, or merge. Keep issues open until their full acceptance criteria have been implemented and verified.

## Agent-Specific Instructions

Do not revert unrelated user changes. Prefer `rg` for repository searches. Treat `extern/` as reference material unless directed otherwise, and keep generated `target*` directories out of commits.

## Local Agent Skills

- Use [uv-python](skills/uv-python/SKILL.md) for Python execution, scripts, modules, dependency management and environments. Run Python through `uv`; use the project's environment and lock where applicable, or `uv run --no-project` for independent utilities.
- Use [kache](skills/kache/SKILL.md) when installing, updating, configuring or verifying the Rust compiler cache. Verify the configured binary and an actual cache hit.
