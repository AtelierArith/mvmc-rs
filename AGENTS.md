# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust port of Julia-mVMC organized as a Cargo workspace. Core crates live under `crates/`: `sfmt19937`, `pfapack`, `mvmc-expert-parsers`, `mvmc-core`, and `mvmc-cli`. Workspace automation is in `xtask/`. Cross-crate fixtures and reference data live in `tests/fixtures/`, while crate-local integration tests live in each crate's `tests/` directory. Benchmark code is under `benchmark/pfapack_compare/`; generated benchmark output belongs in `benchmark/pfapack_compare/results/`. Upstream/reference implementations are kept in `extern/` and `reference/`; avoid editing vendored sources unless the task explicitly requires it.

## Build, Test, and Development Commands

- `cargo check --workspace`: type-check the full workspace.
- `cargo test --workspace`: run all normal Rust tests.
- `cargo test -p pfapack --features 'simd-backend blas-backend'`: test optimized PfaPack backends.
- `cargo clippy --workspace --all-targets -- -D warnings`: lint all workspace targets.
- `cargo fmt --all --check`: verify formatting.
- `cargo run -p mvmc-cli -- <namelist.def>`: run the CLI on an Expert-mode input.
- `cargo run -p xtask -- bench-julia --steps 50 --reps 5 --warmups 1 --threads 1`: compare Rust and Julia workloads.
- `scripts/run_all.sh`: run the PfaPack comparison suite and generate a report.

## Coding Style & Naming Conventions

Rust uses edition 2021 with `rustfmt.toml` enforcing `max_width = 100`, field init shorthand, and `?` shorthand. Use four-space indentation and idiomatic Rust naming: `snake_case` for functions/modules, `PascalCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Keep numerical kernels explicit and localized; preserve scalar reference paths when adding feature-gated optimized implementations.

## Development and Backward Compatibility

This Rust crate and its workspace are under active development. Backward compatibility may be completely ignored during development, including compatibility with existing Rust APIs. Treat the Julia implementation in `extern/` as the source of truth. Change or remove existing APIs and update their callers and tests whenever needed to match that implementation; do not retain compatibility wrappers solely to preserve the old Rust API.

## Testing Guidelines

Use Rust's built-in test framework plus crate-local integration tests. Golden and parity tests compare against Julia/reference fixtures; keep tolerances explicit near assertions. Name tests by behavior, for example `pfaffian_matches_julia_fixture` or `rejects_invalid_header`. For performance-sensitive changes, run both correctness tests and the relevant benchmark variant.

### Deterministic Julia Parity

Use Julia 1.13.1 and the reference workspace's `extern/Julia-mVMC/Manifest-v1.13.toml` for current parity work. Run reference scripts with `julia +1.13.1 --project=extern/Julia-mVMC`. Record the Julia and BLAS versions when generating numerical fixtures; historical Julia 1.11 fixtures must not be presented as newly verified Julia 1.13 results.

The Rust and Julia implementations already use matching random-number algorithms and seeds. Port Julia behavior faithfully: preserve RNG initialization, draw order and count, integer/float conversion, and RNG state throughout parameter initialization, burn-in, move proposals, acceptance/rejection, and sampling. This includes draws on rejected moves and conditional branches. Refactoring and optimization must preserve the same deterministic trajectory for the same inputs and seed.

Do not accept discrepancies as Monte Carlo noise or statistical fluctuations. Compare RNG states, proposed moves, acceptance decisions, saved configurations, and intermediate numerical results to locate the first divergence. Preserve numerical operation order where required for parity. Do not loosen tolerances, reseed, or average repeated runs to conceal a mismatch; floating-point tolerances must never excuse RNG or sampling-trajectory drift. Statistical reference checks supplement deterministic Julia parity checks and do not replace them.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `Optimize utu2 inverse slice access` and `Add SIMD plus BLAS benchmark variant`. Keep commits focused and avoid mixing generated benchmark artifacts with code changes unless the report is requested. Pull requests should describe the numerical behavior changed, list commands run, mention enabled features such as `simd-backend` or `blas-backend`, and link related issues. Include benchmark report paths when performance claims are made.

## Agent-Specific Instructions

Do not revert unrelated user changes. Prefer `rg` for repository searches. Treat `extern/` as reference material unless directed otherwise, and keep generated `target*` directories out of commits.
