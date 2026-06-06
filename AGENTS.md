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

## Testing Guidelines

Use Rust's built-in test framework plus crate-local integration tests. Golden and parity tests compare against Julia/reference fixtures; keep tolerances explicit near assertions. Name tests by behavior, for example `pfaffian_matches_julia_fixture` or `rejects_invalid_header`. For performance-sensitive changes, run both correctness tests and the relevant benchmark variant.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `Optimize utu2 inverse slice access` and `Add SIMD plus BLAS benchmark variant`. Keep commits focused and avoid mixing generated benchmark artifacts with code changes unless the report is requested. Pull requests should describe the numerical behavior changed, list commands run, mention enabled features such as `simd-backend` or `blas-backend`, and link related issues. Include benchmark report paths when performance claims are made.

## Agent-Specific Instructions

Do not revert unrelated user changes. Prefer `rg` for repository searches. Treat `extern/` as reference material unless directed otherwise, and keep generated `target*` directories out of commits.
