# Validation of measured runtime f2645823

All commands used `/tmp/mvmc-629-cpu` and native Cargo target
`/tmp/mvmc-629-target`, with repository-configured compiler caching.

```bash
CARGO_TARGET_DIR=/tmp/mvmc-629-target cargo build --locked --release \
  -p mvmc-cli -p mvmc-core --bin mvmc --example cpu_scaling_contract_586
CARGO_TARGET_DIR=/tmp/mvmc-629-target cargo nextest list --locked --workspace \
  --cargo-profile test-fast --message-format json
git submodule update --init extern/Julia-mVMC
CARGO_TARGET_DIR=/tmp/mvmc-629-target cargo nextest run --locked --workspace \
  --cargo-profile test-fast --no-fail-fast --retries 0
CARGO_TARGET_DIR=/tmp/mvmc-629-target cargo test --locked --workspace --doc
CARGO_TARGET_DIR=/tmp/mvmc-629-target cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
CARGO_TARGET_DIR=/tmp/mvmc-629-cuda-target cargo check --locked \
  --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --all-targets
```

Full workspace: **1,645 passed**, 67 existing ignored helpers/optional gates.
Doctest commands succeeded (zero executable doctests in these crates).
Clippy, formatting and all CUDA targets compiled successfully. Compressed logs retain the initial missing-input
failure and successful full rerun; no test was newly skipped. Additional focused
regression logs retain development failures and their corrected runs. Numerical
bounds and reference expectations were not relaxed.

Measured-runtime PR #633 CI run `41591065374` passed all seven jobs, including
Linux/macOS default and MPI/SIMD tests. The final report commit gets its own CI
before merge; GitHub PR checks are the authoritative final-commit status.

`logs-sha256.json` verifies the saved compressed logs and their original text.
The performance/contract drivers ran before the complete test/check sequence,
not concurrently with measurements. CI runs on remote GitHub runners.
