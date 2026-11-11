# Validation provenance for #731

The selected numerical sources are CPU `da97a821` with public provider fork
`a3dd7b450e81df16065ebf51788414d87bb44171`, and CUDA `d71edcee`. Later report
and lint-only test changes do not alter those numerical kernels. CPU uses the
cached repository Linux x86_64 development image (Rust 1.99.0, MPICH 4.2.0,
OpenBLAS 0.3.26); CUDA uses the recorded CUDA 12.9.2 toolchain image / RTX 3060.
Exact image IDs and hardware/toolchain output are retained in E2E metadata.

- `shared-provider-final-focused.log`: final pin, eight boundary/sign/triangle/
  failure and independent inverse-residual tests pass on host Rust 1.99.1.
- `shared-provider-workspace-final.log`: all-features workspace, `test-fast`,
  locked/offline, no-fail-fast/retries=0: 1,455 pass, 73 documented opt-in skips.
  This includes faer + SIMD + BLAS features together.
- `shared-provider-doc-lint-threaded-final.log`: all-features doctests and
  all-target Clippy with `-D warnings` pass. The subsequent first threaded-test
  invocation fails its opt-in guard because `MVMC_RS_THREADED_182` was missing;
  it does not report a numerical regression. The corrected run is separate.
- `shared-provider-final-recheck.log`: corrected C-referenced threaded runner
  (`MVMC_RS_THREADED_182=1 MVMC_DIAG_605=1 TMPDIR=/tmp`, `--run-ignored only`),
  followed by the final all-features test-fast workspace and formatting checks.
- `cli-reference-opt-out.log`: `mvmc-cli --no-default-features` still builds.
- `cuda-selected-paths-gate.log`: final 132-test CUDA gate, including the actual
  two-device integration test and n=64/126/128/130/192 supplied-tape rebuilds.
  `cuda-selected-paths-build.log` records all-target Clippy and worker build.
- `cuda-cpu-default-isolation.log`: all CUDA targets compile with `--locked`;
  disabling core default features keeps its C-order oracle and original CUDA
  lock SHA-256 `bdbf40a4557b84fe402f16cd3d338a28b04c8481398f9ffcdd4dfaf3fee7f54c`.
- `provider-full-tests.log`: full faer fork tests/doctests after final padding;
  `provider-final-contracts.log` adds expanded/tighter public contracts and
  held-team 4/8/70-worker allocation checks for all four native dtypes.
  `provider-no-std.log` records its minimal no_std library build. Those provider
  commands use Rust 1.96.0. This is downstream candidate validation, not a
  provider-main performance campaign or registry release.

Earlier CUDA/source-stage logs are retained under their original names. They
must not substitute for the selected-source gate. In particular, split-entry
and noinline experiments were measured and reverted; the original overflow
negative control intentionally fails its contract against the old header.

Reproduce CPU commands in `docs/DEV_CONTAINER.md`'s environment with separate
named target/cache volumes, without running reference-submodule setup:

```sh
cargo nextest run --workspace --all-features --cargo-profile test-fast --locked --no-fail-fast --retries 0
cargo test --workspace --all-features --locked --doc --profile test-fast
cargo clippy --workspace --all-targets --all-features --locked --profile test-fast -- -D warnings
MVMC_RS_THREADED_182=1 MVMC_DIAG_605=1 TMPDIR=/tmp cargo nextest run \
  -p mvmc-core --test threaded_issue182 --locked --cargo-profile test-fast \
  --run-ignored only --no-fail-fast --retries 0 \
  -E 'test(runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal)'
cargo fmt --all --check
```

Normal Cargo tests neither read nor invoke the optional C toolbox. Its separate
native INFO probe and upstream defect report are documented in
`c_toolbox/pfaffian_block_info/README.md` and `docs/NUMERICAL_COMPARISONS.md`.
