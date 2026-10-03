# Optional parity gates (#183)

The four phase4/5 fixture gates, Lanczos PhysCal, MPI PhysCal and existing
ctest-equivalent gate are ignored opt-in tests. Default nextest runs report
them as skipped, not passed. Environment variables alone do not enable them:
explicit execution also requires `--run-ignored only`.

This focused milestone does not add the #180 thirteen-model implementation,
new #181 scenarios, threaded coverage or new runner APIs. It does not claim
that all #183 acceptance or numerical parity has been completed.

Normal Rust tests do not build or invoke C, Julia or toolbox programs. These
optional gates read existing reference fixtures and execute Rust. Initialize
the pinned reference submodules before fixture-dependent tests:

```sh
git submodule update --init --recursive
cargo nextest list --locked -p mvmc-core --cargo-profile test-fast \
  -E 'binary(/phase4_zvo_gate/) | binary(phase5_zvo_gate_hubbard) | binary(lanczos_transfer_physcal) | binary(mpi_physcal) | binary(ctest_equivalent)'
```

## Exact invocations

From the repository root:

```sh
MVMC_RS_PHASE4_REAL_ZVO=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate --run-ignored only --no-fail-fast --retries 0
MVMC_RS_PHASE4_CMP_ZVO=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_cmp --run-ignored only --no-fail-fast --retries 0
MVMC_RS_PHASE4_FSZ_ZVO=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_fsz --run-ignored only --no-fail-fast --retries 0
MVMC_RS_PHASE5_HUBBARD_ZVO=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase5_zvo_gate_hubbard --run-ignored only --no-fail-fast --retries 0
MVMC_RS_LANCZOS_PHYSICAL=1 MVMC_RS_LANCZOS_MODEL=hubbard_chain_real MVMC_RS_LANCZOS_MODE=real cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test lanczos_transfer_physcal --run-ignored only --no-fail-fast --retries 0
MVMC_RS_CTEST_MODELS=heisenberg_chain_real cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test ctest_equivalent --run-ignored only --no-fail-fast --retries 0
```

Set `JULIA_MVMC_ROOT` to use a different reference checkout. An explicitly
invalid root fails; it cannot silently fall back to another checkout.
Lanczos models are `hubbard_chain_real`, `hubbard_chain_lanczos` and
`spin_chain_lanczos`; omitted model requests all three. Modes are `real`
(default) and `cmp`. Unknown selections fail before sampling. Required input,
parameter, QQQQ and energy references must exist. Historical missing DC
references remain unverified DC coverage; optional DC comparisons are not
converted into a new numerical/input contract by this reporting change.

The existing ctest harness accepts a comma-separated model selector. Its
existing unsupported model declarations now fail rather than continue to a
passing summary. This milestone does not broaden model support or certify
the long statistical reference workloads.

MPI requires the `mpi` feature, recognized multi-rank launcher metadata and
an actual MPI world of at least two ranks. Fixture preflight precedes MPI
initialization. Missing/singleton launcher metadata fails Unsupported without
initializing MPI; this is a gate restriction, not a restriction on valid
singleton library use. Resolve exactly one
executable via Cargo JSON; never launch a wildcard including stale binaries
or dependency files. This example uses jq:

```sh
mpi_gate_binary=$(cargo test --locked -p mvmc-core --profile test-fast --features mpi --test mpi_physcal --no-run --message-format=json | jq -r 'select(.reason == "compiler-artifact" and .target.name == "mpi_physcal" and .executable != null) | .executable')
test -n "$mpi_gate_binary" && test -x "$mpi_gate_binary" &&
  MVMC_RS_MPI_PHYSICAL=1 mpiexec -n 2 "$mpi_gate_binary" \
  --ignored --exact mpi_physcal_reduces_fixed_parameter_samples --nocapture
```

The existing MPI numerical body is unchanged. A feature-disabled default run
reports the gate skipped. Explicit execution without its selector fails
`NotRun`; with its selector set it fails `Unsupported`:

```sh
env -u MVMC_RS_MPI_PHYSICAL cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
MVMC_RS_MPI_PHYSICAL=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
```

## Reporting contract

The test harness is authoritative: PASS means an executed test completed;
ignored/skipped means the work was not run. Do not count support unit tests
as numerical parity coverage.

| Case | Executed gate behavior |
| --- | --- |
| Absent or empty selector | report NotRun and fail |
| Selector `skip` (case-insensitive) | report ExplicitSkip and fail; omit the ignored test to skip |
| Required fixture absent | report MissingFixture and fail |
| Unknown/unsupported model, mode, feature or rank configuration | report Unsupported and fail |
| Runner, parsing or comparison error | harness failure; never successful early return |
| Required comparisons completed | harness PASS for only those declared comparisons |

There is no compatibility `select_gate` API. New gates must use
`support::require_gate` before fixture or numerical work. Pure helper tests
exercise absent/empty/skip/selected selectors, missing fixtures and unsupported
requests without spawning any reference runtime.

Historical fixture passes are not newly verified Julia 1.13 results or complete
Julia feature coverage. New reference generation is an explicit developer
workflow using `julia +1.13.1 --project=extern/Julia-mVMC` and
`Manifest-v1.13.toml`. Record OS/architecture, compiler/Rust versions,
features/profile, Julia/Manifest, actual BLAS provider/version, seed,
model/steps, ranks/groups/threads, source/reference hashes, command, exit
status and artifact paths. Do not imply MPI or threaded matrices ran from
a serial fixture pass.

## Milestone validation

Validated on Linux x86_64, Rust 1.99.0 (b940084d7), nextest 0.9.146,
locked dependencies and a separate worktree-local target directory, based on
`origin/main` `99e144b`. Pinned submodules were initialized at Julia-mVMC
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, PfaPack.jl
`0dcf52c15caec63516d0703f36bfc8a4bc0e58d0`, and C mVMC
`d73d06bd529d3b2573f38eb5817c4a5f52971006`.

| Check | Captured result |
| --- | --- |
| `cargo nextest run --workspace --locked --cargo-profile test-fast --no-fail-fast --retries 0` | 550 passed, 15 skipped; handle 33235, exit 0 |
| `cargo test --workspace --locked --doc --profile test-fast` | exit 0; no doctest cases declared; handle 35652 |
| `cargo clippy --workspace --locked --all-targets --profile test-fast -- -D warnings` | exit 0; handle 38248 |
| `cargo fmt --all --check`, `git diff --check` | exit 0 |

An existing third-party deprecated atomic-method warning was emitted; no
third-party code or compiler warning policy was changed by this milestone.
Requested numerical oracle/MPI workloads are not implied by reporting-only
negative checks.

Six new pure support test functions are reused by seven focused binaries
(42 helper test instances), plus one existing ctest threshold unit test.
Seven existing optional gate declarations are ignored in either MPI feature
configuration; the two cfg-specific MPI declarations contribute one test per build.

| Focused reporting case | Exact result |
| --- | --- |
| Default seven binaries, MPI feature disabled (32530) | exit 0: 43 passed, 7 skipped; no parity gate ran |
| `--run-ignored only`, all seven selectors unset (87047) | expected exit 100: 0 passed, 7 failed, 43 skipped; each gate reported NotRun |
| Explicit MPI selector, feature disabled (28889) | expected exit 100: 0 passed, 1 failed, 6 skipped; Unsupported |
| Required reference root absent; selectors set (41434) | expected exit 100: 0 passed, 7 failed, 43 skipped; six MissingFixture, feature-disabled MPI Unsupported |
| Existing unsupported ctest declaration, `MVMC_RS_CTEST_MODELS=kondo_chain_real` (34896) | expected exit 100: 0 passed, 1 failed, 7 skipped; Unsupported before numerical work |
| MPI feature enabled, default (62695) and unset-selector request (14060) | build exit 101: mpi-sys could not find mpicc, mpich.pc or ompi.pc; no tests ran |

The focused binaries are `phase4_zvo_gate`, `phase4_zvo_gate_cmp`,
`phase4_zvo_gate_fsz`, `phase5_zvo_gate_hubbard`,
`lanczos_transfer_physcal`, `mpi_physcal`, and `ctest_equivalent`.
The absent-selector run explicitly unset all seven variables listed above
with `env -u`. The missing-reference run set `JULIA_MVMC_ROOT` to the absent
`/tmp/mvmc-issue183-required-fixture-absent` and each selector to 1, except
`MVMC_RS_CTEST_MODELS=heisenberg_chain_real`. All negative cases use
`--run-ignored only --no-fail-fast --retries 0`; none reached a numerical body.
The unsupported ctest result concerns this unchanged harness declaration,
not a new claim that the production Hamiltonian is unsupported.

The initial host MPI builds remain unavailable: no MPI toolchain was installed.
Those build failures are not test passes or expected test-failure evidence.

### Container MPI-feature declaration checks

The existing container `232f26a94452` compiled and checked source commit
`3040c72573ea9a726310e84c20aacf6d7273a35b`, copied with `docker cp` into the fresh
`/tmp/mvmc-issue183-snapshot-2SyKg3/workspace`; no bind-mounted source changed.
The source archive SHA-256 is
`35aad874fafa486340341a012f344ebb8de4b19091fc4503c670dd9aceaa0385`.
The separately copied pinned Julia reference archive SHA-256 is
`008b0ceaa7731e9b723b2df260d9394c57c6fbe7048f1b185c1ac6ef667b695d`.
The tested `mpi_physcal.rs` SHA-256 is
`10820c80e0d46df89418b45334caf325cf11c8ce6f9a0ad2979cc6b989d44292`.

Commands used the existing Rust 1.99.0/MPICH 4.2.0 toolchain, explicit
`MPICC=/usr/bin/mpicc` and separate named-cache target
`CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue183-feature-check`.
Inherited kache and UV configuration were unchanged. In the snapshot directory:

```sh
export CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue183-feature-check
export MPICC=/usr/bin/mpicc
cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --features mpi --test mpi_physcal --no-fail-fast --retries 0
env -u MVMC_RS_MPI_PHYSICAL cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --features mpi --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
MVMC_RS_MPI_PHYSICAL=1 JULIA_MVMC_ROOT=/tmp/mvmc-issue183-snapshot-2SyKg3/missing-reference-root cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --features mpi --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
env -u MVMC_RS_MPI_RANK -u MVMC_RS_MPI_SIZE -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE -u PMIX_RANK -u PMIX_SIZE MVMC_RS_MPI_PHYSICAL=1 PMI_RANK=0 PMI_SIZE=1 JULIA_MVMC_ROOT=/tmp/mvmc-issue183-snapshot-2SyKg3/workspace/extern/Julia-mVMC cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --features mpi --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
```

| MPI-feature reporting case | Exact result |
| --- | --- |
| Default (handle 68349) | exit 0: 6 helpers passed, 1 gate skipped |
| Selector unset (run `0775946e-199f-49fa-9f6f-01cfd1190f8b`) | expected exit 100: 0 passed, 1 NotRun failure, 6 skipped |
| Selected, missing reference root (run `ae5b8fe0-209a-44c5-88a7-8950b802df9d`) | expected exit 100: 0 passed, 1 MissingFixture failure, 6 skipped |
| Selected, singleton launcher metadata (run `b8818849-6ac7-48a4-a2ca-66f60e167428`) | expected exit 100: 0 passed, 1 Unsupported failure, 6 skipped |

All three negative probes failed before MPI initialization. The gate also
checks the actual initialized `MpiContext.world_size()` and rejects values below
two even when launcher metadata claims multiple ranks; that post-initialization
branch was compiled, not runtime exercised. Singleton library use remains valid.
These are compile/feature/gate-metadata checks, **not MPI protocol or numerical
parity validation**. No MPI launcher, reference executable or numerical body ran.
Parent MPI/runtime validation remains separate.

After the MPI guard amendment, the default seven-binary reporting rerun
(handle 7165, source `3040c725`) again passed: 43 passed, 7 skipped, exit 0.
Container `cargo clippy --locked -p mvmc-core --profile test-fast --features mpi
--test mpi_physcal -- -D warnings` passed against that same copied source
(handle 41998, exit 0). Formatting and whitespace checks also passed.
The full-workspace results above precede this MPI-only guard amendment;
the amendment's feature-enabled branch was compiled and linted in the container.

Final milestone validation reran
`cargo nextest run --workspace --locked --cargo-profile test-fast --no-fail-fast --retries 0`
on exact HEAD `7c90becd09b1e4cd620ac5102d9cf651bb06ba92`, reusing the isolated
worktree target. Handle 18158, run `cc9e0718-ce92-40d9-b5a3-571c4582ed28`,
exited 0: **550 passed, 15 skipped** (105.193 seconds, three slow tests).
This final result includes the current gate implementation; the subsequent
validation-record commit changes documentation only. MPI protocol validation
is still not claimed.
