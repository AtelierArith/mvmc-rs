# Optional parity gates (#183)

## Legacy smoke and 50-step follow-up

`run_smoke::heisenberg_chain_real_runs_one_sr_step` remains a normal mandatory
one-step Rust test. An absent reference checkout or required namelist fails
`MissingFixture`; it cannot report a pass without executing the runner.
Initialize the pinned submodules below before ordinary workspace testing.
Fixtures are read-only; no C, Julia or toolbox executable is invoked.

The four `phase5_regression_50step` comparisons are long historical fixture
gates. They are explicitly ignored by default, not removed or weakened:

```sh
MVMC_RS_PHASE5_50STEP=1 cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase5_regression_50step --run-ignored only --no-fail-fast --retries 0
env -u MVMC_RS_PHASE5_50STEP cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test phase5_regression_50step --run-ignored only --no-fail-fast --retries 0
```

The second command deliberately fails all four gates with `NotRun`. Empty
selectors also fail `NotRun`, while `skip` fails `ExplicitSkip`. Selected gates
require the checkout, model namelist and `reference/<model>/zvo_out_first50.dat`;
absence fails `MissingFixture` before sampling. Numerical bodies, seed 1,
50-step counts and existing tolerance budgets are unchanged. These historical
expectations are not newly generated Julia 1.13 or full C parity evidence.

Three new default reporting regressions use isolated subprocess environments
and fresh empty temporary directories, never deleting or replacing real
fixtures. They check the smoke's missing checkout/namelist, all four gates'
unset/empty/skip selectors and missing checkout/namelist, and missing expected
50-step output preflight. Other unit and integration tests remain normal.

### Shared integration compile checkpoint

After merging #197 with the shared numerical drafts, parent workspace process
67117 stopped during compilation, not numerical execution: the gate conflict
resolution omitted Lanczos `report_gate`/`GateStatus` imports and left unused
`require_gate` imports in four phase4/5 files. An import-only repair restored
the Lanczos imports and removed those four unused imports; no scenario body,
algorithm, tolerance or fixture changed. Focused nextest handle 20560 exited 0:
43 passed, 9 ignored (the shared #180 draft has two additional ignored audits).
Run ID: `76357a9c-ec00-43a3-a7f3-b93ef9d7fe59`. Targeted formatting passed.
Strict focused Clippy handle 14051 was blocked by the unrelated shared
`sampling/candidate.rs:400` eight-argument lint; it was not a gate lint pass.
This checkpoint is separate from the isolated milestone validation below and
does not establish shared full-workspace or MPI numerical validation.
Parent committed that five-file import repair as `077231b`; the parent's
separate full-workspace result is not inferred here.

### Follow-up final validation

Validated in a new isolated worktree from main `8ecea4f`, with its own
`target` directory and the same pinned reference submodules recorded below.
Linux x86_64, Rust 1.99.0, nextest 0.9.146, default features, locked dependencies,
`test-fast`. Final Rust implementation is commit `8936a45`; subsequent
validation-record edits change documentation only. No reference runtime ran.

| Command/check | Captured final result |
| --- | --- |
| `cargo nextest run --workspace --locked --cargo-profile test-fast --no-fail-fast --retries 0` | handle 83065, exit 0: 549 passed, 19 skipped, 112.848 seconds; run `49b930df-e200-4f97-bbe2-d9c288f4e92c` |
| `cargo test --workspace --locked --doc --profile test-fast` | handle 25779, exit 0, no doctest cases; no library/doc-test source subsequently changed |
| `cargo clippy --workspace --locked --all-targets --profile test-fast -- -D warnings` | refreshed handle 67691, exit 0 |
| `cargo fmt --all --check`, `git diff --check` | exit 0 |
| Selected 50-step command above | handle 4005, exit 0: 4 passed, 8 skipped; run `29b55453-18fc-4146-a2a8-8903acfe9800` |
| Unset-selector command above | handle 9144, expected exit 100: 0 passed, 4 NotRun failures, 8 skipped; run `3831d2a3-9f45-4e3e-b94c-c3233697a513` |
| Three repetitions of `cargo test --locked -p mvmc-core --profile test-fast --test run_smoke -- --test-threads=8` | handle 53712, exit 0: 8 passed each repetition |

The full default count changes from #197's 550 passed/15 skipped to 549
passed/19 skipped: four long comparisons become explicitly ignored and three
reporting regressions are added. All four long comparisons also passed when
explicitly requested. Negative subprocess checks cover two smoke fixture cases
and twenty long-gate selector/fixture cases; the missing expected-output
preflight regression adds one more negative case. They read no oracle output.
The smoke temporary-directory helper uses a per-call atomic sequence and
exclusive `create_dir`, retries collisions and never deletes existing paths;
the repeated parallel libtest runs verify it outside nextest's isolation.

Tested SHA-256:

- `run_smoke.rs`: `fd8d8ec5e5c3a7e18041b636b0cb37fea22b62ddac1638c48842a0ec6c6b9371`
- `phase5_regression_50step.rs`: `c79194afe61f32c7ccf53a225173a797a09aa06f15c90064f9f4d8a346deecd2`

The earlier full run (96225) passed 549/19 on `62c8768` before the temporary
directory race repair; it is superseded for final-source validation by 83065.
The selected/unset checks cover the unchanged phase5 source hash above.
An existing third-party deprecated atomic-method warning remains unchanged.
These results do not validate shared #179/#180/#181/#182 numerical drafts or
complete all #183 acceptance criteria.

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
MVMC_RS_CTEST_UPSTREAM_MODELS=heisenberg_chain_real cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test ctest_equivalent -E 'test(rust_ctest_upstream_rule_selected_models)' --run-ignored only --no-fail-fast --retries 0
MVMC_RS_CTEST_MODELS=heisenberg_chain_real cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test ctest_equivalent -E 'test(rust_ctest_equivalent_selected_models)' --run-ignored only --no-fail-fast --retries 0
```

`MVMC_RS_CTEST_UPSTREAM_MODELS` (names or `all`) selects the upstream-rule gate
(see [ISSUE180_UPSTREAM_CTEST_RULE.md](ISSUE180_UPSTREAM_CTEST_RULE.md)); it is the
verdict of the `ctest-long` dispatch. `MVMC_RS_CTEST_MODELS` selects the separate
20-step independent-reference gate, which stays MissingFixture until independent
step-20 references exist.

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

The shared MPI scenario body retains its grouped PhysCal rejection assertion;
the isolated milestone checks above do not validate that shared numerical body.
A feature-disabled default run
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

## Preserved shared draft checkpoint (historical evidence)

The following pre-merge draft is retained verbatim for provenance. Its helper
counts and implementation-scope statements describe the earlier checkpoint,
not the merged tree or the validated PR #197 milestone above. In particular,
current shared #180 declarations cover the full matrix; the shared MPI body
checks grouped PhysCal rejection, not grouped numerical parity. The six-helper
support contract and both MPI preflight guards from #197 are retained.

<details>
<summary>Earlier shared draft, including #181 invocation and process evidence</summary>

# Optional parity gates (#183)

Phase 4/5, Lanczos, issue #181 PhysCal and MPI gates are `#[ignore]` tests. A normal nextest run
reports them as skipped/ignored, never as passed. Setting an environment variable
alone does not enable an ignored test: select it with `--run-ignored only` too.
Normal tests do not launch C, Julia or toolbox programs. These optional tests
read existing reference files and run Rust; they do not generate references.

From the repository root, list the declared gates (including ignored tests):

```sh
cargo nextest list -p mvmc-core --cargo-profile test-fast \
  -E 'binary(/phase4_zvo_gate/) | binary(phase5_zvo_gate_hubbard) | binary(lanczos_transfer_physcal) | binary(physcal_issue181) | binary(mpi_physcal)'
```

Run an individual historical 10-step fixture comparison (seed 1):

```sh
MVMC_RS_PHASE4_REAL_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE4_CMP_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_cmp --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE4_FSZ_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_fsz --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE5_HUBBARD_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase5_zvo_gate_hubbard --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Set `JULIA_MVMC_ROOT` to a reference checkout when needed. An explicitly supplied
invalid root fails rather than falling back to another checkout.

Run all three issue #181 PhysCal scenarios (six-model fixed-parameter/output
matrix, deterministic multiple samples, and non-InterAll Lanczos file contracts):

```sh
MVMC_RS_PHYSCAL_181=1 cargo nextest run -p mvmc-core --locked --cargo-profile test-fast --test physcal_issue181 --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Run the Lanczos gate, optionally selecting one model:

```sh
MVMC_RS_LANCZOS_PHYSICAL=1 MVMC_RS_LANCZOS_MODEL=hubbard_chain_real MVMC_RS_LANCZOS_MODE=real \
  cargo nextest run -p mvmc-core --cargo-profile test-fast --test lanczos_transfer_physcal \
  --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Models are `hubbard_chain_real`, `hubbard_chain_lanczos`, `spin_chain_lanczos`;
omitting the model requests all three. Modes are `real` (default) and `cmp`.
Unknown model/mode selections fail. Inputs and the QQQQ/energy references for
the selected model must exist. The existing comparison checks observable
references that are present; historical fixtures omit the DC reference when
there are no DC terms. A pass does not establish DC observable comparison
coverage when that reference is absent. The issue #181 scenario separately
checks the empty-output contract.

MPI needs the `mpi` feature and at least two ranks. Build and resolve exactly
one executable using Cargo's JSON output; do not launch a wildcard that could
include stale binaries or `.d` files. This example requires `jq`:

```sh
mpi_gate_binary=$(cargo test -p mvmc-core --profile test-fast --features mpi --test mpi_physcal --no-run --message-format=json | jq -r 'select(.reason == "compiler-artifact" and .target.name == "mpi_physcal" and .executable != null) | .executable')
test -n "$mpi_gate_binary" && test -x "$mpi_gate_binary" && \
  MVMC_RS_MPI_PHYSICAL=1 mpiexec -n 2 "$mpi_gate_binary" \
  --ignored --exact mpi_physcal_reduces_fixed_parameter_samples --nocapture
```

The MPI gate checks completion of serial PhysCal reduction and rejection of unsupported
grouped PhysCal. It does not establish numerical parity for grouped PhysCal.
Explicitly selecting the MPI gate without the feature fails as `Unsupported`:

```sh
MVMC_RS_MPI_PHYSICAL=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
```

## Interpreting reports

The harness is authoritative: ignored/skipped means coverage was not run;
`PASS` requires the comparison to finish; assertion/runner errors are failures.
Selecting an ignored gate with its selector absent or empty emits `NotRun` and
fails. A selector of `skip` emits `ExplicitSkip` and fails if executed: to skip,
omit that ignored test from selection. Missing required inputs/expected outputs
fail (fixture preflight emits `MissingFixture`; file reads/runner errors may
report a normal harness failure); unsupported selections emit `Unsupported` and fail.
Thus an explicit request cannot succeed by returning without doing the work.
Keep nextest summaries, exit codes and captured output together. Do not count
support-helper unit tests as parity coverage.

`ctest_equivalent` is owned by #180 and its optional declarations use the same
`#[ignore]`/`support::require_gate` contract through that coordinated change.
This #183 change leaves that file untouched. The legacy `select_gate` helper
was removed after all callers migrated, so new gates must use `require_gate`.

Historical fixture passes cover only the named model, mode and outputs. They
are not newly verified Julia 1.13 results or complete Julia feature coverage.
For reference-generation evidence record OS/architecture, compiler and Rust
versions, Rust features/profile, Julia version and Manifest hash, actual BLAS
provider/version, seed, model/steps, ranks/groups/threads, reference revision and
file hashes, command, exit status and artifact paths. New Julia comparisons use
`julia +1.13.1 --project=extern/Julia-mVMC` and `Manifest-v1.13.toml`.
Reference generation is a separate optional developer action. Thread matrices
must be requested and reported explicitly; these gates do not imply they ran.

## Reporting validation checkpoint (2026-10-03)

The following captured processes were started before integration of #190.
Treat their results as gate selection/reporting evidence; they do not
establish numerical parity or validate the integrated numerical implementation:

| Process handle | Selection | Result |
| --- | --- | --- |
| 15826 | Default phase4/5, Lanczos and MPI declarations, `test-fast` | Exit 0: 24 helper tests passed, 6 parity gates skipped |
| 34718 | Default `physcal_issue181`, `--locked`, `test-fast` | Exit 0: 4 helper tests passed, 3 scenarios skipped |
| 28608 | All nine ignored gates explicitly requested with selectors unset, `--locked`, `test-fast` | Expected exit 100: 0 passed, 9 failed at `require_gate`, 28 helper tests skipped |

The missing-selector run used `--run-ignored only --no-fail-fast --retries 0`
and explicitly unset `MVMC_RS_PHASE4_REAL_ZVO`, `MVMC_RS_PHASE4_CMP_ZVO`,
`MVMC_RS_PHASE4_FSZ_ZVO`, `MVMC_RS_PHASE5_HUBBARD_ZVO`,
`MVMC_RS_LANCZOS_PHYSICAL`, `MVMC_RS_MPI_PHYSICAL`, and
`MVMC_RS_PHYSCAL_181`. Every executed gate reported `NotRun` and failed before
entering its scenario body. No C/Julia program, reference generation, MPI
launcher, or numerical parity workload was run by these checks.

After resuming on the integrated tree, focused support-helper verification
(handle 54822) exited 0: 4 helper tests passed, 1 parity gate skipped:

```sh
cargo nextest run -p mvmc-core --locked --cargo-profile test-fast --test phase4_zvo_gate -E 'test(support::tests::)' --no-fail-fast --retries 0
```

Targeted `rustfmt --edition 2021 --check` on the owned Rust files and
`git diff --check` on the owned paths also passed. These checks cover gate
infrastructure and formatting, not numerical comparisons.

</details>
