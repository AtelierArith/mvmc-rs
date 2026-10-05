# Explicit bounded optional-gates CI (#183)

`.github/workflows/optional-gates.yml` runs ONLY via workflow_dispatch; no PR,
push, or schedule trigger. Existing ordinary CI is unchanged. Do not dispatch
before parent review. `family` selects general/lanczos/mpi/thread, or explicitly
all four bounded jobs. A separate planning job creates only the selected matrix
entries; unselected numerical jobs are not instantiated or reported green.

| Job | Actual bounded selection | Not claimed |
| --- | --- | --- |
| general | Two exact ignored tests; General1/2/3/20 and public20 repeat | All13 models or fresh whole-matrix Julia validation |
| lanczos | One ignored test six times, explicit three models x real/cmp; eight DC references + four empty GEx contracts | Empty contracts are not numerical comparisons; InterAll, full Lanczos coverage |
| mpi | One exact test executable, actual worlds2/4, groups1/2, fixed Heisenberg real PhysCal and invalid-NLanczosMode rejection | Full MPI model/solver matrix or independent fullC sampling |
| thread | Existing wrapper selects ONE primary runner test; workers1/2/4, steps2/samples200 | New long20/45-case outcome matrix or all threaded gates |

Lanczos models are `hubbard_chain_real`, `hubbard_chain_lanczos`,
`spin_chain_lanczos`. Their actual namelist closure is checked before execution;
InterAll is explicitly rejected, even if later introduced into a named model.
Historical numerical budgets are unchanged. Reference versions belong to
fixture provenance, not a newly executed Julia runtime: these jobs invoke no
C, Julia or toolbox programs. Offline Cargo remains independent of this script.

The script checks exact ignored identities/counts, never support-only passes.
The DC follow-up requires exact selected-case markers: the six Lanczos calls
must report eight independent `REFERENCE_COMPARED` records and four
`EMPTY_CONTRACT` records, saved separately in `dc-comparisons.json`. The latter
are NOT numerical comparisons: both parsed GEx arrays must be empty, the missing
expectation must be specifically `zvo_ls_cisajscktaltex_001.dat`, and its actual
output must exist as a regular file containing exactly one LF (`b"\n"`).
Zero bytes, spaces, CRLF, multiple LF, nonempty values, missing files and symlinks
are rejected. Other missing expectations fail. C provenance and the optional
whole-function probe are in `c_toolbox/issue183_ls_empty/README.md`; Cargo never
reads or invokes that probe.
This follow-up's six-call validation completed separately (see the final DC
evidence section). The earlier terminal evidence below belongs to the earlier
committed driver and is not retroactively reassigned.
The first DC follow-up attempt, handle79244, is retained at
`/tmp/mvmc-183-dc-frozen.MEObep/lanczos-evidence`: terminal exit1, SHA256
`e21576f08b97cc92c7c0f083d57fb8477bb198e715362604f6949e1d7db76fc8`.
Hubbard real real/cmp completed, but Hubbard Lanczos real failed the erroneous
zero-byte assumption (run `2d16f9fd-b3e3-458d-b5c3-52ed716331f5`). Its actual
GEx file was one LF, which the original C writer requires; this was a test
contract defect, not a production numerical defect. Remaining three selections
were not run. The original source/artifacts remain unchanged. A separate driver
fix reads both stdout and the same selected call's stderr, because nextest puts
captured test output on stderr. Neither correction retroactively converts that
terminal into a pass.

## Final strict C-empty DC follow-up evidence

Handle22750 completed terminal0 on Linux x86_64. Dedicated immutable checkout
`/tmp/mvmc-183-dc-final.vxNlfJ/workspace` used committed production
`c8db43bf461fe6fb426b5fee117319849c1b9c21` plus ONLY the six reviewed paths
(Lanczos test, driver, infrastructure tests, this document, toolbox probe and
README). Its own target is `/tmp/mvmc-183-dc-final.vxNlfJ/target`. The executed
document was the pre-result frozen version; this result section was added only
after terminal0. Reference checkout identity is historical
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, not a fresh Julia execution.

Command from that checkout:

```sh
CARGO_TARGET_DIR=/tmp/mvmc-183-dc-final.vxNlfJ/target OPENBLAS_CORETYPE=HASWELL \
  uv run --no-project python scripts/run_optional_gates_183.py lanczos \
  /tmp/mvmc-183-dc-final.vxNlfJ/lanczos-evidence
```

| Selected model/mode | Actual nextest run | Result |
|---|---|---|
| Hubbard real / real | a180acc5-35ed-4569-98e2-b441278b370d | 1 PASS, 0.109s |
| Hubbard real / cmp | 304c8765-367a-4dba-9241-de66434d3401 | 1 PASS, 0.108s |
| Hubbard Lanczos / real | 3c519fd1-e940-4596-b743-66027c54b319 | 1 PASS, 3.640s |
| Hubbard Lanczos / cmp | 6efdf45b-9242-44ed-80ae-1de0cdf5bfa1 | 1 PASS, 3.591s |
| Spin Lanczos / real | 6966b356-835a-410a-88e9-c23f9b782db8 | 1 PASS, 55.558s |
| Spin Lanczos / cmp | b9f31cd7-8abd-4051-89a7-f3d30540b6f5 | 1 PASS, 51.736s |

Each call selected exactly one ignored numerical gate, with eight other tests
unselected. Twelve distinct DC case/file records were verified: **eight
REFERENCE_COMPARED and four EMPTY_CONTRACT**, not twelve numerical comparisons.
The four empty GEx cases verified the original C's exact one-LF contract.
Independent existing numerical expectations and tolerances were unchanged.

Evidence root: `/tmp/mvmc-183-dc-final.vxNlfJ/lanczos-evidence`.
All35 entries in `artifacts.json` were independently rehashed and matched.

| Artifact | SHA256 |
|---|---|
| terminal.json (exit0, six completed cases) | 29a66818dcdc58907e95109dced643ff89a9b53b801126e4c5f24a72c0d37214 |
| dc-comparisons.json | 99ae693730d185e665c788a9a97e35718411203b3b88213314c87e2e4105c3d1 |
| source.before.json = source.after.json | 1b62cafd5335bb35649e1cf4a0cb535e83df46127c02d0e09cbbc889f7821f23 |
| fixtures.before.json = fixtures.after.json | a1d9daed0dbfc389299860efc634893cf0453f86ca5c8b2499dbb2b9132ee2d5 |
| backend.json | cb86986d76af09aa543db1fefb4ba83e844686e056ab2b4343f7d59ff1439ae1 |

Executed binary `target/test-fast/deps/lanczos_transfer_physcal-02b7565035f61461`
SHA256 `3ba873197c8cf2775a47072257acbd014ab1fffc670defaa7e35b6b049dd6e00`
matched before/after. Actual linked OpenBLAS0.3.26 Haswell runtime threads1,
library SHA256 `bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e`.
Rust1.99, default features, locked test-fast. Infrastructure10PASS and focused
Rust negative2PASS are separate from numerical coverage. No external dispatch,
full13 matrix, fullJulia features, native macOS or InterAll proof is claimed;
issue183 remains open for its remaining reporting/backend-negative audit.

## Driver-only reporting follow-up (not workflow aggregation)

The reporting follow-up adds `family-ledger.json` for **one driver invocation**.
All four bounded families are listed. Unselected families are `NotRun`; an
explicit `--exclude mpi` records `ExplicitSkip` without executing that family.
Duplicate/unknown exclusions and excluding the selected family are rejected.
Selected work starts as `NotRun` (unfinished, not PASS), then finishes as `Pass`,
`MissingFixture`, `Unsupported` or `Failure`. Missing required independent inputs
and explicit selected-gate MissingFixture diagnostics are classified separately
from unsupported inputs/platforms. Rust diagnostic classification requires an
actual selected invocation, an exact nextest `FAIL [duration] (index/count)`
record with the selected crate/test binary and identity, and the matching gate's
line-anchored status marker. The retained real nextest layout is covered by a
regression without depending on its captured inner libtest output. Direct MPI
libtest launches instead require the exact selected `test identity ... FAILED`
record. Build/version/listing failures,
unrelated helper markers or compile-error words cannot acquire a fixture or
unsupported classification through text matching. Execution errors, changed hashes, unfinished
identities and absent/empty completion artifacts are Failure, never PASS.
Preflight-invalid selectors are rejected before creating/reusing evidence roots;
this is a failed request, not a persisted coverage result.

`started_driver_invocations` increments immediately before launching a selected
driver call, including a call that then fails: General1, Lanczos6, MPI2, or
Thread1 for complete runs. Thread is explicitly **outer-wrapper-only**; inner
test/worker executions are not counted. Builds, listings, version queries and
helpers are excluded. `completed_selection_identities` separately lists only
successful verified selections (General two test identities, Lanczos six
model/mode identities, MPI world2/world4, Thread one primary identity).
`selected_test_identities` lists distinct identities actually accepted by
selection preflight (General2, each other family1), not executed model counts.
These fields are not numerical coverage. `helper_tests` is zero for this driver's exact ignored selection;
the separately run infrastructure suite is not included. For a completed
Lanczos family, `numeric_reference_comparisons` records eight **DC** reference
comparisons and `empty_contracts` records four exact C one-LF contracts. These
fields do not count all QQ/energy elements or other families' numerical
assertions. Unknown/uninstrumented numerical comparison counts are JSON null,
not invented zero or inferred PASS counts. Incomplete runs do not claim the
completed-family DC totals. Required artifacts are checked and rehashed before
publishing a successful final ledger; artifact validation failure downgrades
terminal and selected status to Failure.
On artifact/hash failure, the published numerical count becomes null and the
empty-contract count resets to zero with `comparison_evidence=Unverified`.
Started-invocation and completed-selection facts remain available; reset counts
do not claim that no comparisons ran, only that their evidence is not validated.
`comparison_evidence=Verified` is published for the completed, artifact-validated
Lanczos eight-DC/four-empty totals; uninstrumented families remain Unverified.

Backend negative tests use mock libraries/API results, not new numerical gates:
zero/multiple OpenBLAS paths, loader failure, each missing runtime API, null or
invalid config/core strings and non-single-thread runtime reports are rejected.
Runtime config must contain an actual OpenBLAS semantic version, not just the
provider name. Reporting tests cover nonrunning failure terminals, incomplete
or duplicate identities, tampered/missing/out-of-root hash artifacts, and a
simulated completed run downgraded by artifact failure. Simulations are NOT
executed numerical/reference evidence.

The actual handle22750 proof above predates this reporting change and remains
historical source-specific evidence. No numerical gate was rerun for this
driver follow-up; ordinary Cargo behavior is unchanged. The separate workflow
aggregation follow-up below implements bounded four-family reporting, not a
ledger of every optional test identity in the workspace.

## Workflow aggregation: implemented, external dispatch not executed

The dispatch-only workflow now saves a four-family plan and instantiates only
its selected jobs. Unselected families remain `NotRun`, not PASS or ExplicitSkip;
the current dispatch input has no explicit exclusion control. Each job seals a
separate package without modifying original driver evidence. Plan, envelope and
aggregation bind the full head SHA, run ID, attempt, workflow identity and family;
seal independently checks actual `git rev-parse HEAD` and records that checkout.
Bounded execution uses `CARGO_TERM_COLOR=never`, recorded in the envelope, so
strict nextest failure matching is not defeated by ANSI color.

An `always()` aggregation job validates selected package names, envelope/driver
hashes, exact selection identities, source/fixture/binary before-after closure,
runtime backend and the separate eight DC references/four C empty contracts.
Missing evidence (including prebuild failure), cancellation, wrong run/head/
attempt, duplicates, tampering or unsafe paths cannot produce PASS. A verified
failed-driver classification remains MissingFixture/Unsupported/Failure; missing
or unverifiable selected-job evidence is Failure. Numerical/empty/helper counts
remain separate. Ordinary CI and Cargo acquire no oracle/toolbox dependency.

Validation is **infrastructure only**: ten synthetic aggregation tests, eighteen
existing driver tests, actionlint and YAML checks passed. A separate read-only
regression rehashed actual historical handle22750 artifacts and checked their
real driver schema. That producer predates the new ledger: no current run/
attempt binding or ledger was invented, and no numerical gate was rerun.
Commands, actual outputs, terminal codes and source hashes are retained in
`/tmp/mvmc-183-aggregation-headproof.Q5mNX2/*.json`.

```sh
uv run --no-project python scripts/test_optional_gate_aggregation_183.py
uv run --no-project python scripts/test_optional_gates_183.py
uv run --no-project python scripts/test_optional_gate_aggregation_183.py \
  --historical-evidence /tmp/mvmc-183-dc-final.vxNlfJ/lanczos-evidence
```

No external workflow dispatch has been executed. This is not current remote
four-family numerical proof, native macOS proof, full model/Julia coverage or
issue183 completion; remaining acceptance must be audited separately.
Unknown/empty family, missing inputs, source/fixture changes, missing/empty
required artifacts and nonzero gate exits fail. Artifact roots are exclusively
new; output roots are never deleted/reused. MPI additionally receives its
required exclusive `MPI179_PHYSCAL_OUTPUT` path per actual world. Each requested
job captures commands/env, Rust/profile/features, revision/source and fixture
hashes, reference submodule revisions, actual binary hash/linkage and terminal
completion count. Actual linked OpenBLAS is loaded read-only through its
configuration API to record version/core/thread count and library SHA256;
pkg-config or environment alone is not backend-version proof. Ambiguous or
unidentifiable backend fails instead of guessing. No Julia BLAS runtime is
claimed. Artifact upload runs on failure too and missing artifacts fail.
Compilation closure includes `tests/support`, `.cargo` settings when present,
Cargo manifests/lock/toolchain and source; actual compiler/cache flags are
recorded. MPI requires exactly one PASS/zero failures/zero ignored summary per
rank and unique actual world/rank/group-width markers. The thread backend is
queried from the exact executed wrapper archive after exclusive extraction;
its binary hash must match the initial source-associated build, and the nested
wrapper terminal/fixture/source closure must succeed. No current backend is
inferred from an unrelated initial executable.

Local explicit Linux command (requires linked OpenBLAS, nextest; MPI additionally
mpiexec/mpi development libraries):

```sh
CARGO_TARGET_DIR=/absolute/checkout-specific/target \
  uv run --no-project python scripts/run_optional_gates_183.py general /tmp/exclusive-new-evidence
uv run --no-project python scripts/test_optional_gates_183.py
```

Infrastructure tests are not model coverage. Any numerical result must identify
the captured terminal artifact, scope and source snapshot. No external dispatch
or full13/Julia/MPI/thread completion is inferred from adding this workflow.

## Executed bounded validation (2026-10-03)

Final driver SHA256:
`666162c4727444864aee5dc3958cdf0b40f6cdf511df64953fde249520a5a6db`.
Workflow SHA256:
`3b21c01ddf3b4abaf2dadf8584b328832f9d26dc7e341920edee7b80195cf427`.
Infrastructure test SHA256:
`01c7e77ac756ffc92fc621d1cf4bc59361111c2130d0aeb2aaf0c243d36dda95`.
MPI diagnostic-only test SHA256:
`930e54ff1d4194af2a0b41610426bc7da3df3c2c663efc596e9042adbcc8f43e`.
No production kernel, algorithm, tolerance or reference fixture was modified.

General/thread/Lanczos executed in dedicated detached worktree
`/tmp/mvmc-optional183-frozen.e8eQuG/workspace`, base
`f3716ac004b7d82123a810334556c40c51bd69fc`, overlaid with the existing dirty
compile/test sources and these owned implementation files. This is NOT a
committed-main or full-workspace validation. It used its own target
`/tmp/mvmc-optional183-frozen.e8eQuG/target`. Frozen source manifest before/after
SHA256 is `6a06466d8a53388ed8aa81674d119d785b54537c6388896eac4062cfb6281e1b`.
Each family's fixture closure before/after matched. Linux x86_64, Rust1.99.0,
nextest0.9.146, locked/test-fast/default features, actual linked OpenBLAS0.3.26,
Haswell, one backend thread. The CLI form was:

```sh
CARGO_TARGET_DIR=/tmp/mvmc-optional183-frozen.e8eQuG/target \
  uv run --no-project python scripts/run_optional_gates_183.py FAMILY \
  /tmp/mvmc-optional183-frozen.e8eQuG/FAMILY-evidence
```

| Final handle | Actual completed scope | Retained terminal artifact |
| --- | --- | --- |
| 58512, exit0 | General two gates PASS, 3.203s; four independent prefixes plus public20 repeat; run `59db7b9a-80a2-4d7a-8104-1af311d7dde9` | `general-evidence/terminal.json`, SHA `ccb85285b58332b574ba491ec0b3c2c6f7db85bde5b85d045cf5fc363c4a7f5e` |
| 23549, exit0 | Six explicit Lanczos model/mode calls, each one ignored gate PASS | `lanczos-evidence/terminal.json`, SHA `acfc0c3bb72ea452e7c879313eac7da8eeac080855f417b57d2c8a5a08c7f2aa` |
| 36608, exit0 | ONE primary threaded gate PASS, 154.289s; workers1/2/4 actual activation, 1859 compared records; run `00564dd0-7d81-4b6f-9ed9-b5b047b009c9` | `thread-evidence/terminal.json`, SHA `7744e8c09f80de5f7d332e6b35dfaf5ad38010433b9080a8ab791c11c716415f` |

The thread wrapper executed archive SHA256
`e877ec3f06b9bf87aab1e6b5f35b71443fe2c9492a56a1ed6d1a799f2d294270`;
the exclusively extracted actual executable SHA256 is
`7215d2682cda0d2eb34f0e109b7c004695841671683a564741c4ed482b8b7ab7`.
`thread-evidence/thread-executed-archive.json` records the exact archive,
executed binary identity matching the initial build, and nested wrapper path.
Its backend API query was made against this extracted executable's linked
library; no unrelated initial-binary backend is substituted.

Final MPI handle8688 exited0 in container `73c57e563c61`, Rust1.99.0,
MPICH4.2.0, same actual OpenBLAS0.3.26/single thread, features mpi/test-fast.
Command used `CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue183-owner-marker`
and `MPICC=/usr/bin/mpicc`, then `uv run --no-project python
scripts/run_optional_gates_183.py mpi /tmp/mvmc-optional183-mpi-owner-v2-20261003`.
Exact host-persisted evidence root:
`/tmp/mvmc-optional183-mpi-owner-v2-persisted-20261003`.

* `terminal.json`: `476992bcfa51c1f2f66bdf0d49391bcbbdaaaf523061fa3febb33e15b577d144`.
* `mpi-2.stdout`: `ab0ed7a3d09de411c0206acd61006c1769ad22eca0a1fb27a4add82c5791896b`.
* `mpi-4.stdout`: `298a6b199b5c60ec35ece845f90a36ad1d7f51b73042caa297ea4fad1651c42c`.
* Source before/after both `de3fa2f89a2696a58f6e90ddfda6b55cb362e30d1f104722e58d10c505874257`;
  fixture before/after both `c7a3da61d3492636edb2e7155c290121823122066a15bffc19374e8e4e1b9d29`.

Both actual worlds2/4 supplied all zero-based ranks and width1/2 markers, with
actual group size equal to world for width1 and two for width2. Exactly two/four
per-rank libtest summaries reported one PASS, zero failures, zero ignored.
The new eight-line marker observes communicator sizes only after existing
repeat/output assertions; old MPI proofs are not reassigned to this new SHA.

Infrastructure validation: `uv run --no-project python
scripts/test_optional_gates_183.py` passed9 tests (not model execution); parent
independent9 also passed. PyYAML parse/dispatch-only dynamic-matrix schema
checks passed; SHA-verified actionlint1.7.7 passed this workflow (shellcheck
integration disabled, not claimed). Focused MPI Clippy handle67679 exited0;
targeted rustfmt/diff checks passed. No external GitHub dispatch, native macOS,
full13/long45 or fresh C/Julia runtime validation was performed here.

Failures preserved separately: handle93329 failed backend preflight before
numerical work because the driver's observer originally inherited a different
thread environment; fixed by matching its library initialization environment.
The first thread attempt failed preflight on a valid commented namelist; comment
handling and its regression were added. Historical thread handle10682's numeric
gate passed166.010s but its wrapper correctly exited1 with SourceChanged:
an unrelated `constructor_contracts.rs` changed during the shared run.
All evidence remains under `/tmp/mvmc-optional183-thread-owner-v2-20261003`;
it is an infrastructure failure, NOT the final immutable-worktree PASS.
Historical Lanczos59780 and General77178 precede final driver666...;
the final table above supersedes them for this implementation only.

## Full explicit MPI gate set (#179)

A separate dispatch input, `explicit_mpi_gates = true` (use `family = none` to run only
it), starts the `mpi-explicit` job. It runs every explicit MPI gate (the ignored tests of
the `mpi` feature) at 2 and 4 ranks through `scripts/run_explicit_mpi_gates.sh`, including
the rank-wise native-C matrix (`docs/reference/c-to-julia/verification/issue-179-mpi-matrix.md`).
Like the long ctest gate it sits outside the bounded-family ledger: not selected is NotRun,
any failing cell or fewer cells than the enforced minimum fails the job, Julia comparisons are
reported Unverified, and the uploaded artifact carries `provenance.txt`, `cells.txt`,
`summary.md` and all cell logs.
