# Bounded thread plan and case-report infrastructure (#182 / #183)

Related to open #182, #183 and umbrella #185. This offline infrastructure does
not launch numerical gates, modify their selectors, or replace the existing
primary smoke wrapper/dispatch. No workflow changes or current numerical pass
are claimed. Python utilities are run with `uv run --no-project`.

Owned new files: `scripts/thread_gate_plan_183.py`,
`scripts/test_thread_gate_plan_183.py`, and this document. Existing drivers,
thread/MPI wrappers, Rust tests, fixtures and #184 ledgers are untouched.

## Source-bound inventory

The mapping is audited against committed `19a3dd29a8625e0585893609cf82a142ed3da40c`:
`crates/mvmc-core/tests/threaded_issue182.rs`, SHA-256
`41452d5fe1c6deb42c8868bda4b5df1f06c5570f7a15de2668100f269ee20024`.
Changed test source requires a new mapping review; changed production source
is recorded in the plan, not assumed to retain historical gate results.

| Explicit selection key | Actual test identity / mapping | Planned evidence |
| --- | --- | --- |
| `long20` | `runner_twenty_step_workers_repeat_supported_outcomes_and_physcal`; lines 197–210, 1444–1795 | 45 synthetic optimization cells (5 variants × sizes31/32/33 × direct/store0, direct/store1, CG/store0); 42 invariance + 3 expected rejections. Separate 15 synthetic PhysCal cases and 5 independent final-prefix cases. Workers1/2/4, twice each. Steps/window20; samples=size. |
| `failure` | `direct_sr_non_spd_boundary_repeats_workers_one_two_four`; lines 214–230, 1540–1710 | Hubbard real size32/direct/store0, original factor INFO2 at zero-based step10; original failed-update assertions and worker/repeat invariance. Not a successful solve. |
| `prefix` | `independent_runner_prefixes_match_full_normalized_pre_sr_arrays`; lines 234–241, 1902–2209 | Five fixture cases: Heisenberg real/cmp/FSZ, Hubbard real, GeneralRBM CG. Full normalized pre-SR OO/HO plus final parameter/energy/RNG/config comparisons, workers1/2/4. |
| `physcal` | `independent_physcal_workers_match_saved_rng_and_ordered_outputs`; lines 245–257, 2540–2638 | Four original two-frame fixtures, workers1/2/4 twice. Hubbard retains non-InterAll Lanczos2; other models mode0. Exact saved configurations/draw count/future RNG words, numerical ordered output comparisons. |
| `transfer` | `transfer_site_executes_actual_jobs_at_threshold`; lines 262–270, 1833–1900 | Synthetic Hubbard terms31/32/33; four samples, skip SR. Actual 4×size jobs and serial/parallel worker IDs plus invariance. No new public-input/reference oracle. |
| `cg20` | `reviewed_cg_twenty_step_workers_match_and_repeat`; lines 273–302, 2210–2387 | Canonical General CG/store0, steps/window20. Reviewed62b final energy/mapped parameters/config/future RNG words, full declared last window row, actual jobs; workers1/2/4 twice. Not fresh Julia execution. |
| `real-fsz` | `independent_real_fsz_workers_match_public_pre_sr_and_rng`; lines 305–329, 2704 onward | Adapted C-compatible real-FSZ direct/no-store fixture; seeded/initialized/pre-SR/final stages. Written-mask flags and OO/HO where asserted, configurations, RNG/parameters. One QP plane proves serial QP execution, NOT parallel QP activation. Workers1/2/4 twice. |

The three long20 rejection tuples are Hubbard real: size32/store0 step10 INFO2,
size32/store1 step10 INFO4, size33/store1 step16 INFO2. They are recorded separately
from successful optimization and independent numerical comparisons. The long
matrix also calls `independent_runner_prefixes(..., false)`: that embedded
comparison is not the dedicated pre-SR capture performed by `prefix`.

## Schema 1 and fail-closed reports

`plan --root CHECKOUT --select KEY... --output NEW_JSON` requires an explicit,
nonempty, duplicate-free selection. It records actual HEAD, current compile
source closure (including dirty/untracked compile inputs), fixture hashes,
seven exact identities, case IDs and settings. Output is exclusive-create.
Default primary smoke is not selected or changed by this utility.
The emitted exact `cargo nextest list` commands are inert plan data: even test
listing/building is not launched by this utility. Gate execution/records require
a separately approved producer and a stable source snapshot.

`report --plan PLAN --evidence EVIDENCE --output NEW_JSON` validates a retained
schema1 execution envelope. The envelope has matching `head`, `source`,
`fixtures`, a nonempty `binary` hash map, identical `source_after`,
`fixtures_after`, `binary_after`, an exact listed `selection`, `gates` and
`cases`. Each gate has an exact selected `identity`, status, terminal
`exit_status` and outer `invocations` (one when attempted, zero for preflight
failure or NotRun/ExplicitSkip). Pass requires one invocation and terminal0;
an attempted failure requires a nonzero terminal. Child worker invocations are NOT counted as
outer selected driver calls. Counts distinguish selected/observed identities,
outer invocations and reported completed cases.
The envelope also requires `runtime` with the actual Rust version, test-fast,
default features, ranks1 and BLAS version/library hashes plus a single-thread
runtime-query observation. Linkage-only or absent backend metadata cannot pass.
Reference runtime/version lineage remains the pinned fixture provenance, not a
claim that Julia/C ran during Rust tests.

Each case row references an immutable regular JSON artifact by relative path
and SHA-256. The artifact repeats the case/identity and execution bindings,
workers/repeat coverage, required assertion checks, and exact rejection tuple
where applicable. Plans require the entire exact audited Rust preflight file
inventory, including all named reference values/config/RNG/OO/HO/flag/output
files; ANY file under a reference root is insufficient. Every required namelist
binds all referenced definition/overlay files plus an implicit `initial.def`
when present. Prefix oracle `inputs.sha256` files must match the actual loaded
source paths and cover that entire closure; omission/change/duplicates fail.
The current `extern/Julia-mVMC/Manifest-v1.13.toml` is separately labelled
`reference_workspace` (current-workspace-only), never presumed to be a fixture's
historical generation Manifest.
This inventory is not a claim that all semantic input contracts were independently
checked or that any comparator executed.
The checks describe assertions actually present in the audited Rust tests,
not new algorithms, comparison tolerances or Rust-generated expected values.

Unselected or unattempted cases are NotRun. ExplicitSkip remains separate.
Selected MissingFixture/Unsupported/Failure cannot pass. A selected gate PASS
without every case artifact is Incomplete. Even complete hashed `checks:true`
artifacts are only **AttestedOnly**, `evidence_kind=reported_assertion`,
`verification=NotVerified`. All synthetic/claimed execution reports remain
overall Incomplete, with `verified_independent_comparisons=0` and
`approved_emitter=null`. Generic PASS, labels, forged verification flags and
comparison counts cannot establish actual case execution. Only explicitly
`reported_counts_by_kind` are provided; these are NOT executed numerical counts.
Invariance, activation, expected rejection and proposed independent-reference
assertion counts remain separate.
Incomplete/failed report CLI exits nonzero. Malformed/unknown/duplicate records,
missing artifacts, changed hashes/bindings/checks or missing workers fail closed.

Each case requires model, seed, effective steps, ranks/groups and worker settings.
These are still **attested settings**, not certified against all source defaults;
case metadata is never promoted to verified execution. Long20 effective steps
are constrained (20 optimization, 2 synthetic PhysCal), but other metadata needs
future recorder/input binding review.

Independent-reference assertions retain `historical-fixture` metadata: Julia
1.11 references remain 1.11; an unknown version may be null. Missing original
Manifest is explicitly `Unavailable` or `MissingEvidence` with a null hash.
An available Manifest has `manifest_role=historical-generation`; substituting
the current1.13 workspace hash for a 1.11 generation is rejected. No equality
between historical and current Manifest is generally required. Invariance,
activation and expected-rejection groups instead use reference `NotApplicable`:
they do not fabricate an independent Julia fixture oracle. Current workspace
metadata is separate, and `oracle_execution=none` is required throughout.
The current workspace version is parsed from its actual Manifest and must be
a nonempty version string; None/unknown/empty values fail. This requirement
does not apply to a historically unknown Julia version or generation Manifest.
Runtime Rust/profile/features and BLAS metadata are structurally checked but
not verified by this scaffold.

This is a **structural validator of attested assertion reports**, not an
independent recomputation or cryptographic proof that an assertion executed.
Its synthetic controls must never be counted as model/solver execution. The
existing child helper consumes D/N case records internally; no complete new
case-artifact producer is implemented here. Actual verification is deliberately
unavailable until a separately approved emitter source pin, run UUID, actual
nextest selection/terminal logs and gate-binary/per-case log artifact bindings
are implemented and reviewed. The report lists these as required future bindings,
not as already satisfied. Until then reports remain Incomplete, even when every
claimed assertion is true.
No historical report is silently rebound to current source.

## Infrastructure-only verification

```sh
uv run --no-project python scripts/test_thread_gate_plan_183.py
```

Controls cover explicit selection/counts, NotRun/ExplicitSkip/failure categories,
missing/duplicate cases, generic PASS promotion, source/fixture/head/binary
bindings, artifact tampering/absence/symlinks/escape, worker/repeat/check omissions,
reference-root requirements, and incorrect expected-rejection INFO. No Rust test,
C/Julia runtime, fixture generation, workflow dispatch or numerical gate is run.

Next prerequisite after #176/kernel contracts stabilize: separately reviewed
case-record producer retaining actual child assertions and execution bindings,
then selected source-specific independent prefix/PhysCal/real-FSZ runs, followed
by long20/CG/rejection runs. Existing #182/#183 remain open. Full #185 model,
scenario, MPI and public-API coverage is not implied by these seven selections.

## Actual infrastructure checkpoint

Initial 20 synthetic controls passed (0.172s, terminal0). The subsequent
all-seven metadata-only preflight, handle9542, terminated1 with
`MissingFixture: tests/fixtures/threaded_182_real_fsz`; its complete stderr and
terminal remain in `/tmp/mvmc-thread-plan-183-infra.uroF79/plan.stderr` and
`plan-terminal.txt`. No plan, numerical execution or PASS was emitted.
The current default fixture is absent from this checkout; the existing Rust
gate's external `MVMC_RS_THREADED_REAL_FSZ_ROOT` support is not automatically
imported or adopted by this bounded repository-root planner. An explicit
reviewed external-root binding adapter remains separate work.

The first all-case synthetic count control also caught a hand-count error
(15 independent groups instead of 5+5+4+1+4=19). Only that infrastructure
arithmetic expectation was corrected, not any numerical fixture/bound.
Across seven selected identities the planned 83 assertion groups are not 83
distinct models or successful solves: 19 independent-reference groups,
57 invariance groups, 4 expected-rejection groups (including the separate
focused replay of a long-matrix case), and 3 activation groups.

Final reference-role/closure controls: 29/29 passed, 0.929s, terminal0;
`/tmp/mvmc-thread-plan-183-infra.uroF79/reference-roles-tests.stderr` and
`reference-roles-terminal.txt`. These include current1.13-as-historical1.11
rejection, unknown historical Manifest retention, NotApplicable controls,
referenced definition/implicit overlay closure and declared oracle input hashes.
This is infrastructure-only proof, not 29 numerical scenarios. Earlier 20/21/24
checkpoints and the actual MissingFixture preflight failure remain historical
records, not rebound to this final source.
