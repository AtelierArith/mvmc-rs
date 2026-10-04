# Joined evidence for #184 / #185

Current source checkpoint: main `4d39231688b55b22ebbd1c23cac2e4bc0884080b`.
The fab2335 / PR271 reconciliation below is historical; its57 controls and114
execution records remain unchanged. PR273 adds five reader-policy controls and10 records.
The prior `eaa5db6254c721b95e8e63096287cdb74d5124e2` checkpoint and its exact
execution revisions below remain historical, not relabelled as fab2335 runs.
Related to #184/#185; neither umbrella issue is complete.

The [machine-readable matrix](issue-184-joined-evidence.json) preserves all
**2,342 rows**:300 APIs,456 scenarios,1,444 assertions and142 source/document
rows. `original_row` preserves every original field; original owner identifiers
are unchanged. Current coordination is #184/#185 Pauli, not a silent assignment
of all implementation obligations to one agent. Historical source mapping is not
new execution proof. Reconciled top-level fields supersede old placeholders only
for explicitly reviewed conditions.

| Status | Rows | Interpretation |
| --- | ---: | --- |
| CoveredScopedOrdinaryCI | 17 | Named original assertion conditions, not complete API/scenario verification |
| IntentionalDifference | 1 | M0971: zero-QP helper retains zero instead of Julia's one |
| SourceMappedMissingEvidence | 1,617 | Conditions/execution reconciliation incomplete |
| HistoricalExecutionClaimNeedsReconciliation | 80 | Historical claims retained, not newly certified |
| ExplicitScopeExclusionNeedsReview | 627 | Review-required classifications, not automatic exclusions or PASS |

The [38 semantic source reviews](issue-184-semantic38.tsv) remain source-only
conditions and explicit gaps, not file-wide acceptance. Full API overloads,
examples, unit/integration/MPI scenarios and per-cell settings/evidence remain
required. InterAll numerical verification is excluded; generic filename parsing
does not establish its kernels. Current numerical long baseline is20 steps;
historical50-step records are not relabelled.

## Actual execution and durable identities

Recorded ordinary CI command:

```sh
cargo nextest run --workspace "${feature_args[@]}" --locked --cargo-profile ci --no-fail-fast --retries 0
```

Exact-head workflow/job logs define default Linux serial versus MPI/SIMD feature
arguments. Ignored gates are NotRun, not inferred executed from summary counts.
Each covered assertion carries its Rust test identity, original input/literal
settings, Julia revision, tested Rust head, owner issue, command and scoped result.

| Milestone | Durable evidence | Scope |
| --- | --- | --- |
| [PR244](https://github.com/AtelierArith/mvmc-rs/pull/244) | [15 control identities/ordinals](issue-184-pr244-controls.tsv) | Reporting/path/hash controls and real offline bundle verification; contextual S289–S293 only, not original model assertions. Tested `ea0c0a95276ceaa0d09b9209536788f06f04919b`, merged0130fdf; serial968/MPI970 PASS,40/58 skipped. |
| [PR247 run37165386569](https://github.com/AtelierArith/mvmc-rs/actions/runs/37165386569) | [6 identities](issue-184-pr247-controls.tsv) | M0437/M0438 literal conditions PASS; M0436 different-default evidence remains pending. Jobs111327112883/111327112877:971/973 PASS,40/58 skipped. |
| [PR250 run37165949026](https://github.com/AtelierArith/mvmc-rs/actions/runs/37165949026) | [13 identities](issue-184-pr250-controls.tsv) | M0426–M0429 strict complex literals PASS; typed Result/default utility, not C reader grammar. Jobs111328734625/111328734789:979/981 PASS,40/58 skipped. |
| [PR253 run37166619372](https://github.com/AtelierArith/mvmc-rs/actions/runs/37166619372) | Matrix M0971–M0980 exact identities, inputs, lengths and ordinals | M0972–M0980 scoped PASS; M0971 intentional difference. Other S188 conditions remain pending. Jobs111330733350/111330733376:984/986 PASS,40/58 skipped. |
| [PR254 run37167056269](https://github.com/AtelierArith/mvmc-rs/actions/runs/37167056269) | [Separate initialization receipt](issue-184-pr254-initialization.json) | 13 inputs/parameter/QP/raw-state/count initialization contracts, six checkpoints; Linux980/982 PASS. Workspace-query checks RNG identity only. No sampling/SR/output20-step promotion. |

Original Julia revision is `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`.
Earlier execution heads remain unchanged after subsequent merges. This
reconciliation ran no Julia, C oracle, Cargo or model tests.

## Historical parser condition reconciliation at fab2335

The [57 named controls](issue-184-parser-milestone-controls.tsv) bind complete
reviewed Rust test bodies to actual PASS records in both Linux jobs:114 execution
records, not114 independent scenarios. Their conditions were reviewed against
the original Julia8bb methods and C reader/caller lifecycle. File/function names
and CI ordinals are checked mechanically; they are not a semantic token matcher.
The table retains tested head, test-source SHA256 and line, literal settings,
command, job/log SHA256, durable links, owner issue and explicit remaining scope.

| PR | Scope | Linux default / all-features totals | Run |
| --- | --- | --- | --- |
| 258 | 9 Transfer controls: complete complex/spin coordinates, hex conversion, imaginary carry, required file/count errors and separate headerless helper | 994/996 PASS;40/58 skipped | [37168415174](https://github.com/AtelierArith/mvmc-rs/actions/runs/37168415174) |
| 261 | 9 Coulomb controls: intra/inter literals, hex/carry, parser-only nonfinite values, required-file/count/safety boundaries | 1005/1007 PASS;40/58 skipped | [37169038401](https://github.com/AtelierArith/mvmc-rs/actions/runs/37169038401) |
| 263 | 9 Hund/Exchange controls: native diagonal/order/sign retention, scalar carry and separate convenience helper | 1014/1016 PASS;40/58 skipped | [37169525530](https://github.com/AtelierArith/mvmc-rs/actions/runs/37169525530) |
| 265 | 9 PairHop definition controls, one archived count1/seven-row public-loader rejection, one core rejection-path smoke | 1024/1026 PASS;40/58 skipped | [37171100066](https://github.com/AtelierArith/mvmc-rs/actions/runs/37171100066) |
| 268 | 9 LocSpin controls: complete indexed table, informational count, negative/nonbinary values and defined integer carry | 1033/1035 PASS;40/58 skipped | [37171828186](https://github.com/AtelierArith/mvmc-rs/actions/runs/37171828186) |
| 270 | 10 indexed TransSym/QPTrans controls: complete weight/map/sign fields, AP accessor, aliases and safety boundaries; short-zero discrepancy below | 1043/1045 PASS;40/58 skipped | [37172837109](https://github.com/AtelierArith/mvmc-rs/actions/runs/37172837109) |

These controls add partial condition or architecture-context links to19 existing
rows; four additional
generic QPTrans API/type rows have their incorrect representation mapping
corrected to explicit missing counterparts. No whole original API/scenario or
assertion row is promoted solely from these new tests. The status counts above
historically remained15 scoped covered assertions, one intentional difference and2326 rows
still requiring reconciliation/review. Original fields/owners remain preserved.

Native reader conditions link to original file-helper rows as architecture
context, not equivalent helper execution. Direct content-helper tests link only
to the corresponding content API; no original term API is certified indirectly.
Required-loader controls cover A001/A007 only for the named family boundary.

### Native input versus architecture and safety

`DefinedCRead`, `DefinedCCarry` and `DefinedCZero` identify supported complete
reader conditions. `RustSafetyRejection` names safe rejection of overflow,
bounds or unwritten storage; it does not claim C defined diagnostics for unsafe
input. Count and public-loader controls are bounded Rust acceptance/error tests,
not full C executable validation. Parser acceptance of infinity/NaN is not
runnable-model approval. `HelperArchitecture` remains separate from stricter
native file readers and does not certify every original helper error overload.

Original generic Julia `QPTransTerm(site,momentum,phase)` APIs A053–A055/A107
are not equivalent to Rust's indexed `QPTransSection/QPTransEntry`. Indexed
A056/A057 likewise need explicit representation and overload reconciliation.
The core PairHop smoke uses archived component composition and checks only that
both metadata branches do not reject solely issue43; it is **not** S409's
independent energy/RNG/output or20-step scenario proof. Original S028 composite and S128/S129 normalization remain pending. M0110/M0134
receive only the individually reviewed cardinality joins below.

At the historical fab2335 checkpoint, one source-identified discrepancy remained: Rust's QPTrans
zero-count test requires five header lines, while C `ReadBuffInt` reads the count
from two lines and the caller ignores subsequent header EOF before
`GetInfoTransSym` skips all body reads when NArray=0. Thus that historical test's
two-line-zero rejection is recorded as `UnresolvedCContractDifference`, not
native parity PASS. Source: `readdef.c:118–124,874–876,2232–2265`, SHA256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.
No native QPTrans probe or production repair was executed in that historical reconciliation.

### Scoped reader-policy update at4d392 / PR273

[PR273](https://github.com/AtelierArith/mvmc-rs/pull/273) resolves the short-zero
reader discrepancy under #272: two consumed count lines precede zero return;
positive full-header checks remain. Five named controls append10 actual CI
records to the preserved57/114 table, for62 controls/124 records total.
Tested head `35ac648ebf3e2b0c561ce5d592a89e3ff28c0f00`; [default job](https://github.com/AtelierArith/mvmc-rs/actions/runs/37174830753/job/111355185604)1047/1047 PASS,40 skipped,282.109s;
[all-features job](https://github.com/AtelierArith/mvmc-rs/actions/runs/37174830753/job/111355185560)1049/1049 PASS,58 skipped,223.637s.
The table binds full raw-log hashes, ordinals, source hashes, settings and commands.
C authority remains readdef118–124,874–876,2232–2265; this is source review and
ordinary Rust CI, **not native execution**. Count0 parsing is not runnable-model
approval. The old rejection/control remains historical in the unchanged table
prefix and JSON policy_history. All2342 rows, statuses and owners remain unchanged.
M0110/M0134 subsequently received full original/source review: only their cardinality assertions are promoted as described below.

### Original cardinality assertions: M0110 and M0134

Parent full-source review approved exactly two original assertions, not S028,
whole-model, inverse or full API acceptance. M0110 maps Julia test186 length2 to
Rust `original_four_site_identity_cyclic_and_default_opttrans_are_loaded` line118;
M0134 maps Julia test257 length4 to
`original_heisenberg_sixteen_site_translation_sample_and_default_identity` line170.
Original revision is Julia8bb; tested Rust head is
`4d89e2a4b5c160bb3469f4b5bbde40fe1c7cdef5`, not a new4d392 execution.
[Default job](https://github.com/AtelierArith/mvmc-rs/actions/runs/37172837109/job/111349156674)
ordinals868/1043 and870/1043; [all-features job](https://github.com/AtelierArith/mvmc-rs/actions/runs/37172837109/job/111349156624)
ordinals871/1045 and873/1045 supply four historical actual PASS records.
These are separate original-assertion receipts, not additions to the62/124 control table.

M0110 uses original Nsite4/NElec2/NLocSpin0 versus Rust Nsite4/NElec1;
M0134 uses original Nsite16/NElec8/NLocSpin16 versus Rust Nsite16/NElec1
and default local-spin settings. NElec/local-spin are not inputs to this
translation-count reader; no equivalent complete model is claimed. The four-site
numeric payload is independently equal despite header/whitespace differences;
the16-site qptransidx.def is byte-original (SHA256
`25715a81bbc8127724cbd666f55a4c9187c7907240ce03e201989fce963c0d16`).
The JSON rows bind source revisions/hashes, exact conditions, settings, command,
raw log hashes, four PASS lines and durable job links. Original records, owners,
parent links and both historical exclusion flags remain intact. Current counts
are17 scoped covered assertions, one intentional difference and2324 pending rows.

### Remaining full acceptance

All300 APIs,456 scenarios,1444 assertions,125 source files and17 documents remain
in the deliverable. All629 historical scope flags remain preserved;627 still
require individual review after the two cardinality joins, not blanket InterAll
numerical exclusions. The38 semantic source reviews remain
source-only. Full method overloads, examples, unit/integration/MPI callsites,
settings, commands, ownership and independent evidence are still required.
Normal all-features CI does not establish actual MPI launch/crossover; skipped
gates remain NotRun. PR254's13-model initialization is not13-model20-step loops.
Four missing independent FSZ fixtures and full thread/sampling/solver scenarios
are not filled by reader controls. Related #184/#185 remain open.

## Validation and remaining evidence

Current metadata validation uses Node, not a Python/Cargo/oracle run. It checks
all2342 keys/classes/original records/owners/parent links, preserves all15+1 prior
covered mappings plus the two reviewed cardinality joins and semantic38 bytes, and binds57 named test sources to114
actual job records. Fifteen negative controls reject wrong revision/anchor/hash,
absent identity, wrong job/ordinal, duplicate join, context or unresolved C
discrepancy promoted to parity, absent settings/command, changed owner, blanket
exclusion and whole-row promotion. These are metadata controls, not57 new Rust
executions or semantic acceptance of every original row.

Counts still needing complete reconciliation:1617 source-mapped missing evidence,
80 historical execution claims and627 scope classifications to review (2324
rows total). The original629 flag texts remain preserved; only two individually
reviewed cardinality conditions changed current classification. The19 partial/context joins and four representation-gap annotations
do not independently promote rows. Remaining300-API/456-scenario breadth is not reduced.

Historical validation of the earlier eaa5 delivery:

Mechanical metadata checks:2,342 unique keys, original records/owners preserved,
38 semantic rows byte-identical, all15 covered assertions with coherent
PASS/entry/settings/reference/command fields and durable run URLs. Python ran
through `uv run --no-project --no-python-downloads python -B`; these checks are
not semantic or numerical acceptance of remaining rows.

Active receipts use repository-relative identity tables and GitHub links.
`original_row` may retain historical `/tmp` receipts; those are historical
provenance, not required external inputs for reading this deliverable. Missing
evidence remains explicit. Aggregate CI totals never promote every model cell.
