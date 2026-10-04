# Joined evidence for #184 / #185

Current checkpoint: main `eaa5db6254c721b95e8e63096287cdb74d5124e2`.
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
| CoveredScopedOrdinaryCI | 15 | Named original assertion conditions, not complete API/scenario verification |
| IntentionalDifference | 1 | M0971: zero-QP helper retains zero instead of Julia's one |
| SourceMappedMissingEvidence | 1,617 | Conditions/execution reconciliation incomplete |
| HistoricalExecutionClaimNeedsReconciliation | 80 | Historical claims retained, not newly certified |
| ExplicitScopeExclusionNeedsReview | 629 | Review-required classifications, not automatic exclusions or PASS |

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

## Validation and remaining evidence

Mechanical metadata checks:2,342 unique keys, original records/owners preserved,
38 semantic rows byte-identical, all15 covered assertions with coherent
PASS/entry/settings/reference/command fields and durable run URLs. Python ran
through `uv run --no-project --no-python-downloads python -B`; these checks are
not semantic or numerical acceptance of remaining rows.

Active receipts use repository-relative identity tables and GitHub links.
`original_row` may retain historical `/tmp` receipts; those are historical
provenance, not required external inputs for reading this deliverable. Missing
evidence remains explicit. Aggregate CI totals never promote every model cell.
