# Issue #185 draft milestone status

Related to #185, #184, #183, #176, #177, #178, #175, #174, #181, #180, #179 and #182. No issue closure is claimed.

## Integration and acceptance

Parent integrated main `3e18cd5` at shared HEAD `52d025e`, restored the draft changes and retained the stash. Parent reports `cargo check --workspace --locked` exit0 and 15 SR tests PASS. Full-suite handle **7629 terminated100**, nextest ID `102b3eb0-a1e2-476b-8727-60a45a95be4f`: **749 run,748 PASS,1 FAIL,238.059s**. The failure was OptTrans-CG full-shape92 versus mapped89. Ramanujan's subsequent individual repair checkpoint `15253db9` PASS does not turn this failed full-suite run into PASS. Subsequent13-CG checkpoint `2c4dfaae` terminated100: **6 PASS,7 FAIL**, missing top-step `zvo_out` references; reference import is assigned Ramanujan, not a numerical-tolerance defect or authorization to loosen bounds. Parent formatting exit0, Clippy exit0 and doctests exit0 (**0 cases**, compile evidence only) preceded the latest patch and require final revalidation. No current full-workspace PASS is available here. Earlier full-suite failures remain historical evidence.

The **52/52 fresh ctest references were acquired externally**; this is acquisition, not 52 Rust comparisons or model acceptance. Cargo tests must remain independent of C/Julia/toolbox execution. InterAll is excluded from new #185 scope; repair of historical InterAll consumers does not expand acceptance.

Latest parent selected CG checkpoint `61e661c6`:terminal0,13/13 PASS,196 excluded,14.670s, compiled after the new preflight. This supersedes the selected13-test missing-reference failure after repair, not the terminal100 full-suite result. Goodall PhysCal checkpoint `ea2061c5-f99e-4ca8-8ef0-b1e6fa0df790`:terminal0,42/42 PASS,0skip,84.561s, six models/nine combinations/two frames/all-term gates. Its compiled binary is the tested artifact; run.rs source changed from `a4f586e9` at launch to `0c789661` at harvest, so this is **not source-frozen proof** and cannot validate later runner patches.

Follow updated AGENTS and [numerical policy](../../../NUMERICAL_COMPARISONS.md): C algorithm/input authority, Julia architecture, explicit justified numeric bounds, exact primitive RNG on identical control paths, and 20-step same-input/seed/configuration repeatability. A located numerical acceptance divergence can explain later cross-implementation trajectories; it cannot excuse an RNG or algorithm defect.

## Implemented scope, evidence and remaining work

Results below are bounded parent-reviewed or owner-reported checkpoints, not reruns of all draft changes at `52d025e`. Mutable-tree results without a compilation hash must not be retrospectively attributed to this HEAD.

| Issue | Implemented milestone / available evidence | Remaining acceptance |
| --- | --- | --- |
| #184 | Assertion/API/scenario/source inventories, C-versus-Julia distinctions and focused analytical contracts. Parent core contracts19 PASS (`bff4789e`), hop1 PASS (`da5c6f76`), DH2 literal1 PASS (`0a6d8024`); SFMT4 PASS and analytical Pfa6 PASS in each normal/SIMD/BLAS/combined profile. Tiny Pfa cases do not prove optimized-kernel activation. | Ledger:632 syntax-only unreviewed,596 settings-reviewed/proof-pending,184 focused contracts with differences,23 source-reviewed,6 loop expansions,3 excluded. Complete semantic review and exact public API/example/scenario mappings; no 1444-row coverage claim. |
| #183 | Optional gates and explicit reporting are being migrated. Phase5 runtime20 normal reporting checkpoint `1d9a521c`:8 PASS,4 explicit SKIP. | Execute selected opt-in gates explicitly; missing fixtures/unsupported selection must fail or report incomplete. Nonfinite false-pass regression and provenance strengthening are assigned Banach; reporting PASS is not numerical model20 PASS. |
| #176 | Real-FSZ layout/shadow authority reviewed. Latest REAL-g `722de57e`:1 PASS with explicit written/defined flag policy; standalone retained-input C direct-SR reproduction agrees for the matched operator/increment/post-sync scope. | Full supported real-FSZ sampling/SR/PhysCal scope and portability, not merely setup, ABI or standalone kernel agreement. Preserve historical flag failure and distinguish adapted C-real SR from unmodified Julia. |
| #177 | Root-resolved seed implementation/verification belongs to MPI owner; strict primitive RNG contracts remain intact. | Supply current exact positive/zero/negative seed, root broadcast and rank-state evidence. No blanket #177 PASS is asserted by CLI launch tests. |
| #178 | Current offline public rejection tests: parent `2b7a304c`10 PASS, retained data/state/RNG/output boundaries for matched settings. MPI CLI negative verifier `5b1143` exit0, worlds2/4: rejected launches, error once per rank, no output. | Complete collective/grouped failure matrix and accepted controls. Meaningful Rust diagnostic families need not copy Julia exception literals. Accepted S196/M0998 actual-reducer proof remains separately coordinated. |
| #175 | Independent callback fixtures and tests observe actual in-place success/error state, draw counts, next624, callback indices/status/averaged energy and retained output boundaries; focused12/12 PASS reported before integration. | Fresh integrated verification and native MPI callback-failure evidence; no claim that an owned RNG replay substitutes for actual runner state. |
| #174 | Positive public CLI reference comparisons implemented; parent historical full-suite callback/CLI selections passed. MPI CLI positive `e960ec` exit0, worlds2/4: one summary and one output row. | Integrated serial/CLI scope and indexed Green/Lanczos output checks; CLI process tests cannot claim unobservable internal RNG state. MPI process/output success is not numerical parity. |
| #181 | IO134 synthetic contracts and C-indexed formatting/lifecycle checks: parent `990e6bf3`12 filtered tests PASS; owner mixed15 is a different selection. | Match remaining private-versus-public representative assertions explicitly; third-Hubbard19 is separate, not IO134 closure. Complete non-InterAll PhysCal/Lanczos scenario acceptance. |
| #180 | Fresh reviewed-CG reference acquisition; selected complex/Hubbard/canonical, real, DH2 and pairhop short/20-prefix consumers have bounded PASS checkpoints. Provenance helper `55f833af`3 PASS includes implicit-Auto initial-file hash rejection and dependency closure. | Import/verify remaining consumers and full model/configuration coverage. 52 acquired references and three helper tests are not model numerical verification. Fresh20 windows/final parameters cannot be truncated historical50 results. |
| #179 | Same-configuration MPI repeatability:108 CG pairs and corrected387 direct pairs have separate successful records; parent checked corrected388-line TSV/nonzero0. Literal MPI reducer/product and public CLI checks have narrow independent records. | Numeric correctness/residuals and full declared success/rejection scenario mapping remain distinct from 495 repeatability pairs. Record world/group/worker/storage/seed settings; no blanket cross-language trajectory or tolerance upgrade. |
| #182 | Parent five one-step models/fullOOHO/final workers1/2/4 `821f94c3`1 PASS; selected long20 repeatability `1ba4d50b`1 PASS. Actual 14-site activation evidence belongs to threading owner and is separate from tiny/QP1 serial cases. | Complete all callsite/branch/configuration coverage and current integrated gates. Selected five models or one long20 model are not whole #182; preserve explicit ignored selections and independent activation evidence. |

Detailed commands, settings and historical limitations are in [the current #184 matrix](issue-184-evidence-matrix.md) and the domain-owned records. Final merge requires current full tests, separate doctests, formatting/lint and applicable platform/feature/optional gates; narrow results cannot substitute for these.

Goodall reports fresh post-restoration bounded **43 PASS**, with owned source hashes unchanged: core `9a449fa7-2cd6-4ac6-a1c7-dbb53e4c69ec`36 PASS,0.243s (5 Slater,10 public boundaries,6 output blocks,15 IO134/third-Hubbard selections); IO-library `fd275c50-1654-4e20-9a5f-9131f23dd9c4`6 PASS,0.025s; CLI `e8e4cd94-cb3c-4ac9-b7db-27f67d42fb89`1 PASS,0.057s, exercising four subprocess cases. These are owner-reported selections, not independent parent reruns or full #181/#185 acceptance. Original sixteen-site Slater lifecycle gap is covered; raw public-runner Green predivision and S196 actual-MPI control remain pending. Goodall's separate `issue-181-owned-paths.txt`/draft-packaging record owns its1157-path manifest; none of those paths is implicitly added to Chandra's allowlist below.

## Exact Chandra-owned packaging manifest

The aggregate review artifact is [issue-185-owned-paths.txt](issue-185-owned-paths.txt):8008 explicit sorted changed/new paths assembled from owner allowlists, SHA256 `08ffa9d427b0f6f39be8cca1bf38234d7dfca51eb8c3bb7f7aa7ea81531f674a`. No generation/build process is live for this manifest. Sort and diff checks passed; no staging was performed. Pending freshct20/phase5 namespaces are excluded. Ramanujan's packaging ACK adds shared `crates/mvmc-core/src/run_mpi_tests.rs`, `crates/mvmc-core/src/state.rs` (energy_squared schema/documentation jointly dependent on Goodall) and `crates/mvmc-core/tests/physcal_callback.rs`; this is not exclusive whole-file ownership or authorization to overwrite other contributions. Unclaimed `crates/mvmc-expert-parsers/src/types.rs` remains reported to parent and **excluded**. Owner-declared expected `zvo` fixture payloads are retained; generated runtime `crates/mvmc-core/zvo*` files are excluded. Parent review is required before staging; the manifest does not authorize completion or merge.

Paths below are an ownership allowlist, **not** authorization to stage other agents' production dependencies. The fixture subtree includes its inputs, provenance and validation records. Other owners must provide their own manifests for the aggregate draft.

```text
crates/mvmc-core/tests/issue184_assertion_contracts.rs
crates/sfmt19937/tests/julia_sfmt_assertion_contracts.rs
crates/pfapack/tests/julia_assertion_contracts.rs
crates/mvmc-core/tests/physcal_callback_reference.rs
crates/mvmc-cli/tests/physcal_reference.rs
scripts/generate_physcal_callbacks_175.jl
tests/fixtures/physcal_callbacks_175/**
docs/reference/c-to-julia/verification/issue-184-evidence-matrix.md
docs/reference/c-to-julia/verification/issue-184-public-apis.tsv
docs/reference/c-to-julia/verification/issue-184-scenarios.tsv
docs/reference/c-to-julia/verification/issue-184-sources.tsv
docs/reference/c-to-julia/verification/issue-184-assertion-audit.tsv
docs/reference/c-to-julia/verification/issue-184-io134-review-handoff.md
docs/reference/c-to-julia/verification/issue-185-draft-status.md
```

Exclude unrelated `.cargo/**`, `.codegraph/**`, generated `crates/mvmc-core/zvo*` (including out/var files), any `__pycache__/**`, `wrapper.mod`, Cargo target/cache directories and external acquisition scratch artifacts. Do not stage by broad workspace glob or `git add .`. Security documents, other agents' tests/fixtures/toolbox/production changes and external reference modifications are not in this ownership allowlist. No commit or push was performed for this draft record.
