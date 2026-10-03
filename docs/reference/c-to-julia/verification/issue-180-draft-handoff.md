# Pauli draft handoff after main integration

This is a draft ownership and evidence handoff, not issue completion. Main was
integrated at HEAD `52d025e`. Updated AGENTS policy has been read: independent RNG
contracts remain exact on identical control paths; demonstrated numerical
acceptance differences may change cross-implementation trajectories. The current
long acceptance baseline is 20-step same-implementation/configuration repeatability.
No independent commit or push was performed.

## Owned implementation and tests

- `crates/mvmc-core/src/average.rs`
- `crates/mvmc-core/src/sr.rs` — observation hooks only; preserve any other owner's changes.
- `crates/mvmc-core/src/sr_observer.rs`
- `crates/mvmc-core/tests/sr_observer.rs`
- `crates/mvmc-core/tests/ctest_equivalent.rs`
- `crates/mvmc-core/tests/ctest_model_prefixes.rs`
- `crates/mvmc-core/tests/ctest_window_fixtures.rs`
- `crates/mvmc-core/tests/support/ctest_provenance.rs`
- `crates/mvmc-core/tests/dh2_runtime.rs`
- `crates/mvmc-core/tests/dh4_runtime.rs`
- `crates/mvmc-core/tests/runner_config.rs`
- `crates/mvmc-expert-parsers/src/lib.rs` — Green loader validation edits only.
- `crates/mvmc-expert-parsers/src/utils/validation.rs` — Green bounds/routing edits only.
- `crates/mvmc-expert-parsers/tests/issue184_green_public_bounds.rs`
- `crates/mvmc-expert-parsers/tests/green_two_ex.rs` — only
  `malformed_referenced_file_is_reported_by_expert_loader`; other edits are shared.

## Owned offline toolbox paths

- `c_toolbox/ctest_prefix_oracle.jl`
- `c_toolbox/ctest_reviewed20.md`
- `c_toolbox/ctest_orbital_contracts.md` — provenance/generation documentation edits.
- `c_toolbox/ctest_aggregate_windows.jl`
- `c_toolbox/ctest_opt_window.c`
- `c_toolbox/ctest_opt_window.md`
- `c_toolbox/ctest_opt_window_upstream.inc`
- `c_toolbox/ctest_window_bounds.c`
- `c_toolbox/ctest_window_bounds.md`
- `c_toolbox/ctest_dh_opt_window.c`
- `c_toolbox/ctest_dh_reaggregate.jl`
- `c_toolbox/ctest_dh_window_oracle.jl`
- `c_toolbox/ctest_dh_window_oracle.md`
- `c_toolbox/ctest_direct_sr_capture.jl`
- `c_toolbox/ctest_direct_sr_metrics.jl`
- `c_toolbox/ctest_direct_sr_metrics.md`
- `c_toolbox/ctest_audit_direct_sr_metrics.jl`
- `c_toolbox/ctest_audit_mpi_sr_capture.jl`
- `c_toolbox/ctest_audit_mpi_cg_operator.jl`
- `c_toolbox/ctest_audit_mpi_cg_complex_operator.jl`
- `c_toolbox/ctest_audit_real_cg_stopping.jl`
- `c_toolbox/real_fsz_direct_sr_audit.c`
- `c_toolbox/run_real_fsz_direct_sr_audit.jl`
- `c_toolbox/real_fsz_direct_sr_audit.md`
- `c_toolbox/reviewed_rbm_initializer_arithmetic.c`
- `c_toolbox/reviewed_rbm_initializer_arithmetic.jl`
- `c_toolbox/reviewed_sfmt_state.c`
- `c_toolbox/reviewed_sfmt_state.jl`
- `c_toolbox/reviewed_sfmt_state.md`
- `c_toolbox/check_reviewed_sfmt_state.jl`

## Owned fixture and documentation collections

All existing files under these exact fixture directories are this work's archived
collections; preserve their provenance and historical step labels:

- `tests/fixtures/ctest_model_prefixes/`
- `tests/fixtures/ctest_dh_windows/`
- `tests/fixtures/ctest_direct_sr_metrics/`

The newly generated 20-step matrix has **not** been imported into these directories.
Do not include unrelated generated root `zvo` files or toolbox binaries.

- `docs/reference/c-to-julia/verification/issue-176-current-evidence-audit.md`
- `docs/reference/c-to-julia/verification/issue-180-model-coverage.md` — shared document; preserve parent/Chandra edits.
- `docs/reference/c-to-julia/verification/issue-180-sr-observer.md`
- `docs/reference/c-to-julia/verification/issue-180-rbm-first-initializer-operation.md`
- `docs/reference/c-to-julia/verification/issue-180-retained-cg-operator-audit.md`
- `docs/reference/c-to-julia/verification/issue-180-complex-cg-operator-audit.md`
- `docs/reference/c-to-julia/verification/issue-180-real-cg-stopping-audit.md`
- `docs/reference/c-to-julia/verification/issue-184-sr-semantic-review-pauli.md`
- `docs/reference/c-to-julia/verification/issue-185-real-sr-weight-average.md`
- `docs/reference/c-to-julia/verification/issue-180-draft-handoff.md`

The 18 files currently under
`docs/reference/c-to-julia/verification/evidence/` belong to these offline audits.
Other agents' MPI, IO, runner, CLI, threading and semantic ledger paths are not
claimed here. The reviewed full-parameter C audit tooling/fixtures is coordinated
with Ramanujan and needs joint ownership confirmation before selecting whole files.

## Actual execution evidence

- Parser suite: `6a175916-c5fc-4df4-8751-d593a9585289`, 115/115 passed,
  zero skipped, terminal 0; parent independently reproduced 115/115 before integration.
- Provenance SHA/closure tests: `7916cebd-9e1f-4c5c-81a4-8c488e1c1136`,
  3/3 passed, ten other tests unselected, terminal 0. Includes unlisted Auto
  `initial.def`, explicit overlays and definition omission/tampering. Support tests
  are not numerical model execution.
- Focused parser and ctest-equivalent Clippy passed before integration.
- Parent observer binary: `bf06e2dd-d662-415d-825d-212d1a430b76`, 5/5 passed,
  zero skipped; factor/substitution failures and on/off invariance included.
- Parent DH2/DH4/runner configuration: `209651a8`, 12/12 passed, terminal 0
  before integration; this is not a new post-integration result.
- Strict 108 actual direct-SR pairs and real/complex CG residual audits are
  archived diagnostics, not MPI full-trajectory acceptance or a new tolerance.
- Original C real-FSZ direct solve audit `15087`: terminal 0, INFO 0,
  conditioning/backward-residual evidence and matched post-sync boundary;
  standalone kernel using captured reference buffers, not a native C sampler.

Fresh frozen batch **58651 finished with terminal 0**, all 52 cases executed:
13 canonical models times prefixes 1/2/3/20, zero `ORACLE UNVERIFIED` records.
Stage: container `73c57e563c61`,
`/tmp/mvmc-ctest20-reviewed62b.D3rDdz-matrix`.
Actual stdout SHA-256:
`915437efc0b6637fcc53ab570680413353af0e1e231a87ee6cfde57c0e3e23c6`.
Executed generator SHA-256:
`9355984f3b7762f82a82e49455d841ddc7ca93c9a69d5e3d0f54f9384c2a6a8e`.
The full log is the stage path plus `.stdout.txt`. Source manifest, all 63
identities, actual loaded package paths, thread/backend metadata and non-consuming
SFMT state/index/draw count were checked. Native C FSZ energy is explicitly mixed
with the Julia runner; other energy paths are reviewed Julia kernels.

## Remaining acceptance

1. Offline C aggregate every fresh window, verify output schemas and provenance,
   and import standalone checked-in fixtures through reviewed implementation edits.
2. Run fresh short-prefix numerical comparisons and all 13 20-step repeatability
   gates on the integrated Rust snapshot. Apply current numerical/trajectory policy
   without weakening independent RNG or hiding algorithm/input defects.
3. Finish parser72/input56 semantic audit, especially public multi-record
   initialization/overlay ordering/fixed-value contracts; coordinate Chandra's ledger.
4. Final integrated workspace test-fast, Clippy, formatting, doctests and requested
   MPI/thread/backend matrix checks belong to the parent/coordinated owners.
5. Native macOS portability remains unrun by this agent. Historical evidence must
   not be represented as current 20-step or new Julia 1.13.1 validation.

Use `Related to #180` (and applicable open issues), not `Closes #180`, for the draft.
