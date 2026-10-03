# Draft PR handoff — Related to #179, not complete

Parent integrated main at HEAD52d025e. This handoff does not commit, stage or push.
Exact path inventory: [issue-179-owned-paths.txt](issue-179-owned-paths.txt).
The initial production paths in that list are scoped ownership contributions,
NOT exclusive whole-file ownership: reducer/sampler interfaces have coordinated
runner contributions. Parent must review/stage the combined shared draft rather
than overwrite another owner's edits. Own verification files are the named MPI
tests, scripts, toolbox probes and issue179 docs. Do not package __pycache__,
container artifacts/targets, host scratch output, .codegraph, or other owners'
fixtures/test/doc drafts merely because they are untracked.

Runner run.rs/CLI/lib.rs, sr_cg.rs, sr.rs/sr_observer.rs, threading.rs and reference
fixture migrations belong to their assigned owners. Optional replay depends on
Ramanujan's separately owned `ctest_cg_main_upstream.inc` /
`ctest_cg_dot_upstream.inc` and Pauli's residual helpers; parent must include their
reviewed dependency paths in the combined PR. Normal Cargo tests never invoke
C/Julia/toolbox runtime.

## Verification status before final freeze

- Corrected direct handle83487 terminal0:387 pairs; retained original full
  handle51328 terminal1:108 CG pairs passed,387 harness failures. Independent
  union audit proves495 declared configurations with no gaps/duplicates/selected
  failures. Same-implementation repeatability only, not independent accuracy.
- Literal two-rank real/complex CG product gate passed; parent39cd1b independently
  repeated it. Actual root poison broadcast and product phases are covered.
- Public CLI worlds2/4 positive boundary gate passed; parent e960ec confirmed it.
  Actual no-feature rejection passed; parent5b1143 confirmed it. No Julia explicit
  MPI-mode policy equivalence claim.
- Accepted S196 singleton: corrected six world/worker configurations passed two
  public PhysCal runs each, with exact primitive/repeat discrete checks, bounded
  computed repeats, actual worker observations, density identity, and indexed
  output checks. [Full settings/failures/proof](issue-179-S196-singleton.md).
- Retained expanded callback498 rows passed the stronger identity/reason/count/
  staged-output gate on its documented older snapshot; not current-main proof.
- Independent C fixed-published-Julia-operand CG replay/residual diagnostics pass
  their narrow protocol scope; Rust/C/Julia comparison covers first common solve
  only. No arbitrary downstream CG forward tolerance or exact-iteration gate.
- After restoration: utility suite46 passed via uv; newly owned literal/singleton
  Rust files pass rustfmt. These focused checks are not final full-workspace fmt,
  Clippy, nextest or portability validation.

All numerical/runtime claims above retain their captured source hashes, primarily
qAZUvg. They are NOT relabelled current-main52d025e proof. Parent's cargo check and
15 SR tests after integration are separately reported, not owned MPI final proof.

## Remaining acceptance

1. New frozen post-integration combined source: MPI matrix, failures, formatting,
   strict Clippy, workspace nextest/doctests, feature gates and portability checks.
2. Long20 MPI/worker same-seed repeatability with consistent effective final
   window; current495 evidence is prefixes1/2/3. No truncation of historical50
   artifacts into20-step final parameters.
3. Independent short numerical/state/SR/root-output checks across required model,
   projection/storage/worker scopes, with meaningful operation/residual budgets;
   do not use repeatability as independent accuracy.
4. Remaining parallel74/QPsplit9/MPI87 individual semantic mappings: serial public
   invocation boundaries, public non-root result architecture, named PhysCal
   outputs/messages, and the C-versus-Julia diagnostic warning distinction.
5. Update prospective reference scripts to the clarified policy where old
   cross-language exact trajectory gates still require evidence/diagnostic
   separation; retain primitive fixed-path RNG protections and failed records.

No owned validation process is live at this checkpoint. #179 stays open; draft
body should say **Related to #179**, not Closes.
