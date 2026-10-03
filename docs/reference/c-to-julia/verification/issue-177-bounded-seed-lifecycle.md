# Bounded #177 public negative-seed lifecycle

Related to #177/#179/#185; no issue closure or full numerical parity claim.
New test: `crates/mvmc-core/tests/mpi_issue177_seed_lifecycle.rs`. No production
API, clock, seed payload, RNG algorithm or numerical tolerance was changed.

## Actual final proof

Container `73c57e563c61`, root
`/home/vscode/.cache/mvmc/issue177-verified-outcome.mVfNw1`.
Complete committed archive `60c86eb039ac64e22d118796a00d5e5a35ee364b` plus ONLY
the new test. Reference gitlinks were hydrated at their exact pinned commits:
Julia `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, PfaPack
`0dcf52c15caec63516d0703f36bfc8a4bc0e58d0`, C
`d73d06bd529d3b2573f38eb5817c4a5f52971006`. No dirty production/reference writes
were imported. This is not a later-current-main proof.

- Test SHA `36bf88b554a07414eca7e7f5f702fc51c20e44f3082c8ca2a1d084e87bd4a9b6`.
- Immutable `bin/lifecycle` SHA `e21a0d069d6c6df0d675015197fc2e384b1335d8d7c6723527cc53e3136d762b`.
- Source manifest SHA `f7bd0d386d45be6d4f814977cb2842cf4a138b94c94cf44b2c10eb53563ed9e6`.
- `results.tsv` SHA `1aa0a22330925c5ad834e63bb275f0aac448ce07f4aa3bc622ae6e5867666549`.

Handle46483 terminal0: build0 (4.90s), world2/world4 launcher0, per-rank
exactly1 PASS/0 FAIL/0 ignored, width1/2 completion markers for every rank;
targeted clippy0 (1.11s), source/binary postchecks0. Dependency
tenferro-runtime emits an atomic `fetch_update` deprecation warning; this is
not a claim of whole-workspace warning-free lint.

Retained commands: `commands.txt`; backend/version: `rust.txt`, `mpi.txt`,
`platform.txt`, `environment.txt`, `libraries.sha256`, `binary-ldd.txt`;
source/build: `base.txt`, `overlay-lock.sha256`, `source.sha256`, `build.json`,
`build.log`, `build.exit`, `source-check.txt`; execution: `results.tsv`,
`world2.log`, `world4.log`; lint: `clippy.log`, `clippy.exit`.

Exact build/lint commands (named target, BLAS/OMP1, inner workers1):

```sh
cargo test --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue177_seed_lifecycle --no-run --message-format=json
cargo clippy --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue177_seed_lifecycle -- -D warnings
MPI177_EVIDENCE_DIR=/NEW/world2 timeout --kill-after=5s 90s mpiexec -n 2 \
  bin/lifecycle --ignored --exact enabled::public_negative_seed_opt_and_physcal_lifecycle --nocapture
```

Repeat the last command with world4 and a distinct NEW evidence directory.
Both compile/lint commands were bounded600s with kill grace10s.
The test directory must not exist. Feature-disabled builds have an explicit
Unsupported sentinel; selecting zero tests is not a proof.

## Contract and limits

Public OPT and PhysCal use a copied offline real normal Heisenberg fixture,
InterAll excluded, QP8, seed declaration−1, world2/4, widths1/2, direct/store0,
warmup1, interval1. There is ONE measurement frame/optimization step/window,
with EIGHT saved chain points: `NVMCSample=8` is not eight output frames.
OPT removes the PhysCal-only TwoBodyGEx input declaration before public loading.
Actual input files are retained in `worldN/wWIDTH-mMODE/`.

The reducer observes but never modifies the actual `[status, base]` integer
broadcast, forwards every operation, asserts global base agreement and the
actual offset `rank / width`. Width1 is independent per-rank chains: its
within-group equality check is self-only, NOT world-wide sampler equality.
Width2 compares actual complete saved-chain checkpoints across both group ranks.

Separate public positive-override runs use the observed base plus the SAME
reducer offset. They do not replace/reseed the negative run's live RNG. OPT
requires identical complete sampling event streams, seven-field event8
checkpoint shape (saved five planes, counters, next624), and SR observations.
PhysCal checks primitive seeded raw624/cursor/next624 before initialization,
actual sampled raw624/cursor/count/next624 and explicit saved planes/counters,
then final RNG/configuration against its separate positive run.

OPT's existing event8 does NOT expose raw624/cursor or total initialization draw
count. No fake raw state is inferred from next624. Its next624 and full actual
sampling event stream are the bounded OPT proof. No new production observer
was introduced, and no cross-language model trajectory reference was generated.

The seed-focused OPT outcome classifier permits success, or ONLY exact step0
direct-SR error after a complete sampling checkpoint with independently observed
positive factor INFO, finite original system, UPLO=U, dimension consistency,
no substitution INFO and RHS/increment bits unchanged on failing ranks. A
successful peer must have status0; collective failure needs a real factor
failure somewhere. Separate positive/negative runs must have equal categories
and identical SR observations/trace. No arbitrary error is accepted. In this
final capture all four root scenarios succeeded: the conditional factor-failure
acceptance branch was NOT exercised. No successful SR convergence or returned
parameter immutability is inferred from an Err (the API drops its private data).
The ordinary #179 expected-success repeat script remains unchanged and rejects101.

Additional public boundary cases, widths1/2 on both worlds: all-rank UInt32 upper
overflow, last-peer UInt32 overflow, and last-peer invalid usize offset metadata.
They require returned seed/offset errors, zero sampling events, no seed broadcast
and absent OPT output directories. These are fault-only API/configuration cases,
not modifications of the actual negative-clock payload. `boundary-*.txt` retains
the PhysCal error on every rank (12 files world2,24 world4). Valid peers may
construct and discard an untouched primitive RNG before collective rejection;
do not claim no primitive object allocation or a pre-MPI.Init failure.

## Existing fresh clock/failure protocol and preserved earlier attempts

Committed `822c35e221a2ccac0b12ea9215473bf7e90709fb`, root
`/home/vscode/.cache/mvmc/issue177-current-protocol.7sxFbz`: handle1502 terminal0,
`run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures`, world2/4
per-rank1 PASS (0.14/0.21s). Root-only injected clocks, rank delays, root clock
failure and peer seed-conversion failure are real MPI protocol proof, not the
new public real-clock sampling test. `protocol-results.tsv`, `protocol2.log`,
`protocol4.log`, binary/source manifests retain this separate lineage.

Earlier public roots are retained, not relabelled:

- `issue177-public-lifecycle.r6KNVg`: chain1, world2 SR step0 FAIL101, world4 PASS.
- `issue177-public-lifecycle-chain8.UobuLT`: chain8, handle46181 terminal0;
  before explicit world-base/conversion-boundary/failure-aware refinements.
- `issue177-outcome-public.we56b3`: handle40176 pre-build wrong ldd path failure;
  corrected metadata phase25186 had MPI2/4 PASS but wrapper1 from owned clippy
  `bind_instead_of_map`. Preserve both statuses separately.
- `issue177-final-outcome.uQ7jR1`: handle7635 build101 from incomplete snapshot
  (`xtask` missing), no MPI execution. Final proof uses a complete git archive.
