---
date: 2026-10-06
model: Claude Opus 5.5
status: verified (native C reference, Linux x86_64 Dev Container)
topic: MPI optimization scenario matrix (#179)
---

# Issue #179: MPI optimization scenario matrix

C (`extern/mVMC-1.3.0`) is the numerical and input authority; Julia-mVMC supplies the
feature list. **No new Julia run is claimed**: every comparison below is against native
C, and Julia comparisons stay Unverified (the older Julia-side evidence of #179 comments
is unchanged and not re-presented). InterAll is excluded. PhysCal against native C under
MPI/threads is owned by #397 and is not repeated here; the PhysCal grouped cells already
in the repository are the #349 fixtures (`tests/fixtures/grouped_nsplit_349/`).

## What is executed

| Layer | Where | Cells |
|---|---|---|
| Rank-wise matrix vs native C (one SR step) | `crates/mvmc-core/tests/mpi_issue179_matrix.rs`, `tests/fixtures/mpi_matrix_179/` | 83 cells (32 at 2 ranks, 51 at 4 ranks) |
| Multi-step repeatability and one-group-equals-serial | same file, `multi_step_runs_are_repeatable_and_one_group_equals_serial` | 4 models x {d0 width 1, d0 width N, CG width 1} at 2 and 4 ranks |
| Failures, preflight, seeds, collectives, state, summary, CLI | the #177/#178/#179/#182/#184/#196/#234/#274/#283/#349 explicit gates | all of them, via the runner |
| Executable gate | `scripts/run_explicit_mpi_gates.sh`; dispatchable job `mpi-explicit` of `.github/workflows/optional-gates.yml` | 86 cells, all Pass |

## The rank-wise matrix

Models: `real` and `cmp` Heisenberg chain; `fsz1`, `fsz2` FSZ (`NQPFull` 1 and 2);
`ot` Hubbard chain with DH factors and OptTrans (`NQPOptTrans = 3`, `-o`).
Solvers: `d0` direct `NStore 0`; `d1` direct `NStore 1`; `cg` `NSRCG 1`.
Each model x solver at world 2 with `NSplitSize` 1, 2 and world 4 with 1, 2, 4
(7 saved samples: uneven 2/2/2/1 and 4/3 sample partitions; QP ranges uneven for
multi-QP models (`NQPFull` 8, 6, 2, ...) and empty for `fsz1` at width 2 and 4). Extra empty-work cells: 3
saved samples on 4 ranks for `real` and `fsz1` at widths 1, 2, 4, and 3 samples on 2
ranks at width 2.

Per rank, **exact** (integers, generator state): `Counter[0..6]` after `ReduceCounter`
(accept decisions are counted in the proposal/accept slots, so a wrong decision would
change them), every saved `EleIdx` (configurations), the whole SFMT state and cursor
(the next 624 words are a function of it; this also fixes the draw count and order,
including the parameter-initialization draws of every group and the rejected-move draws).
Per rank and reduced, **toleranced** (`|a-b| <= 1e-13 + 1e-12 |b|`, last-bit summation
order, `docs/NUMERICAL_COMPARISONS.md`): `<HO>` and `<OO>` on every rank (allreduce, so a
duplicate or missing reduction on any rank is visible), `<O>` on rank 0 (C reduces `OO`
and `HO` only; `SROptO` is the rank-local last-sample derivative), and the step energy.

Result: all 83 cells pass at 2 and 4 ranks, including grouped real/complex/FSZ with
uneven and empty work. The grouped sampling chain of group g equals the width-1 chain of
rank g, and the solver does not change step-0 sampling (both asserted on the C fixtures
themselves by the always-running test `c_fixtures_have_the_group_and_solver_invariants_of_the_c_contract`).

## C-defective and C-convention cells

* **Grouped stored O and grouped SR-CG** (`NSplitSize > 1` with `NStore != 0` or
  `NSRCG != 0`): `VMCMainCal` writes `SROptO_Store` at the global sample slot
  (`vmccal.c:241,248`) but `calculateOO_Store` reads columns `0..sampleEnd-sampleStart`
  from the base pointer (`vmccal.c:314-318`), so ranks after the first of a group read
  unwritten memory. Reported upstream as tmisawa/Julia-mVMC#61. Evidence kept: grouped
  `d1` `<OO>` loses those ranks' samples (the fixture test asserts the loss in at least 8
  cells), and two independent C generations differ only in the grouped-CG `<OO>` entries
  (denormal garbage), which are therefore not stored. Rust: grouped `d1` `<OO>` is asserted
  equal to Rust grouped `d0` `<OO>` (Rust is correct; its `finalize_oo_store` takes the
  sample offset); grouped SR-CG is rejected before initialization with the C reason, on every
  rank. Their sampling, counters, `<HO>`, `<O>` and energy are still compared with C.
* **Real-mode OptTrans derivative slots**: C writes them through a mis-offset complex
  pointer (`vmccal.c` `calculateOptTransDiff`, #370, tmisawa/Julia-mVMC#55); the last two
  parameter slots (rows and columns of `<OO>`) are excluded, as in the serial gate.
* **Complex stored `<OO>` layout**: Rust's complex `NStore = 1` `<OO>` is the conjugate
  (memory-transposed) of C's `ZGEMM('N','C')` result; real parts, the only part SR uses,
  agree and serial `NStore 0` and `1` trajectories are bit-identical. The comparison
  conjugates before comparing. Not a C defect; a representation difference.
* **Tiny sample counts**: with 3 samples on one group the C SR solve fails
  (`c_sr_error = 1`, cells `real-d0-r2w2-s3` and `real-d0-r4w4-s3`); Rust must fail too and
  does (collective error, no state).

## Beyond step 1

SR-CG parameter trajectories are ill-conditioned (#358), so no multi-step trajectory is
forced onto C. Instead, for real, cmp, FSZ (`NQPFull` 2) and OptTrans: a 4-step run
(40 samples) is bitwise repeatable on every rank (generator state, counters, energy, root
`zvo_out.dat`) for direct width 1, direct one-group and CG width 1; and the one-group
direct run equals the serial Rust chain of the same seed to `1e-8` scaled (observed
7e-10 FSZ, 3e-13 otherwise). Grouped-vs-C PhysCal and ParaOpt energies over several steps
are in the #349 fixtures.

## Acceptance criteria

| Criterion | Status |
|---|---|
| Enumerate 2/4 ranks, split/grouped, real/complex/FSZ, direct/CG, NStore 0/1, QP and OptTrans | Done above; invalid combinations: grouped CG rejected (C-undefined), `NStore` with grouped is C-defective and Rust-correct, `NSRCG >= 2` without `NStore` rejected by `validate_sr_storage_contract` (`undefined in mVMC C`) |
| Compare seeds, configurations, accept decisions, next 624 words, local/reduced accumulators, SR updates, root output | Against native C, not Julia. Seeds/configurations/decisions/RNG: exact per rank. Reduced accumulators: toleranced. SR update: operands at step 1 (the update itself is ill-conditioned), repeatability beyond. Root output: energy, `zvo_out`/Green files in #349 and #181 |
| Uneven partitions, empty work/weights, collective order, duplicate reductions | Uneven and empty cells above; collective order and duplicate reductions: `mpi_issue179_collectives`, `mpi_issue184_parallel_scalar_contracts`, and every-rank operand comparison |
| Coordinated failures with timeouts; new vs prior evidence | The #178 gates (preflight, preparation, sampling failure, SR failure) run in the runner with a 300 s per-cell timeout; the matrix is new evidence generated from a retained native build recorded in `PROVENANCE.txt` |
| Executable MPI gate; unsupported cells recorded as rejections | `scripts/run_explicit_mpi_gates.sh` and the `mpi-explicit` job; rejected cells (grouped CG) are asserted rejections, not passes |

## Reproduction

```sh
# Dev Container, repository root
c_toolbox/mpi_matrix_179/build.sh /tmp/mvmc-179-c        # instrumented native C
python3 c_toolbox/mpi_matrix_179/generate.py --vmc /tmp/mvmc-179-c/build/src/mVMC/vmc.out \
  --work /tmp/mvmc-179-runs --out tests/fixtures/mpi_matrix_179   # regenerate fixtures
scripts/run_explicit_mpi_gates.sh /tmp/new-dir                    # all explicit gates, 2 and 4 ranks
```

CI: dispatch `.github/workflows/optional-gates.yml` with `family = none` and
`explicit_mpi_gates = true`. The job is outside the bounded-family ledger, like the long
ctest gate: not selected means NotRun, any failing or missing cell fails it (minimum cell
count enforced), and the uploaded artifact holds `provenance.txt`, `cells.txt`,
`summary.md` and every cell log.
