---
date: 2026-07-07
datetime: 2026-07-07 17:56 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC V05-3 SR-CG with NSplitSize > 1 design plan
updated:
  - datetime: 2026-07-07 18:20 JST
    model: GPT-5 Codex
    note: Incorporated plan-review findings and the completed C semantics audit; the target is corrected Julia sample-offset semantics, not C parity for this broken C combination.
target:
  repository: Julia-mVMC
  base: develop
  scope: VMCParaOpt SR-CG grouped sample split
---

# Julia-mVMC V05-3 SR-CG + NSplitSize > 1 Design Plan

## Decision

Do not start implementation by simply removing the current validator rejection.
The plan-review audit has established that C-mVMC accepts
`NSplitSize > 1 && NSRCG != 0` syntactically but is not a valid numerical oracle
for that combination. Julia should therefore implement **corrected sample-offset
semantics** and gate the feature by MPI self-consistency, not by C-reference
comparison for split SR-CG.

This mode crosses two sensitive contracts:

- `VMCMainCal(comm1)` stores sample-local O vectors.
- `StochasticOptCG(comm0)` consumes those stored O vectors through collective
  matrix-vector products.

The initial positive scope should be narrow:

- `VMCParaOpt`
- `NSRCG = 1`
- `NSplitSize > 1`
- `NStore = 0` or `NStore = 1`, both using the SR-CG stored-O path
- sz-conserved, non-BackFlow inputs
- existing `heisenberg_chain_real_nsrcg`-style standard projection
  (`NQPFull = 8`) first
- real path first; complex path only after the real gate is stable

Keep these rejected in the first implementation PR:

- `NSRCG >= 2`
- `useDiagScale != 0`
- `RescaleSmat != 0`
- FSZ standard-projection split (`NMPTrans != 0` / `NQPFull > 1`)
- OptTrans-derived QP split
- BackFlow
- any attempt to change sampling RNG semantics

`NQPFull = 1` can be added later with a new fixture, but it is not required for
the first gate. The existing SR-CG reference fixture has `NQPFull = 8`; using it
avoids fixture churn while exercising the new SR-CG split risk.

## Background

Current Julia state:

- MPI SR-CG is supported for `NSplitSize = 1`.
- `operate_by_s!` broadcasts the search vector over `comm0`, allreduces the
  raw sampled product over `comm0`, then applies the `<O>` correction once.
- `vmc_main_cal!` and `vmc_main_cal_fsz!` already split sample ranges with
  `split_loop(n_vmc_sample, ctx.rank1, ctx.size1)`.
- `finalize_oo_store!` / `finalize_oo_store_real!` accept `sample_start`, so the
  Julia stored-O finalization can read only the samples owned by a split rank.
- Julia `sr_opt_o_store` buffers are zero-initialized and cleared through the
  accumulator lifecycle, so non-owned global sample columns are zero in the
  full-size store shape.
- `validate_supported_para_opt_modpara` currently rejects
  `NSplitSize > 1 && NSRCG != 0`.

Relevant files:

- `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/weight_average.jl`
- `Julia-mVMC/test/mpi/run_mpi_smoke.jl`
- `Julia-mVMC/test/mpi/mpi_srcg_e2e_smoke.jl`

C reference points:

- `mVMC/src/mVMC/vmcmain.c` calls `VMCMainCal(comm_child1)`,
  `WeightAverageWE(comm_parent)`, `WeightAverageSROpt(comm_parent)`, then
  `StochasticOptCG(comm_parent)`.
- `mVMC/src/mVMC/vmccal.c` and `mVMC/src/mVMC/vmccal_fsz.c` store sample O
  vectors during `VMCMainCal`.
- `mVMC/src/mVMC/stcopt_cg_impl.c` implements `operate_by_S` with `comm_parent`
  broadcast / barrier / allreduce.

Known historical concern and policy context:

- The earlier MPI SR-CG plan intentionally kept `NSplitSize > 1` rejected
  because stored-O sample splitting and SR-CG were not audited together.
- The project policy has treated SR-CG and `NSplitSize > 1` as non-coexisting
  until a separate design/review exists. This plan and its paired review provide
  that separate design.
- The design decision is now explicit: Julia should follow corrected
  sample-offset semantics. C-mVMC is not used as the split SR-CG oracle.

## Phase 1: C Semantics Audit Result

The paired plan review completed the C semantics audit for `mVMC @ 622166a`.
Summary:

- C has no guard for `NSplitSize > 1 && NSRCG != 0`, but also has no public
  regression fixture for that combination.
- C writes `SROptO_Store*` by global sample index in `VMCMainCal`.
- C `calculateOO_Store*` reads compact local columns and does not account for
  `sampleStart`.
- C SR-CG matvec reads all `NVMCSample` columns from `SROptO_Store*`.
- C `SROptO_Store*` is not zero-initialized, so split ranks read non-owned
  uninitialized columns.
- Even if zero-initialized, the C full-column matvec shape would require an
  explicit ownership contract to avoid double-counting or stale-column
  ambiguity.
- `WeightAverageSROpt(comm_parent)` does globally reduce `<O>`, `<O^2>`, and
  `HO`, so the broken part is the stored-O split/matvec contract, not the
  global SR average itself.

Design conclusion:

- Do not create a C-reference split SR-CG gate.
- Do not claim C parity for this combination in public docs.
- Gate Julia by same-chain-count MPI self-consistency and focused local-window
  unit tests.
- If implementation proceeds, update the project policy note that currently
  marks SR-CG + `NSplitSize > 1` as non-coexisting.

## Phase 2: Julia Runtime Design

### 1. Keep CG Collective Scope On `comm0`

`operate_by_s!` should continue to use:

- `bcast!(ctx, x; root = 0, which = :comm0)`
- local sampled product using this rank's `stcOs`
- `barrier(ctx; which = :comm0)`
- `allreduce_sum!(ctx, z; which = :comm0)`
- one application of `inv_w * z - coef * stcO + DSROptStaDel * sdiag * x`

Do not move these collectives to `comm1`. `comm1` splits samples within a chain
group; SR-CG solves one global SR system over all ranks in `comm0`, matching
C's `comm_parent` path.

### 2. Keep Or Make The Stored-O Sample Window Explicit

`stochastic_opt_cg!` currently initializes a `CGWorkspace` with
`n_vmc_sample = data.modpara.nvmc_sample`, assuming each rank has a full
sample-sized local store.

For split support, there are two valid Julia shapes:

1. Keep `CGWorkspace` sized to global `NVMCSample`, with non-owned samples zero.
2. Size `CGWorkspace` to the local `sample_size` and pass the local window into
   `stochastic_opt_cg_init!`.

Correctness baseline:

- Shape 1 is already close to the current implementation because Julia store
  buffers are zero-initialized and cleared before use.
- If shape 1 is kept, add tests that prove non-owned global sample columns stay
  zero under split and that the full-column matvec matches a hand-computed
  local-window result after `comm0` allreduce.

Optional optimization:

- Use local `sample_size` for the BLAS work arrays.
- Pass `sample_start` and `sample_size` into `stochastic_opt_cg_init!`.
- Read `sr_opt_o_store[(sample_start + s) * stride + i]`.
- Keep `state.energy.wc` and `stcO` / `sdiag` globally averaged over `comm0`.
- Allreduce the raw sampled product over `comm0`, so all ranks contribute only
  their local sample window.

This avoids doing BLAS over zeros for non-owned samples and makes the split
contract visible in function signatures. It should be treated as a performance
and clarity improvement, not a prerequisite for correctness if shape 1 is
explicitly tested.

If `stochastic_opt_cg_init!` is reworked, replace the current
`store_idx <= length(...)` silent skip with an explicit dimension assertion.

### 3. Preserve Lock-Step CG

All ranks in `comm0` must enter the same collective loop with the same
dimensions.

Required invariants:

- `n_smat` is identical on every rank after redundant-direction filtering.
- `smat_to_para_idx` is identical on every rank.
- `state.energy.wc` is the globally reduced `Wc`.
- `sr_opt_oo` / `sr_opt_ho` values used to build `stcO`, `sdiag`, and `g` are
  globally reduced.
- no rank-local early return occurs after entering the CG loop.
- rank0 parameter update is followed by `sync_modified_parameter!(ctx, data)`.

### 4. Keep Negative Contracts First-Class

Before removing the current `NSplitSize > 1 && NSRCG != 0` rejection, add or
keep targeted rejections for combinations outside the first implementation:

- `NSRCG >= 2`
- `useDiagScale != 0`
- `RescaleSmat != 0`
- `NSplitSize > 1 && FSZ standard projection`
- `NSplitSize > 1 && OptTrans-derived QP sectors`
- BackFlow

The validator tests should verify the exact rejected combination, not just that
some error occurs.

## Phase 3: Test Plan

### Unit Tests

Add focused tests for:

- `stochastic_opt_cg_init!` with `sample_start` and `sample_size`, proving it
  reads only the owned sample window, if shape 2 is implemented.
- non-owned global sample columns remain zero and the full-size shape matches a
  local-window hand calculation, if shape 1 is retained.
- `operate_by_s!` MPI worker where each rank has a different local contribution
  and the global result matches a hand-computed `S*x`.
- `operate_by_s!` still broadcasts rank0 `x` before local multiplication.
- `operate_by_s!` allreduces before the `<O>` correction by using a fixture
  where reducing after correction would produce a different answer.
- validator matrix for accepted/rejected `NSplitSize` + `NSRCG` combinations.

Some `operate_by_s!` broadcast/allreduce coverage already exists in
`Julia-mVMC/test/mpi/mpi_srcg_operate_smoke.jl`; extend it only for split-local
store/window cases that are not yet covered.

### MPI Integration Gates

Add a self-consistency gate with the same chain-count invariant used by direct
SR split tests:

- reference: `mpiexec -n 2`, `NSplitSize = 1`
- split: `mpiexec -n 4`, `NSplitSize = 2`

Initial fixture:

- start from `Julia-mVMC/test/integration/reference/heisenberg_chain_real_nsrcg`
- force one optimization step
- use `NSRCG = 1`
- keep the existing fixture's standard projection (`NQPFull = 8`)
- real path first

Compare:

- first `zvo_out` row
- `zqp_opt` after the first step
- `zvo_SRinfo` / CG iteration status if stable enough for the fixture; if text
  equality is too brittle, compare numeric fields with documented tolerance

Tolerance:

- Start from the existing SR-CG first-step tolerance class.
- Document any looser tolerance with observed max differences, because SR-CG is
  sensitive to BLAS and reduction order.
- Because the self-consistency gate uses the same chain set, start stricter than
  the C-reference parameter tolerance where feasible and loosen only from
  observed split-vs-unsplit differences.

### Negative MPI Gates

Keep or add explicit failure-mode workers for:

- `NSRCG >= 2`
- `NSplitSize > 1 && useDiagScale != 0`
- `NSplitSize > 1 && RescaleSmat != 0`
- `NSplitSize > 1 && OptTrans-derived QP split`
- `NSplitSize > 1 && FSZ standard projection`

## Implementation Sequence After Design Approval

1. Add or update tests that pin the chosen store-window shape.
2. If keeping full-size global store, prove the zero-column invariant and avoid
   changing `stochastic_opt_cg!` dimensions in the first PR.
3. If moving to local-window workspaces, thread `sample_start` / `sample_size`
   into `stochastic_opt_cg!` and `stochastic_opt_cg_init!`.
4. Keep `operate_by_s!` on `comm0`; do not move CG collectives to `comm1`.
5. Replace the blanket SR-CG split rejection in
   `validate_supported_para_opt_modpara` with the narrow support matrix.
6. Add MPI self-consistency gate for the existing real `NQPFull = 8` SR-CG
   fixture.
7. Update README/manual compatibility tables without claiming C parity for this
   combination.
8. Update the project policy note that currently marks SR-CG + `NSplitSize > 1`
   as non-coexisting, after the implementation is accepted.
9. Consider complex path only after the real gate is stable.

## Verification Commands

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
julia --project=. test/integration/runtests.jl
JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
git diff --check
```

If a new standalone MPI worker is added, run it directly through the smoke
harness and at least once as a focused command during development.

## Risks And Review Focus

- A wrong sample offset can pass serial and `NSplitSize = 1` tests while failing
  only under split.
- C-mVMC cannot be used as the split SR-CG oracle for this combination.
- Redundant-direction filtering must be globally identical; otherwise ranks
  deadlock in the CG collectives.
- Applying the `<O>` correction before allreduce silently adds the correction
  once per rank.
- Tolerances for post-CG parameters may need to remain coarse, but energy and
  update-status differences should still be bounded and explained.
- Complex SR-CG doubles the effective parameter dimension and should not be
  bundled with the first split implementation.
- Public documentation must avoid claiming C parity for split SR-CG unless the
  C-side stored-O contract is fixed and separately tested.

## Exit Criteria For The Plan Phase

- C semantics audit is documented in the plan/review pair.
- The intended target is explicit: corrected Julia sample-offset semantics, not
  C parity for the broken C combination.
- Positive and negative support matrix is fixed before code edits.
- The first MPI self-consistency fixture is identified as the existing real
  `NQPFull = 8` SR-CG fixture.
- The implementation PR can proceed without changing scope midstream.
