---
date: 2026-07-07
datetime: 2026-07-07 11:43 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC V05-4 VMCPhysCal NSplitSize > 1 implementation plan
updated:
  - datetime: 2026-07-07 13:09 JST
    model: GPT-5 Codex
    note: Incorporated plan-review findings on gate detection scope, tolerance, PhysCal NStore/FSZ test scope, and shared OptTrans split predicate.
target:
  repository: Julia-mVMC
  base: develop
  scope: VMCPhysCal grouped split for sz-conserved normal Green output
---

# Julia-mVMC V05-4 VMCPhysCal NSplitSize > 1 Implementation Plan

## Decision

`VMCPhysCal` の `NSplitSize > 1` は、まず **sz-conserved normal Green**
に限定して実装する。

R1 supported scope:

- `VMCPhysCal`
- `NSplitSize >= 1`
- sz-conserved path (`i_flg_orbital_general = 0`)
- `NLanczosMode = 0`
- real path first, complex sz-conserved path second if the same smoke shape is stable
- standard-projection `NQPFull > 1` from `NSPGaussLeg` / `NMPTrans`
- existing PhysCal outputs:
  - energy / variance files
  - one-body Green (`zvo_cisajs_*.dat`)
  - direct two-body Green (`zvo_cisajscktalt_*.dat`)
  - non-FSZ factored/product two-body Green (`zvo_cisajscktaltex_*.dat`)

The following remain outside R1:

- FSZ / general-orbital PhysCal split
- FSZ `TwoBodyGEx`
- `NLanczosMode = 1/2` with `NSplitSize > 1`
- OptTrans-derived QP sectors with split, if such inputs reach PhysCal data
- BackFlow
- new C-reference generation for split runs

The initial correctness gate is **self-consistency with the same Markov-chain
count**, not serial-vs-MPI or equal-rank comparison:

- reference: `mpiexec -n 2`, `NSplitSize = 1`
- split: `mpiexec -n 4`, `NSplitSize = 2`

Both runs have two `comm1` groups, hence the same chain seed set
(`RndSeed + group1`). This matches the comparison invariant already used by
ParaOpt `NSplitSize` gates.

## Background

C-mVMC `VMCPhysCal(comm_parent, comm_child1, comm_child2)` uses the same split
axes as parameter optimization:

- sampling: `VMCMakeSample*` with `comm_child1`
- main calculation: `VMCMainCal*` with `comm_child1`
- energy weight average: `WeightAverageWE(comm_parent)`
- Green-function average: `WeightAverageGreenFunc(comm_parent)`
- counters: `ReduceCounter(comm_child2)`
- output: rank0 only

Relevant C reference points:

- `mVMC/src/mVMC/vmcmain.c`
- `mVMC/src/mVMC/vmccal.c`
- `mVMC/src/mVMC/vmccal_fsz.c`
- `mVMC/src/mVMC/average.c`

Current Julia state:

- `Julia-mVMC/MVMCOptimizers.jl/src/run_phys_cal_from_namelist.jl` already
  parses `NSplitSize`, validates PhysCal input, builds `ParallelContext`, seeds
  RNG via `resolve_rnd_seed(ctx, ...)`, and passes `ctx` to `vmc_phys_cal!`.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_phys_cal.jl` already passes `ctx` to
  `vmc_main_cal!`, `vmc_main_cal_fsz!`, `weight_average_we!`,
  `weight_average_green_func!`, `reduce_counter!`, and rank0 output gates.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` already splits samples with
  `split_loop(n_vmc_sample, ctx.rank1, ctx.size1)`.
- `Julia-mVMC/MVMCOptimizers.jl/src/weight_average.jl` already implements
  C-shaped Green reduce-to-root over `comm0`, including Lanczos arrays.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl` already accepts `ctx` in
  the four normal sampling entry points and implements QP split / `comm1`
  scalar reductions.
- The current blocker is the PhysCal validator plus four sampling calls in
  `vmc_phys_cal!` that still call sampling without `ctx`.

The primary existing fixture for R1 is
`Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref/`.
It is sz-conserved, uses `NLanczosMode = 0`, includes standard projection
(`NSPGaussLeg = 8`, `NMPTrans = -1`), and includes `TwoBodyGEx`.

## Non-Goals

- Do not compare `NSplitSize = 1` under `mpiexec -n 2` to a serial run; the
  chain count differs.
- Do not compare `mpiexec -n 2, NSplitSize = 1` to
  `mpiexec -n 2, NSplitSize = 2`; the chain count differs.
- Do not widen FSZ/general-orbital PhysCal split until there is a dedicated
  fixture and review for the FSZ Green path.
- Do not add a ParaOpt-style `NStore = 0/1` x `NSplitSize = 1/>1` matrix for
  this PhysCal R1. `NStore` controls SR O/OO/HO storage in optimization mode;
  the PhysCal Green path does not branch on it. The committed fixture's
  `NStore = 1` is acceptable for this gate.
- Do not widen Full Lanczos split in the same PR. The reductions are close, but
  `NLanczosMode = 2` Green output is an adjacent residual and should get its own
  focused gate after the normal-Green split gate is stable.
- Do not introduce independent-chain sampling or RNG-stream redesign.

## Implementation Tasks

### Task 1: Narrow The PhysCal Validation Contract

Files:

- Modify `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
- Modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`

Steps:

- [ ] Replace the current blanket `validate_supported_phys_cal_modpara`
      rejection of `NSplitSize > 1` with an entry-point contract that allows the
      value at ModPara level.
- [ ] Move the R1 scope guard into `validate_supported_phys_cal_data(data)`:
  - reject `data.modpara.nsplit_size > 1 && data.i_flg_orbital_general != 0`
  - reject `data.modpara.nsplit_size > 1 && data.modpara.lanczos_mode > 0`
  - reject active OptTrans-derived QP data with split if such data is present
    on PhysCal inputs (`n_qp_opt_trans > 1`, multiple `opt_trans`, or multiple
    `qp_opt_trans`)
- [ ] Factor the OptTrans-derived split predicate into a shared helper and call
      it from both ParaOpt and PhysCal validators. Preserve the existing
      ParaOpt semantics: a single identity/trivial OptTrans entry is accepted;
      only `n_qp_opt_trans > 1`, `length(opt_trans) > 1`, or
      `length(qp_opt_trans) > 1` makes the split guard active.
- [ ] Keep the existing FSZ/general-orbital Lanczos guard.
- [ ] Keep `validate_factored_green_supported(data)` as the FSZ `TwoBodyGEx`
      guard; do not mix it with split validation.
- [ ] Update unit tests:
  - `NSplitSize > 1` is accepted by global and PhysCal ModPara validation.
  - `NSplitSize > 1` is accepted for a minimal sz-conserved
    `ExpertModeData` with `NLanczosMode = 0`.
  - `NSplitSize > 1` rejects FSZ/general-orbital PhysCal data.
  - `NSplitSize > 1` rejects Lanczos PhysCal data.
  - trivial single-entry OptTrans data remains accepted with split.
  - active OptTrans-derived QP data is rejected with split.
  - Existing ParaOpt split validation tests remain unchanged.

Acceptance:

- Unsupported PhysCal split combinations fail before MPI context construction in
  `run_phys_cal_from_namelist`.
- Direct calls to `vmc_phys_cal!` still enforce the same data-level contract.
- Error messages name the exact unsupported combination and say the current
  supported split scope is sz-conserved normal Green.

### Task 2: Pass `ParallelContext` Into PhysCal Sampling

Files:

- Modify `Julia-mVMC/MVMCOptimizers.jl/src/vmc_phys_cal.jl`

Steps:

- [ ] Pass `ctx = ctx` into the four non-BackFlow sampling calls:
  - `vmc_make_sample_real!(data, state, rng, CTIMER_DISABLED; ctx = ctx)`
  - `vmc_make_sample_fsz_real!(data, state, rng, CTIMER_DISABLED; ctx = ctx)`
  - `vmc_make_sample!(data, state, rng, CTIMER_DISABLED; ctx = ctx)`
  - `vmc_make_sample_fsz!(data, state, rng, CTIMER_DISABLED; ctx = ctx)`
- [ ] Keep BackFlow out of scope; do not add a split path to `vmc_bf_*`.
- [ ] Do not change `vmc_main_cal!`, `weight_average_we!`,
      `weight_average_green_func!`, or `reduce_counter!` unless a failing gate
      identifies a real bug.
- [ ] During review, explicitly verify the four non-BackFlow sampling call sites
      in `vmc_phys_cal!` all pass `ctx = ctx`.

Acceptance:

- `NSplitSize = 1` PhysCal behavior remains unchanged.
- Under `NSplitSize = 2`, sampling uses QP split and `comm1` scalar reductions
  rather than each rank recomputing the full QP range.
- Main calculation continues to split samples over `comm1`.
- Green output remains rank0-only.

### Task 3: Add A PhysCal NSplit MPI Worker

Files:

- Create `Julia-mVMC/test/mpi/mpi_physcal_nsplit_smoke.jl`

Worker contract:

- Usage:
  `julia --project=<workspace> test/mpi/mpi_physcal_nsplit_smoke.jl <fixture> <mode> <nsplit> <output_dir>`
- Copy the fixture `physcal_ref/inputs` into a temporary work directory.
- Rewrite only runtime-control values in copied `modpara.def`:
  - `NDataQtySmp = 1`
  - `NVMCWarmUp = 1`
  - `NVMCSample = 4` or `8`
  - `NSplitSize = <nsplit>`
  - `NLanczosMode = 0` for R1 fixtures
- Use the fixture's committed `zqp_opt.dat`.
- Run `MVMCOptimizers.run_phys_cal_from_namelist`.
- On rank0, assert the expected files exist:
  - `zvo_out.dat`
  - `zvo_var.dat`
  - `zvo_cisajs_001.dat`
  - `zvo_cisajscktalt_001.dat`
  - `zvo_cisajscktaltex_001.dat`
- Print exactly one root success label and one non-root success label per
  non-root rank.

Acceptance:

- Worker exits cleanly under `mpiexec -n 2` with `NSplitSize = 1`.
- Worker exits cleanly under `mpiexec -n 4` with `NSplitSize = 2`.
- A rank-local exception aborts MPI rather than hanging a collective.

### Task 4: Register The Self-Consistency Gate

Files:

- Modify `Julia-mVMC/test/mpi/run_mpi_smoke.jl`

Steps:

- [ ] Add a `physcal_nsplit_worker` constant.
- [ ] Add `run_physcal_nsplit_case(fixture, mode, nsplit, nranks)` that:
  - launches the worker
  - asserts the five PhysCal output files are present
  - parses numeric contents from each output file
  - returns a tuple of parsed vectors
- [ ] Add a testset for `heisenberg_chain_real`:
  - reference: `run_physcal_nsplit_case("heisenberg_chain_real", "real", 1, 2)`
  - split: `run_physcal_nsplit_case("heisenberg_chain_real", "real", 2, 4)`
  - compare all five output vectors with a PhysCal split tolerance
    (`5e-8` initially, matching the existing standard-projection split gate's
    tolerance class for `NQPFull > 1` summation-order changes; tighten only if
    the measured maxdiff supports it).
- [ ] If runtime is acceptable, add the same shape for
      `heisenberg_chain_cmp`; otherwise leave complex sz-conserved as R2 and do
      not document it as split-supported yet.

Acceptance:

- The `heisenberg_chain_real` split run matches the same-chain-count reference.
- The test checks all normal Green output families, including factored
  `TwoBodyGEx`.
- The test detects split-mechanism errors such as missing `comm1` scalar
  reductions, QP slice double-counting, seed/group mistakes, and rank0 output
  regressions.
- The self-consistency gate is not a direct proof that `vmc_phys_cal!` passed
  `ctx` into sampling: if `ctx` were omitted, ranks would recompute the full QP
  range in lockstep and this numeric gate could still pass. That wiring is
  protected by Task 2's explicit call-site review and by the existing ParaOpt
  `NSplitSize` gates that exercise the same sampling machinery.
- No FSZ split case is included in R1 because FSZ/general-orbital PhysCal split
  remains rejected by validation.

### Task 5: Update User-Facing Compatibility Docs

Files:

- Modify `Julia-mVMC/README.md`
- Modify `Julia-mVMC/MVMCOptimizers.jl/README.md`
- Modify `Julia-mVMC/docs/manual/04_physics_calc.md`
- Modify `Julia-mVMC/docs/manual/05_compatibility.md`
- Optionally modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/INDEX.md`

Public wording must not mention local review or plan paths.

Required message:

- `VMCPhysCal` supports `NSplitSize > 1` only for the exact R1 scope that is
  tested.
- FSZ/general-orbital PhysCal split remains unsupported.
- Full Lanczos with `NSplitSize > 1` remains unsupported until a dedicated gate
  lands.
- ParaOpt SR-CG with `NSplitSize > 1` remains unsupported.

Acceptance:

- Docs no longer say blanket "`VMCPhysCal` remains limited to `NSplitSize = 1`"
  once the implementation lands.
- Docs do not imply FSZ, Lanczos, or OptTrans split support.
- Public text contains only public repository paths and behavior summaries.

### Task 6: Verification Matrix

Run from `Julia-mVMC/` unless noted.

Required before PR:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
julia --project=. test/integration/phys_cal_equivalent.jl
JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
```

Recommended if the branch also contains recent Full Lanczos changes:

```bash
julia --project=. test/integration/lanczos_equivalent.jl
julia --project=. test/integration/runtests.jl
```

Acceptance:

- Unit tests pass.
- Existing serial/C PhysCal reference gates pass.
- MPI smoke includes the new PhysCal `NSplitSize` self-consistency test.
- Existing ParaOpt `NSplitSize`, SR-CG, and failure-mode tests still pass.

## Follow-Up Tracks

### R2: Complex Sz-Conserved PhysCal Split

If `heisenberg_chain_cmp` is not included in R1, add it as the next small PR
using the same self-consistency shape. Do not broaden docs to complex
sz-conserved split until this gate is present.

### R3: Full Lanczos With PhysCal Split

After the normal-Green split gate is stable and the mode2 Lanczos Green fixture
has landed, add a focused split gate for `NLanczosMode = 2`.

Risk points:

- Lanczos arrays already participate in `weight_average_green_func!(ctx, state)`,
  but the output and canonical-list behavior should be tested explicitly.
- Keep the first Lanczos split fixture sz-conserved and non-FSZ.

### R4: FSZ / General-Orbital PhysCal Split

Handle FSZ separately because it intersects with known `TwoBodyGEx` limitations
and the FSZ Green path has a different state layout.

Required before widening:

- a non-factored FSZ PhysCal split fixture
- explicit guard that FSZ `TwoBodyGEx` remains rejected
- review of `vmc_make_sample_fsz!`, `vmc_make_sample_fsz_real!`,
  `vmc_main_cal_fsz!`, and output layout under split

## Review Checklist

- [ ] Does every PhysCal entry point enforce the same split contract?
- [ ] Does the diff show `ctx = ctx` on all four non-BackFlow sampling call
      sites in `vmc_phys_cal!`?
- [ ] Does validation reject unsupported split combinations before MPI
      initialization when using the namelist runner?
- [ ] Does the self-consistency test preserve chain count?
- [ ] Does rank0-only output still hold under `mpiexec -n 4`?
- [ ] Are all five normal PhysCal output files compared, not just `zvo_out.dat`?
- [ ] Do docs state the exact supported scope without implying FSZ/Lanczos split?
