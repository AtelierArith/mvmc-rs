---
date: 2026-06-22
datetime: 2026-06-22 16:18 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC v0.5 NSplitSize/NStore direct-SR implementation plan
updated:
  - datetime: 2026-06-22 16:45 JST
    model: GPT-5 Codex
    note: Incorporated review findings A-F: PhysCal namelist validator, unreleased C branch caveat, chain-count invariant, NQPFull guard placement, fixture blocker, and stored-O R1 clarifications.
---

# Julia-mVMC v0.5 NSplitSize / NStore plan

## Decision

次の v0.5 実装主題は、C-mVMC の新方針に合わせた `NSplitSize > 1`
direct-SR support とする。

最初の release scope は以下に限定する。

- `VMCParaOpt`
- direct SR (`NSRCG = 0`)
- `NStore = 0` and `NStore = 1`
- non-FSZ path and FSZ path
- `NSplitSize >= 1`
- one software thread per MPI rank for parity gates

`NSplitSize > 1` と SR-CG (`NSRCG != 0`) は当面非共存とする。SR-CG の
stored-O matvec path は内部 MPI split 未対応として扱い、`NSplitSize = 1`
のみを対象にする。

`VMCPhysCal` の `NSplitSize > 1` はこの計画の primary scope に含めない。
`validate_supported_modpara` を緩和すると PhysCal 側まで誤って許可しやすい
ため、entry-point specific validator を分けて、未実装 entry point は明示的に
reject する。

## Background

C-mVMC 側では、以前の
`NSplitSize > 1 && (NStore != 0 || NSRCG != 0)` 一律 reject から、
以下の方針へ変更された。

- direct SR (`NSRCG = 0`) では `NSplitSize > 1` と `NStore = 0/1` を許可する。
- `NSplitSize > 1 && NSRCG != 0` は reject する。
- 非 FSZ / FSZ の両方で、`NSplitSize = 1/2` と `NStore = 0/1` の比較 test を
  MPI test として走らせる。

C 側の確認元:

- `../Dev_mVMC/mVMC/src/mVMC/readdef.c:991`
- `../Dev_mVMC/mVMC/src/mVMC/vmccal.c:84`
- `../Dev_mVMC/mVMC/src/mVMC/vmccal_fsz.c:39`
- `../Dev_mVMC/mVMC/test/python/runtest_nstore_nsplit.py:119`
- `../Dev_mVMC/mVMC/test/python/CMakeLists.txt:274`
- `../Dev_mVMC/mVMC/doc/en/source/standard.rst:770`

この確認元は未公開の C-mVMC 開発 branch であり、公開 Julia-mVMC の README /
manual / changelog では `../Dev_mVMC/...` のような local-only path を出さない。
また、公開 C-mVMC release/tag に入るまでは、公開文面で「公開 C release との
parity」とは書かず、Julia 側の supported scope として説明する。

Julia-mVMC 側の現状:

- `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl` は `comm0` / `comm1` /
  `comm2` と `group1` をすでに持っている。
- `resolve_rnd_seed(ctx, ...)` は `ctx.group1` offset を使うため、同じ
  `comm1` group 内 rank は C と同じ seed 共有に近い形を作れる。
- ただし `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl` は
  `NSplitSize > 1` を全体 reject している。
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl` は `ctx` を
  `vmc_main_cal!` / `vmc_main_cal_fsz!` へ渡していない。
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` の non-FSZ path は
  stored-O finalize を持つが、現在は rank-local full sample range
  `0:(NVMCSample-1)` 前提である。
- FSZ path は direct `calculate_oo!` のみを使っており、`NStore = 1`
  stored-O path をまだ通していない。

## Non-goals

- SR-CG with `NSplitSize > 1`
- `NSRCG >= 2`
- `useDiagScale != 0`
- `RescaleSmat != 0`
- BackFlow
- full Lanczos
- `VMCPhysCal` grouped MPI/QP split
- independent-chain sampling or stochastic semantics redesign

## Key Design Points

### 1. Contract: global validator と entry-point validator を分ける

`NSplitSize < 1` は引き続き invalid value として global に reject する。

`NSplitSize > 1` はそれ自体では unsupported としない。代わりに、
runtime entry point ごとに対応可否を判断する。

- `VMCParaOpt`:
  - accept: `NSRCG = 0`, `NSplitSize >= 1`, `NStore = 0/1`
  - reject: `NSplitSize > 1 && NSRCG != 0`
- `VMCPhysCal`:
  - keep reject for `NSplitSize > 1` in this milestone, unless a separate
    PhysCal plan implements and gates it.
- serial run with `NSplitSize > 1`:
  - C effectively continues with a size-1 communicator plus warning/load
    imbalance semantics. Julia may accept it as a no-op split, but tests should
    focus on actual MPI launchers.

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_phys_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/run_phys_cal_from_namelist.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`
- `Julia-mVMC/test/mpi/mpi_failure_modes.jl`

Acceptance:

- `NSplitSize = 2, NSRCG = 0` is accepted by the para-opt path.
- `NSplitSize = 2, NSRCG = 1` is rejected before MPI initialization.
- Both para-opt entry points (`vmc_para_opt!` and
  `run_para_opt_from_namelist`) enforce the para-opt contract.
- Both phys-cal entry points (`vmc_phys_cal!` and
  `run_phys_cal_from_namelist`) continue to reject `NSplitSize > 1`.

### 2. Pass `ParallelContext` into main-cal

`vmc_para_opt!` should pass `ctx` to:

- `vmc_main_cal!(data, state, timer, ctx)`
- `vmc_main_cal_fsz!(data, state, timer, ctx)`

Inside main-cal, derive the local sample range from C-equivalent
`SplitLoop(NVMCSample, rank1, size1)`.

Current Julia helper options:

- `split_loop(n, ctx.rank1, ctx.size1)` returns 0-based half-open boundaries.
- `split_range(n, ctx.rank1, ctx.size1)` returns a 1-based Julia range.

The existing `process_sample_range!` uses 0-based sample ids, so use
`split_loop` and call it with `sample_start:(sample_end - 1)`.

Acceptance:

- With `NSplitSize = 1`, sample range remains the current full rank-local range.
- With `mpiexec -n 4` and `NSplitSize = 2`, ranks in each `comm1` group split
  `NVMCSample` in the same way as C.
- Empty sample ranges are accepted and do not call stored-O finalize for direct
  SR store mode.
- Four-way comparison runs must keep the number of Markov chains fixed:
  `nproc / NSplitSize` must match between reference and split runs. The planned
  `-n 2, NSplitSize = 1` vs `-n 4, NSplitSize = 2` comparison has two groups in
  both cases and therefore the same chain seed set (`base + group1`).

### 3. Fix stored-O finalize for split sample ranges

C fixed `NStore = 1, NSRCG = 0, NSplitSize > 1` by keeping the global
sample-indexed store writes and passing a `sampleStart` offset to
`calculateOO_Store*`.

Julia can follow the same rule:

- keep the stored-O buffers allocated with full `NVMCSample` width;
- clear only the active local sample range before the sample loop;
- keep `calculate_oo_store*!` writes indexed by the global sample id;
- extend `finalize_oo_store!` and `finalize_oo_store_real!` with a
  `sample_start::Int = 0` keyword, or pass a view starting at the active local
  sample offset;
- for direct SR (`nsrcg = false`), only finalize `sample_size` columns starting
  at `sample_start`;
- for `nsrcg = true`, this path remains `NSplitSize = 1` only and must not be
  used with `NSplitSize > 1`.

This preserves current serial behavior because `sample_start = 0` for the
existing full-range path.

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_vmc_main_cal_sr.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_threading.jl`

Acceptance:

- Unit test with stored samples written only in columns `sample_start:sample_end`
  fails without the offset and passes with it.
- Existing serial BLAS/store tests still pass.
- Direct SR `NStore = 0` is unaffected.

### 4. Add FSZ stored-O path

The FSZ main-cal path currently uses direct `calculate_oo!` even when
`NStore = 1`. To match C's direct-SR `NStore` support, FSZ needs the same
store/finalize split as non-FSZ.

Tasks:

- Compute `use_sr_store = data.modpara.nstore_o != 0 || data.modpara.nsrcg != 0`
  in `vmc_main_cal_fsz!`.
- Use `calculate_oo_store!` for FSZ complex SR data when `use_sr_store` is true.
- Call `finalize_oo_store!` after the local sample range, with `sample_start`
  and `sample_size`.
- Keep `NSRCG != 0 && NSplitSize > 1` rejected, so FSZ SR-CG is not widened by
  this work.

Acceptance:

- FSZ `NStore = 0` and `NStore = 1` produce matching direct-SR results under
  `NSplitSize = 1`.
- FSZ `NStore = 1, NSplitSize = 2` matches the corresponding C comparison
  within the direct-SR tolerance.

### 5. Sample generation and QP split scope

C's `NSplitSize` has two effects:

1. `VMCMakeSample*` splits the QP projection work within `comm1`.
2. `VMCMainCal*` splits Monte Carlo samples within `comm1`.

Julia currently does not pass `ctx` into `vmc_make_sample!` /
`vmc_make_sample_fsz!`. Therefore, the first safe implementation should not
claim general QP-split support unless this is implemented.

Recommended staged scope:

- R1: support direct-SR `NSplitSize > 1` for fixtures with `NQPFull = 1`.
  In this case QP split is a no-op and the main correctness issue is sample
  splitting plus stored-O finalize. R1 intentionally lets ranks in the same
  `comm1` group redundantly generate the same full chain and then processes a
  local sample subset in main-cal. This is correct for `NQPFull = 1` but not a
  performance model for full C grouped QP splitting.
- R2: implement full C-compatible QP split in sampling:
  - pass `ctx` to `vmc_make_sample!` and `vmc_make_sample_fsz!`;
  - use `split_loop(NQPFull, ctx.rank1, ctx.size1)` for `CalculateMAll`;
  - make `CalculateIP` / accept-reject probability use the comm1-reduced QP sum;
  - verify all ranks in a `comm1` group stay in Markov-chain lock-step.

Until R2 is implemented, add an explicit runtime guard for
`NSplitSize > 1 && NQPFull > 1` in `VMCParaOpt`, with an error that says QP split
sampling is not implemented yet. This avoids silently running a redundant
full-QP calculation while advertising C's grouped MPI/QP split.

Implementation note: `validate_supported_para_opt_parallel_modpara(ctx,
modpara)` currently does not receive `NQPFull`. The R1 guard must therefore be
placed after `get_n_qp_full(data)` is available in `vmc_para_opt!` and
`run_para_opt_from_namelist`, or the validator signature must be extended to
receive the derived `NQPFull`.

Acceptance for R1:

- `NSplitSize > 1` direct-SR tests use fixtures with `NQPFull = 1`.
- A guard test verifies `NSplitSize > 1 && NQPFull > 1` is rejected.

Acceptance for R2:

- A fixture with `NQPFull > 1` passes C-reference MPI comparison.
- A synthetic or small integration test proves ranks in the same `comm1` group
  agree on accept/reject decisions and saved samples.

### 6. MPI comparison tests

Add a Julia MPI worker mirroring C's `runtest_nstore_nsplit.py` structure.

Candidate file:

- `Julia-mVMC/test/mpi/mpi_nsplit_nstore_smoke.jl`

Run cases:

- `direct_ref`: `NSplitSize = 1`, `NStore = 0`, `NSRCG = 0`
- `store_ref`: `NSplitSize = 1`, `NStore = 1`, `NSRCG = 0`
- `direct_split`: `NSplitSize = 2`, `NStore = 0`, `NSRCG = 0`
- `store_split`: `NSplitSize = 2`, `NStore = 1`, `NSRCG = 0`

Use `mpiexec -n 2` for reference cases and `mpiexec -n 4` for split cases, as
in the C test environment (`MVMC_MPI_PROCS_REF=2`,
`MVMC_MPI_PROCS_SPLIT=4`).

Models:

- non-FSZ: use an existing small direct-SR integration fixture, preferably
  `hubbard_chain_real` or a hand-authored `hubbard_chain_spin_jastrow`
  equivalent with `NQPFull = 1`.
- FSZ: use `hubbard_chain_fsz` or another small FSZ fixture with `NQPFull = 1`.

R1 blocker: before implementation, confirm the chosen non-FSZ and FSZ fixtures
actually have `NQPFull = 1`. If the closest analogue of C's
`HubbardChain_SpinJastrow` uses `NQPFull > 1`, create or derive a small
`NQPFull = 1` fixture first; do not leave fixture selection as a post-implementation
open question.

Comparisons:

- compare `zqp_opt.dat` after skipping the same warm-up/header convention used
  by the existing integration helpers;
- compare `zvo_out.dat` rows for direct/store consistency;
- keep tolerances tight for direct SR, but do not require bit identity across
  BLAS/MPI reductions unless the existing MPI direct-SR gate already guarantees
  it for that fixture.

Integrate into:

- `Julia-mVMC/test/mpi/run_mpi_smoke.jl`

Acceptance:

- non-FSZ four-way comparison passes.
- FSZ four-way comparison passes.
- `NSplitSize > 1 && NSRCG = 1` remains a failure-mode test.
- Existing SR-CG `NSplitSize = 1` MPI tests continue to pass.

### 7. Documentation updates

Public docs must describe the supported scope accurately.

Update:

- `Julia-mVMC/README.md`
- `Julia-mVMC/MVMCOptimizers.jl/README.md`
- `Julia-mVMC/docs/manual/03_optimization.md`
- `Julia-mVMC/docs/manual/04_physics_calc.md`
- `Julia-mVMC/docs/manual/05_compatibility.md`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/INDEX.md`

Required message:

- `VMCParaOpt` direct SR supports `NSplitSize > 1` with `NStore = 0/1`, subject
  to the implemented QP-split scope.
- `NSplitSize > 1 && NSRCG != 0` is unsupported.
- `VMCPhysCal NSplitSize > 1` remains unsupported unless implemented in a
  separate workstream.
- If R1 ships before full QP split, document the `NQPFull = 1` limitation.
- Do not cite `../Dev_mVMC/...` or any local review/report path in public
  docs. Until the corresponding C changes are in a public C-mVMC release/tag,
  avoid wording that claims parity with a public C release for this new scope.

## Verification Commands

Minimum local checks:

```bash
julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl")'
julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_vmc_main_cal_sr.jl")'
julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_parallel.jl")'
julia --project=. test/mpi/run_mpi_smoke.jl
git diff --check
```

If C references are regenerated locally, C-mVMC runs on macOS must use:

```bash
OMP_NUM_THREADS=1
```

## Implementation Order

1. Create branch `feature/v0.5-nsplit-nstore`.
2. Add validator tests for:
   - `NSplitSize > 1, NSRCG = 0` accepted by para-opt contract;
   - `NSplitSize > 1, NSRCG = 1` rejected;
   - both para-opt entry points enforce the para-opt contract;
   - both phys-cal entry points still reject `NSplitSize > 1`.
3. Pass `ctx` into `vmc_main_cal!` / `vmc_main_cal_fsz!`.
4. Use `ctx.rank1/ctx.size1` sample split in main-cal.
5. Add stored-O finalize offset support and unit tests.
6. Add FSZ stored-O path.
7. Select/derive non-FSZ and FSZ `NQPFull = 1` fixtures.
8. Add `NQPFull > 1` guard for R1, or implement QP split sampling before
   widening the support claim.
9. Add non-FSZ and FSZ MPI four-way comparison workers.
10. Update README/manual compatibility docs.
11. Run targeted unit tests and full MPI smoke.
12. Write review note under `docs/reviews/` before commit/PR.

## Open Questions

- Should R1 intentionally ship with `NQPFull = 1` only, or should we implement
  sampling QP split before any public `NSplitSize > 1` support claim?
- Does the current FSZ path's always-complex treatment cover the C FSZ test
  scope, or do we need an additional real-FSZ stored-O unit before integration?
- Should `VMCPhysCal NSplitSize > 1` be scheduled immediately after para-opt
  direct SR, or remain a separate v0.5+ candidate?
