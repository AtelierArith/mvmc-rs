---
date: 2026-06-22
datetime: 2026-06-22 10:18 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC v0.5 CG implementation plan
updated:
  - datetime: 2026-06-22 10:29 JST
    model: GPT-5 Codex
    note: Incorporated review corrections for validator names, test paths, barrier policy, and CG global-average preconditions.
---

# Julia-mVMC v0.5 CG implementation plan

## Decision

v0.5 の主題は CG/SR-CG path の C parity 強化と MPI 対応とする。

最初の release scope は以下に限定する。

- `VMCParaOpt`
- non-FSZ path
- `NSplitSize = 1`
- `NSRCG = 1`
- `NStore = 0` or `NStore = 1`
- `useDiagScale = 0`
- `RescaleSmat = 0`
- one software thread per MPI rank for parity gates

FSZ CG、`NSplitSize > 1`、preconditioned CG、S-matrix rescale は v0.5 の実装対象にしない。

## Background

Serial `NSRCG = 1` はすでに動作しているが、post-CG parameter update は BLAS / FMA /
reduction-order 差に敏感で、現状は coarse tolerance gate で扱っている。

MPI `NSRCG != 0` は現在 intentionally unsupported である。理由は Julia の
`operate_by_s!` が rank-local であり、C の `operate_by_S` が持つ
`comm_parent` broadcast/allreduce semantics をまだ実装していないため。

既存計画とレビュー:

- `docs/plans/2026-06-17-julia-mvmc-mpi-srcg-plan.md`
- `docs/reviews/2026-06-17-julia-mvmc-mpi-srcg-plan-review.md`
- `docs/reports/2026-06-12-julia-mvmc-nsrcg1-serial-cg-residual-rootcause.md`

## Non-goals

- `NSplitSize > 1`
- `NSplitSize > 1 && (NStore != 0 || NSRCG != 0)`
- FSZ CG
- BackFlow
- Full Lanczos
- `NSRCG >= 2`
- `useDiagScale != 0`
- `RescaleSmat != 0`
- independent-chain sampling or stochastic semantics redesign

`NSplitSize > 1 && (NStore != 0 || NSRCG != 0)` は C 側にも silent wrong SR
optimization の疑いがあるため、CG 実装とは切り離して継続 unsupported とする。

## Implementation Workstreams

### A. Parser and runtime contract

Goal: unsupported CG submodes を serial / MPI の両方で明示的に reject し、silent
divergence を防ぐ。

Tasks:

- `ModParaParameters` に `use_diag_scale` と `rescale_smat` を追加する。
- `modpara_parser.jl` で `useDiagScale` と `RescaleSmat` を parse する。
- `NSRCG = 2` を plain `NSRCG != 0` として扱わず、unsupported submode として検出する。
- These fields do not exist today; this is a new parser/data-model addition, following existing snake_case field names such as `nsrcg`, `nstore_o`, and `nsplit_size`.
- `validate_supported_modpara` で以下を reject する。
  - `NSRCG >= 2`
  - `useDiagScale != 0`
  - `RescaleSmat != 0`
- `validate_supported_para_opt_parallel_modpara(ctx, modpara)` は、`NSRCG != 0`
  全体 reject から、未実装 submode の reject へ縮小する準備をする。
- Update the current MPI error message that still says `v0.4` when this support lands.

Touch points:

- `Julia-mVMC/MVMCExpertModeParsers.jl/src/types/expert_types.jl`
- `Julia-mVMC/MVMCExpertModeParsers.jl/src/parsers/modpara_parser.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`

Acceptance:

- `NSRCG = 1` is accepted in serial.
- `NSRCG >= 2`, `useDiagScale != 0`, and `RescaleSmat != 0` fail before computation.
- Rejection tests cover both direct validator calls and runtime entry points where practical.

### B. Serial CG semantic gate

Goal: MPI 化前に serial CG の semantic contract を固定する。

Tasks:

- `operate_by_s!` の small workspace unit test を追加する。
- `NStore = 0` と `NStore = 1` の `NSRCG = 1` store input equivalence を確認する。
- `stcOs_real`, optional `stcOs_imag`, `stcO`, `sdiag`, `invW`, `x`, `z`
  の最小ケースで C-compatible `S*x` を固定する。
- `xdot` が source-order accumulation を維持することを regression test で明確にする。

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_stochastic_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_weight_average.jl`

Acceptance:

- Current serial `NSRCG = 1` integration fixture does not regress.
- Unit tests pin the sampled-product and correction semantics independently of MPI.
- Parameter update tolerance remains documented as numerical sensitivity, not unresolved control-flow mismatch.

### C. Thread `ParallelContext` through CG

Goal: serial behaviorを保ったまま、CG path に MPI communication hook を通す。

Tasks:

- `operate_by_s!` に `ctx::ParallelContext = serial_context()` を渡す。
- `stochastic_opt_cg_main!` に `ctx` を渡す。
- `stochastic_opt_cg!` に `ctx` を渡す。
- `vmc_para_opt!` の `NSRCG != 0` branch から `stochastic_opt_cg!(data, state, timer, ctx)` を呼ぶ。

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl`

Acceptance:

- `serial_context()` では existing serial tests が全て同じ挙動を保つ。
- `ctx` threading だけの commit では MPI behavior をまだ広げない。

### D. Implement C-compatible MPI `operate_by_s!`

Goal: C の `operate_by_S(comm_parent)` と同じ collective semantics を Julia に入れる。

Required order:

1. `bcast!(ctx, x; root = 0, which = :comm0)`
2. compute local sampled product into `z`
3. `barrier(ctx; which = :comm0)` for strict C ordering
4. `allreduce_sum!(ctx, z; which = :comm0)`
5. compute `coef = xdot(ws.stcO, x)`
6. apply `z[si] = invW * z[si] - coef * ws.stcO[si] + DSROptStaDel * ws.sdiag[si] * x[si]`

Important invariant:

- The allreduce must wrap only the raw sampled product. Do not allreduce after the
  `<O>` and diagonal correction, otherwise the correction is applied once per rank.

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl`
- `Julia-mVMC/test/mpi/`

Acceptance:

- Synthetic two-rank test proves rank0 `x` broadcast.
- Synthetic test proves all ranks produce identical `z`.
- Synthetic test catches the wrong ordering where correction is reduced across ranks.
- `comm0` is used; `comm1` and `comm2` are not used in this milestone.

### E. Rank0 parameter update and lock-step

Goal: C と同じく rank0 が CG result を parameter に反映し、既存 sync で全 rank へ配る。

Precondition:

- Before entering CG, `stcO`, `sdiag`, `g`, and `Wc` must already be global
  comm0 averages/sums on every rank. `operate_by_s!` only reduces the raw sampled
  `<OO>x` product per CG iteration; it does not make rank-local SR averages global
  by itself.

Tasks:

- Under MPI, update variational parameters only on output rank / rank0.
- Serial mode continues local update.
- Keep `sync_modified_parameter!(ctx, data)` in the main optimization loop as the propagation point.
- Broadcast `info` consistently after CG, matching C's rank0 decision followed by `MPI_Bcast`.
- Avoid rank-local early returns inside the collective CG loop.

Collective invariants:

- `n_smat` must be identical on all ranks.
- cut decisions must be based on global SR averages and therefore identical.
- CG break conditions must be rank-consistent.
- any `info` decision must remain globally synchronized before the next collective.

Touch points:

- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/parameter_sync.jl`

Acceptance:

- Unit or MPI worker test asserts `n_smat` and CG call counts match on all ranks.
- No rank-local CG early exit can skip a broadcast/allreduce called by other ranks.

### F. MPI integration fixtures

Goal: `NSRCG = 1`, `NSplitSize = 1` を MPI で C reference と比較する。

Tasks:

- Existing `NSRCG != 0 under MPI` failure-mode test を pass case へ置き換える。
- `NSplitSize > 1` failure-mode test は維持する。
- Add first-step or short-run MPI fixtures:
  - `mpiexec -n 2`
  - `mpiexec -n 4` if runtime is acceptable
- Compare against C-mVMC MPI reference, not just Julia serial.

Gate outputs:

- first `zvo_out` row with tight energy/Sz tolerances
- `zvo_SRinfo.dat` shape and key fields
- `zqp_opt` parameter update with documented CG tolerance

Touch points:

- `Julia-mVMC/test/mpi/`
- `Julia-mVMC/test/integration/`
- `Julia-mVMC/MVMCOptimizers.jl/test/runtests.jl`

Acceptance:

- MPI `NSRCG = 1`, `NSplitSize = 1` no longer errors.
- First-step C/Julia MPI gate passes for rank 2.
- Rank 4 is added if runtime and CI/site stability are acceptable.

### G. Documentation and release notes

Goal: user-facing scope を実装と一致させる。

Tasks:

- README / manual compatibility table を更新する。
- `NSRCG = 1` under MPI support を `NSplitSize = 1` 限定で明記する。
- `NSRCG >= 2`, `useDiagScale`, `RescaleSmat`, `NSplitSize > 1` は unsupported と明記する。
- `NSRCG = 1` parameter-update tolerance の理由を concise に残す。

Touch points:

- `Julia-mVMC/README.md`
- `Julia-mVMC/MVMCOptimizers.jl/README.md`
- `Julia-mVMC/docs/manual/03_optimization.md`
- `Julia-mVMC/docs/manual/05_compatibility.md`
- `Julia-mVMC/CHANGELOG.md`

Acceptance:

- Docs no longer say all MPI CG is unsupported once implementation lands.
- Docs do not imply FSZ CG or `NSplitSize > 1` support.
- Public text must not mention local-only paths or internal report paths.

## Suggested PR / commit split

1. `cg-contract`: parser fields, unsupported submode guards, serial tests.
2. `cg-serial-semantics`: focused `operate_by_s!` semantic tests and any necessary refactor.
3. `cg-context-threading`: pass `ParallelContext` through CG path without enabling MPI CG.
4. `cg-mpi-operate-by-s`: broadcast/allreduce in `operate_by_s!` plus synthetic MPI tests.
5. `cg-mpi-integration`: C-reference MPI fixture, failure-mode update, docs.

Remote push / PR creation / merge / release operations require explicit user confirmation.

## Release Gate

Minimum local gate, run from `Julia-mVMC/`:

- `julia --project=@. MVMCExpertModeParsers.jl/test/runtests.jl`
- `julia --project=@. MVMCOptimizers.jl/test/runtests.jl`
- `julia --project=@. test/integration/runtests.jl`
- `mpiexec -n 2 julia --project=@. test/mpi/run_mpi_smoke.jl`
- targeted parser / unsupported-input unit tests
- targeted `stochastic_opt` unit tests

Site gate:

- Genkai first, one software thread per MPI rank.
- Ohtaka second if time permits.
- Compare C-mVMC and Julia-mVMC for the first-step MPI SR-CG fixture.

## Risks

- CG parameter update is numerically sensitive. Do not use a strict multi-step
  bit-parity gate as the primary acceptance criterion.
- Reduction order differs across BLAS / compiler / MPI implementations. First-step
  energy and SRinfo shape are more stable than long-run parameter trajectories.
- Incorrect allreduce placement in `operate_by_s!` silently computes the wrong
  system matrix.
- Rank-local early return inside CG can hang MPI collectives.
- `NSplitSize > 1` must remain out of scope because it intersects the known C-side
  store/SR-CG issue.

## Next Step

Start with Workstream A. It is small, high-value, and removes an existing serial
silent-divergence risk before MPI CG implementation expands the supported surface.
