---
date: 2026-06-30
datetime: 2026-06-30 13:18 JST
updated_datetime: 2026-06-30 13:35 JST
model: GPT-5 Codex
status: design
topic: Julia-mVMC V05-3 NSplitSize standard-projection QP split
---

# Julia-mVMC V05-3 NSplitSize Standard-Projection QP Split Design

## Scope

This design covers the next V05-3 step for Julia-mVMC ParaOpt MPI support:
allow `NSplitSize > 1` together with standard-projection `NQPFull > 1`.

Supported scope:

- `VMCParaOpt` only.
- Direct SR only: `NSRCG == 0`.
- `NStore == 0` and `NStore == 1`.
- Non-FSZ and FSZ paths.
- Complex and real Pfaffian paths.
- `NQPFull > 1` only when it comes from standard projections:
  `NQPOptTrans == 1` and either `NSPGaussLeg > 1` or `abs(NMPTrans) > 1`.

Rejected or future scope:

- `NQPOptTrans > 1` and `OptTrans` derived `NQPFull > 1` with
  `NSplitSize > 1`.
- `NSRCG != 0` with `NSplitSize > 1`.
- `VMCPhysCal` with `NSplitSize > 1`.
- BackFlow.
- Any change to stochastic semantics, RNG stream ownership, or independent
  Markov-chain splitting.

The implementation goal is C-mVMC fidelity. The split axis must match C:
sampling splits the QP sum across the child communicator, while the main
calculation keeps a full QP loop per sample and splits samples instead.

## Background

`NQPFull` is not one feature. It is the product of the ordinary projection
space and optional optimization transformations:

```text
NQPFix = NSPGaussLeg * abs(NMPTrans)
NQPFull = NQPFix * NQPOptTrans
```

The standard projection part (`NSPGaussLeg` and `NMPTrans`) is a normal
mVMC feature. The optional `NQPOptTrans` / `OptTrans` path is rarely used,
is active only through the optimization transformation input path, and was
documented locally as under-tested and not suitable for relaxing together
with `NSplitSize` in this step.

Relevant evidence checked for this design:

- `mVMC/src/mVMC/vmcmain.c`: C builds `comm0`, `comm1`, and `comm2`, and
  seeds Markov chains by `RndSeed + group1`.
- `mVMC/src/mVMC/vmcmake.c`, `mVMC/src/mVMC/vmcmake_real.c`,
  `mVMC/src/mVMC/vmcmake_fsz.c`, and
  `mVMC/src/mVMC/vmcmake_fsz_real.c`: C sampling calls `SplitLoop` over
  `NQPFull` on the inner communicator and passes that QP range into
  `CalculateMAll*`, `CalculateNewPfM*`, and `UpdateMAll*`.
- `mVMC/src/mVMC/qp.c` and `mVMC/src/mVMC/qp_real.c`: C computes the local
  QP partial sum and then `MPI_Allreduce`s over the communicator supplied
  by the caller.
- `mVMC/src/mVMC/vmcmake.c`: C sampling passes the inner communicator to
  `CalculateLogIP*`, and reduces the initial `CalculateMAll*` retry flag
  with `MPI_MAX` so every rank in the group regenerates together.
- `mVMC/src/mVMC/vmccal.c` and `mVMC/src/mVMC/vmccal_fsz.c`: C main
  calculation splits samples over the inner communicator, uses the full QP
  range for each local sample, and passes `MPI_COMM_SELF` to `CalculateIP*`
  so the full-QP inner product is not reduced across `comm1`.
- `mVMC/src/mVMC/stcopt_cg_impl.c`: SR-CG stored-O matrix-vector products
  are reduced over `comm0`, and are not part of this relaxation.
- `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl`: Julia already has the
  communicator model, `split_loop`, seed grouping, and allreduce wrappers.
- `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`: Julia currently
  rejects all `NSplitSize > 1 && NQPFull > 1`; this is too broad for the
  supported standard-projection case.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`: Julia sampling
  currently sets `qp_start = 1` and `qp_end = n_qp_full + 1`, so it does
  not yet perform the C-like QP split.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`: Julia main
  calculation already splits samples through `split_loop`, which matches
  C for this step.
- `../Dev_mVMC/docs/2026-06-21-nqpopttrans-investigation.md`: local
  investigation that separates normal projection `NQPFull > 1` from the
  rarely used `NQPOptTrans > 1` path.
- `../Dev_mVMC/docs/2026-06-21-pr110-nsplit-implementation-verification.md`:
  local verification that C's PR #110 style split uses QP splitting in
  sampling and sample splitting in main calculation.
- `../Dev_mVMC/docs/plans/2026-06-22-mvmc-nstore-nsplit-implementation-plan.md`
  and `../Dev_mVMC/docs/reviews/2026-06-22-mvmc-nstore-nsplit-plan-review.md`:
  local notes confirming that direct SR with `NStore == 0/1` can be
  treated separately from SR-CG.

## Runtime Contract

The user-visible contract should be precise:

- `NSplitSize > 1` is allowed for ParaOpt direct SR when
  `NQPOptTrans == 1`, even if `NQPFull > 1`.
- `NSplitSize > 1` remains rejected when `NQPOptTrans > 1` or when parsed
  `OptTrans` data indicates the optimization-transform path is active.
- `NSplitSize > 1 && NSRCG != 0` remains rejected.
- `NSplitSize > 1` remains rejected for PhysCal.
- BackFlow remains rejected independently of this work.

Implementation should change the guard in
`Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl` from the current
blanket `NSplitSize > 1 && NQPFull > 1` reject to an OptTrans-specific
reject. The guard should prefer explicit parsed fields over inference:

- Use `data.n_qp_opt_trans` as the primary discriminator.
- Treat nonempty or otherwise active `data.opt_trans` / `qp_opt_trans`
  state as OptTrans-derived and unsupported with split.
- Continue using the existing `get_n_qp_full(data)` behavior for total QP
  count, but add small helpers if needed to make `NQPFix` versus
  `NQPOptTrans` visible in validation and tests.

Do not relax the global unsupported features in
`validate_supported_modpara`, such as unsupported SR-CG modes or rescale
features. This step is only about replacing a too-broad `NQPFull` guard
with a narrower `NQPOptTrans` guard.

## Sampling Design

Sampling must mirror C's QP split.

The current Julia sampling entry points in
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl` should receive either
the `ParallelContext` itself or a small derived QP split object:

```text
if ctx.size1 == 1
    qp_start = 1
    qp_end = n_qp_full + 1
else
    qp_start0, qp_end0 = split_loop(n_qp_full, ctx.rank1, ctx.size1)
    qp_start = qp_start0 + 1
    qp_end = qp_end0 + 1
end
```

The local QP range is half-open in Julia call sites: `[qp_start, qp_end)`.
This preserves the existing convention already used by `calculate_m_all*`
and related kernels.

The implementation must support empty QP ranges. C's `SplitLoop` permits
empty work when `ctx.size1 > NQPFull`; the local contribution is then zero
and the allreduce recovers the full value. Julia must not assume every
rank owns at least one QP point.

Existing kernel signatures already carry QP ranges:

- `calculate_m_all*`
- `calculate_new_pf_m*`
- `update_m_all*`

The implementation should wire the computed QP range into those existing
arguments instead of duplicating kernels.

Initial Pfaffian retry is part of the split contract. After
`calculate_m_all*` returns the local status for the rank's QP range,
sampling must allreduce the status across `comm1` with max/OR semantics
before deciding whether to regenerate the sample. If any rank sees a
singular local Pfaffian, every rank in the same `comm1` group must
regenerate in lockstep.

The same collective-alignment rule applies to every sampling branch that
can change control flow. Retry, regeneration, `isfinite(log_ip_old)`,
error/early-return decisions, and `burn_flag` transitions must depend only
on values that are already local to the whole group or have been reduced
over `comm1`. A single rank must not return, retry, or advance to a
different collective sequence independently.

All sampling variants must use the split range consistently. In particular,
the current FSZ complex sampling call sites at
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl:4046`,
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl:4166`, and
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl:4279` pass a literal full
range `1, n_qp_full + 1`. Those calls must be changed to the local split
range; otherwise full-range summation followed by `comm1` reduction will
double-count by `ctx.size1`.

## IP And LogIP Design

`CalculateIP` / `CalculateLogIP` parity is the critical part of the QP
split. In C, each rank computes a partial QP weighted sum and then reduces
that sum over the inner communicator.

Julia should extend the corresponding functions in
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl` to perform the same
operation:

- `calculate_ip_fcmp`
- `calculate_log_ip_fcmp`
- `calculate_ip_real`
- `calculate_log_ip_real`

The functions must receive an explicit reduction mode or communicator.
Passing `ctx::ParallelContext` alone is not enough, because sampling and
main calculation can share the same `ctx.size1 > 1` while requiring
different reduction behavior.

The required modes are:

- Sampling: reduce over `comm1`.
- Main calculation: no reduction, equivalent to C's `MPI_COMM_SELF`.
- Serial and non-MPI callers: no reduction by default for backward
  compatibility.

The important contract is:

1. Sum the supplied QP range.
2. If the caller requested `comm1`, allreduce the local sum over `comm1`.
3. If the caller requested no reduction, leave the local sum unchanged.
4. Take the logarithm after the optional reduction.

The real path should not change its sign or logarithm convention in this
step. If a separate real-log parity bug is found, handle it as a separate
fix with its own tests. This V05-3 task is only about adding the missing
communicator reduction.

## Main Calculation Design

Main calculation must not be QP-split for this step.

C uses `comm1` differently in main calculation than in sampling: it splits
samples over the child communicator, and each local sample uses the full
QP range. Julia already follows this pattern in
`Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` by computing
`sample_start, sample_end = split_loop(n_vmc_sample, ctx.rank1, ctx.size1)`.

Therefore the implementation should:

- Keep full QP loops in `vmc_main_cal!` and `vmc_main_cal_fsz!`.
- Keep the existing sample split over `ctx.rank1` / `ctx.size1`.
- Call `calculate_ip*` / `calculate_log_ip*` from main calculation with the
  no-reduce mode, matching C's `MPI_COMM_SELF` behavior.
- Audit all main-calculation `calculate_ip*` call sites so none of them
  accidentally reduce over `comm1`.
- Leave stored-O finalization behavior intact except for tests that prove
  `NStore == 1` still agrees under `NSplitSize > 1`.

This avoids changing PhysCal semantics and avoids conflating sample
splitting with sampling-time QP splitting.

## Tests

Unit and contract tests:

- Validation accepts ParaOpt direct SR with `NSplitSize > 1`,
  `NQPOptTrans == 1`, and `NSPGaussLeg > 1`.
- Validation accepts ParaOpt direct SR with `NSplitSize > 1`,
  `NQPOptTrans == 1`, and `abs(NMPTrans) > 1`.
- Validation rejects `NSplitSize > 1 && NQPOptTrans > 1`.
- Validation rejects active `OptTrans` data with `NSplitSize > 1`.
- Existing `NSplitSize > 1 && NSRCG != 0` rejection remains covered.
- Any new QP range helper covers normal ranges and empty ranges.
- Existing `NQPFull == 1 && NSplitSize > 1` behavior remains covered by a
  regression test, because the new code path changes it from redundant
  full-range sampling to split-range plus reduction.
- IP helper tests cover both modes: sampling-style `comm1` reduction and
  main-calculation no-reduce behavior.
- Initial Pfaffian retry tests cover comm1-wide max/OR of local failure
  flags, including the case where only one rank sees a failure.
- If practical in the existing MPI test harness, synthetic IP allreduce
  tests should verify that rank-local partial QP sums reproduce the serial
  full QP sum after `comm1` reduction.

MPI integration tests:

- Add or adapt a standard-projection fixture with `NSPGaussLeg > 1`,
  `NMPTrans == 1`, and `NQPOptTrans == 1`.
- Add or adapt a standard-projection fixture with `abs(NMPTrans) > 1` and
  `NQPOptTrans == 1`.
- For each fixture, compare four cases with equal Markov-chain count:
  - direct reference: `NSplitSize == 1`, `NStore == 0`
  - stored reference: `NSplitSize == 1`, `NStore == 1`
  - direct split: `NSplitSize == 2`, `NStore == 0`
  - stored split: `NSplitSize == 2`, `NStore == 1`
- Compare `zvo_out.dat`, `zvo_var.dat`, `zqp_opt.dat`, and
  `zvo_SRinfo.dat`.
- Start with `max_abs <= 1e-8`; document and justify any looser tolerance.
- Add a negative integration check that `NQPOptTrans > 1` with split fails
  during validation before divergent MPI collectives can occur.

For macOS C-mVMC reference runs, use `OMP_NUM_THREADS=1`.

## Documentation

Public Julia-mVMC documentation should describe the supported scope without
mentioning local investigation paths:

```text
NSplitSize > 1 supports standard-projection NQPFull > 1 when
NQPOptTrans = 1. OptTrans-derived NQPFull > 1 remains unsupported with
NSplitSize > 1.
```

Candidate files to update during implementation:

- `Julia-mVMC/README.md`
- `Julia-mVMC/MVMCOptimizers.jl/README.md`
- relevant manual or compatibility documentation under `Julia-mVMC/`
- changelog or release notes if the roadmap expects them for this item

Public PR text must not mention local-only docs or local workspace paths.
It should cite only public repository paths, tests, and result summaries.

## Risks And Mitigations

- Empty QP ranges: cover with helper tests and make IP partial sums return
  zero before allreduce.
- Main-calculation double-counting: make IP reduction mode explicit and
  keep main-calculation IP calls on the no-reduce path.
- Collective divergence: ensure sampling retry, regeneration, error-return,
  and burn-state transitions use comm1-agreed values before changing
  control flow.
- FSZ hard-coded full ranges: treat the known full-range sampling call
  sites as correctness fixes, not optional audit items.
- Performance expectations: QP splitting introduces scalar allreduces in
  the sampling loop. This is C-faithful but can be latency-bound; it should
  be treated as useful mainly when per-QP work dominates allreduce latency.
- Real path sign/log convention: do not change it in this task.
- `NMPTrans < 0`: preserve the existing absolute-value convention used by
  `get_n_qp_full`.
- Accidental PhysCal relaxation: keep PhysCal split rejection unchanged.
- SR-CG leakage: keep `NSRCG != 0 && NSplitSize > 1` rejected until stored-O
  CG split semantics are separately designed.

## Acceptance Criteria

- Julia validation no longer rejects standard-projection `NQPFull > 1`
  with `NSplitSize > 1` for ParaOpt direct SR.
- Julia still clearly rejects `NQPOptTrans > 1` / active `OptTrans` with
  `NSplitSize > 1`.
- Julia still rejects `NSRCG != 0 && NSplitSize > 1`.
- Sampling uses C-like QP split over `comm1`, including allreduce in
  `calculate_ip*` / `calculate_log_ip*` only for sampling-time calls.
- Initial Pfaffian retry and regeneration decisions are synchronized across
  `comm1`.
- Main calculation remains sample-split and full-QP per local sample, and
  all main-calculation `calculate_ip*` calls use no-reduce behavior.
- FSZ complex sampling no longer contains hard-coded full-range
  `calculate_log_ip_fcmp(..., 1, n_qp_full + 1, ...)` calls on paths that
  should use the local split range.
- MPI smokes pass the four-case matrix for at least one `NSPGaussLeg > 1`
  fixture and one `NMPTrans > 1` fixture.
- `NQPFull == 1 && NSplitSize > 1` remains covered and does not regress.
- Existing non-split tests continue to pass.
- Public docs and PR text do not expose local-only investigation paths.

## Self-Review

- Placeholder scan: no placeholder markers or open placeholders remain.
- Scope check: the design is limited to ParaOpt direct SR and does not mix
  in SR-CG, PhysCal, BackFlow, or independent-chain sampling.
- Consistency check: sampling is QP-split, main calculation is sample-split,
  matching the C behavior observed in the implementation.
- Collective check: sampling-time reductions are explicit, while
  main-calculation IP calls remain no-reduce to avoid double-counting.
- Ambiguity check: `NQPFull > 1` is explicitly split into standard
  projection (`NQPOptTrans == 1`) versus OptTrans-derived
  (`NQPOptTrans > 1`) cases.
- Testability check: acceptance criteria include validation, unit-level
  range behavior, MPI positive fixtures, and MPI negative validation.
