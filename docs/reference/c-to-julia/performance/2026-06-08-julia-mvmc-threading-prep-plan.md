---
date: 2026-06-08
datetime: 2026-06-08 13:51 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC v0.3 threading preparation, 1-thread gate, and implementation review points
---

# Julia-mVMC threading prep plan

## 1. Baseline confirmed before threading

Repository state:

- `Julia-mVMC` branch: `develop`
- HEAD: `89745c5`
- Worktree: clean
- Julia: `julia version 1.12.1`
- Julia threads: default `Threads.nthreads() == 1`; explicit `JULIA_NUM_THREADS=1` also reports `1`
- C/OpenMP guard for local tests: `OMP_NUM_THREADS=1`

Baseline commands and results:

| Gate | Command shape | Result |
| --- | --- | --- |
| `SFMT.jl` | `OMP_NUM_THREADS=1 julia --project=@. -e 'using Pkg; Pkg.test()'` | 27/27 pass |
| `PfaPack.jl` | same | 5699/5699 pass |
| `MVMCExpertModeParsers.jl` | same | pass; `Read Input Parameters` 71/71, `Orbital and QPTrans` 118/118 |
| `MVMCOptimizers.jl` | same | `Unit Tests` 399/399 pass, `Slater Update Tests` 25/25 pass |
| root integration | `OMP_NUM_THREADS=1 julia --project=@. test/integration/runtests.jl` | 120/120 C-reference pass + helper/runner contract pass |
| PhysCal e2e | `OMP_NUM_THREADS=1 julia --project=@. test/integration/phys_cal_equivalent.jl` | 26/26 pass |
| ctest-equivalent | `OMP_NUM_THREADS=1 julia --project=@. test/integration/ctest_equivalent.jl` | 61/61 pass |

This is the baseline to preserve with `JULIA_NUM_THREADS=1`.

## 2. C threading shape vs Julia mutation points

### C-mVMC reference shape

C-mVMC does not simply OpenMP-parallelize the whole Markov chain over saved
samples. In `VMCMakeSample*`, the Markov chain state (`TmpEleIdx`,
`TmpEleCfg`, `TmpEleNum`, `TmpEleProjCnt`, `InvM`, `PfM`, `BurnFlag`,
`Counter`) is updated sequentially, with OpenMP mostly inside array loops and
kernel loops. `VMCMainCal` splits samples across MPI ranks via `SplitLoop`,
while OpenMP is again used inside expensive loops such as Hamiltonian,
Green-function, Slater, Pfaffian/update, and averaging kernels.

Implication: sample-level threading is safe first for `VMCMainCal` after
sampling has produced saved configurations. Sample-level threading of
`VMCMakeSample` changes the Markov chain/RNG contract unless redesigned as
independent chains and accepted as a new numerical mode.

### Julia mutation points

`vmc_para_opt!` step order:

1. `[20]` update Slater elements and QP weights.
2. `[3]` `vmc_make_sample*`.
3. `[4]` `vmc_main_cal*`.
4. `[21]` `weight_average_we!`, `[25]` SR average, `reduce_counter!`.
5. `[22]` output, `[5]` stochastic opt, `[23]` parameter sync.

`vmc_make_sample*` currently mutates shared state directly:

- single temporary electron configuration arrays in `state.electron_config.tmp_*`
- single burn sample arrays and `counter[11]` burn flag
- single `state.workspace` buffers (`proj_cnt_new`, `pf_m_new*`)
- global `state.slater_matrix.inv_m*` / `pf_m*` updated during the chain
- saved sample arrays (`ele_idx`, `ele_cfg`, `ele_num`, `ele_proj_cnt`, `ele_spn`)
- sampling counters
- one RNG stream
- one shared `CTimer`

`vmc_main_cal*` currently mutates shared accumulators directly:

- `state.energy.{wc, etot, etot2, sztot, sztot2}`
- `state.sr_opt.sr_opt_o` as a per-sample scratch buffer
- `state.sr_opt.sr_opt_oo*`, `state.sr_opt.sr_opt_ho*`, `state.sr_opt_o_store*`
- `state.phys_quantities.local_*` scratch and `phys_*` accumulators
- `state.slater_matrix.inv_m*` / `pf_m*` during each sample's `calculate_m_all*`
- one shared `CTimer`

The existing `ThreadedPfaPackWorkspace` only covers Pfaffian local buffers
inside `calculate_m_all`. It does not make `VMCOptimizationState` safe for
sample-level threading.

## 3. 1-thread reproducibility gate

Every threading PR must first pass an explicit 1-thread gate:

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
for pkg in SFMT.jl PfaPack.jl MVMCExpertModeParsers.jl MVMCOptimizers.jl; do
  (cd "$pkg" && JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. -e 'using Pkg; Pkg.test()')
done
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. test/integration/runtests.jl
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. test/integration/ctest_equivalent.jl
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. test/integration/phys_cal_equivalent.jl
```

Acceptance:

- Existing tolerances and row counts must remain unchanged.
- `JULIA_NUM_THREADS=1` must use the current code path or a strictly equivalent
  code path. It must not introduce new RNG splitting, new reduction order, or
  timer semantics.
- If a threading abstraction is added, tests must cover that the 1-thread
  workspace/reduction path writes byte-identical or tolerance-identical output
  to current `develop`.

## 4. Implementation outline and review points

### Phase T0: design-review PR, no behavior change

Add only abstractions and tests that are inactive by default:

- `VMCThreadConfig` or equivalent helper deciding effective thread count.
- Thread-local accumulator types for energy, SR, PhysCal, counters, and timer.
- Pure reduction helpers that merge local accumulators into `state`.
- Unit tests for empty, one-local, and multi-local deterministic reductions.

Review point R0:

- Confirm scope: first implementation parallelizes `VMCMainCal` sample reduction
  and safe inner loops, not `VMCMakeSample` Markov-chain generation.
- Confirm 1-thread gate command list and acceptance tolerances.
- Confirm timer policy: local timers are merged at the end; no shared `+=` in
  threaded regions.

### Phase T1: 1-thread refactor of main-cal accumulators

Refactor `vmc_main_cal!` and `vmc_main_cal_fsz!` so the sample loop writes to a
single local accumulator even when there is only one thread. Keep the loop
sequential.

Required tests:

- Existing full baseline gate.
- Unit tests for energy/SR/PhysCal accumulator equivalence.
- A minimal public-path test that `JULIA_NUM_THREADS=1` output matches current
  integration references.

Review point R1:

- Check for accidental shared scratch use (`sr_opt_o`, `local_cis_ajs`,
  `inv_m/pf_m`) that would become a race in T2.
- Check reduction order is deterministic and matches current sequential order
  when there is only one local accumulator.

### Phase T2: parallel `VMCMainCal` sample loop

Split saved samples across Julia threads. Each thread gets:

- local Slater/Pfaffian work arrays or a safe per-thread state view
- local `SROptO` scratch
- local energy/SR/PhysCal accumulators
- local timer accumulator

After the threaded loop, merge locals in thread-id order.

Required tests:

- 1-thread baseline gate.
- `JULIA_NUM_THREADS=2` and `4` integration / ctest-equivalent / PhysCal gates.
- A numerical tolerance gate for expected floating-order differences in
  multi-thread reductions.

Review point R2:

- Race audit: no writes to shared `state.energy`, SR arrays, PhysCal arrays,
  `CTimer`, or sample-local scratch from inside the threaded loop.
- Numerical audit: multi-thread differences are bounded and explained by
  deterministic reduction order, not RNG or sample-set changes.

### Phase T3: safe inner-loop threading cleanups

Where C already uses OpenMP inside kernels, add or tighten Julia threading only
for loops with independent writes:

- Slater element updates and real/complex copy loops.
- Hamiltonian / Green-function term loops when local reduction buffers are clear.
- Weight-average loops after reduction.

Review point R3:

- Check each `Threads.@threads` loop has independent writes or explicit local
  reductions.
- Check nested threading does not oversubscribe badly with existing
  `calculate_m_all` threading.

### Phase T4: sampling-threading decision point

Only after T0-T3 should `VMCMakeSample` be reconsidered. There are two possible
paths:

- Keep sampling sequential and optimize kernels around it. This best preserves C
  behavior and RNG semantics.
- Add an explicit independent-chain mode. This requires new RNG stream rules,
  new acceptance criteria, and cannot be sold as 1-thread-identical C
  equivalence for multi-thread sample generation.

Review point R4:

- Decide whether independent-chain sampling belongs in v0.3 or should be
  deferred.
- If pursued, require a design review before implementation because it changes
  stochastic semantics more than T1-T3.

## 5. Recommended next step

Start with Phase T0 on a new branch, e.g. `feature/v0.3-threading-maincal`.
The first PR should contain only inactive infrastructure and reduction tests.
That gives a review boundary before any hot-loop behavior changes.
