# L128 interval Opt development measurements, 2026-10-12

The finalized release candidate takes **25.742293 s** (median of three calls
after the first call). Measurement and SR still run on the CPU; the Julia
performance target and C proposal parity remain open.

Related to #550, #552 and #555. These measurements track the experimental
`cuda-interval` multiwalker path. The public single-chain Rust/Julia GPU
comparison is recorded separately in [../L128-opt-20261012](../L128-opt-20261012/README.md):
47.8588 s versus 5.12493 s. The overall performance target remains open.

## Workload

RTX 3060 (GPU 1), real FP64 Hubbard L128, 32 QP planes, 64 walkers,
five retained configurations per walker per SR step, 20 SR steps, final
averaging window one, seed 1. Warmup and interval are each 128 proposals.
The total retained count is 320 per step. The interval sampler runs on CUDA;
measurement and SR use the C-order CPU implementations, one BLAS thread per
call. The existing interval proposal contract uses five SFMT words per hop
and empty-destination selection. It is distinct from the C rejection-draw
contract; preservation of this development baseline is not C trajectory parity.

## Whole-run progression

The following use the ignored `interval_group_opt_hubbard_l128_timing` test
(test profile, opt-level 2). Timing includes the public multichain optimizer
and ordinary output; diagnostic receipt serialization happens afterward.
Each row is one observation, including device initialization for that process.

| Candidate | Seconds | Notes |
| --- | ---: | --- |
| PR #549 source, observational harness only | 348.086 | Peak RSS 24,626,468 KiB |
| Reduce contributions once + grouped QP kernel + batch readback | 43.816 | Peak RSS 5,638,476 KiB |
| Root-only model-table staging + batched device initialization | 41.047 | Device initialization succeeds |
| Same candidate, bulk real Slater shadow copy | 27.196 | 1.51 times faster than preceding row |

The last isolated change borrows the real Slater table once per copy. The
previous nested loop called `SlaterElmFlat::set` for every entry, advancing
a single process-wide atomic version each time. At 64 walkers, 32 QPs and
L128, this means 134,217,728 contended atomic updates per SR step. A single
mutable slice borrow keeps cache invalidation valid and copies the same
real components in the same flat layout without numerical arithmetic.

A separate release worker runs four complete optimizations in one process.
The first takes 28.910559 s; subsequent calls take 28.294159, 28.575973 and
28.587529 s (median 28.575973 s). These calls create a new walker thread team,
so their thread-local CUDA initialization is still included. They use the
same public rank runner as the multichain optimizer. They should not be
reported as a same-trajectory comparison to Julia, which uses 320 XORWOW
walkers, seed 79 and GPU measurement/SR in the public comparison.

## Numerical validation

Against the frozen PR #549 baseline, every one of 64 walkers has exactly
matching final SFMT state (624 words and index), total words consumed,
electron configuration, occupation, projection counts and counters. All
20 steps' energies and variational parameters, plus all numerical entries
of `zvo_out.dat`, `zvo_var.dat` and `zqp_opt.dat`, are checked with explicit
`atol=rtol=1e-12`; the observed maximum difference is zero in every field.
This is a within-Rust optimization check on an identical control path.

Focused actual-CUDA gates cover the independent Julia shared tape, CPU
Pfaffian rebuilds on valid skew inputs, non-power-of-two QP counts and
matrix sizes through 256, batched initialization, root staging order,
transactional failed rebuilds, and upload/readback identity. The coordinator
error test verifies all peers return and can enter another pass. A separate
test switches across both visible CUDA devices and checks the actual selected
device ID, then verifies an unavailable ordinal returns an error.

The older synthetic interval gate and microbenchmarks initialized arbitrary
non-skew matrices. Their generators are corrected under #555. Historical
microbenchmarks on those tables are mechanism measurements only; use this
physical Hubbard workload for E2E claims.

## Reproduction

Use the `gpu/mvmc-gpu-cuda` standalone workspace. Copy the repository's
`benchmark/hubbard_chain/inputs/hubbard_chain_L128` input and set
`NVMCWarmUp=1`, `NVMCInterval=1`, `NVMCSample=5`, `NSROptItrStep=20`,
`NSROptItrSmp=1`. Keep the same QP/model files. Set `CUDA_VISIBLE_DEVICES=1`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and library paths for the
recorded CUDA toolkit (see the public comparison's environment report).

```sh
MVMC_INTERVAL_E2E_WALKERS=64 MVMC_INTERVAL_E2E_STEPS=20 \
MVMC_INTERVAL_E2E_SMP=1 MVMC_INTERVAL_E2E_NAMELIST=/absolute/input/namelist.def \
MVMC_INTERVAL_E2E_OUTPUT=/absolute/output \
MVMC_INTERVAL_E2E_RECEIPT=/absolute/receipt.jsonl \
timeout 1800 cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --locked --test interval_e2e_l128 --run-ignored only --test-threads 1 \
  --no-fail-fast --retries 0 --success-output immediate
```

For the release worker, build `--release --example opt_timing_worker`, set
`MVMC_OPT_TIMING_SEED=1`, `MVMC_RS_SAMPLER_BACKEND=cuda-interval`, and
`MVMC_RS_{SR,MEASURE,MEASURE_PF}_BACKEND=c-order`. Pass the namelist,
output directory, and `64` as arguments. Feed `0`, `1`, `2`, `3`, `quit` on
separate stdin lines. `MVMC_C_TIMER=1` enables the C stage timers; stage
profiling is recorded separately from the four-call run above.

## Remaining time

One separate release call with `MVMC_C_TIMER=1` takes 28.830848 s:

| Stage | Seconds |
| --- | ---: |
| Sampling | 11.48999 |
| Initialization within sampling (nested) | 3.69290 |
| Observables | 5.06428 |
| Collective reduction and weighted averaging | 7.69059 |
| SR | 1.16123 |
| Slater update | 0.97871 |

The C timer's initialization label includes shadow copying, configuration
preparation, GPU allocation/compilation and batched factorization. It is not
a CPU-factorization-only timer. The remaining collective copies, measurement
and device interval arithmetic require further work to approach Julia's time.

The source hash manifest identifies the production files of this measured
candidate on `perf/550-cuda-interval`, based on PR #549 commit `48e04e89`,
with core collective commit `10418a28` and the subsequent source changes.
Raw logs, the complete timer and receipt hashes accompany this report.

A subsequent candidate moves collective input copies to per-rank locks and
output copies to a shared read lock. The reduction retains the same rank
order; a final barrier protects both generations. One test-profile E2E call
takes 25.981 s, versus 27.196 s above. Its complete receipt/output comparison
also has zero observed difference (`concurrent-copy-validation.json`). This
single observation is complemented by the final release repetitions below. Production C
proposal/draw parity is tracked explicitly in #557.

The finalized production candidate is commit
`d2f73ee9ebe796e8244f0d6afea825ca58788bc5` (including the concurrent-copy
change); the timing worker is commit `e29343c6`.
`concurrent-copy-source-sha256.txt` records its file hashes. Full workspace
nextest: 1,609 passed, 63 skipped; workspace doctests, root/CUDA fmt and
Clippy pass. Both corrected skew-input microbenchmarks complete on CUDA;
`skew-bench-validation.log` is a correctness run made during a CPU build,
so its timings are not used as performance evidence.

This Rust walker workload makes 983,040 proposals (49,152 per SR step).
The public Julia comparison makes 860,160 proposals and uses a different
walker topology/RNG. Equal retained samples are not equal trajectories or
effective statistical sample sizes.

### Final release repetition

The finalized candidate (`d2f73ee9`, timing worker `e29343c6`) takes
26.239604 s on its first call, then **25.516081, 25.767082 and 25.742293 s**
(median **25.742293 s**). C stage timing is enabled; all four calls create
fresh walker teams. All numerical output entries across all 20 steps of
all four calls match the frozen parent's outputs within `atol=rtol=1e-12`,
with observed maximum difference zero. This is separate from the exact RNG
and configuration receipt check of the test-profile E2E run.

Raw logs, per-call timers and `concurrent-copy-release-summary.json` record
this run. No local build, test or second benchmark ran concurrently. The
previous release candidate with serialized collective copies measured
28.575973 s median; these sequential cohorts are development observations,
not an interleaved A/B experiment. The Julia comparison and its caveats above
remain unchanged.
