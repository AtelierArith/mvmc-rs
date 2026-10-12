# L128 Opt with the C-compatible resident CUDA sampler

The release worker takes **26.205515 s**, the median of three calls after its
first call. The numerical/RNG correction in #557 is verified; the Julia E2E
performance goal in #550 remains open. Sampling runs on CUDA; measurement and
SR still use the CPU.

Owner direction after this measurement permits Rust GPU parity with Julia GPU
and prioritizes avoiding host/device communication. This report archives the
validated C-stream experiment; C draw parity is not a prerequisite for the next
GPU performance implementation. The next path uses Julia GPU as its reference.

Production and harness source: `8219d290845a3cc6df87f3700ae223353fdb743a`.
This follows the experimental interval measurements in
[../L128-interval-20261012](../L128-interval-20261012/README.md). Those older
measurements use Julia's five-word proposal tape and a different trajectory;
their 25.742293 s median is not an identical-work before/after comparison.

## Workload and timing

Same RTX 3060 (physical GPU 1) and Linux host as the
[public Rust/Julia comparison](../L128-opt-20261012/README.md): real FP64 Hubbard
L128, 32 QP planes, 64 independent SFMT walkers, seed 1, 20 SR steps, five retained
configurations per walker per step (320 total), 128 proposals per warmup/interval,
and one final SR averaging step. There are 983,040 proposals in the complete run.
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Input and production/binary hashes
are recorded here. The CUDA/OpenBLAS environment is the same as
[the public environment record](../L128-opt-20261012/environment.json).

Each timed call constructs a new walker team, so thread-local CUDA initialization
is included. The worker keeps its process alive. C stage timers and interval
instrumentation are enabled in all four calls. No other local build, test or
benchmark ran during this measurement.

| Call | Seconds |
| --- | ---: |
| First | 26.389008 |
| Warm 1 | 26.045286 |
| Warm 2 | 26.211788 |
| Warm 3 | 26.205515 |
| Warm median | **26.205515** |

Independent warm medians from the root's C timers:

| Stage | Seconds |
| --- | ---: |
| Sampling | 11.75264 |
| Initialization within sampling (nested) | 3.50475 |
| Observables | 4.56314 |
| Collective reduction and weighted averaging | 5.84263 |
| SR | 1.33251 |
| Slater update | 0.83116 |

These root timers include synchronization waits; they do not isolate kernel-only
costs or independently establish which rank is the slowest. Their medians are
not additive. The worker wall time includes ordinary output and runner overhead.

The published Julia GPU target is 5.124929 s (320 XORWOW walkers, seed 79,
GPU measurement/SR, 860,160 proposals). The public single-chain Rust GPU baseline
is 47.858827 s. The present prototype has a different walker topology and proposal
count, so it is not a controlled speedup against either public workload. Equal
retained sample counts also do not imply equal effective statistical samples.

## Numerical validation

The same L128 input, seed and 64-walker topology were run through the C-order Rust
CPU sampler and the new CUDA sampler. Both test-profile runs complete 20 steps.
Their single-observation wall times are 24.114 s and 25.227 s, respectively; these
are validation timings, separate from the release repetitions above.

All 64 final SFMT states (624 words plus index), consumed-word counts, electron
configurations, occupancies, projection counts and counters match exactly. Final
retained optimization state and every number in all 20 steps' `zvo_out.dat` and
`zvo_var.dat`, plus `zqp_opt.dat`, are compared with `atol=rtol=1e-12`. The observed
maximum difference is **zero** in every field. The four release runs also match
all of those numerical output files, with maximum difference zero.

The independent [extracted C probe](../../../../c_toolbox/interval_c_rng_557/README.md)
checks the draw contract at occupied-destination retries, the retry cap,
Metropolis rejection, partial-proposal refill and accepted-window rebuilds.
This is a kernel reference, not a full C executable Opt comparison. The CPU
candidate's separate retry-cap discrepancy is tracked in #559; the controlled
CUDA fixture follows C directly and the L128 observed path does not encounter it.

`verify.py` reads the archived receipts and outputs without rerunning CUDA:

```sh
uv run --no-project python benchmark/host_gpu_comparison/results/L128-c-stream-20261013/verify.py
```

Full workspace: 1,609 tests passed, 63 skipped. Workspace doctests, root/CUDA fmt
and Clippy pass. Actual-CUDA gates cover C boundary fixtures, mixed-walker refill,
20-step host/GPU/repeat Opt, the historical independent Julia tape, full occupancy,
QP tiling, transfer identity, device ordinal and failed-rebuild transactionality.
The first full-occupancy test setup had a zero consecutive pivot in the C initializer;
its corrected nonsingular skew fixture passes. Both the initial log and corrected
run are retained, along with the final complete CUDA gate.

## Reproduction

Copy the physical L128 inputs from `benchmark/hubbard_chain/inputs/hubbard_chain_L128`
and replace `modpara.def` with this directory's version. Check `input-sha256.txt`.
Use the recorded CUDA library path, `CUDA_VISIBLE_DEVICES=1`, and one BLAS/OpenMP
thread. Build:

```sh
cargo build --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --release --example opt_timing_worker
```

Run the worker with the absolute namelist path, an output directory, and `64`.
Set `MVMC_OPT_TIMING_SEED=1`, `MVMC_RS_SAMPLER_BACKEND=cuda-interval`,
`MVMC_RS_SR_BACKEND=c-order`, `MVMC_RS_MEASURE_BACKEND=c-order`,
`MVMC_RS_MEASURE_PF_BACKEND=c-order`, `MVMC_C_TIMER=1`, and
`MVMC_INTERVAL_TIMING=1`. Feed `0`, `1`, `2`, `3`, `quit` on separate stdin lines.

For the exact receipt comparison, use the ignored `interval_e2e_l128` test with
`MVMC_INTERVAL_E2E_WALKERS=64`, `_STEPS=20`, `_SMP=1`, `_NAMELIST`, `_OUTPUT`, and
`_RECEIPT` set to the desired absolute paths. Set `_BACKEND=c-order` for the CPU
reference and `_BACKEND=cuda-interval` for the CUDA candidate. The source of the
same test records full final RNG/configuration receipts after timing.
