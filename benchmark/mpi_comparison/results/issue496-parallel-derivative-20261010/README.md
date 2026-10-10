# Parallel real Slater derivative confirmation (#496)

Linux x86_64 Ryzen 9 PRO 8945HS Dev Container; periodic half-filled Hubbard
chain, t=1, U=4, Lsub=4, NSPGaussLeg=8. MPI 4 ranks, 4 compute threads per
rank, BLAS 1 thread. Each optimization has 300 steps and 300 total saved
configurations per step (75 per rank). Heavy workloads ran separately.

| Sites | C median (s) | Rust median (s) | Rust time reduction |
|---|---:|---:|---:|
| 32 | 3.496530 | 3.401366 | 2.72% |
| 64 | 13.756170 | 13.594021 | 1.18% |

Three C/Rust batches alternate execution order. C receives an initial discarded
warmup, then each measured run uses a fresh process. Rust receives a full
300-step warmup before each measured run; its separate 20-step execution
observer is outside measured repetitions. All three corresponding Rust runs
were faster than C at each size in this batch. The L64 margin is small; this
does not establish a robust advantage across independent batches.

C reports its rank-zero internal All timer, starting after MPI_Init; Rust
reports the synchronized maximum rank time for the warmed production API,
including parsing, initialization, sampling, SR and output. Startup/JIT is
excluded. This lifecycle difference remains relevant when comparing times.
Providers remain C/Rust OpenBLAS 0.3.26 LP64. C executable hash, Rust executable
hash, source patch, environment and original Git HEAD are in provenance.json.

Rust uses the first optimization milestone from PR #498 plus parallelization
of independent real Slater derivative QP planes. Each plane's scatter-addition
order and the final ascending QP reduction remain unchanged. C distributes
these planes in SlaterElmDiff_fcmp (src/mVMC/slater.c). No draws are added.

All three Rust output files at both sizes have 300 rows and six finite columns;
their observed maximum absolute difference against the original Rust output
is zero. C output differences are recorded as observations, with no tolerance
chosen here; they do not substitute for locating numerical divergences.
Separate 20/300-step four-rank audits match the previously validated Rust
candidate exactly for all 624 RNG words, index, consumed-word count, electron
configuration/index/occupancy/spin, projector counts and acceptance counters.
Root final energies also have observed absolute difference zero.

Julia is not rerun in this batch. The consistent persistent-scheduler
experiment was rejected after full-workload regressions, preserving the
previous QPv5 candidate. See the sibling current-comparison report for its
separate measurement batch; do not interpret those values as simultaneous
three-way measurements for this candidate.
