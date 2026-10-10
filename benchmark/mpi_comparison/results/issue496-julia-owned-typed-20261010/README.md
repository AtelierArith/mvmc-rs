# Julia owned SR buffers and typed parameter loops (#496)

Linux x86_64 Dev Container, Ryzen 9 PRO 8945HS, Julia 1.13.1, MPICH4.2.
LinearAlgebra uses OpenBLAS0.3.30 ILP64; native PfaPack links
OpenBLAS0.3.26 LP64. The project uses the pinned
`benchmark/julia_comparison/Manifest-v1.13.toml`. Julia source
`b672e05587cf9d4cd72f567c3ac43b8a4542e1e6`, submitted in upstream
[julia-patch PR #75](https://github.com/tmisawa/Julia-mVMC/pull/75).
This report preserves the validated milestone; it does not establish the
full goal of beating fresh C at both sizes.

Periodic half-filled Hubbard chain, t=1, U=4, Lsub=4, NSPGaussLeg=8,
NMPTrans=-1 (normalized to 1), NQPOptTrans=1. There are eight QP planes:
sampling splits them into two per MPI rank, while measurement evaluates all
eight for each rank's local configurations. MPI4, four compute threads per
rank, BLAS1, Opt300, 300 saved configurations per step (75/rank), averaging
window300. One full warmup, three repetitions, sequential isolated timing.

| Sites | Repetitions (s) | Julia median (s) |
|---|---|---:|
| 32 | 3.553393927, 3.507448726, 3.507301127 | 3.507448726 |
| 64 | 13.339505944, 13.403614018, 13.344671638 | 13.344671638 |

These are maximum-rank warmed production API times, including parsing,
initialization, sampling, SR and output. Startup, JIT, builds and the separate
worker observation are excluded. No process-local method replacement or
experimental timer is installed in these runs. C's internal rank-zero All
timer has a different boundary; a fresh alternating C comparison is required
before interpreting the remaining gap.

The production change aliases uniquely owned SR aggregates and stores rather
than clearing, swapping and merging a second aggregate buffer. Derivative
scratch remains independent. Ownership changes or replaced public buffers
rebuild the private accumulator; zero-plus-local publication is retained.
Typed inner parameter helpers preserve family iteration, duplicate assignment
order, declared slots and parameter values.

`fullsuite.log` in `raw-evidence.tar.gz` records 41,306/41,306 passing tests with unchanged numerical
budgets. `audit-summary.json` records all 16 L32/L64 ×20/300-step ×4-rank
cases matching the prior reviewed QPv9 reference for SFMT624, index, consumed
word count, reseeds, saved/temporary/burn-in configurations and counters.
Source hashes, the patch from the preceding head and raw primary timing logs
are retained here. `base-commit.txt` names that preceding head; the candidate
commit is the b672e05 identifier above.

Full outputs and untimed worker profiles remain in the container cache
`/home/vscode/.cache/mvmc/issue496-julia-owned-typed-production/`.
Timing and final energy alone do not establish C trajectory parity.

## Fresh alternating C comparison

Three alternating batches per size completed successfully; see `fresh-c-pair/summary.json`, raw logs, source/binary hashes, and output observations.

| Sites | C median (s) | Julia median (s) | Julia/C |
|---|---:|---:|---:|
| 32 | 3.426570 | 3.581715848 | 1.045 |
| 64 | 13.505080 | 13.550214897 | 1.003 |

This paired comparison does not demonstrate Julia beating C. It motivates further Slater update optimization. These runs precede the later joint Opt/PhysCal report; keep their distinct measurement identifiers. Upstream PR #75 merged as `7b7f63a3548a489fddecd17f9ec92dee085fe9cf`, whose tree is identical to the tested b672 commit.

`raw-evidence.tar.gz` retains unchanged suite/worker/timing logs, C timers, and source.diff. Extract with `tar -xzf raw-evidence.tar.gz` in a separate directory.
