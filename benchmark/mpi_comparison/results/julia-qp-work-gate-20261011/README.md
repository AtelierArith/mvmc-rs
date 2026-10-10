# Julia real-QP partial-pool dispatch

Related to [issue #496](https://github.com/AtelierArith/mvmc-rs/issues/496).
Implementation: [Julia PR #78](https://github.com/tmisawa/Julia-mVMC/pull/78).

The ordinary real inverse routine previously selected serial execution when
there were fewer QPs than pool threads. Permit independent QP matrices to use
the available workers only when QP_count * matrix_size^3 >= 32768 * pool_threads.
This calibrated gate retains serial execution for small partial teams whose
dispatch cost caused a measured PhysCal regression in an earlier prototype.
Retain the caller's work-size gate, private scratch,
static scheduling, nested/worker fallbacks, serial reductions and RNG calls.

Half-filled periodic Hubbard chains, t=1, U=4, Lsub=4, eight QPs. Opt uses 300
steps; PhysCal uses 100 groups with fixed C-generated parameters shared between
variants. Both use 320 total configurations per step/group, NSplitSize=1:
320 samples per rank at one rank, 80 at four ranks. Three alternating pairs
after fully warming both variants; medians in seconds, smaller is faster.

| Workload | Ranks | Threads/rank | Sites | Before (s) | After (s) | Change |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 1 | 16 | 32 | 11.691 | 11.596 | -0.8% |
| Opt | 1 | 16 | 64 | 57.083 | 36.594 | -35.9% |
| Opt | 4 | 4 | 32 | 3.434 | 3.489 | +1.6% |
| Opt | 4 | 4 | 64 | 13.232 | 13.170 | -0.5% |
| PhysCal | 1 | 16 | 32 | 4.547 | 4.620 | +1.6% |
| PhysCal | 1 | 16 | 64 | 22.459 | 16.947 | -24.5% |
| PhysCal | 4 | 4 | 32 | 1.415 | 1.386 | -2.1% |
| PhysCal | 4 | 4 | 64 | 5.203 | 5.370 | +3.2% |

These are within-Julia paired comparisons, not new C-versus-Julia-versus-Rust
measurements. The maximum-rank production API duration includes parsing,
initialization, computation, outputs and completion barrier; startup, compilation
and warmups are excluded. A typed Ref selects the original or candidate guard
before each run. No profiler runs during primary timing. Small changes in the
four-rank control cases should not be interpreted as established speedups.

Linux x86_64 Dev Container, Ryzen 9 PRO 8945HS (8 physical cores / 16 logical
CPUs), Julia 1.13.1, MPI.jl 0.20.27, MPICH 4.2.0. Julia OpenBLAS 0.3.30 ILP64
and native helper OpenBLAS 0.3.26 LP64 each use one thread. CPU-heavy builds,
tests and other benchmarks do not overlap timed comparisons.

Baseline is `85f9ea12c49e46639b8284334388116d5f16a337`, identical in tracked tree
to merged upstream `d493f113ecc009d44e70327ffcbb500a566a1380` (SR-store PR #77).
Candidate is `7a8e1fa65a12d20507417509c00b1b1e5b8202e2`. Source/kernel/input
hashes and patches are preserved beside the logs. Earlier ungated prototype
measurements are separate historical evidence and are not final-source results.
Historical fixtures and timing reports are not relabelled or regenerated.

PR #78 merged after all seven upstream checks passed. The reference submodule
is pinned to upstream main `62416e30a04e33a6d3641351fa8487dd640473cb`, whose
tracked tree is identical to the measured candidate (`d96bbb49ce077123307fa6b5bcaa81996dd46f90`).

Full unit tests passed 45,350 optimizer assertions at sixteen threads, plus
base and Slater suites. Focused tests passed at one and four threads, including
independent inverse residuals, zero/uneven QP counts, nested callers and integer
pivot sentinels proving actual worker execution below pool capacity.

Twenty-step L32/L64 audits at 1×16 and 300-step audits at 1×16 and 4×4 used the
actual baseline/candidate sources without the timing selector. All six cases
and twelve paired ranks matched exact native SFMT words/index/draw/reseed counts,
integer configurations and acceptance counters. Floating outputs use explicit
absolute/relative bounds from the repository numerical-comparison policy, not
bitwise comparison. JSON summaries record the observed maximum differences.

Raw timings, energy outputs, test logs and optional oracle audits accompany this
report. `audit-evidence/uncompressed-sha256.json` identifies the original bytes
of compressed audit evidence. The scripts retain the original Linux container
paths; the paired Julia harnesses can also be invoked with explicit project and
input paths as documented in the upstream PR report.
