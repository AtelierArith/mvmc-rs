# Rust optimization milestone, Linux x86_64 (#496)

Periodic half-filled Hubbard chains, `t=1`, `U=4`, 300 SR steps, 300 total
saved configurations per step, MPI 4 ranks × 4 computation threads, BLAS 1.
One full warmup precedes three measured repetitions. Measurements use the
Ryzen 9 PRO 8945HS Dev Container, Rust 1.99, GCC 13.3, MPICH 4.2.0,
OpenBLAS 0.3.26 LP64. Julia optimization remains in progress and is not
claimed complete by this Rust milestone.

| Sites | Original Rust median (s) | Optimized Rust repetitions (s) | Optimized median (s) | Fresh C repetitions (s) | Fresh C median (s) |
|---|---:|---|---:|---|---:|
| 32 | 4.406248422 | 3.473796563, 3.481769835, 3.539394901 | 3.481769835 | 3.58350, 3.73056, 3.71526 | 3.71526 |
| 64 | 19.163782063 | 14.034864926, 14.131739858, 14.152117444 | 14.131739858 | 14.82200, 14.25877, 14.55564 | 14.55564 |

Rust elapsed time decreased about 21.0% and 26.3% relative to its original
baseline. Its median is about 6.3% and 2.9% below the fresh C controls.
The L64 margin is small and needs confirmation in the final comparison.
These are sequential batches on an otherwise idle machine, not interleaved
paired trials. Rust records the maximum elapsed time across ranks around
the warmed production runner, including parsing, initialization, sampling,
SR and output. C records its rank-zero internal All timer, after MPI_Init
and before teardown, using a fresh process for each run. Startup and Julia
JIT are excluded from runner comparisons; the timer boundaries differ.

The parallel real Transfer path previously used a general Green-function
helper that allocated scratch for each term. It now uses the existing real
fast-kernel arithmetic for eligible inputs, reuses output storage, and folds
term results in the original order. Generic/RBM paths retain their existing
algorithm. Accepted real rank-one updates use stack scratch through matrix
order 128 and runtime-selected AVX2 for independent rows on supported x86_64
CPUs. Separate multiply/subtract/add operations preserve the scalar arithmetic
order; there is no FMA or reduction reassociation. Other CPUs use the scalar
path, and larger matrices retain heap scratch.

An instrumented 20-step L64 profile attributes approximately 1.149 s to QP
factor/inverse dispatch and 0.736 s to accepted rank-one update dispatch over
its combined warmup/observation/measurement runs. Transfer dispatch falls to
approximately 0.130 s. These accumulated dispatch times cover multiple runs
and are diagnostic; they are not percentages of a single primary timing.
Instrumentation is disabled in the timing table.

The numerical and RNG checks are described in [README.md](README.md).
Raw local execution evidence is retained under
`/home/vscode/.cache/mvmc/issue496-profile/rust-avx2/`,
`/home/vscode/.cache/mvmc/issue496-profile/rust-rng-audit/`,
`/home/vscode/.cache/mvmc/issue496-profile/c-rng-audit/`, and
`/home/vscode/.cache/mvmc/issue496-c-control/` in the container named-volume
cache. Portable JSON summaries and provenance are checked in alongside this
report. The timed executable SHA-256 was
`89bc67682cf690406d2fac63ccb6deabaa37ac580187b10f3d72591666d78d26`;
subsequent changes affected the test assertion policy and diagnostic examples.
