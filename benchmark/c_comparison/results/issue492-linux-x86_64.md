# Native C parity and Rust inner-thread improvement (#492)

Linux x86_64 Dev Container, AMD Ryzen 9 PRO 8945HS, 2026-10-10.
Related to [#492](https://github.com/AtelierArith/mvmc-rs/issues/492).

The hoisted cold-region gate was suppressing profitable L32 Pfaffian and
measurement work. Changing its estimated-work threshold from 400 to 40 us
reduces four-thread L32 optimization and PhysCal times by about 26% each.
An independent instrumented run observes all four `mvmc-inner` workers in
both workloads, rather than inferring utilization from configured capacity.
Small regions retain the work gate, and outside-pool thresholds are unchanged.
This gate is calibrated on this Ryzen host; different dispatch costs can favor
different thresholds. No claim of a speedup on unmeasured hardware is made.

## Timing

Each implementation/configuration has one discarded warm-up and three measured
repetitions. Rust and native C are interleaved within each workload. The baseline
round precedes the final round; both rounds include C as a control. C is the
unmodified authoritative executable, one MPICH rank; BLAS/BLIS/MKL use one thread.
Values are median internal `[0] All` seconds, including initialization and
computation, excluding MPI launch. Probe/worker-observation times are excluded.

| Workload | Threads | Rust before | Rust after | C in final round |
|---|---:|---:|---:|---:|
| Hubbard optimization L16 | 1 | 0.5160 | 0.5198 | 0.6977 |
| Hubbard optimization L16 | 4 | 0.5230 | 0.4774 | 0.7523 |
| Hubbard optimization L32 | 1 | 1.5056 | 1.5373 | 1.8037 |
| Hubbard optimization L32 | 4 | 1.5062 | 1.1114 | 1.0554 |
| Hubbard optimization L64 | 1 | 2.4967 | 2.5346 | 2.8580 |
| Hubbard optimization L64 | 4 | 1.1812 | 1.1834 | 1.0957 |
| Hubbard PhysCal L32 | 1 | 4.2830 | 4.3105 | 6.6257 |
| Hubbard PhysCal L32 | 4 | 4.2910 | 3.1688 | 3.2491 |

The L32 four-thread speedups over the old Rust are 1.36x and 1.35x. Relative
to final Rust at one thread they are 1.38x and 1.36x. C remains about 5% faster
for L32 optimization and 7% for L64 optimization; Rust is about 3% faster for
L32 PhysCal, a small difference that needs cautious interpretation. L64 changes
by only 0.2%. The L16 C control varied appreciably between rounds (0.58 to 0.75 s
at four threads), so its cross-round ratios do not establish a reliable speedup.
CPU frequency, affinity and other host activity are not controlled.

Raw internal/wall timers and host loads: [baseline](issue492-baseline-runs.json),
[final](issue492-final-runs.json). Compiler/BLAS/MPI versions, linked libraries,
binary hashes and dirty-tree state: [baseline environment](issue492-baseline-environment.json),
[final environment](issue492-final-environment.json). The baseline Rust binary
was recovered from `git archive 72ca4e99` in an isolated target and its SHA-256
matches the original measured binary. Actual worker-entry snapshots are in
[the separate observation run](issue492-workers.json).

## Numerical comparison

All four full benchmark workloads, at one and four threads, pass the existing
per-output budgets in both rounds, including warm-ups: 32 C/Rust comparisons
per round. Parameter/SR checkpoints use `1e-11` absolute plus relative bounds,
energies `1e-12`, Green operators `1e-13`. Missing files, shapes, discrete indices
and nonfinite values fail independently. The [baseline](issue492-baseline-numerical.json)
and [final](issue492-final-numerical.json) retain every result and its bounds.
Computed floating-point results are never required to match bitwise.

A broader first-step comparison keeps original sample counts and one PhysCal
data set across all 13 workloads at one and four threads, with one warm-up and
one measured repetition. Eleven workloads pass. Real and complex Heisenberg
parameter files exceed the short-prefix budget; these remain explicit failures,
and the runner returns exit status 1. No tolerances were increased. See
[all 52 prefix comparisons](issue492-prefix-numerical.json) and
[versions/source hashes](issue492-prefix-environment.json) and
[input hashes and runs](issue492-prefix-runs.json).

Read-only full C/Rust captures locate these discrepancies at floating-point SR
operands, with matching primitive RNG consumption, next 624 RNG words, saved
electron configurations and common acceptance/rejection counters. Active SR
component maps match exactly. This verifies identical control paths through
the captured first sampling step, not every later optimization trajectory.

| First-step system | Active components | Max S difference | Condition number, infinity norm | Rust actual solve backward error |
|---|---:|---:|---:|---:|
| Heisenberg real | 10 | 4.16e-17 | 3.84e7 | 2.09e-17 |
| Heisenberg complex | 20 | 4.11e-15 | 2.50e7 | 3.23e-17 |
| Hubbard RBM OptTrans | 57 | 2.05e-15 | 9.39e4 | 1.07e-17 |

Independent solves of the separately captured C/Rust systems produce maximum
solution differences of `2.16e-10`, `6.78e-10` and `7.77e-14` respectively.
The native Heisenberg parameter-file differences are about `1.77e-10` and
`6.79e-10`. The large condition numbers explain this amplification while the
actual Rust solves have small backward errors. These are measured errors and
conditioning evidence, not a blanket acceptance of all parameter differences.
The independent audit uses pinned NumPy 2.3.4; its own BLAS configuration is
recorded separately from the application OpenBLAS/BLIS environment.

Operand/residual records: [real](issue492-heisenberg-real-system.json),
[complex](issue492-heisenberg-complex-system.json), [RBM](issue492-rbm-system.json).
Exact state checks: [real](issue492-heisenberg-real-checks.json),
[complex](issue492-heisenberg-complex-checks.json), [RBM](issue492-rbm-checks.json).
C's complex Hermitian moment dump has the opposite storage orientation;
conjugation is documented in the capture command. The unfactored real direct
SR matrix is compared without that transformation.

## Correctness changes and reproduction

The initial historical FSZ timing auto-loaded neighboring `initial.def` only
on Rust. The maintained runner explicitly disables that convenience when C has
no positional parameter file; the corrected FSZ prefix comparison passes.
Its old timing must not be described as equivalent-work performance.

RBM OptTrans had two numerical defects: derivatives used Julia's doubled
real/imaginary layout instead of C's consecutive sector writes, and SR selection
remapped native OptFlag writes into the parameter tail. Correcting both makes
the active map match C (57 components rather than 59), and reduces normalized
operand differences to about `1e-15`. Native unwritten tail flags are explicitly
inactive: C allocates these entries with malloc and does not write them, so
their allocator bytes are not a portable numerical contract. Rust no longer
inherits active Julia reservation defaults for them. Historical Julia fixtures
remain explicitly separate from the native input path.

Run the maintained commands in [the benchmark README](../README.md). Optional
full executable instrumentation is in `c_toolbox/native_comparison_492`; it is
never used by Cargo builds or tests. The independent checked-in C derivative,
SR-system and exact RNG/configuration fixtures are documented under
`tests/fixtures/native_comparison_492` with source, binary and patch provenance.

Final Linux validation: 1,584 workspace unit/integration tests passed, 61
optional tests skipped; workspace doctests, Clippy with warnings denied,
formatting and six Python comparator tests passed. Native macOS was not
available in this environment and was not verified. No vendored C source was
edited, and the Rust runtime remains independent of the C toolbox.
