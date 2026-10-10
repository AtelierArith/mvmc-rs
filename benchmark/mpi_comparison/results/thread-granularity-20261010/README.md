# Thread dispatch investigation, Linux x86_64 (2026-10-10)

Julia's SR sample-store dispatch gate reduces warmed Opt time by 34.7% for L32 at
one MPI rank and sixteen computation threads. Rust's recursive-join prototype
regressed and was rejected. These are separate changes and comparisons.

Related to [issue #496](https://github.com/AtelierArith/mvmc-rs/issues/496).
Julia implementation, independent C-equation checks, kernel crossover probe and
full verification evidence: [Julia PR #77](https://github.com/tmisawa/Julia-mVMC/pull/77).

## Matched Julia confirmation

Periodic half-filled Hubbard chain, t=1, U=4, Lsub=4, eight quadrature points;
300 Opt steps, 320 total samples per step. NSplitSize=1; four-rank inputs use
80 samples per rank. Ryzen 9 PRO 8945HS, eight physical cores/sixteen logical CPUs.
Julia 1.13.1, MPI.jl 0.20.27, MPICH 4.2.0, BLAS one thread in both Julia's
OpenBLAS 0.3.30 ILP64 and the native helper's OpenBLAS 0.3.26 LP64.

| MPI ranks | Threads/rank | Sites | Julia before (s) | Julia after (s) | Change |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 32 | 17.803 | 11.627 | -34.7% |
| 1 | 16 | 64 | 60.181 | 56.652 | -5.9% |
| 4 | 4 | 32 | 3.634 | 3.494 | -3.8% |
| 4 | 4 | 64 | 13.509 | 13.311 | -1.5% |

Three alternating baseline/candidate pairs after warming both paths in each
process; order baseline/candidate, candidate/baseline, baseline/candidate.
All twelve pairs favored the candidate. Time is the maximum rank's production
API duration, including parsing, initialization and outputs, excluding process
startup, compilation and warmups. No profiler ran during the confirmation.
A typed Ref selector installed before warmup chose either the original 64-item
threshold or the candidate thresholds, with common selector overhead. Separate
unmodified-source L32/one-rank/sixteen-thread pilots measured medians of 18.139 s
and 11.653 s. Treat the small four-rank effects conservatively.

Only dispatch changes: serial storage below 65,536 real or 32,768 complex values;
large vectors retain static threading. Elementwise C arithmetic, sample offsets,
RNG calls and other threading gates remain unchanged. The baseline source is
043ec52b9c9a1542083c233bd1923fe24f36d03c, with the same tracked tree as upstream
main 02afdae0a19732c727e0d09132d9761c57d68fcb. Candidate PR head is
85f9ea12c49e46639b8284334388116d5f16a337.

Local verification passed 41,392 optimizer assertions, 25 Slater assertions and
15 base assertions. The new 53-assertion storage suite also passed at one and
sixteen threads. Eight 20-or-300-step size/layout audits checked twenty paired
rank snapshots: all native SFMT words/index/reseed/draw count, saved and temporary
electron configurations, and acceptance/rejection counters matched exactly.
Computed floating outputs use explicit justified numerical tolerances; their
maximum absolute difference was observed as zero. Historical fixtures are not
regenerated or relabeled.

The earlier [complete C/Julia/Rust rank/thread table](../rank-thread-sweep-20261010/README.md)
retains its original source identifiers and timing protocol. Its cells are not
replaced with measurements from this paired experiment. No new C, PhysCal or
all-six-layout performance claim is made here.

## Rust prototypes and faer reference

Rust already hoists the driver into a dedicated Rayon pool, uses static block
partitions and keeps small regions serial with estimated work gates. Its
`fork` still broadcasts to the entire pool when a region has fewer active blocks
than workers. The eight-QP workload can therefore synchronize sixteen workers.

On the same L32/one-rank/sixteen-thread/320-sample input, three fresh warmed
repetitions gave:

| Rust dispatch | Median Opt (s) | Change |
|---|---:|---:|
| Existing static broadcast | 8.508 | baseline |
| Partial-pool recursive join prototype | 9.033 | +6.2% |

All three prototype repetitions were slower than all three baseline repetitions.
The prototype is not adopted. The source patch and binary hashes accompany the
logs. Loss of cache locality is a hypothesis, not an established explanation.

A second prototype kept broadcast and assigned partial groups relative to the
driver worker, guaranteeing its participation. Sequential baseline/candidate
batches measured 8.409 s and 8.350 s (0.7% lower). This small preliminary result
does not establish a speedup; the candidate has not been adopted. Its three
repetitions, source patch and binary hashes are preserved separately.

The inspected [faer source](https://github.com/sarah-quinones/faer-rs/tree/ddbd2ae5280825fe97f86735618fed2f6f4eb0b3)
passes worker budgets to child operations, uses algorithm-specific thresholds,
and currently invokes Spindle in `thread::join_raw`. A separate integer-only
dispatch probe compared broadcast, iterator, recursive join and persistent
Spindle 0.2.6 worker groups. With sixteen pool workers, eight blocks and a
60-microsecond gap, broadcast took 17.848 microseconds and an eight-worker
Spindle group took 4.888 microseconds for the synthetic 4096-iteration payload.
This excludes gap time and setup; it is not a whole-VMC speedup. All outputs and
exact call counts were checked. Holding Spindle workers while issuing the existing
Rayon broadcasts can prevent completion, so it cannot replace a single helper
without coordinating the rest of dispatch and blocking MPI calls.

Julia static threading creates and waits for one task per pool thread; the new
store gate avoids that overhead for small vectors. Julia's MPI reductions call
MPI.jl's blocking in-place Allreduce. MPI FUNNELED requires calls on the
initializing thread, constraining attempts to move communication into tasks.

## Evidence and reproduction

`paired-summary.json` contains every primary repetition, output comparison and
completion checks; `audit-summary.json` records the eight deterministic audits.
The upstream PR includes the full paired harness, input/source/provider hashes
and compressed verification logs. This directory preserves Rust's rejected
prototype and the standalone dispatch probe, with its locked dependencies.

Run the optional standalone probe in the same Linux environment:

```sh
CARGO_TARGET_DIR=/tmp/mvmc-spindle-probe-target \
  cargo run --release --locked --manifest-path spindle-probe/Cargo.toml
```

Spindle source archive SHA-256:
673aaca3d8aa5387a6eba861fbf984af5348d9df5d940c25c6366b19556fdf64.
See `provenance.json` for library hashes and complete environment identifiers.
Exploratory four-rank runs with incorrect per-rank sample counts are excluded.
