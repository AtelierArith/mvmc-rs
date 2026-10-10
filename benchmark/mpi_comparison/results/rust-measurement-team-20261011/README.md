# Rust bounded saved-sample team

Related to [#496](https://github.com/AtelierArith/mvmc-rs/issues/496). This milestone concerns normal-real saved-sample evaluation in Opt and PhysCal. The issue's separate fresh-C acceptance criterion is 300 total samples at 4 MPI ranks × 4 threads; this study preserves the requested 320-total-sample rank/thread sweep and does not close that issue.

## Scope and numerical behavior

The real CPU `CalcMAll` saved-sample calculation reuses a Spindle 0.2.6 team when the QP count is smaller than the Rayon pool and the existing cold-work gate permits parallel work. Its maximum budget is the QP count. The benchmark has eight QPs, so a requested pool of 16 retains capacity 16 while this calculation has a team budget of at most eight. Spindle starts workers lazily; the budget does not guarantee eight simultaneous participants. Sampling, SR, MPI calls, callbacks, complex/FSZ and stage backends retain their existing execution paths. The team is released before MPI communication.

Public `install` keeps its Rayon behavior. Static block boundaries, sole ownership of each output, scalar calculations and ordered accumulations remain unchanged. Spindle may schedule a block on different workers. Nested block work executes inline; guards restore the block/scope flags during unwinding. No C/Julia runtime dependency or numerical kernel replacement is introduced.

Spindle's `with_lock` creates a local manager and a Rayon scope, rather than acquiring a global mutex. The leader processes and steals its own team's pending jobs, so progress does not require all budgeted workers to have started. Its exit wakes and releases participating workers and joins scoped jobs. Integer ownership tests cover QP counts 0/1/2/3/8/15/16/17, uneven blocks, nested regions, root/item panics, ordinary 16-worker Rayon work after cleanup, and 32 concurrent callers on the 16-worker pool. Artificial barriers inside a lazy bounded team are deliberately absent; such barriers require a gang-execution contract that this private scope does not expose.

## Protocol

Linux x86_64 Dev Container, AMD Ryzen 9 PRO 8945HS (8 physical cores, 16 logical CPUs), MPICH/UCX, OpenBLAS with one thread per rank. Periodic half-filled Hubbard chain, hopping t=1, interaction U=4, Lsub=4, eight QPs, sites 32 and 64. Each step/group has 320 total Monte Carlo samples, divided across MPI ranks with `NSplitSize=1`. Opt uses 300 SR steps; PhysCal uses 100 groups with the same fixed C-optimized input parameters for each baseline/candidate pair. One full warmup precedes three measured runs; tables show median seconds. Rust times include production parsing, initialization, sampling, calculation and output, use the maximum rank time, and exclude executable startup/build/warmups. The 20-step execution-observer run is outside primary measurements.

The original before/after controls use production Rust from f7ac3e58; main 370cf7a3 changes only the reviewed Julia reference pin and reporting. The baseline RNG audit is rebuilt from a pristine 370cf7a3 checkout. An older audit binary with unverified source provenance was excluded and its exploratory files retained outside this report. The final candidate confirmation is rebuilt after the Clippy closure-borrow cleanup. Source, binary and input hashes accompany results.

Final rebuilt-binary confirmation at 1 MPI rank × 16 requested threads (team maximum budget 8):

| Calculation | Sites | Baseline before | Candidate | Baseline after | Change vs before |
|---|---:|---:|---:|---:|---:|
| Opt | 32 | 8.414 | 7.188 | 8.484 | −14.6% |
| Opt | 64 | 31.787 | 28.083 | 32.582 | −11.7% |
| PhysCal | 32 | 3.396 | 2.647 | 3.462 | −22.1% |
| PhysCal | 64 | 12.371 | 10.548 | 12.428 | −14.7% |

Every candidate repetition is faster than every before/after control repetition for these four cases. These are whole-production timings. The initial pre-cleanup binary also improved all four cases; the table uses the final release binary.

Other requested layouts (this eight-QP team is bypassed when threads per rank are at most eight):

| Ranks × threads | Sites | Opt baseline | Opt candidate | PhysCal baseline | PhysCal candidate |
|---|---:|---:|---:|---:|---:|
| 1 × 1 | 32 | 11.973 | 11.938 | 4.588 | 4.637 |
| 1 × 1 | 64 | 55.579 | 56.692 | 22.289 | 22.273 |
| 2 × 8 | 32 | 4.759 | 4.669 | 2.099 | 2.106 |
| 2 × 8 | 64 | 18.243 | 18.252 | 7.815 | 7.932 |
| 4 × 4 | 32 | 3.394 | 3.354 | 1.463 | 1.420 |
| 4 × 4 | 64 | 14.219 | 14.101 | 5.835 | 5.866 |
| 8 × 2 | 32 | 3.043 | 3.055 | 1.130 | 1.120 |
| 8 × 2 | 64 | 13.486 | 13.372 | 4.883 | 4.876 |
| 16 × 1 | 32 | 3.511 | 3.581 | 1.064 | 1.051 |
| 16 × 1 | 64 | 16.147 | 16.918 | 4.971 | 4.927 |

The serial L64 Opt regression remains in the additional control: baseline before 55.579 s, candidate 56.692 s, baseline after 55.317 s (+2.0% / +2.5%). Bypassing the team does not rule out changes in compiler output or other timing effects; this small regression is an explicit limitation of the parallel improvement.

The 16×1 L64 Opt repetitions also require distribution context: baseline = [16.933, 16.147, 16.081] s; measurement = [16.378, 19.936, 16.918] s; baseline-after = [16.449, 16.480, 16.396] s. The candidate median is +4.8% against the before control and +2.9% against the after control. The final JSON retains every repetition, including the approximately 20 s candidate outlier; these controls are not presented as a speedup.

See [timings and numerical checks](rust-measurement-final-summary.json), [rank-local audit checks](rust-measurement-final-audit-summary.json), [provenance](provenance.json), and the `raw/` logs. The compressed `audit-snapshots.json.gz` retains the exact rank snapshots whose source hashes are recorded in the audit summary.

## Verification

All computed floating-point comparisons use explicit absolute plus relative tolerances. Dispatch-only benchmark output checks use 64 machine epsilons for each component; the per-element arithmetic and reduction order are unchanged. The focused ordinary Rust regression compares short Opt and PhysCal runs under 1/16 threads using the repository's 1e-11 absolute and relative short-SR/repeatability envelope, independently requiring exact full SFMT state/index/draw count and discrete electron state/counters. It uses checked-in fixtures and invokes no external numerical reference executable. Full MPI audits compare exact rank-local SFMT/discrete snapshots for 20/300 steps, with floating-point output checks kept separate.

Completed checks include default workspace (1584 passed), MPI workspace (1586 passed), a repeated MPI workspace after main/reference update (1586 passed), formatting, workspace type checking, MPI all-target Clippy with warnings denied, and workspace documentation tests. Process-scoped helper tests remain normally ignored and are launched by their parent tests; no failing check is skipped. Earlier local failures from the empty private Julia fixture checkout and an incorrect eager-worker test expectation were fixed before verification.

Final checks compare 17,136 file pairs / 31,753,440 computed values across all 24 measured layout cases using the stated bounds. The 8 independent 20/300-step MPI audit cases contain 20 paired rank snapshots; full SFMT words/index/draw count, electron configuration and counters agree exactly. All compared computed outputs have observed maximum absolute difference 0. This includes the final rebuilt benchmark outputs; it is a tolerance check, not a bitwise floating-point assertion. PhysCal final-state SFMT/discrete equality is additionally covered by the ordinary short regression; the full 100-group benchmark checks all physical outputs.

## Rejected and separate experiments

Global minimum-work gates of 80/120 us cut L64's 20-step parallel-call count from 171,138 to 21,571, but regressed whole 300-step runs. Per-dispatch transient Spindle teams also regressed L32 Opt (8.414 → 10.325 s). Pool-wide persistent teams offered larger 1×16 speedups but regressed 4×4 L32 and changed the public worker-availability behavior; they were not adopted. Padding the full team and batching PF workspace leases remain separate exploratory or rejected probes. Only the bounded saved-sample scope is proposed here; the larger global-team result must not be attributed to this implementation.

## Follow-up investigation

The unused global-team speedup suggests remaining savings in sampling. A future milestone should identify bounded local sampling chunks, keep proposal/RNG advancement and ordered reductions on their existing caller, release workers before every MPI collective, and preserve MPI's initializing-thread requirement. Prefer extracting a genuinely MPI-free CPU operation over weakening `MpiContext`/`Reducer` thread-safety types or changing public `install`. The global gate experiment is a negative control: fewer dispatch calls alone did not yield a faster full run. A future candidate needs its own Opt/PhysCal whole-run measurements, all requested rank/thread controls and independently exact RNG/discrete audits before adoption. No sampling-scope implementation is included here.

## Reproduction and archived checks

The measured release build enables `mpi` with the locked workspace dependencies and the existing release profile. Runtime settings and compiler/backend versions are recorded in `provenance.json`; the archived timing scripts record the exact input and binary hashes. Their named-volume paths identify the original private checkouts and can be adjusted to freshly built baseline/candidate checkouts. The standard `bash bench/run.sh` remains the common C/Julia/Rust comparison entry point.

Final Rust verification commands (with this checkout’s separate named-volume Cargo target):

```sh
cargo fmt --all --check
cargo check --locked --workspace --features mpi
cargo clippy --locked --workspace --all-targets --features mpi -- -D warnings
cargo nextest run --locked --workspace --cargo-profile test-fast --features mpi --no-fail-fast --retries 0
cargo test --locked --workspace --doc
cargo build --locked --release -p mvmc-core --features mpi --example mpi_benchmark --example mpi_physcal_benchmark --example mpi_rng_audit
```

Compressed build/test logs accompany this report. `analyze-rust-measurement-final.py` is the optional external benchmark/audit analyzer; normal Rust tests do not read this report or invoke reference runtimes.
