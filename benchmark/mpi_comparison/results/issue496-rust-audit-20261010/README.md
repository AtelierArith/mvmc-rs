# Rust optimization numerical audit (#496)

These are separate correctness runs, excluded from primary performance timing.
The Linux x86_64 Dev Container ran the periodic half-filled Hubbard chain
(`t=1`, `U=4`, `L=32/64`) with MPI world 4, four computation threads per rank,
one BLAS thread, 75 saved configurations per rank per step, and seed 1 plus
the existing rank seed offsets. Both 20-step and 300-step runs were checked.

`rust-discrete-comparison.json` compares the frozen pre-optimization Rust
implementation at `07a2f9f9a78271df7ffa73fff61b7b359345dfe3` against the
optimized implementation. All 16 size/step-count/rank combinations agree
exactly in 624 SFMT words, word index, counted draws, final configurations,
electron indices/spins, projection counts and sampling counters. Root-rank
final energy differences were zero in these runs; other ranks do not return
a final energy. The complete production states were captured by the optional
`mpi_rng_audit` example, not by a replacement sampling kernel.

`c-rng-comparison.json` compares the unmodified C executable's final 624 SFMT
words and index against optimized Rust for the same 16 cases. Every case
agrees. `rust_counted_words` is measured by Rust; the C observer does **not**
measure a C draw counter. See the provenance JSON files and
`c_toolbox/mpi_rng_496/README.md` for executable hashes, ELF offsets, source
origins and reproduction instructions. No C source was modified.

`output-comparison.json` checks all 300 output rows and all six columns for
three uninstrumented runs at each size against the frozen Rust benchmark.
The observed maximum absolute differences were zero. This observation does
not establish cross-platform bitwise floating-point parity; numerical tests
continue to use explicit tolerances.

Local workspace validation also passed: default-feature nextest (1,585
passed), all-feature nextest (1,586 passed), both with `test-fast`, locked
dependencies, no retries and no fail-fast; all-feature workspace clippy,
formatting, and documentation tests. Existing ignored tests remain ignored.
Native macOS validation is still required before merging the milestone.
