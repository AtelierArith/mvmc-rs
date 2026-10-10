# Direct SR accumulation experiment — issue #496

Linux x86_64, 2026-10-10, full L64 Hubbard Opt300, MPI4 x threads4, BLAS1.
One warmup and three measurements per mode; same compiled binary with an
experimental environment flag selecting the direct path. Sequential mode
batches are an initial confirmation, not a fresh C performance comparison.

Current runner clears all aggregate and scratch arrays immediately before
measurement, then its private SrMeasurement cache clears another set, measures
into it, and adds it back to the zero destination. The diagnostic direct path
uses the existing uniquely owned destination and preserves the zero-add
publication (including signed-zero handling) after measurement. Sampling,
SR arithmetic and reduction order remain unchanged. PhysCal is unchanged.

| Path | Median seconds | Individual measurements |
|---|---:|---|
| Private cache | 13.869856086 | 13.839606470, 13.869856086, 13.889442735 |
| Direct SR | 13.247782067 | 13.124357207, 13.273518574, 13.247782067 |

Reduction: 4.49%. All three 300-row, six-column output comparisons observed
zero delta. This does not establish exact RNG or full numerical correctness;
targeted golden/contract tests and MPI state audits are subsequent gates.
No numerical tolerance was changed. The experimental flag is temporary and
must be removed from the eventual production implementation.

The earlier preliminary run accidentally used three warmups and one measured
repetition; its 14.318/13.434s observation is not the confirmation table above.
Scripts record the corrected driver argument order.

## Subsequent verification of the diagnostic direct path

`targeted-tests.log`: all 58 existing run::callback_tests passed with the
in-place flag, including independent reference fixtures, SR modes/stores,
nonfinite samples, invalid walkers, and observer/unwind contracts. This is
not a full-workspace test result.

`rng-comparison.json`: all 16 L32/L64 x20/300-step x4-rank cases matched all
624 SFMT words, index, consumed-word count, final electron indices/configuration,
occupation/spin/projector counts and move counters exactly. Production output
rows (six finite columns) had zero observed delta for all four cases. Non-root
summary energy is intentionally NaN/JSON null and is not a numerical comparison.

Final production cleanup removes the private cache and experimental flag in
branch perf/issue496-direct-sr, based on main f04cfca7. It retains the public
owned-accumulator merge API. That final source requires separate workspace,
platform and fresh C/Rust benchmark verification before a performance claim.

## Final source checks

Final production source plus the separate #505 FSZ observer-expectation fix
passed all 1584 workspace tests (82 existing ignored) with MPI + SIMD + BLAS
features, test-fast, four inner workers and one BLAS thread. The first archived
checkout was missing submodule input files; after restoring the pinned source
archives, its sole remaining failure was the FSZ instrumentation expectation
under the automatic gate. #505 / PR #506 preserves independent C fixture
values and tolerances and adds automatic/forced 1/2/4-worker coverage.

`final-workspace-tests.log`, `final-clippy.log`, and `final-doc.log` record the
successful final full test, Clippy -D warnings, and separate workspace doctest
runs. Final benchmark/audit examples also built successfully with the production
MPI/default numerical features. Fresh final-source timing and state comparisons
are still required; the earlier diagnostic comparison is not substituted for them.

## Fresh final-source C/Rust Opt confirmation

The final-source audit again matched all 16 rank cases exactly for RNG words,
index/count, final configuration and counters; all four output comparisons had
zero observed delta. See `final-rng-comparison.json`.

Full Opt300, L32/L64, periodic half-filled Hubbard t1/U4, 300 saved samples per
step (75/rank), MPI4 x4 workers, BLAS1, and averaging window300 in both ports.
Three alternating C/Rust batches per size; C fresh processes with an initial
warmup, Rust one full warmup plus one measured repetition in each batch.
No other local CPU-heavy job ran during timing. All three paired comparisons
were faster for Rust at each size.

| Sites | C median (s) | Rust median (s) | Rust reduction |
|---|---:|---:|---:|
| 32 | 3.445990 | 3.145541025 | 8.72% |
| 64 | 13.725770 | 13.044224062 | 4.97% |

C is the unchanged independent reference executable (SHA256 recorded and
verified by the script), not the former instrumented one-process emulation.
Its timing is rank0 internal All after MPI_Init; Rust is the warmed production
API maximum over ranks, including parse/init/sampling/SR/output, excluding
startup/build and the separate execution observer. These boundaries differ.
Linux x86_64 Ryzen9 PRO8945HS, Rust1.99, MPICH4.2, GCC13.3, OpenBLAS0.3.26 LP64;
actual final linked libraries are in final-libraries.log.

`final-c-confirmation/` contains all 12 measured times, source patch/hashes,
binary hashes, raw logs and numerical output observations. Rust outputs all
match the previous production-path batch with observed delta0. Repeated C
processes show small differences versus that previous batch (up to3.41e-7
across reported columns); these are recorded without applying or changing a
tolerance, and the timing table is not a deterministic C-trajectory parity
claim. Independent native-C fixtures are covered by the passing workspace tests.

Julia Opt remains outside this milestone and issue#496 remains open. This
report establishes the measured Rust Opt milestone under the stated conditions;
PhysCal follows the unchanged production path, covered by workspace tests.
