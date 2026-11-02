# Per-next-site Rayon gap diagnostics (#581)

`MVMC_RS_INNER_GAP_SITE_PROFILE=1` enables the existing same-thread top-level scope gap collector and additionally groups counts by the *next* call site and its hot/cold flag. A gap is elapsed time from the previous nonempty top-level profiled scope's end to the next such scope's start **on the same thread**. First scopes have no preceding gap, nested scopes are excluded, and caller/worker streams are aggregated even if they overlap. The previous site is not recorded. This is not global serial time, worker idle time, a causal attribution, or a dispatch latency measurement. Profiling locks a shared map on each attributed gap; it perturbs scheduling and the measurements. Both settings are off by default. The English/Japanese manuals describe the opt-in control.

## Instrumented cases

One diagnostic process per case used the same fixed L32/L64 inputs and model settings in [rayon-profile-20261102](../rayon-profile-20261102/README.md): L32 Opt and PhysCal, L64 Opt, 300 samples, warmup 10, interval 1, seed 1, 20 Opt steps/window 1 or 20 fixed-parameter PhysCal windows. Inner workers 1 or 4, Rayon capacity 4, BLAS/OMP one, tenferro SR, default wait/gate settings, CPUs 0–3. This is a single profiled observation, not a timing or speedup comparison. The host was not isolated or frequency controlled, and load was not recorded. Source was `ccfce45ff1ace4ef76836759273fdafbc62ffcf1` plus [source.patch](source.patch) (`git apply --unidiff-zero`); the release binary SHA-256 was `e4924982137bc73c016a142cde345680c4ded42dbb6ae947d1e662eeffc46994`. Linux x86_64 6.8.0-147, Xeon E5-2699 v3, Rust 1.99.0, MPICH 4.2.0, pthread OpenBLAS, dev-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`.

Buckets are `<1`, `<5`, `<20`, `<60`, `<100`, `<1000`, `>=1000` µs. Examples below are the next call sites, not the work preceding the gap:

| Case | Workers | Next site / scope | Bucket counts |
|---|---:|---|---|
| L32 Opt | 1 | `sampling/updates.rs:116` / hot | 189151, 3560, 216, 1, 0, 0, 0 |
| L32 Opt | 4 | `sampling/updates.rs:116` / hot | 173637, 18223, 1065, 3, 0, 0, 0 |
| L32 PhysCal | 4 | `sampling/updates.rs:425` / cold | 0, 26136, 131, 641, 1221, 101, 245 |
| L64 Opt | 1 | `slater_derivative.rs:504` / cold | 0, 0, 5328, 671, 1, 0, 0 |
| L64 Opt | 4 | `slater_derivative.rs:504` / cold | 0, 0, 0, 5999, 1, 0, 0 |

For example, the L32 PhysCal row has 1567 next-site observations at or above 60 µs, but those observations can arise on different threads and do not prove 1567 global stalls or identify their cause. The per-site bucket totals matched the aggregate hot/cold counters in all six runs. The numeric non-timing outputs (4 files for each Opt case and 80 for PhysCal) matched a separately built uninstrumented 40 µs-gate reference at printed precision, maximum absolute token difference zero. This is not full RNG-state or proposal/decision comparison; the runner verification below covers separate fixed cases. An initial diagnostic sequence was discarded because stdout and stderr interleaved in the same file, corrupting some lines. The archived sequence sends them to separate files.

[profile](profile/) contains source-input hashes, raw outputs, logs, full site/aggregate summaries and exact commands. Prepare the three input directories as described in the referenced report. Build the candidate in its own dev-container target with `cargo build -p mvmc-cli --features mpi --release --locked`, adjust the fixed paths, target name and CPU affinity in `profile/run.sh`, and use a fresh output directory. `uv run --no-project python profile/analyze.py /tmp/mvmc-581-gap-sites-clean` regenerates the per-site summary and verifies aggregate bucket totals. `profile/compare-output.py PROFILE_ROOT REFERENCE_ROOT` compares finite numeric output files to the separately prepared uninstrumented 40 µs-gate report output. Raw archive extraction allows reanalysis without rerunning. The reference binary was the earlier gap-collector candidate SHA-256 `5ee7c581f626932bb88a73cf02f77667e8314f74843f768182f376c8654359cf`.

## Disabled-profile overhead check

A separate paired sequence used the actual parent and candidate binaries, with all gap profiling disabled, same L32 inputs, one warmup and three alternating measurements each, CPUs 0–3 and otherwise the same environment. Wall time includes process startup and output, excludes Docker startup. Parent binary SHA-256 `019487c2a4c4a1b1db1de8b915a527102bc8535bb5f55b57df0d10e08fc816ae`; candidate SHA-256 is above.

| Workload | Workers | Parent median | Candidate median |
|---|---:|---:|---:|
| Opt | 1 | 2.440 s | 2.428 s |
| Opt | 4 | 2.093 s | 2.062 s |
| PhysCal | 1 | 2.146 s | 2.174 s |
| PhysCal | 4 | 2.383 s | 2.301 s |

The small three-run sample does not establish the absence of a small regression or a speedup. The 4 Opt and 80 PhysCal finite non-timing output files matched numerically at printed precision across the six measured runs per row. A preliminary overhead sequence was interrupted and discarded after discovering that its baseline included unrelated code differences; the reported sequence used the true parent binary and started after its build completed. [overhead](overhead/) retains individual wall/user/system samples, numeric output, hashes and the exact script. Run `overhead/run.sh IMAGE PARENT_TARGET CANDIDATE_TARGET INPUT_DIRECTORY FRESH_OUTPUT_DIRECTORY` and `uv run --no-project python overhead/analyze.py FRESH_OUTPUT_DIRECTORY`; adjust CPU affinity and paths for the host. Host load was not sampled or controlled.

## Correctness checks

On Linux, the focused tests verify top-level attribution and exclusion of nested scopes, plus the existing hot/cold and unwind behavior. A separate 1/2/4-worker runner matrix enabled the site profiler and experimental shared-team path, asserted actual multi-worker team activation, and captured candidate/decision/raw-draw event streams for five fixed QP32 direct-SR cases. The existing runner compared the complete discrete event records exactly across workers and same-worker repeats, and C-referenced counters/snapshots and numerical values using the repository's existing justified bounds. All cases passed; this is a correctness diagnostic, not a timing run or a full C event-by-event oracle. The compressed [runner log](runner.log.gz) records the gate. Reproduce with `MVMC_RS_SR_BACKEND=tenferro MVMC_RS_TENFERRO_THREADS=4 MVMC_EXPERIMENT_SHARED_CPU_TEAM=1 MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1 MVMC_EXPERIMENT_ASSERT_SHARED=1 MVMC_EXPERIMENT_TRACE_MOVES=1 MVMC_RS_INNER_GAP_SITE_PROFILE=1 OPENBLAS_NUM_THREADS=1 MVMC_RS_THREADED_182=1 cargo nextest run -p mvmc-core --test threaded_issue182 --run-ignored all -E 'test(=runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal)' --cargo-profile test-fast --locked --no-fail-fast --retries 0 --no-capture`. The Julia fixture submodule is required by this explicit developer gate; ordinary Rust tests remain independent of reference runtimes.

The same Linux dev-container candidate passed the full `cargo nextest run --workspace --cargo-profile test-fast --locked --no-fail-fast --retries 0` suite (1627 tests), `cargo fmt --all --check`, `cargo test --workspace --doc --locked`, and `cargo clippy --workspace --all-targets --locked -- -D warnings`. Exact-head cross-platform CI is still required before merge.

## Follow-up priorities and gate candidates

1. Investigate L32 PhysCal's `sampling/updates.rs:425` and the Green-measurement region together. The next-site histogram above locates many long **thread-local** gaps in the instrumented case, while the matched uninstrumented 1/4-worker comparison in [rayon-profile-20261102](../rayon-profile-20261102/README.md) showed no clear four-worker wall-time gain. Instrument actual scope transitions and worker activity with less perturbation before assigning a cause.
2. Investigate the L64 Opt derivative and Pfaffian regions, comparing their actual work and dispatch paths. The saved gate sweep put every 80/160 µs L64 Opt sample slower than every 20/40 µs sample on this host. Treat 20 and the existing 40 µs as candidates for a controlled crossover study; do not adopt either from this one host and small sample.
3. Examine the frequent `sampling/updates.rs:116` hot scopes after the larger sites. Their gap counts show volume, not a direct timing saving; use the uninstrumented end-to-end case and fixed-seed decision checks to evaluate any proposed change.

These are ordered experiments, not claims that the named next sites caused the preceding gaps. The diagnostics do not provide per-site previous-scope identities, full application idle time or a portable new default.
