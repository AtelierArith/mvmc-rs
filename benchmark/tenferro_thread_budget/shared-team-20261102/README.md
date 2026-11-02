# Experimental shared CPU team, L=32 (#582)

This experiment compares the **same Rust binary** with `MVMC_EXPERIMENT_SHARED_CPU_TEAM` disabled and enabled. When enabled, fixed inner-kernel blocks may use the already held tenferro CPU Engine team via a nonblocking lease; a busy Engine executes the blocks serially on the caller, while an unavailable Engine falls back to the existing Rayon path. The change is experimental, does not provide a global pool budget, and does not cover the `stage` team-lifetime setting. It must not be interpreted as completion of #582, #583, or #584.

Both workloads use the fixed L=32 inputs and settings in [mpi-placement.md](../mpi-placement.md): 300 samples, warmup 10, interval 1, fixed seed 1, Opt 20 SR steps/window 1 and fixed-parameter PhysCal 20 windows. Each pair uses the same executable, OpenBLAS and OMP one thread, Rayon 4, tenferro SR, inferred tenferro budget, default wait settings, and one warmup plus three alternating measurements. Inner workers are 1 or 4. Bash wall time includes process launch, initialization, calculation and output, and for MPI includes MPICH startup; Docker startup is excluded. The single process is pinned to CPUs 0–3; four MPI ranks are each pinned to disjoint four-CPU masks among CPUs 0–15. The CPUs occupy separate physical cores on socket 0. No frequency control was applied.

| Workload | MPI ranks | Inner workers/rank | Baseline median | Shared-team median |
|---|---:|---:|---:|---:|
| Opt direct SR | 1 | 1 | 2.469 s | 2.429 s |
| Opt direct SR | 1 | 4 | 2.072 s | 1.551 s |
| PhysCal | 1 | 1 | 2.144 s | 2.176 s |
| PhysCal | 1 | 4 | 2.678 s | 1.546 s |
| Opt direct SR | 4 | 1 | 2.828 s | 2.853 s |
| Opt direct SR | 4 | 4 | 2.697 s | 1.806 s |
| PhysCal | 4 | 1 | 2.614 s | 2.624 s |
| PhysCal | 4 | 4 | 3.166 s | 1.896 s |
| Opt SR-CG | 4 | 4 | 2.945 s | 2.176 s |

This short experiment shows a lower wall-time median in these four-worker cases, not general CPU efficiency: MPI Opt aggregate user time at its median increased from 23.185 to 26.546 seconds, while system time fell. User/system figures are whole-child totals, not thread or phase attribution. The one-worker rows are negative controls. SR-CG uses a separate sequence with NSRCG changed to 1; its script's historical `held`/`stage` labels mean probe off/on and **do not** select the separate Engine team-lifetime mode. Individual timing logs, settings and input hashes are in `single/`, `mpi/`, and `cg/`; the JSON summaries include per-run timings and finite formatted output comparisons. Across each direct-SR/PhysCal pair the compared non-timing output values were identical at printed precision; CG output was byte-identical. These comparisons do not establish RNG-state, individual proposal/decision, or unprinted floating-point equality.

A separate diagnostic run enabled profiling counters, so its time is not compared with the timing runs. It observed all attempts succeed and four distinct participating thread IDs: single-rank Opt 20158/20158, PhysCal 18452/18452; MPI Opt per rank 20160–20171 and PhysCal 18396–18452. Thread IDs count participants, not utilization or simultaneous execution. The profiling-disabled MPICH failure/recovery test passed all ranks in 2- and 4-rank jobs; it is a correctness test, not a performance measurement.

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Rust 1.99.0, MPICH 4.2.0, pthread OpenBLAS. Source baseline was `0921e1513fd1ba79bfa51f2e647bf0e08a49ba56`. The single-process runs used [`uninstrumented.patch`](uninstrumented.patch), executable SHA-256 `14d6416fce38d80cf701b74f0b927de7efd5f4d3804cd25ff0d5e4982aae43d8`. MPI and CG used [`instrumented.patch`](instrumented.patch), executable SHA-256 `c792ed59e6ce980b3e312d1f45d003455df62da40b8750bc000eae04b96da556`, with profiling disabled for timings. Both patches are zero-context (`git apply --unidiff-zero`) and apply to that baseline; do not infer the later head's performance from these binaries.

## Reproduce

Build each patch in a separate checkout and target volume with the dev container's MPI toolchain: `git apply --unidiff-zero PATCH`, then `CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked`. Prepare the input copies as in [mpi-placement.md](../mpi-placement.md), including the PhysCal window change. Use a fresh output directory each time and the same target volume for both modes of a same-binary comparison:

```sh
benchmark/tenferro_thread_budget/shared-team-20261102/single.sh \
  DEVELOPMENT_IMAGE UNINSTRUMENTED_TARGET UNINSTRUMENTED_TARGET /tmp/placement-input /tmp/shared-single
benchmark/tenferro_thread_budget/shared-team-20261102/mpi.sh \
  DEVELOPMENT_IMAGE INSTRUMENTED_TARGET INSTRUMENTED_TARGET /tmp/placement-input /tmp/shared-mpi
benchmark/tenferro_thread_budget/shared-team-20261102/cg.sh \
  DEVELOPMENT_IMAGE INSTRUMENTED_TARGET /tmp/placement-input/opt /tmp/shared-cg
```

Scripts require allowed consecutive CPUs 0–3 or 0–15 by default; `MVMC_BENCH_CPU_BASE` changes the first CPU. Check the rank affinity for a different launcher or host. Further per-kernel crossover, spin/wake policy, shared-resource fallback and full fixed-seed validation are necessary before adopting this experiment by default.

After the C-reference repair proposed in #606 was applied locally, the optional runner matrix also passed with tenferro SR and the shared-team probe enabled. A separate activation-instrumented rerun asserted that the 2- and 4-worker observed children actually used the team: each recorded 427860 attempts, 427859 successful dispatches and respectively 2 or 4 distinct participants. One dispatch per run fell back to Rayon. This is a correctness/activation check with instrumentation overhead, not a timing result; it does not demonstrate elimination of competing pools.

All measurements below through the L64 section used earlier candidate binaries where a busy Engine fell back to Rayon; the final busy-Engine candidate is measured separately at the end.

## MPI wait-policy comparison

A separate same-binary L32 experiment used four MPI ranks on disjoint four-core masks and four inner workers per rank. It compared the shared dispatch disabled/enabled with tenferro's default spin policy and `MVMC_RS_TENFERRO_SPIN_POLLS=0`. Each of the four combinations received one warmup and three measured runs with alternating order. All other input, binary, thread and timing boundaries match the MPI experiment above. CPU values are medians of the aggregate child user/system times, independently of the wall median.

| Workload | Shared dispatch | tenferro spin polls | Wall median | User CPU median | System CPU median |
|---|---|---|---:|---:|---:|
| Opt | off | default | 2.737 s | 23.432 s | 3.804 s |
| Opt | on | default | 1.905 s | 27.688 s | 0.506 s |
| Opt | off | 0 | 2.550 s | 22.974 s | 3.799 s |
| Opt | on | 0 | 2.734 s | 23.551 s | 3.733 s |
| PhysCal | off | default | 3.051 s | 25.579 s | 5.159 s |
| PhysCal | on | default | 1.956 s | 28.474 s | 0.381 s |
| PhysCal | off | 0 | 3.124 s | 25.723 s | 5.270 s |
| PhysCal | on | 0 | 4.468 s | 30.640 s | 5.697 s |

In this sample, zero spin removed the shared team's wall-time advantage and increased PhysCal time; it did not consistently reduce total CPU time. These data do not isolate wake or idle spinning and do not establish an optimal policy. The 4 Opt and 80 PhysCal finite formatted output files (excluding timing files) matched to printed precision across the 12 measured runs per workload. Complete RNG states and decisions were not recorded. The [raw logs, input hashes, analysis script and summary](spin/) are preserved as gzip where applicable.

Run [spin/run.sh](spin/run.sh) with the previously prepared L32 inputs and instrumented target volume, adjusting the fixed container/image paths and CPU masks for the host; its output directory must be fresh. After the run, `uv run --no-project python spin/analyze.py OUTPUT_DIRECTORY` recomputes the timing and output summary. The script's fixed `/tmp/mvmc-582-spin-comparison` path and other arguments document the original execution rather than serving as an automatic system-wide policy.

A separate spin-zero diagnostic enabled the probe counters, so its run times are not compared with the table. Each rank reported four distinct participants; all 20160–20171 Opt and 18396–18452 PhysCal dispatch attempts succeeded. Its finite formatted outputs matched the corresponding measured spin-zero run for 4 Opt and 80 PhysCal files. See [diagnostic.sh](spin/diagnostic.sh) and its compressed logs; this still does not establish actual simultaneous CPU use or complete RNG parity.

## Preserve the bounded saved-sample team

The first experimental implementation attempted to enter the shared Engine team before checking whether the existing bounded Spindle saved-sample team was active. The revised candidate lets an existing bounded team dispatch its own fixed blocks first. A focused test entered a two-worker Spindle team and verified that it completed the work without calling the separate Engine team. This avoids that particular nested-pool path; it is not a global resource cap and does not eliminate the busy/unavailable Rayon fallback.

With that correction, a fresh same-binary Opt/PhysCal comparison used one warmup and three alternating measured runs. MPI used the same disjoint four-core/rank layout as before; the single-process cases used CPUs 0–15, `RAYON_NUM_THREADS=16`, inner workers 4 or 16 and an explicitly matching tenferro thread count. Other settings and timing boundaries are as above. The source was the prior candidate plus [bounded-team.patch](bounded-team.patch) (SHA-256 `4303e963df9e3e9a9f2e6bd3eba39f4b77993bb0bf764fecfdeb05ef2b979781`), and the release binary SHA-256 was `5e4ffe3e238e9b3de4f0577f5e1d29a7a7f46d6741d81b531b6310826b6e1d7b`.

| Workload | Ranks | Inner workers/rank | Baseline median | Shared median |
|---|---:|---:|---:|---:|
| Opt | 4 | 1 | 2.866 s | 2.884 s |
| Opt | 4 | 4 | 2.691 s | 1.858 s |
| PhysCal | 4 | 1 | 2.661 s | 2.645 s |
| PhysCal | 4 | 4 | 2.969 s | 1.902 s |
| Opt | 1 | 4 | 2.332 s | 1.584 s |
| Opt | 1 | 16 | 2.292 s | 1.896 s |
| PhysCal | 1 | 4 | 2.247 s | 1.562 s |
| PhysCal | 1 | 16 | 1.587 s | 1.539 s |

The 4 Opt and 80 PhysCal non-timing formatted numeric files matched to printed precision across all six measured runs for each row. The 16-worker PhysCal difference is small relative to three-run variability and does not establish a gain. [MPI](bounded-mpi/) and [single-process](bounded-single/) include timing logs, summaries, input hashes and the single-process script. To reproduce, apply `instrumented.patch` to baseline `0921e1513fd1ba79bfa51f2e647bf0e08a49ba56`, then apply `bounded-team.patch` with `git apply --unidiff-zero`, build as above, and invoke `mpi.sh` with the same target volume for before and after, and `bounded-single/mvmc-582-bounded-single.sh` with the same candidate volume for both modes. Its paired analyzer accepts the output directory.

A separate [counter-instrumented run](bounded-diagnostic/) used the corrected candidate and is not included in timings. MPI ranks each recorded four distinct participants, with 20160–20171 successful Opt and 18396–18452 successful PhysCal shared dispatches. The single-process 16-worker case observed 16 distinct participants, with 2158/2158 Opt and 452/452 PhysCal attempts succeeding. Its 4 Opt and 80 PhysCal numeric files matched the corresponding timed run at printed precision. The lower attempt counts for 16 workers cannot by themselves be attributed to any single path. Neither participant counts nor formatted output establish complete RNG parity or simultaneous core utilization.

## L64 optimization check

The corrected candidate was also tested on the repository Hubbard L64 Opt input with 300 samples, warmup 10, interval 1, seed 1, 20 steps/window 1. A single process used CPUs 0–15, Rayon capacity 16 and inner/tenferro worker counts 1, 4 or 16. The same binary and enabled/disabled path were alternated for one warmup and three measured runs each. Time includes CLI startup and output, excludes Docker startup; no CPU frequency control or dedicated-host isolation was used. The first attempted sequence was stopped and excluded because an unrelated build overlapped it; the recorded sequence started after the build ended. Host load average was approximately 2.3 before and 3.7 after the clean sequence (the sequence itself can contribute to load).

| Inner/tenferro workers | Shared dispatch | Wall median | User CPU median | System CPU median |
|---:|---|---:|---:|---:|
| 1 | off | 11.521 s | 11.417 s | 0.096 s |
| 1 | on | 11.581 s | 11.490 s | 0.103 s |
| 4 | off | 5.427 s | 18.338 s | 2.696 s |
| 4 | on | 5.380 s | 20.973 s | 0.229 s |
| 16 | off | 4.791 s | 45.760 s | 9.627 s |
| 16 | on | 5.051 s | 41.098 s | 0.385 s |

The four finite formatted non-timing output files matched to printed precision across the six measured runs at each thread count. Neither this check nor timings establish full RNG-state or proposal/decision parity. These three samples show no clear benefit at four workers and a slower median at 16; the result supports retaining the default path rather than generalizing from the L32 gains. The [L64 directory](l64/) contains input hashes, raw logs, scripts, individual timing samples and the numeric-output summary.

A separate counter-enabled diagnostic, excluded from timing, confirmed 160182/160182 successful shared dispatches and four distinct participants with four workers, and 142182/142182 with 16 distinct participants at 16 workers. Its four numeric output files matched the corresponding timed run at printed precision. This is evidence that the path was exercised, not a measurement of concurrent CPU utilization. Reproduce with `l64/run.sh` followed by `uv run --no-project python l64/analyze.py /tmp/mvmc-582-l64-clean`; `l64/diagnostic.sh` is the separately instrumented command. Adjust the fixed container, CPU and input paths, and use a fresh output directory.

## Busy Engine fallback candidate

The final candidate preserves an already active bounded Spindle team, executes blocks on the calling thread when the shared Engine is busy, and uses Rayon when the Engine is unavailable. This avoids launching another Rayon dispatch on the busy path but does not establish a global thread budget. The candidate is revision `36035125` plus [busy-team.patch](busy-team.patch); apply the zero-context patch with `git apply --unidiff-zero`. The MPI release executable SHA-256 was `538ceef33b84b1c5f371025089e71d16557b3076e53c8370d3a3624835aac508`. The same binary was used for both columns, with shared dispatch off and on respectively.

Using the same L32 MPI placement, timing scope and one warmup plus three alternating measurements described above, a clean run yielded:

| Workload | Inner workers/rank | Baseline median | Shared median |
|---|---:|---:|---:|
| Opt | 1 | 2.866 s | 2.868 s |
| Opt | 4 | 2.722 s | 1.809 s |
| PhysCal | 1 | 2.617 s | 2.644 s |
| PhysCal | 4 | 3.017 s | 1.944 s |

The 4 Opt and 80 PhysCal non-timing output files matched at printed precision across measured runs; this alone does not establish RNG-state or decision parity. In separate counter-enabled diagnostics, excluded from timings, each MPI rank reported four distinct participants and every dispatch succeeded: 20160–20171 Opt and 18396–18452 PhysCal. Participant count does not prove simultaneous CPU utilization. The three-run measurements are workload-specific and do not supersede the L64 limitations above. A preceding run was discarded because binary inspection overlapped its timing; the results above come from a fresh run with no builds or binary inspection during timing.

The [busy-mpi directory](busy-mpi/) preserves the input hashes, raw output and timing logs, analyzer, summary, diagnostic logs and scripts. Reproduce with `busy-mpi/run.sh DEVELOPMENT_IMAGE TARGET_VOLUME TARGET_VOLUME INPUT_DIRECTORY FRESH_OUTPUT_DIRECTORY`, followed by `uv run --no-project python busy-mpi/analyze.py FRESH_OUTPUT_DIRECTORY`. The diagnostic script uses fixed local paths which must be adjusted for another host.

## Candidate, decision and draw trace

A separate Linux correctness run enabled the final shared candidate, including the busy fallback, and captured the existing observational sampler event stream in the Hubbard real, QP32, direct-SR, `store=0`, two-step runner case. The runner compared the complete recorded event lists exactly across 1, 2 and 4 inner workers (including same-worker repeats), alongside its existing C-referenced and RNG/configuration checks. Events include update kinds, proposed candidate fields and rejection flags, Metropolis decisions and exact RNG words/draws; no extra RNG draws are introduced by recording. A nonempty Metropolis decision assertion prevents a vacuous trace. The 1/2/4 matrix passed. This is one case, not proof for other models or every numerical control path, and is not a timing run.

To reproduce on the candidate merged with #606's reference repair, enable `MVMC_EXPERIMENT_TRACE_MOVES=1`, `MVMC_EXPERIMENT_SHARED_CPU_TEAM=1`, `MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1`, `MVMC_RS_SR_BACKEND=tenferro`, `MVMC_RS_TENFERRO_THREADS=4`, `MVMC_RS_THREADED_182=1`, and `OPENBLAS_NUM_THREADS=1`, then run `cargo nextest run -p mvmc-core --test threaded_issue182 --run-ignored all -E 'test(=runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal)' --cargo-profile test-fast --locked --no-fail-fast --retries 0`. Also enable `MVMC_EXPERIMENT_ASSERT_SHARED=1` to assert actual shared dispatch and at least two participants in multi-worker child processes; the assertion is diagnostic and is not part of the runtime. The [test log](move-trace-test.log.gz) reports the gate result; the full event lists are compared in-process and are not logged on success. This checks same-implementation event equality, while the runner's existing separate C-referenced fixtures cover counters and snapshots; it is not a full C event-by-event oracle.

A further paired four-worker check executed the same internal runner child once with the shared path disabled and once enabled, and compared its full Hubbard sampling-event record byte for byte. Both records were 386433 bytes, SHA-256 `4fc4e90284d80f4b323152d11c7166a06c6d8e94e2291f98a705717703ba9004`. The enabled child asserted 591392 successful shared dispatches from 593026 attempts and four distinct participants over the **entire runner matrix**, not only the traced Hubbard case. The attempt/success difference combines busy and unavailable cases, so it does not identify which fallback occurred. No timing is inferred. [move-ab](move-ab/) contains the exact container-side command, output and compressed event records. To reproduce, mount this checkout at `/work`, its own writable target at `/target`, a fresh output directory at `/results` and the script at `/run-probe` in the documented development container; run `CARGO_TARGET_DIR=/target bash /run-probe` from `/work`. The reference Julia submodule and `jq` and `rg` must be available. The child compares only the sampling-event stream; broader numerical comparison remains the job of the full runner matrix.

The trace selector was subsequently broadened to all five QP32/direct-SR/store0 two-step cases: Heisenberg real, complex, complex-FSZ, real-FSZ, and Hubbard real. The complete 1/2/4-worker runner matrix with activation asserted passed again. A new paired four-worker off/on run compared all five event records byte-for-byte: each combined record was 1,809,221 bytes and had SHA-256 `3a18916a0c5f98b37b5fae979ae007a1922629d435ab21ae90a9614abaae984a`. The enabled whole-matrix activation counts remained 591392/593026 with four participants. [move-ab-expanded](move-ab-expanded/) preserves the paired event records and matrix result; use the `move-ab/run.sh` command above with the current trace selector. These are five fixed same-implementation paths, not a general C event oracle or exhaustive parameter coverage. The preceding single-case hashes describe the earlier, narrower selector.

## Classified fallback counters

The final diagnostic candidate, source revision `71fa6619` plus [classified-counters.patch](classified-counters.patch), reports successes, busy/failed-lease cases and unavailable-engine/team cases separately when `MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1`. The busy category includes a poisoned lock and an `exec()` failure as well as lock contention; these counters classify control flow, not duration, utilization or the underlying cause. Unit coverage forces the unavailable and busy branches and checks that completed attempts equal the sum of the three outcomes. Profiling is disabled by default.

A fresh same-binary MPI4 L32 comparison on CPUs 0–15, one warmup and three alternating measurements, yielded:

| Workload | Inner workers/rank | Shared off | Shared on |
|---|---:|---:|---:|
| Opt | 1 | 2.853 s | 2.854 s |
| Opt | 4 | 2.717 s | 1.808 s |
| PhysCal | 1 | 2.635 s | 2.621 s |
| PhysCal | 4 | 3.010 s | 1.876 s |

These are wall medians with the scope and caveats described above; the 4 Opt and 80 PhysCal non-timing numeric files matched at printed precision across all measured runs per setting. The release executable SHA-256 was `019487c2a4c4a1b1db1de8b915a527102bc8535bb5f55b57df0d10e08fc816ae`. A separate counter-enabled diagnostic, excluded from the timing table, observed four participants on each rank; all 20160–20171 Opt and 18396–18452 PhysCal attempts succeeded, with zero busy and zero unavailable. [classified-mpi](classified-mpi/) retains raw results, source inputs/hashes, scripts and summaries. Apply the zero-context patch with `git apply --unidiff-zero`, build the MPI release binary as above, and run `classified-mpi/run.sh DEVELOPMENT_IMAGE TARGET TARGET INPUT_DIRECTORY FRESH_OUTPUT_DIRECTORY`; `uv run --no-project python classified-mpi/analyze.py FRESH_OUTPUT_DIRECTORY` recreates the summary. Adjust the fixed paths in the separately instrumented diagnostic script.

The five-case 1/2/4-worker runner comparison with both the gap profiler and the shared probe enabled also passed. In its observed 2- and 4-worker children, respectively, 427859 of 427860 attempts succeeded, zero took the busy/failed-lease path, and one took the unavailable path; two and four participant IDs were recorded. This explains those observed children, not the earlier unclassified 593026-attempt run. The per-process counters are relaxed diagnostic snapshots after synchronous work completes, and are not a global thread budget.

Finally, rerunning the `move-ab/run.sh` four-worker on/off comparison with these classified counters reproduced the same five-case event hash `3a18916a0c5f98b37b5fae979ae007a1922629d435ab21ae90a9614abaae984a` on both paths. The enabled whole-matrix child reported 593026 attempts, 591392 successes, zero busy/failed-lease and 1634 unavailable; four participant IDs were observed. `unavailable` includes both no published shared Engine and an Engine without a usable team, so this does not by itself identify which condition applied. The two compressed full event lists and output are in `classified-mpi/ab-*`; this remains a correctness diagnostic, not a performance measurement.
