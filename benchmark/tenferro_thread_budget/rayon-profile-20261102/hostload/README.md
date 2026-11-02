# Host-load-recorded Rayon scaling repeat (#581)

This repeats the earlier [scaling experiment](../README.md#matched-14-worker-end-to-end-comparison) with the identical existing binary and inputs, adding a separate host observer. It is not a measurement of the new process-budget implementation or current main. Binary SHA-256: `5ee7c581f626932bb88a73cf02f77667e8314f74843f768182f376c8654359cf`; its source, software/hardware and build provenance are in the parent report. The image remains `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615` (Linux x86_64, Xeon E5-2699 v3, Rust 1.99.0, pthread OpenBLAS).

CPU mask 0–3, Rayon capacity 4, inner workers 1/4, BLAS/OMP 1, tenferro SR with inferred budget, default spin and 40 µs work gate. L32 Opt/PhysCal and L64 Opt use the same fixed input copies as the earlier report; archive hashes identify them. Each cell has one excluded warmup and three alternating measured processes. Wall time includes CLI startup, calculation and output, excluding Docker startup. No concurrent build was started locally during this repeat. Frequency was not controlled and the shared host was not isolated.

| Workload | Inner workers | Wall median | User CPU median | System CPU median |
|---|---:|---:|---:|---:|
| l32-opt | 1 | 2.425 s | 2.860 s | 0.073 s |
| l32-opt | 4 | 2.080 s | 4.741 s | 0.663 s |
| l32-phys | 1 | 2.307 s | 2.281 s | 0.037 s |
| l32-phys | 4 | 2.966 s | 6.358 s | 1.344 s |
| l64-opt | 1 | 11.566 s | 12.441 s | 0.107 s |
| l64-opt | 4 | 5.491 s | 18.069 s | 2.678 s |

All finite non-timing numeric output tokens matched at printed precision across the six measured processes per workload (4 files per Opt, 80 per PhysCal), maximum absolute difference 0. This is not a full RNG/decision trace; independent runner parity checks are recorded elsewhere.

The host observer recorded 118 samples over approximately 117.093 seconds, including Docker launches. System-wide one-minute load average ranged 0.48–2.27. This is host load, not utilization of the four assigned CPUs or proof of isolation. Raw `/proc/stat` includes per-CPU counters, and CPU pressure snapshots are retained; they do not identify a particular competing process or establish a scheduling cause. Sampling overhead is included in the shared-host environment. In this repeat L32 PhysCal again had a higher four-worker wall median, while L64 Opt benefited; the small sample does not select a portable gate or explain the PhysCal slowdown.

`raw.tar.gz` retains all inputs' hashes, logs, timing and numeric outputs; `observer.jsonl.gz` retains host identity/topology, affinity, load, CPU accounting and pressure samples plus the successful child exit. No old measurement has been retroactively assigned these load values.

To reproduce, prepare the same inputs and binary using the parent report. The archived scripts are the exact executed commands with local paths. Copy `run.sh` to `/tmp/mvmc-581-scaling-hostload-run.sh`, then run `uv run --no-project python observe.py` with a fresh output directory and observer filename (edit both scripts' fixed paths for another host). Reanalyse the generated or extracted outputs with `uv run --no-project python ../scaling/analyze.py OUTPUT_DIRECTORY`. The saved `summary.json` contains every measured wall/user/system sample. Read `observer.jsonl.gz` as JSON lines to inspect individual load observations.
