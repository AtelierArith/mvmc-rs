# Function-level GPU/CPU suite (issue #450)

A portable suite that first checks the **numerics** of every GPU (and tenferro/CPU) function
against the C-order CPU oracle, and only then records timings. Run it on any CUDA machine
(for example an A100) and send back one archive.

* Primary output: a PASS/FAIL verdict per function and size with an explicit, justified bound.
  The run exits non-zero if any check fails.
* Secondary output: timings (median, min, max after warm-ups), speedups, break-even sizes and a
  "GPU-ize?" recommendation per function.

## Quick start (English)

Requirements: an NVIDIA driver (`nvidia-smi` works), **Rust >= 1.96** (`rustup update stable`),
git submodules (`git submodule update --init --recursive`), and either

* **native**: CUDA toolkit >= 12.6 (libnvrtc, libcublas, libcusolver, e.g. `/usr/local/cuda`) and
  OpenBLAS (`apt install libopenblas-dev liblapack-dev pkg-config`), or
* **docker**: docker with the NVIDIA container toolkit (uses `nvidia/cuda:12.9.1-devel-ubuntu24.04`,
  installs OpenBLAS inside, mounts your `~/.rustup` and `~/.cargo`; override the image with
  `MVMC_RS_CUDA_IMAGE`).

```bash
# native, 10-minute smoke run
scripts/bench/run_all.sh --native --quick --out bench-out
# the real A100 run (about 2 hours), native or docker
scripts/bench/run_all.sh --native --full --gpu 0 --out bench-out
scripts/bench/run_all.sh --docker --full --gpu 0 --out bench-out
```

`--native` checks the toolchain first and stops with a clear message when something is missing
(`--preflight-only` runs just the checks). Run on a quiet machine; CPU rows are load sensitive.
Do not set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`: the suite sets the 1-core baseline itself and
records the thread environment.

Expected runtime: build 5-15 min (first time), `--quick` about 10 min, `--full` about 2 h (one GPU).

**Sending results back:** send `bench-out/results-<host>-<date>.tar.gz` (a few hundred KB). It contains
`report.md`, one CSV per family, raw logs and `metadata.txt` (GPU, driver, CUDA/cuBLAS/cuSOLVER/NVRTC,
CPU, OS, rustc, git revision, tenferro/cudarc versions, thread settings). To compare machines:

```bash
uv run --no-project scripts/bench/analyze.py results-a100.tar.gz results-rtx3060.tar.gz --out comparison.md
```

## Quick start (Japanese / 日本語)

必要なもの: NVIDIA ドライバ (`nvidia-smi` が動くこと)、Rust 1.96 以上、サブモジュール
(`git submodule update --init --recursive`)、および次のどちらか。

* native: CUDA Toolkit 12.6 以上 (libnvrtc / libcublas / libcusolver) と OpenBLAS
  (`apt install libopenblas-dev liblapack-dev pkg-config`)
* docker: NVIDIA Container Toolkit 付き docker (`nvidia/cuda` イメージ。OpenBLAS はコンテナ内で導入)

```bash
scripts/bench/run_all.sh --native --quick --out bench-out        # 約10分の動作確認
scripts/bench/run_all.sh --native --full --gpu 0 --out bench-out # A100 本番 (約2時間)
scripts/bench/run_all.sh --docker --full --gpu 0 --out bench-out # docker の場合
```

最初に数値の正しさ (C 順序の CPU オラクルとの偏差が明示した上限以内か: PASS/FAIL) を検証し、
速度は参考値として記録します。FAIL があると終了コードは非 0 です。結果として
`bench-out/results-<host>-<date>.tar.gz` ができるので、それをそのまま送り返してください。

## What is run

| family | functions | oracle and numerical bound | timings |
|---|---|---|---|
| `pfaffian` | batched Pfaffian + inverse, f64/c64, n = 16..256, NQP = 8, B up to the memory limit; zero-pivot/NaN statuses | pfapack C order; per plane `max(inv rel, pf rel) <= 16 n eps cond(A)`; independent invariants `A inv = I`, skew `inv`, `Pf^2 = det` | pfapack 1 thread / rayon all cores / tenferro CPU / CUDA total, kernel, upload, download |
| `sr` | Gram, S/g assembly, Cholesky solve, CG matvec, full CG solve (fixed iterations), `dot_general`, `cholesky` (f64, c64) | `COrderSr` (OpenBLAS); bounds `4 k eps`, `8 eps`, `8 n eps kappa`, `4 (k+n) eps`, `2 K kappa 4 (k+n) eps`; NPara and samples up to 2e4 in `--full` | C order 1 core and all cores, tenferro CPU, tenferro CUDA |
| `sr_resident` | device-resident SR (#447) | hook: runs `examples/bench_sr_resident.rs` when present; otherwise one `NotAvailable` row (never invented numbers) | by that program |
| `sampler` | teacher-forced and free-run lock-step sampler vs CPU, resident inverse; stage timings (propose, accept, recompute); end to end vs CPU multichain, W up to 4096, L up to 256 | exact RNG state/draw count/configuration, decision flips located by the #424 teacher (defects must be 0), weights `<= 2*16*(n+s)*eps*kappa/sqrt(w_min)` per walker, derived from the recompute plus rank-1/2 update error (kappa of the Slater planes, s updates between recomputes; ratios observed/allowed are reported), resident inverse `<= 1e-6` (a stale resident download is reported separately as `sampler_download_synchronization`) | CPU 1 thread, CPU multichain (all cores), CUDA pinned/pageable |
| `transfers` | pinned, pageable, tenferro, copy/kernel overlap, ping-pong | bitwise exact round trip (a copy has no arithmetic) | per-call time and bandwidth vs size |

Inputs for the sampler (Hubbard chain L = 16..256) are generated with the Rust StdFace port
(`mvmc --dry-run`) into the output directory.

## CSV schema

One file per family under `csv/` (all share the header):

`family,function,variant,dtype,params,reps,median_s,min_s,max_s,dev_metric,dev_value,dev_bound,dev_ratio,verdict,note`

* `params`: `key=value;...` (for example `n=64;NQP=8;B=64;planes=512`, `NPara=2000;samples=20000`,
  `L=64;W=512`). CPU rows of the Pfaffian family time a prefix of `planes` planes (planes are
  independent); compare per-plane times.
* `median_s,min_s,max_s,reps`: seconds; min/max empty when only a median is available.
* `dev_*`: deviation versus the oracle, its bound and `dev_value/dev_bound`.
* `verdict`: `PASS`, `FAIL`, `ERROR` (unexpected runtime error; counts as failure), `ORACLE` (the
  reference itself), `INFO` (timing only), `NotAvailable` (function absent in this build; no
  numbers), `SKIPPED` (memory limit etc.; the note says why).

The #447 hook: a program `gpu/mvmc-gpu-cuda/examples/bench_sr_resident.rs` that accepts
`--quick|--full --out FILE.csv`, writes this schema and exits non-zero on FAIL is picked up
automatically.

## Report

`report.md` (per archive) and `analyze.py` (merged) list, in order: 1. Numerical validation
(verdicts and worst deviation/bound per function and machine), 2. Timing (reference) with speedup
tables and break-even sizes, 3. "GPU-ize?" recommendation (a function with any FAIL is never
recommended), 4. NotAvailable/skipped rows, then the metadata block. Example:
[`results/rtx3060-reference.md`](results/rtx3060-reference.md) (RTX 3060, FP64 at 1/64 of FP32).

## Caveats

* The `--full` sampler timing grid is capped at `W * L^2 <= 2.1e6` to keep the 2 h budget (larger points are
  `SKIPPED` rows); the numerical checks run for L = 16..256. Raise the cap with `MVMC_BENCH_WORK_CAP=1e7 scripts/bench/run_all.sh ...`.

* Timings of the sampler depend on host cores as much as on the GPU: its host side runs one thread
  per walker; compare against the multichain row of the same machine.
* The numerical bounds are the standard forward-error bounds recorded in the source headers of
  `gpu/mvmc-gpu-cuda/examples/function_suite/*.rs`; they were not tuned to the GPUs. Do not loosen
  them to make a run pass; send the failing archive instead.
