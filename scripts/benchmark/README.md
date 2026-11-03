# CPU / GPU benchmarks in one command

From the checkout (or using the script's absolute path from any directory):

```bash
bash scripts/benchmark_cpu.sh
bash scripts/benchmark_gpu.sh
```

Both measure the same half-filled real Hubbard chain L=128 definitions:
20 SR steps of optimization and one fixed-parameter PhysCal window without
Lanczos, seed 79, eight QP planes and 320 samples/walkers. CPU runs C, Julia and
Rust with **1, 4, 8 and 16 physical cores**. GPU runs Julia and Rust on one NVIDIA
GPU; C has no GPU implementation. Each has one warmup and three measured runs.
Python is always invoked with **`uv run --no-project python`**.

The scripts prepare the pinned reference submodules, instantiate/build Julia's
locked packages, build Rust release executables and (CPU only) the C reference,
run measurements sequentially, and print Markdown tables. Results go to a fresh
`target/bench/cpu-<UTC>-<pid>` or `gpu-<UTC>-<pid>` directory. See `report.md`,
`summary.json`, `timings.json`, `metadata.json`, per-implementation logs and raw
outputs. The shared `target/benchmark-entrypoints` build cache avoids rebuilding
unchanged artifacts. Keep this cache separate for every checkout. C cache entries
are keyed by source and image ID and verify the executable hash before reuse.

## Prerequisites

- Linux, Rust/Cargo, `uv`, Juliaup with **Julia 1.13.1**, `git`, `taskset`, and the
  ordinary native BLAS/LAPACK/compiler dependencies in `docs/DEVELOPMENT.md`.
  No Python dependencies need installation outside `uv`.
- CPU: a running Docker daemon. An existing mvmc-rs Dev Container image is
  auto-detected. Otherwise `mvmc-rs-benchmark:local` is built automatically from
  `.devcontainer/Dockerfile`; this first build can take substantial time and
  needs network access. `--image IMAGE` or `MVMC_BENCH_IMAGE` selects an existing
  image. The first full sweep needs at least 16 allowed physical cores; on a
  smaller machine select `--threads` explicitly. Allowed core IDs are discovered
  from Linux topology and affinity, rather than assuming CPU 0 is available.
- GPU: NVIDIA driver, `nvidia-smi`, a supported CUDA GPU and compatible CUDA 12
  runtime/NVRTC/cuBLAS/cuSOLVER/cuRAND libraries for the Rust CUDA 12.8 build.
  Native libraries may already be on `LD_LIBRARY_PATH` or in `/usr/local/cuda`.
  The script also discovers installed CUDA 12 Julia artifacts via Julia's actual
  depot paths; no user name or artifact hash is hard-coded. Set
  `MVMC_BENCH_CUDA_LIBRARY_PATH` if needed. Julia retains its own pinned GPU
  software stack; different CUDA library versions are not silently equated.

The scripts never install system packages, modify the pinned manifests, or
silently skip a missing/failed implementation. A global per-user file lock
prevents these CPU/GPU entrypoints from running together, including during
build/preparation. Manual benchmarks and other users are outside that lock;
avoid other builds or loads while measuring. CPU frequencies are not locked.

## Options and quick verification

```bash
bash scripts/benchmark_cpu.sh --help
bash scripts/benchmark_gpu.sh --dry-run
bash scripts/benchmark_cpu.sh --smoke
bash scripts/benchmark_gpu.sh --smoke --device 1
bash scripts/benchmark_cpu.sh --threads 8 16 --output /tmp/mvmc-cpu-results
bash scripts/benchmark_gpu.sh --device GPU-UUID --output /tmp/mvmc-gpu-results
```

`--smoke` uses two SR steps, 16 samples/walkers and one measured repetition
(after one warmup); CPU uses 1 and 4 cores. **Smoke results are not the published
full-workload performance figures.** `--threads`, `--reps`, `--workloads opt phys`,
`--build-dir` and `--output` can narrow/reproduce a run. Output directories must
be fresh; failure retains partial outputs/logs. `--dry-run` writes nothing.

GPU selection uses `--device` first, then the visible set from
`CUDA_VISIBLE_DEVICES`, otherwise all devices. Within the allowed set it picks
the device with least memory allocated at selection time and records its UUID. Both implementations receive the same visible physical device
and use visible ordinal 0. This observation is not an exclusive device lock.

## Timing and sampling contracts

- CPU C/Rust: process wall including startup/exit, C also `mpirun -np 1`; Docker
  launch is excluded on both sides. C/Rust order alternates by repetition.
  Julia: warm public API including parsing and file output, excluding startup,
  initial JIT, explicit GC and result serialization. These are different scopes.
- CPU Rust: explicit PF/SR `c-order`, matching the recent performance comparison;
  these are not measurements of every backend or the default `calc-m-all` path.
  All use one BLAS thread, with the configured number of internal workers.
- GPU: warm public APIs, no per-call output, excluding startup/JIT, repetition
  zero, explicit Julia GC and post-timing serialization. Rust Opt's public API
  lifetime includes its internal result/state teardown. CUDA completion is
  included by the Rust API's host-result contract and explicit Julia synchronize.
- Opt uses the dedicated resident ensemble (320 walkers in the full case) on
  GPU. CPU runs one chain with 320 samples. PhysCal Rust GPU retains the single
  SFMT chain with `(1+320)*128 = 41,088` configured proposals, whereas Julia GPU
  uses 320 XORWOW walkers and `320*(128+128) = 81,920` proposals. Identical input
  files/parameters **do not imply equal RNG trajectories or sampling work**.
  CPU and GPU wall ratios must not be called matched kernel speedups.
- Fixed parameters are copied unchanged and SHA-256 checked against
  `9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90`.

The driver requires complete finite timing groups and runtime success/output
checks. It does not prove cross-language numerical/RNG equivalence or test the
1.1x performance acceptance criterion; that separately validated cohort lives in
`benchmark/cpu_scaling_l128/results/20261103-issue629`.

CPU Lanczos is a separate workload. To reproduce that report, use
`benchmark/lanczos_refresh/README.md`; it is not folded into L128 timings.
