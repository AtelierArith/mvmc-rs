# Julia versus Rust MPI comparison (#490)

[Linux x86_64 measurements, 2026-10-09](results/linux-x86_64-20261009.md)
cover L16/L24/L32 at 300 SR steps, including both sample policies and the
Dev Container maintenance checks. The commands below default to a shorter
20-step development run; use `--steps 300` for the recorded workload.

Run in the repository's Linux x86_64 Dev Container. Both implementations use
its MPICH 4.2.0 (Hydra, PMI1, no PMIx), one BLAS thread per rank, and the same
Expert-mode Hubbard input. Julia is pinned to 1.13.1 and the unchanged lock in
`benchmark/julia_comparison/Manifest-v1.13.toml`. Rust uses release mode and
the production C-order CPU backend. No GPU is used.

```sh
bash .devcontainer/install-julia.sh
bash .devcontainer/verify-mpi.sh
# Optional setup/build only, before the machine is idle enough for measurement:
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-setup --prepare-only
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-total --model 16 --model 24 --model 32
# Fixed samples per rank, instead of fixed total work:
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-per-rank --sample-policy per-rank --model 16
# Longer optimization:
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-300steps --steps 300 --model 16
# Hybrid layouts, fixed total samples and production Rust work gates:
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-hybrid-auto --layout 1x4 2x2 4x1 \
  --steps 300 --model 16 --model 24 --model 32
# Separately measure actual pooled Rust kernels even on small inputs:
uv run --no-project python scripts/bench_mpi.py \
  --output target/bench/mpi-hybrid-pooled --layout 1x4 2x2 4x1 \
  --rust-inner-threshold 1 --steps 300 --model 16 --model 24 --model 32
```

Defaults: MPI ranks 1 and 4, 20 SR steps, 300 samples, one complete warmup,
three measured repetitions. `--sample-policy total` divides 300 samples among
ranks (300 at rank count 1; 75 per rank at rank count 4). `per-rank` keeps 300
per rank, so four ranks perform 1,200 samples per step. Total samples must be
divisible by every rank count. Both policies retain `NSplitSize=1`, `NStore=1`,
`NSRCG=0` and `RndSeed=1`. Four ranks use four independent seeded chains.

`--layout` specifies processes and computation threads independently. Julia
sets both `JULIA_NUM_THREADS` and `JULIA_MVMC_INNER_THREADS=1`; native PfaPack
threading remains disabled (`JULIA_MVMC_PFAPACK_THREADS=0`). Rust sets
`MVMC_RS_INNER_THREADS` and runs the whole MPI lifecycle inside `threading::install`,
matching the CLI's FUNNELED ownership. Its production work gates can keep small
inputs serial despite a configured pool. `--rust-inner-threshold 1` explicitly
requests pooled dispatch and is a separate performance policy, not the default.
Both implementations complete their full warmup and then run a separate
diagnostic of at most 20 steps. Rust records actual kernel worker entries;
Julia records sampled MVMC stacks grouped by thread. Both are excluded from timings;
measured repetitions have observation disabled. Julia uses only the default
compute pool (`JULIA_NUM_THREADS=N,0`) and one GC thread.
Every rank reports its actual BLAS thread count, which must be one. Julia and
explicit-threshold Rust runs require evidence of at least two kernel workers
when computation threads exceed one. Automatic Rust dispatch records any
serial fallback instead of treating pool capacity as parallel work.

`--julia-source` can select a separate optimization worktree. The pinned lock
is retained; source commit/diff and a copy of the executed optimizer sources
are saved with the result, so upstream changes cannot replace baseline evidence.

Each implementation runs warmup and measurements in the same MPI process,
starting each optimization from the input/seed again. The timer surrounds the
production optimization runner with MPI barriers and records the maximum
elapsed time across ranks. Input parsing, allocation, initialization, sampling,
SR and file output are included. Launcher startup, package setup and warmup
are excluded, including first-call Julia JIT. Rust and Julia use matching
runner-level timing boundaries; no rounded CLI timings are used. These are
full optimization timings, not isolated sampling-kernel timings.

The runner validates every actual MPI world/rank, completed steps, finite
energy and output files. CSVs, a median timing/throughput report, logs, copied
inputs, exact Julia lock, isolated MPI preferences, binary hashes and library
metadata stay in a new result directory. The Julia project uses symlinks to
the reference's local packages; its preferences and lock never overwrite the
reference checkout. Missing native SFMT/PfaPack libraries are built before
measurement. Cargo honors the container's named-volume target directory.
The executed Python/Rust/Julia worker sources are copied into
`executed-sources/` so later edits do not lose the measured version.

The report compares Rust and Julia at the same rank/sample count. For `total`,
the 1-to-4 time ratio measures scaling at fixed sample work; warmup costs and
independent chains still change. For `per-rank`, use sample throughput gain:
the total work increases fourfold, so the time ratio alone is not strong
scaling. Changing rank count changes the optimization trajectory. Energies
are diagnostic output, not numerical parity or convergence evidence; do not
interpret timing agreement as proof of RNG/algorithm parity.

Failures and timeouts abort measurement; partial CSV/logs remain for diagnosis.
Each command has a configurable `--timeout` (default 1,800 seconds). Setup,
build and measurement run sequentially. Avoid other CPU workloads while timing.

Light runner checks (no MPI/reference runtime required):

```sh
uv run --no-project python scripts/test_bench_mpi.py
```

2026-10-10 hybrid measurements and Julia optimization: [report](results/linux-x86_64-hybrid-20261010.md).
