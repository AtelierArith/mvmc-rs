# Host E2E comparison: Rust MPI16 (Dev Container) vs Julia GPU vs Rust GPU

Reproducible whole-runner timing for the Hubbard-chain E2E workload
(variational optimization + PhysCal) on the Linux x86_64 host with an
NVIDIA A100. Issue: [#544](https://github.com/AtelierArith/mvmc-rs/issues/544).

This benchmark answers "which GPU path is faster end to end" by timing the
**public runner** (input parse + initialization + Monte Carlo + SR + output)
for three configurations:

| case | binary / runner | device | location |
| --- | --- | --- | --- |
| `rust-mpi16` | `mvmc --features mpi`, `mpiexec -n 16` | CPU (MPICH 4.2.0) | repository Dev Container |
| `julia-gpu` | `extern/Julia-mVMC/test/gpu/benchmark_end_to_end.jl` | A100 GPU | host |
| `rust-gpu` | `gpu/mvmc-gpu-cuda/target/release/mvmc-cuda` | A100 GPU | host |

## Default configuration

All cases share one input and budget (override through the environment):

* model `L128` (`benchmark/hubbard_chain/inputs/hubbard_chain_L128`),
* `STEPS=20` SR steps, `SAMPLES=320` retained samples per step,
* `WARMUP_SWEEPS=1`, `INTERVAL_SWEEPS=1`, `SEED=79`,
* PhysCal: `NVMCCalMode=1`, `PHYSCAL_RUNS=1` (`NDataQtySmp=1`),
* `WARMUPS=1`, `REPS=3`.

Samples are distributed differently by design and this is stated, not hidden:

* `julia-gpu` uses `SAMPLES` **independent GPU walkers** (XORWOW chains),
* `rust-gpu` runs **one production chain** with `NVMCSample=SAMPLES`,
* `rust-mpi16` runs **16 chains** of `SAMPLES/16` samples (`NSplitSize=1`,
  the C `RndSeed + rank` group seed).

Equal retained samples do not imply equal trajectories or effective sample
sizes; times are observed lifecycle timings.

## Prerequisites

Host (Julia GPU and Rust GPU):

```sh
# Julia 1.13.1 + the test/gpu project (downloads CUDA.jl artifacts on first run)
julia --project=extern/Julia-mVMC/test/gpu -e 'using Pkg; Pkg.instantiate(); Pkg.build("PfaPack"); Pkg.build("SFMT")'
# Rust GPU binary (standalone workspace)
(cd gpu/mvmc-gpu-cuda && cargo build --release --locked --bin mvmc-cuda)
```

Dev Container (Rust MPI16): the standard `.devcontainer` provides MPICH 4.2.0.
Start it once with `devcontainer up --workspace-folder .`.

## Running

```sh
# 1) Rust GPU on the host
bash benchmark/host_gpu_comparison/run_rust_gpu.sh

# 2) Julia GPU on the host
bash benchmark/host_gpu_comparison/run_julia_gpu.sh

# 3) Rust MPI16 inside the Dev Container
devcontainer exec --workspace-folder . \
  bash benchmark/host_gpu_comparison/run_rust_mpi16.sh

# 4) Combine the recorded times
bash benchmark/host_gpu_comparison/summarize.sh
```

Every case writes `<case>.csv` (`case,phase,rep,seconds` plus an internal
`All [0]` column where the run provides `zvo_CalcTimer.dat`) and raw logs into
`benchmark/host_gpu_comparison/results/<UTC timestamp>/`, together with an
`environment.txt` metadata block (uname, nvidia-smi, CUDA/NVRTC, rustc, Julia,
git revision, input hashes).

## Fixed-parameter checkpoint

PhysCal needs a fixed-parameter file. The Rust optimizer writes real/zero
**pairs** when the output window has a single point and triples otherwise
(`crates/mvmc-core/src/io.rs`), while the PhysCal reader expects triples
(`crates/mvmc-core/src/initial_params.rs`). `run_rust_gpu.sh` therefore
generates the checkpoint with a two-point window (`--nsmp 2`) and both Rust
cases reuse it. Julia generates its own checkpoint internally.

## Caveats

* `rust-gpu` and `rust-mpi16` use the production C-order CPU baseline for the
  parts not on the device; `julia-gpu` runs its own GPU sampling. The sampling
  structure differs (see above).
* Numbers depend on host load; run on a quiet machine and prefer medians.
* The Dev Container and the host are the same physical machine here.
