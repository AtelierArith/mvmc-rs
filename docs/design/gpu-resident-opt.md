# GPU-resident Opt (#560)

`mvmc_gpu_cuda::opt::run_gpu_para_opt_from_namelist` is the ensemble GPU Opt
entry point. It follows Julia GPU's lifecycle and returns the energy history
and final parameter vector:

```rust,ignore
use mvmc_gpu_cuda::opt::{run_gpu_para_opt_from_namelist, OptConfig};
let result = run_gpu_para_opt_from_namelist("namelist.def", OptConfig {
    device: 0,
    nsteps: 20,
    walkers: Some(320),
    warmup: Some(128),
    interval: Some(128),
    seed: Some(79),
    ..Default::default()
})?;
```

Warmup and interval are proposal counts **per walker**. Omitted counts use
`NVMCSample`, `NVMCWarmUp * NVMCInterval * Nsite`, and `NVMCInterval * Nsite`.
One configuration per walker contributes to each SR step. Returned energies
are total energy averaged over walkers, before that step's parameter update;
divide by `Nsite` to obtain energy per site. Parameters include the final SR
update and gauge normalization. The existing CPU/C-order runner remains a
separate API with its original numerical authority.

The reference is Julia-mVMC GPU commit
`47452f9061ec3de049d891a7aa7fd9cfeac228a4`. Setup uses its SFMT parameter
initialization, initial.def/input overlays, gauge and translation normalization.
Configuration initialization and sampling use one ensemble-owned cuRAND
XORWOW generator. Sampling uses the Julia cached sampler's empty-destination
proposal and log-space acceptance with the `1e-100` overlap floor (the default
Julia Opt proposal at L128). Each proposal consumes five device UInt32 words.
The supplied-tape sampler remains available for independent checks.

## Ownership and transfers

A run owns one walker ensemble, rather than a host optimizer per walker.
Configurations, occupancy, inverse/Pfaffian caches and RNG stay resident.
Measurements consume those inverse tables directly. Sparse Slater-basis
contraction produces derivatives on the GPU, followed by FP64 BLAS moments,
masked/stabilized covariance, native Cholesky solve, parameter gauge, Slater
assembly and inverse refresh. No large sensitivity matrix or host sample
collection is required.

Covariance preserves Julia's separately rounded mean outer product and
subtraction. Fusing these into an FMA would create a spurious variance for a
single sample, activating rows that should be identically zero. The pruning
diagonal and solver matrix share this operation, with a one-walker regression.

The whole warmup/optimization series executes in one raw CUDA scope and ends
with one stream synchronization. It issues zero explicit host/device copies.
Setup uploads the immutable model once; initialization reads one pending-count
scalar per attempt. Finalization reads a sticky device status and, on success,
the energy history and parameter vector. Failed SR or inverse refresh cannot
publish a successful host result. RawExec counters measure explicit copies and
barriers, excluding provider/library internals and kernel argument traffic.

Compiled modules and native handles are tied to an exact tenferro runtime
identity. A thread-local runtime cache retains this identity per selected
device, avoiding repeated compilation on consecutive API calls. RNG and model
arrays belong to individual calls and are not stored in this cache. Tests
interleave seeds and device ordinals to check isolation.

## Supported scope and checks

The current runner requires real normal orbitals, conserved spin in each
operator pair, Gutzwiller/Jastrow projection, ordinary hopping (`NExUpdatePath=0`),
direct SR (`NSRCG=0`) and serial execution. Local spins, RBM, DH, OptTrans,
complex/general orbitals and grouped execution return errors. Hamiltonian
support includes Transfer, CoulombIntra/Inter, Hund, PairHop, Exchange and
spin-conserving InterAll. Individual QP matrices must remain nonsingular;
singular-sector cofactor fallback and delayed/paired inverse optimizations are
remaining work under #560/#550. Failed state reconstruction or SR is reported.
Nonfinite proposal overlaps are rejected before updating the configuration,
inverse, Pfaffian or cached overlap (#564); finite overlaps retain log-space
acceptance for extreme ratios and zero uniform draws.

Independent Julia fixtures cover L128 initialization, energy, derivatives and
one SR update, plus nonzero correlations and all 324 two-pair overlays on three
sites in an L64 model. See
[`opt_julia.md`](../../gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia.md) for provenance
and bounds. Additional tests check the SR residual, failed-solve publication,
extreme log-space acceptance, transfer counts and repeated complete 20-step
optimizations. CPU workspace tests continue to check the C paths.

For a persistent-process E2E timing worker:

```sh
cargo build --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --example resident_opt_worker
# Send repetition numbers followed by "quit" on stdin.
gpu/mvmc-gpu-cuda/target/release/examples/resident_opt_worker \
  /absolute/input/namelist.def /absolute/results 320 20 128 128
```

Every timed call includes parsing, state construction, GPU initialization,
warmup, all Opt steps, final transfers and state destruction. The first call
includes kernel/library startup; warm calls reuse the execution runtime.
Result receipt serialization occurs after the timer.
