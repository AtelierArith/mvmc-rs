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

For matrices of side 128 or larger, sampling retains four accepted rank-two
updates as vectors before materializing the inverse. Ratios include every
pending correction, including the separate electron-slot unit vectors.
Each return, partial interval and rebuild boundary flushes the queue. A second
inverse buffer uses the existing factorization output allocation, whose lifetime
does not overlap sampling; only small per-walker vector packets are added.
The resident kernel uses four warps per walker, with QP groups limited by the
shared-memory budget. It keeps the five-word proposal tape and acceptance rule.

The same matrix sizes use paired skew elimination after parameter updates or
sampling rebuilds. Up to four rank-two elimination changes are held in shared
memory, and the inverse grows two columns at a time. A failed factor, nonfinite
inverse, or projected overlap cancellation above 1024 triggers the conventional
factorizer on the device before publication. The cancellation threshold follows
Julia's conservative fallback policy; it is not a universal forward-error bound.
Smaller matrices retain the original operation order. FMA and warp reductions
on the larger-matrix path change rounding, with independent CPU factorization,
inverse residual, supplied-tape and complete Opt numerical checks under #550.

## Supported scope and checks

The current runner requires real normal orbitals, conserved spin in each
operator pair, Gutzwiller/Jastrow projection, `NExUpdatePath` 0, 1 or 2,
direct SR (`NSRCG=0`) and serial execution. Local spins (#660) and DH2/DH4
doublon-holon factors (#665) are supported. RBM, OptTrans, complex/general
orbitals, `NExUpdatePath=3` and grouped execution return errors.

### Doublon-holon factors DH2/DH4 (#665)

Julia GPU rejects DH (`ext/cuda_state.jl:41-42`), so this is new functionality
validated against the Rust CPU C-order path. The sampler keeps the Julia GPU
tape contract; the physics follows C `projection.c` and `parameter.c`:

- The neighbor tables (DH2 `[n2][Nsite][2]`, then DH4 `[n4][Nsite][4]`) are
  uploaded once at setup (`src/resident_dh.cuh`). A center contributes only as
  a holon or doublon, to counter `offset + x + (c/2 + 2*xm)*N` (C `MakeProjCnt`).
- `opt_counts` adds the DH counters; projection derivatives are the counters.
- A hop's DH log-weight change is the block-parallel sum over all
  (table, center) items of `P[after] - P[before]`, reduced in a fixed order and
  added to the Gutzwiller/Jastrow delta. An exchange changes no DH counter.
  `opt_delta` adds the same change for one- and two-move operators, so the
  Hamiltonian, Green functions and the full-Pfaffian fallback include it.
- `opt_gauge` applies C `SyncModifiedParameter`: `shiftDH2`/`shiftDH4` per
  d/h generator, the summed shift added to the Gutzwiller factors, then
  `shiftGJ`, with the flags of C `SetFlagShift`.

See `gpu/mvmc-gpu-cuda/tests/fixtures/dh/README.md` for inputs, bounds and timings.

### Local spins and exchange proposals (#660)

The proposal mix follows Julia GPU (`ext/cuda_runners.jl:52-58`,
`cuda_sampling_update.jl:215-341`, pin `7314015`): the exchange probability is
0, 0.5 or 1 for `NExUpdatePath` 0, 1 or 2, decided by tape word 3
(`u32/2^32 < p`). This GPU contract does not reproduce the C `getUpdateType`
draw. A hop whose source is a local-spin site is invalid, and the empty
destination is ranked among the empty sites of that spin without a local spin.
An exchange picks the opposite-spin partner with word 2 and is valid when both
sites are singly occupied by opposite spins. Its overlap is recomputed from full
Pfaffians, its projection ratio is exactly one (no charge or doublon count
changes), and an accept refactors the walker inside the hop kernel. Any queued
delayed updates are discarded rather than flushed, as for the #573 refactor.
Initialization ports `initial_configuration_kernel!`: a label permutation
assigns local-spin sites in site order, then each spin shuffles the remaining
sites. A uniform local-spin mask is uploaded once at setup, only when a local
spin exists; there are no new transfers during sampling. See
`tests/fixtures/localspin/README.md` for the Julia GPU fixtures and bounds. Hamiltonian
support includes Transfer, CoulombIntra/Inter, Hund, PairHop, Exchange and
spin-conserving InterAll. Failed state reconstruction or SR is reported. The per-family
[support matrix](gpu-resident-model-support.md) (#572) lists each rejected family,
its rejection site, CPU and Julia GPU status and the planned milestone order.

### Singular QP planes (#573)

An individual QP plane can be exactly singular while the weighted overlap
`sum_q w_q Pf_q` is finite and nonzero. The resident path follows the per-plane
recovery of the accepted Julia GPU reference (Julia-mVMC `7314015`):

- **Validity.** A device mask `valid[w][q]` marks a plane valid when its
  factorization has no zero pivot and a finite nonzero Pfaffian
  (`cuda_batched_inverse.jl:68`). The tests are exact; there is no conditioning
  threshold. An invalid plane publishes `Pf = 0`, and no all-valid path reads its
  inverse storage. A rebuild publishes a walker whenever its total overlap is
  finite and nonzero. Initialization still resamples walkers with any invalid
  plane (`cuda_initial_sampling.jl:118-153`).
- **Hops.** All-valid walkers use the inverse ratio and rank-two updates.
  - An accept whose candidate Pfaffian in some plane is zero or nonfinite
    refactors the walker inside the hop kernel instead of updating
    (`cuda_sampling_update.jl:343-371`).
  - A walker with an invalid plane computes every candidate Pfaffian directly
    and refactors on every accept (`:291-331`).
  - The refactor uses one block, per-walker scratch in the factorization
    buffers, a skew Parlett-Reid Pfaffian with partial pivoting, and a
    Gauss-Jordan inverse published as a mirrored skew pair. Queued delayed
    updates are discarded, not flushed. The accepted window restarts, so the
    host's bounded round schedule still holds; a completion check enforces it.
- **Energy/Green.** `opt_energy_fallback` evaluates every operator of such a
  walker with full Pfaffians of the moved configuration
  (`cuda_observables.jl:109-122,283-295`). Opt and PhysCal share it.
- **Derivatives.** For an invalid plane, `opt_cofactors` stores the exact
  cofactors `dPf(A)/dA(i,j) = (-1)^(i+j+1) Pf(A without rows/columns i,j)`
  (0-based `i<j`) where the inverse would otherwise be read
  (`cuda_sr.jl:154-187`). Valid planes keep `Pf * inverse`.
- **Failures.** A zero or nonfinite total overlap remains a hard failure and
  publishes nothing for that walker.
- **Cost.** Each fallback costs `O(n^3)` per term or per minor on one block,
  so it is meant only for exceptional walkers. On an RTX 3060, one invalid
  plane of one L128 walker adds about 5.5 s to an observation. The 8128 minors
  dominate that time. Walkers without an invalid plane skip the fallbacks after
  a mask check.

**Divergence from C.** C drops a sample whose `CalculateMAll` fails
(`vmccal.c:135-138`), and the Rust CPU/C-order path mirrors that
(`SlotStatus::Failed`). The resident GPU path follows Julia instead: such a
walker keeps contributing through cofactors and full Pfaffians. This is
deliberate under the GPU reference policy and does not change the CPU path.

Failures latch one sticky device status word; the first failing stage is kept
(`src/resident_status.cuh`, `ResidentFailure`): 1 rebuild, 2 energy/observable,
3 SR solve, 4 parameter update, 5 schedule, 6 derivatives. Initialization
reports its retry limit separately. On failure, the result boundary downloads
only the status word.

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
