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

The current runner requires normal orbitals, conserved spin in each
operator pair, Gutzwiller/Jastrow projection, `NExUpdatePath` 0, 1 or 2,
and serial execution. SR is direct (`NSRCG=0`) or C's conjugate gradient
(`NSRCG≠0`, #670). Local spins (#660), DH2/DH4
doublon-holon factors (#665), real RBM factors (#667) and real OptTrans
weights (#674) are supported with real orbitals. Complex normal orbitals
(#678) are supported with Gutzwiller/Jastrow factors, complex QP weights
and coefficients, exchanges and direct SR. General orbitals,
`NExUpdatePath=3` and grouped execution return errors.

### Complex normal orbitals (#678)

Julia GPU rejects complex parameters (`ext/cuda_state.jl:39`), so this is new
functionality validated against the Rust CPU C-order complex path. The
sampler keeps the Julia GPU tape contract. The physics is C's complex normal
path (`AllComplexFlag≠0`):

- **Layout.** Parameters, Slater tables, inverses, Pfaffians, QP weights,
  overlaps and Hamiltonian coefficients are interleaved `(re, im)` FP64
  pairs (`src/resident_complex.cuh`). Projection factors use the real parts,
  as C `LogProjVal`/`LogProjRatio` (`creal(Proj)`); the sampler reads a real
  projection copy that the gauge refreshes. Every complex product and sum is
  separately rounded.
- **Rebuild.** C `CalculateMAll_fcmp` runs on the device as the batched complex
  PFAPACK kernel already in `pfaffian_batched.cu`
  (`pfinv_plane<Cx<double>>`: `zsktf2`, `utu2pfa`, `utu2inv`), followed by the
  #573 validity commit (`interval_complex.cu`).
- **Sampling.** `interval_complex_hop` proposes with the five-word tape,
  scores a hop with C `CalculateNewPfM2`
  (`pfMNew = -pfM sum_j invM[a][j] sltE[a][j]`), accepts in log space on
  `2 (LogProjRatio + log|ip_new| - log|ip_old|)` (C `vmcmake.c`,
  `creal(clog(ip))`, with the `1e-100` floor) and applies C
  `UpdateMAll_child_fcmp` immediately. There is no delayed queue or paired
  factor. Exchanges and walkers with an invalid plane use full complex
  Pfaffians and refactor in the kernel (complex `resident_dense.cuh`
  counterparts), as the real path does.
- **Observables.** Operator values are C `locgrn.c`'s
  `conj(ProjRatio ip_new / ip)`, i.e. `<psi|O|x>/<psi|x>`; the local energy is
  `sum_t c_t v_t` with complex Transfer/InterAll coefficients (C `calham.c`).
  PhysCal's TwoBodyGEx row is C `calgrn.c`'s `G1[idx0] conj(G1[idx1])`, and
  `PhysCalResult` reports the imaginary parts in `*_imag`.
- **Derivatives.** C `SlaterElmDiff_fcmp`: `O_k = -sum W_q coeff Pf_q
  invM[j][i] / ip` for d/dRe, and `i O_k` for d/dIm; the projection slots are
  the counters and 0 (C `vmccal.c`). Invalid planes use the exact complex
  minors.
- **SR.** C's complex `StochasticOpt` solves a real system over `2*NPara`
  components: `S_ij = Re<O_i^* O_j> - Re<O_i> Re<O_j>` and
  `g_i = -2 dt (Re<e O_i> - Re<e> Re<O_i>)` (`stcopt.c`, `stcopt_dposv.c`).
  With the stacked real matrix `D = [Re O; Im O]` (`2W x 2NPara`),
  `Re<O_i^* O_j> = (D^T D)_ij / W`, `Re<O_i>` is the mean of the first `W`
  rows and `Re<e O_i> = (D^T [Re e; -Im e])_i / W`. The direct-SR kernels and
  the Cholesky solve run unchanged on `D`; the parameter update adds the
  solution to the interleaved parameters, slot for slot.
- **Gauge.** C `SyncModifiedParameter` with complex parameters (`opt_gauge_c`):
  `shiftGJ` on the real parts, then the Slater block scaled by
  `D_AmpMax / max|z|` (`cabs`).
- **Not supported in complex mode** (rejected before allocation): RBM,
  OptTrans, DH2/DH4, local spins and SR-CG for Opt. PhysCal ignores `NSRCG`.
  A complex `ParaQPTrans` weight in real mode stays rejected; C's real path
  and Julia GPU take its real part.

See `gpu/mvmc-gpu-cuda/tests/fixtures/complex/README.md` for inputs, bounds
and timings.

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

### RBM correlation factors (#667)

Julia GPU rejects RBM (`ext/cuda_state.jl:40`), so this is new functionality
validated against the Rust CPU C-order kernels. The sampler keeps the Julia GPU
tape contract. The physics is C `rbm.c` on the complex normal path
(`vmcmake.c`, `calham.c`, `calgrn.c`, `vmccal.c`) with real coefficients, where
every quantity is real. C's real path omits the RBM factor from sampling and
the local energy, and its `RBMDiff` reads unsaved counters
(tmisawa/Julia-mVMC#59); the resident path does not reproduce that.

- The parameter layout is C's: projection, RBM (`NRBM`, nine sections in C
  order), Slater. Complex coefficients and `Nneuron≠0` are rejected; every
  present section must be completely mapped, as the C readers require.
- Index tables (physical counters per site, family, hidden coefficient and
  couplings per neuron) and the coefficients are uploaded once at setup
  (`src/resident_rbm.cuh`). Each walker keeps its hidden fields
  `theta[w][h]` on the device.
- `opt_rbm_theta` evaluates C `MakeRBMCnt` after initialization and after
  every SR update (C rebuilds at each `VMCMakeSample`); an accepted hop or
  exchange applies C `UpdateRBMCnt` (an exchange is two hops). Both use C's
  operation order without FMA, so the fields equal the CPU C-order values bit
  for bit.
- A proposal adds C `LogRBMRatio` to the projection log-ratio: the physical
  layer in counter order, and per neuron `|θ'|-|θ|+log((1+e^{-2|θ'|})/(1+e^{-2|θ|}))`,
  reduced over the block. C takes one `log` per block product instead; the
  difference is rounding only. `opt_delta` adds the same ratio for one- and
  two-move operators (right pair first), so energies, Green functions and the
  full-Pfaffian fallback include it.
- `opt_rbm_diff` writes the real part of C `RBMDiff` between the projection
  and Slater columns. The gauge leaves RBM unchanged and rescales Slater only,
  as C `SyncModifiedParameter` does, then refreshes the device coefficients.

See `gpu/mvmc-gpu-cuda/tests/fixtures/rbm/README.md` for inputs, bounds and timings.

### OptTrans weights (#674)

Julia GPU rejects variational OptTrans (`ext/cuda_state.jl:43`), so this is
new functionality validated against the Rust CPU C-order kernels. The sampler
keeps the Julia GPU tape contract; the physics follows C `qp.c`, `slater.c`,
`vmccal.c` and `parameter.c`:

- **Layout.** The `NQPOptTrans` weights are the last parameter block, after
  projection, RBM and Slater (C `setmemory.c:302`). The planes are
  `NQPFull = NQPFix * NQPOptTrans` with `q = opt*NQPFix + NSPGaussLeg*mp + sp`.
  Each Slater table composes the OptTrans map before QPTrans, with both signs
  (C `slater.c:52-90`); setup builds the plane index tables once.
- **Weights.** The fixed weights `QPFixWeight` are uploaded once at setup.
  `opt_gauge` applies C `UpdateQPWeight` (`qp.c:130-148`),
  `QPFullWeight[i*NQPFix+j] = OptTrans[i] * QPFixWeight[j]`, after every SR
  update and before the rebuild, so sampling, energies and derivatives always
  use the current weights. There are no new transfers.
- **Derivatives.** `opt_derivatives` writes C `calculateOptTransDiff`
  (`vmccal.c:639-652`), `O_i = sum_j QPFixWeight[j] Pf[i*NQPFix+j] / ip`,
  summed in C's order and divided as C does. An invalid plane publishes
  `Pf = 0` (#573). The Slater derivatives sum every plane with its full
  weight, as C `SlaterElmDiff_fcmp` does. The resident path keeps the
  mathematically correct real-mode slots; C's real path shifts them (#370,
  tmisawa/Julia-mVMC#55).
- **Gauge.** As C `SyncModifiedParameter` (`parameter.c:163-176`), the Slater
  rescale covers the Slater block only, and then
  `OptTrans *= 1/max|OptTrans|`. If the maximum is zero, the weights are not
  scaled, as in the Rust C-order sync (C would divide by zero).
- **Activation and flags.** The resident parser activates OptTrans from an
  `OptTrans` definition and optimizes every real component, as Julia does.
  C activates OptTrans only with `-o`. Its `GetInfoOptTrans` writes
  `OptFlag[fidx]` with an undoubled index (`readdef.c:2400`), so the OptTrans
  component flags are never written. That is a C defect (#676,
  tmisawa/Julia-mVMC#94): the Rust C-order CLI (`-o`) also optimizes the real
  weight components now. A resident model with those flags cleared keeps the
  weights at their input values (`resident_opt_opttrans_repeats_twenty_steps`).

PhysCal uses the normalized input weights (C normalizes them at setup for both
modes, `vmcmain.c:276`) and never changes them. See
`gpu/mvmc-gpu-cuda/tests/fixtures/opttrans/README.md` for inputs, bounds and
timings.

### SR-CG (`NSRCG≠0`, #670)

Julia GPU Opt rejects CG SR (`ext/cuda_runners.jl:27-28`), so this is new
functionality. C `StochasticOptCG_real` (`stcopt_cg_impl.c`) defines the
algorithm, and the Rust CPU `mvmc_core::sr_cg::SampledSrOperator` ports it.
The kernels are in `src/opt/cg.cu`:

- `cg_moments` forms `<O>`, `<OO>_ii`, `<HO>` and `<H>`. One thread per
  parameter sums the samples in order, then divides by `Wc` (one sample per
  walker, `w=1`).
- `cg_prepare` applies C's diagonal cut. The maximum is non-absolute and
  starts at `S_00`; a parameter is cut when `S_ii < max * DSROptRedCut`, and
  `OptFlag` also applies. It then forms `g = -(2 dt)(<HO> - <H><O>)` and sets
  `x = 0`, `d = r = g`, `delta = r.r`. The threshold is
  `DSROptCGTol^2 nSmat^2` and `max_iter` is `NSROptCGMaxIter` or `nSmat`.
- Each iteration forms `q = S d` with `cg_ty` (`y = O^T d`, one block per
  walker, plus `<O>.d`) and `cg_oy` (`z = y O / Wc - (<O>.d)<O> +
  DSROptStaDel S_ii d`). It then runs `cg_iterate` (`alpha`, `x`, `r`,
  `beta`, `delta = beta*delta`, `d`).
- Every 20th iteration, `cg_alpha` runs instead, followed by `r = g - S x`
  (`cg_refresh`), as in C.
- The active set is a mask instead of C's compacted index arrays. Masked
  entries add exact zeros, and elementwise updates use `_rn` intrinsics
  without FMA.

**Residency.** The stop test runs on the device: the kernel that ends an
iteration sets a stop flag when `iter >= max_iter` or `delta < threshold`.
The host enqueues an upper bound, `NSROptCGMaxIter` or `NPara >= nSmat`
iterations. Once the flag is set, the remaining kernels return at once, so
there is no per-iteration host read.

- A 20-step run keeps 0 explicit transfers and 1 synchronization.
- CG never forms the `np x np` Gram matrix or the Cholesky workspace.
- The extra memory is `6 np + Wc` doubles, plus a `delta` trace and the
  per-step iteration counts.
- The no-op tail costs about 3 launches per iteration. At L128
  (`nSmat = 772`, about 470 iterations) a solve takes 12 ms.

**Results and failures.** `OptResult::cg_iterations` returns C's iteration
count for each step, the last `zqp_SRinfo.dat` column. A nonfinite
increment is C's `info=1`, after which C stops (`vmcmain.c:501-504`). The
device latches `RESIDENT_CG` (status 7) and publishes nothing.

**Validation** (`opt::cg_tests`). The references are the CPU CG on the GPU's
own derivatives and energies, and a second CPU run with the samples in
reverse order. The reordered run is the same operator with a different
summation order, so it measures CG's own rounding sensitivity.

- **Moments and gradient.** These match the host C-order sums within 4-8 ulp
  (same operations in the same order). The active set and `nSmat` are exact;
  the cut margin is reported.
- **Stopping rule.** It is checked exactly on the device's own `delta`
  trace: every completed iteration started at `delta >= threshold`, and the
  loop ended at `max_iter` or below the threshold.
- **Iteration count.** It equals the CPU count whenever the stop falls in
  the deterministic regime. This holds for `NSROptCGMaxIter` 7, 25 and 40,
  for tolerances 1e-4 and 2e-5 (3 and 10 iterations), and for 1e-3
  (0 iterations).
- **Deterministic regime.** The two runs differ only in reduction order, a
  relative perturbation of at most `(W + nSmat) eps` per operation. The
  per-iteration `delta` must agree within `8 k kappa (W + nSmat) eps`. Here
  `kappa <= 2 lambda_max / (DSROptStaDel min S_ii)`; the lower eigenvalue
  bound is rigorous because the covariance is positive semidefinite, and
  `lambda_max` comes from power iteration. Observed: L16 3e-9 against a
  budget of 7e-7 through iteration 21, and L128 4e-5 against 1e-4 through
  iteration 33.
- **First divergence.** Once `||r||` has fallen well below `||g||`, CG
  amplifies rounding about 10x per iteration. The CPU's reordered run then
  diverges from the CPU run as much as the GPU does: from iteration 22-23
  (L16), 12 (DH L16), 17 (RBM L16) and 34-35 (L128). The first jump comes
  from the explicit residual refresh at iteration 20. Iterates are compared
  only before the reordered run reaches 1/64 of the budget.
- **After the divergence.** Counts that end inside the amplifying regime
  differ legitimately. With the default tolerance at L128 they are GPU 440
  and CPU 473 (the reordered CPU run also differs), and both runs satisfy
  C's stopping rule.
- **Solution.** With true residuals `r = g - S x` evaluated on the host,
  `||x_gpu - x_cpu|| <= (||r_gpu|| + ||r_cpu||) / lambda_min`. Observed
  relative differences are 1.5e-3 (L16, 100 iterations) and 7.6e-4 (L128)
  after the divergence, and below 6e-15 when the run ends first.
- **Residual drift.** The device's true residual stays within its
  recurrence residual plus `8 k (W + nSmat) eps (||S|| ||x|| + ||g||)`.
- **Failure and repeatability.** A NaN local energy gives a NaN `delta`,
  which never passes C's stop test. The run reaches `max_iter`, the device
  latches status 7, and the parameters are unchanged. 20-step runs repeat
  exactly after an interleaved run.

**Timing** (RTX 3060, CUDA 12.9 image `b129bd583bbe`). The workload is seed
79, 64 walkers, warmup and interval of 128 proposals per walker, and 20 steps.
GPU times are warm complete namelist API calls
(`resident_cg_opt_repeats_twenty_steps`). The CPU time is the release `mvmc`
CLI (C-order runner, one thread, wall time including output). It runs with
`NVMCSample 64`, `NVMCWarmUp 1`, and `NVMCInterval 8` (L16) or `1` (L128).

| Workload | GPU CG | GPU direct | CPU CG | CPU direct |
|---|---|---|---|---|
| L16 (`nSmat` 100) | 0.14 s | 0.11 s | 0.45 s | 0.45 s |
| L128 (`nSmat` 772) | 1.49 s | 1.36 s | 14.7 s | 14.1 s |

At L128, the GPU runs 444-537 iterations per step and the CPU 509-684. The
counts differ because the samples differ.

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
