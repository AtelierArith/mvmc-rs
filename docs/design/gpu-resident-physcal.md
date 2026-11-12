# Resident GPU PhysCal

`mvmc_gpu_cuda::physcal::run_gpu_phys_cal_from_namelist` computes fixed-parameter
ensemble means on CUDA, using the Julia GPU initialization and sampling contract.

```rust,ignore
use mvmc_gpu_cuda::physcal::{run_gpu_phys_cal_from_namelist, PhysCalConfig};
let result = run_gpu_phys_cal_from_namelist("namelist.def", PhysCalConfig {
    opt_para: Some("zqp_opt.dat".into()),
    walkers: Some(320),
    nsamples: Some(20),
    warmup: Some(128),
    interval: Some(128),
    seed: Some(79),
    ..Default::default()
})?;
println!("{}", result.energy);
```

The explicit `warmup` and `interval` lengths count proposals per walker.
Omitted settings follow `NVMCSample`, `NDataQtySmp`, `NVMCWarmUp` and
`NVMCInterval`; the last two use the same site-count conversion as Julia GPU.
Each window advances all walkers, measures energy and all requested one-body,
direct two-body and product-side Green functions, then adds the ensemble mean.
The returned arrays follow the shared Rust parser's operator order and are
averaged over windows. With `TwoBodyGEx`, one-body output uses that parser's
C-compatible canonical list (deduplication plus missing constituents), which
can differ from Julia GPU's raw one-body list. Direct/product-side outputs
retain their parsed order. Fixed parameters are loaded before `In*.def` overlays,
so explicit overlays take precedence as in C PhysCal. This input rule is retained
even though Julia GPU currently loads those files in the opposite order.

## Windows and second moments (#729)

One window contains one configuration from each of W walkers, after `interval`
proposals per walker. The ensemble is initialized and warmed once; successive
windows continue the same walkers. Defaults are W=`NVMCSample` and
K=`NDataQtySmp`, with `interval = NVMCInterval * Nsite` and
`warmup = NVMCWarmUp * NVMCInterval * Nsite`. A window therefore has W samples;
combine consecutive windows into larger bins when comparing with C's bins.
Windows can be correlated and are not independent restarts.

`PhysCalResult::energy_windows` contains K `EnergyWindow` records with mean
energy (real and imaginary parts) and `energy_second_moment`. The latter is
`mean(|E_loc|²)`, following C 1.4.0 `vmccal.c:200` (`conj(e)*e`) and `:443`
(`e*e` for real energies). Its imaginary part is identically zero. For a
Hermitian Hamiltonian, the variance estimate is
`energy_second_moment - energy² - energy_imag²`. This is the second moment
over walkers, rather than the square of their mean. The overall second moment
is also returned, alongside the existing overall means.

`PhysCalConfig::window_greens = true` additionally returns
`PhysCalResult::green_windows = Some(Vec<GreenWindow>)`; otherwise it is `None`.
Each record has the same parsed operator order and real/imaginary components
as the overall Green arrays. Defaults store only three energy values per window.
Optional Green history costs `8 * K * G * lanes` extra bytes on the device and
host (`G = OneBodyG + TwoBodyG + TwoBodyGEx` rows, `lanes=1` real or `2` complex).
All histories share the final output allocation/download with the overall
means. No per-window transfer or stream synchronization is added.

The arithmetic mean of window values agrees with the overall mean within
`8*K*epsilon*max(1, max_window_magnitude)`, including accumulation and host
rounding (`epsilon=f64::EPSILON`). The variance is nonnegative within
`64*W*epsilon*max(1, energy_second_moment)`; a number-operator eigenstate checks
zero variance at a nonzero eigenvalue. Separate L16 real/complex and L128 tests
first check per-configuration energies against CPU C-order kernels using the
existing condition-number bound, then isolate warp-reduction rounding on the
same operator values. Green-history on/off preserves all means and energy
windows. These checks do not replace the independent RNG/tape gates.

The existing CUDA binary exposes an explicit subcommand:

```bash
cargo build --locked --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --bin mvmc-cuda
CUDA_VISIBLE_DEVICES=0 gpu/mvmc-gpu-cuda/target/release/mvmc-cuda \
  resident-physcal namelist.def zqp_opt.dat new-output --green-windows
```

This uses the sampling lengths and seed from the input. `new-output` must not
exist. `energy.dat`, `green_one.dat`, `green_two.dat` and `green_two_ex.dat`
contain one FP64 mean per line in the parsed order above. `energy_imag.dat` and
`energy_second_moment.dat` contain the overall imaginary energy and second moment.
`energy_windows.dat` has a header followed by
`window energy_real energy_imag energy_second_moment` (zero-based window indices).
The optional `--green-windows` flag creates `green_one_windows.dat`,
`green_two_windows.dat` and `green_two_ex_windows.dat`, with header and rows
`window operator real imaginary`; operator indices are zero-based within each
parsed list. Without the flag these three history files are not created.
`sampling.txt` records the resolved seed, walker/window counts, warmup and
interval in proposals, RNG, history flag and second-moment convention.
These documented GPU layouts differ from the CPU runner's full C-format
moment/parameter files. The CPU executable is `mvmc`. Process startup and file
output belong to CLI wall time; the warmed API benchmark excludes both.

The [#729 paired API report](../../benchmark/gpu_physcal_windows_729/README.md)
records L16/L128 timings with Green history off, unchanged transfer and
synchronization counts, and the additional 24 bytes per energy window.

Supported inputs are real normal orbitals, Gutzwiller/Jastrow projections,
operators whose individual pairs conserve spin, `NExUpdatePath` 0, 1 or 2 with optional local spins
(#660, see [GPU-resident Opt](gpu-resident-opt.md)), optional DH2/DH4 factors (#665), optional real RBM factors (#667), optional real OptTrans weights (#674, fixed at their input values),
`NSplitSize=1`, and no Lanczos. Complex normal orbitals (#678) follow C's
complex path with Gutzwiller/Jastrow projections, complex QP weights and
coefficients and exchanges; `PhysCalResult` then reports the imaginary parts
in its `*_imag` fields (zeros for real inputs), and TwoBodyGEx is C's
`G1[idx0] conj(G1[idx1])`. General orbitals / FSZ (#714), real or complex,
follow C's FSZ path with Gutzwiller/Jastrow projections, any `2Sz` and
exchanges, and, with real parameters, local spins and local spin flips
(`NExUpdatePath=3`, path 2 with `2Sz=-1`). Spin-flip OneBodyG, TwoBodyG and
TwoBodyGEx pairs are measured (C `calgrn_fsz.c`). `NSRCG` is accepted and ignored, as in Julia GPU
PhysCal (`ext/cuda_runners.jl:27`), because PhysCal never runs SR. Unsupported modes fail explicitly,
before any device allocation; see the
[support matrix](gpu-resident-model-support.md) (#572).
The ordinary CPU/CLI PhysCal retains its existing contract; the legacy
single-SFMT CUDA path was removed in #643. The new public API is selected by `benchmark_gpu.sh`.

The model and initialization are shared with resident GPU Opt; PhysCal omits
the sparse derivative basis and SR matrix/solver storage. Julia PhysCal uses
immediate inverse updates, whereas Opt opts into delayed updates. That
distinction is preserved. One ensemble owns its cuRAND XORWOW generator,
configurations, inverse/Pfaffian caches and observable accumulators on device.
No complete host optimizer state is constructed for a walker. Sampling and
measurement run in one stream scope without explicit intermediate uploads,
downloads or host decisions. Setup downloads initialization status, and the
final result downloads one status word and the observable means.

Numerical authority for this explicit GPU path is Julia-mVMC
`611031147f1bad9704c55084f6817feeba481b13`, with the pinned GPU manifest and
Juliaup `+1.13`. CPU/C-order execution remains C-authoritative. See the
[independent fixture provenance](../../gpu/mvmc-gpu-cuda/tests/fixtures/physcal_julia.md)
for numerical bounds, exact RNG/discrete checks and optional regeneration.
Transfer counters cover explicit raw-execution operations; they do not count
every operation inside native CUDA libraries. The benchmark report records the
full warmed API boundary, which includes Rust's internal state teardown.
