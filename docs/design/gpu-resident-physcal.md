# Resident GPU PhysCal

`mvmc_gpu_cuda::physcal::run_gpu_phys_cal_from_namelist` computes fixed-parameter
ensemble means on CUDA, using the Julia GPU initialization and sampling contract.

```rust,ignore
use mvmc_gpu_cuda::physcal::{run_gpu_phys_cal_from_namelist, PhysCalConfig};
let result = run_gpu_phys_cal_from_namelist("namelist.def", PhysCalConfig {
    opt_para: Some("zqp_opt.dat".into()),
    walkers: Some(320),
    nsamples: Some(1),
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

The existing CUDA binary exposes an explicit subcommand:

```bash
cargo build --locked --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --bin mvmc-cuda
CUDA_VISIBLE_DEVICES=0 gpu/mvmc-gpu-cuda/target/release/mvmc-cuda \
  resident-physcal namelist.def zqp_opt.dat new-output
```

This uses the sampling lengths and seed from the input. `new-output` must not
exist. `energy.dat`, `green_one.dat`, `green_two.dat` and `green_two_ex.dat`
contain one FP64 mean per line in the parsed order above; `sampling.txt` records the seed,
walker/window counts and RNG. These are ensemble means rather than the ordinary
CLI's full C-format moment/parameter files. Omitting the subcommand selects the
usual CLI and its original sampling/output contract. Process startup and file
output belong to CLI wall time; the warmed API benchmark excludes both.

Supported inputs are real normal orbitals, Gutzwiller/Jastrow projections,
operators whose individual pairs conserve spin, hopping-only sampling, no local spins/RBM/DH/OptTrans,
`NSRCG=0`, `NSplitSize=1`, and no Lanczos. Unsupported modes fail explicitly,
before any device allocation; see the
[support matrix](gpu-resident-model-support.md) (#572).
The ordinary CPU/CLI PhysCal and legacy single-SFMT CUDA path retain their
existing contracts. The new public API is selected by `benchmark_gpu.sh`.

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
