# Julia GPU local-spin / exchange fixtures (#660, #572 milestone 1)

These are independent Julia GPU references for resident GPU Opt and PhysCal
with local spins and exchange proposals. Julia GPU had stage-level local-spin
tests only (6-site `parity.jl`), so these large-lattice fixtures are new.

## Inputs

`gen_inputs.py` writes both inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/localspin/gen_inputs.py`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | Local spins | `2Ne` | `NExUpdatePath` (exchange p) | Exercises |
|---|---|---|---|---|---|
| `heisenberg_chain_L64` | 64 | 64 | 64 | 2 (1.0) | exchange-only sampling, full-Pfaffian proposals, in-kernel refactor on every accept |
| `kondo_chain_L64` | 128 (0-63 conduction, 64-127 local) | 64 | 128 | 1 (0.5) | hops restricted by the local-spin mask, mixed hop/exchange, Opt delayed queue and paired inverse (`2Ne >= 128`) |

Both use real normal orbitals, Gutzwiller/Jastrow, identity TransSym and
`NSPGaussLeg=8` (8 QP planes). Couplings follow the StdFace Heisenberg sign
convention of `tests/fixtures/original_heisenberg_parser_184`. These are engine
tests, not physics benchmarks.

## Reference and generation

- Julia-mVMC `73140156ce813efb8b528516b4a9b5640247f23b` (the `extern/Julia-mVMC`
  pin), unmodified. `test/gpu/Manifest-v1.13.toml` SHA-256
  `86c69ea3a0c04c2b6af6f48a3f36ec2a08fa2bc564ac4845f5799e53244fc573`.
- Julia 1.13.1, CUDA.jl 6.4.2, CUDA runtime 13.4.0, cuRAND 10.4.4,
  cuBLAS 13.8.0, cuSOLVER 12.3.4; Linux x86_64, RTX 3060 (physical GPU 1),
  driver 580.178.04.
- Rust verification ran in the CUDA 12.9.2 devcontainer image (cuBLAS 12.9.2,
  NVRTC 12.9) on the same RTX 3060.

```sh
for m in heisenberg kondo; do for f in opt physcal; do
  CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    julia +1.13.1 --startup-file=no --project=extern/Julia-mVMC/test/gpu \
    gpu/mvmc-gpu-cuda/tests/fixtures/localspin/gen_localspin_julia.jl \
    gpu/mvmc-gpu-cuda/tests/fixtures/localspin/${m}_chain_L64/namelist.def \
    gpu/mvmc-gpu-cuda/tests/fixtures/localspin/${m}_$f.txt $f
done; done
```

`gen_localspin_julia.jl` follows the Julia namelist runners stage by stage.
It uses 8 XORWOW-initialized walkers with seed 79. The Opt flavor uses
`prepared_gpu_data`, a delayed cache with a paired inverse at `2Ne >= 128`,
derivatives and one direct SR update plus gauge. The PhysCal flavor uses
`physcal=true`, immediate updates and Green functions. Both flavors then run
`3*Nsite` proposals of a fixed formula tape (`(k*0x9e3779b97f4a7c15 +
0xbf58476d1ce4e5b9) >> 32`, `k = 1..5*walkers*steps`) through `gpu_sample!`
with the runner's exchange probability, `empty_destinations`,
`incremental_correlation` and `triangle_update = 2Ne >= 128`. The Rust test
regenerates that tape and checks it against the stored head and checksum.
Normal Rust tests only read these files and never invoke Julia.

## Contract and bounds

Exact (integers):

- Initial configurations. These check the XORWOW draw order/count of the
  local-spin label permutation and the per-spin nonlocal shuffles.
- Final configurations after the supplied tape, including every exchange and
  local-spin hop decision.
- Per-walker acceptance counts.
- Single occupancy of every local-spin site.

FP64 values use the existing resident stage bounds (`../opt_julia.md`,
`../physcal_julia.md`):

- Initial parameters: `1e-14 + 1e-14|b|`. The observed difference is 0, from
  the shared SFMT `init_parameter`.
- Pfaffians and overlaps: `1e-12 + 5e-11|b|`.
- Energies, derivatives, Green-function means, SR energy mean and updated
  parameters: `1e-10 + 1e-10|b|`.

These bounds cover independent factorization and dot/QP reductions with
different CUDA libraries. They are not a tolerance on RNG or proposal
differences, which are compared exactly.

Observed maximum differences (RTX 3060):

| Quantity | Heisenberg Opt | Kondo Opt | Heisenberg PhysCal | Kondo PhysCal |
|---|---|---|---|---|
| final Pfaffian, relative | 1.77e-13 | 2.90e-13 | within bound | within bound |
| initial energy / observables, abs | 5.78e-13 | 8.95e-13 | 4.09e-14 | 1.49e-13 |
| initial derivatives, abs | 1.09e-13 | 5.37e-14 | n/a | n/a |
| final energy / observables, abs | 1.29e-14 | 9.95e-14 | 2.78e-15 | 1.42e-14 |
| final derivatives, abs | 3.27e-14 | 1.44e-14 | n/a | n/a |
| updated parameters, abs | 6.66e-15 | 2.30e-13 | n/a | n/a |
| SR backward error (CPU reconstruction, `< 1e-11`) | 1.05e-16 | 8.44e-17 | n/a | n/a |

The first floating difference is in the independent factorization, as for the
L128 fixtures. Every supplied-tape decision still matched. The full-Pfaffian
exchange proposals and in-kernel refactors kept the trajectories identical
over `3*Nsite` proposals. The projection ratio of an exchange is exactly one
in Rust. Julia recomputes it from unchanged counts, which can differ only by
summation-order roundoff. It did not change any decision here.

## End-to-end timing and residency (RTX 3060, GPU 1)

Matched workloads: seed 79, `initial_def` none, warmup 128 and interval 128
proposals per walker. Opt runs 20 SR steps and PhysCal runs 20 windows. Each
time covers one complete namelist API call. The warm time is the second call
in the same process. Rust times come from `resident_opt_localspin_repeats_twenty_steps`
and `resident_physcal_localspin_repeats_twenty_windows`. Julia times come from
`MVMCOptimizers.run_gpu_{para_opt,phys_cal}_from_namelist` at the pin above.

| Workload | Walkers | Rust first / warm (s) | Julia first / warm (s) |
|---|---|---|---|
| Heisenberg Opt | 128 | 72.0 / 38.0 | 141.8 / 69.8 |
| Kondo Opt | 64 | 54.0 / 54.0 | 108.6 / 100.9 |
| Heisenberg PhysCal | 64 | 40.4 / 19.7 | 46.0 / 45.4 |
| Kondo PhysCal | 64 | 53.9 / 53.9 | 102.2 / 98.9 |

The whole sampling, measurement and SR series uses 0 explicit uploads,
0 downloads and 1 stream synchronization (RawExec counters).
Every accepted exchange refactors one walker (`O(nq n^3)` on one block), and
every exchange proposal evaluates `nq` full Pfaffians. This dominates both
implementations, as Julia's design prescribes.

The resident XORWOW stream matches Julia GPU, so these complete runs also
agree with Julia beyond the supplied-tape stages. The printed values agree to
all digits shown:

- Opt energy history `E[0], E[19]`: Heisenberg `-0.439335, -0.977516`;
  Kondo `57.352032, 50.511622`.
- PhysCal energy: Heisenberg `-0.846249966`; Kondo `57.017556136`.

This is an observation only. The tests require same-implementation
repeatability and do not freeze a long trajectory.

Fixture SHA-256:

```text
e92ab1511c5246b0e66fd51048dc48a50d13a7f27726a1ff3fd34d1f5c9e3f5d  heisenberg_opt.txt
acb01a081c1472b45616d747df9b9b77b510675cbefd492e09110b7ffdad9869  heisenberg_physcal.txt
dc4fbe2464ad2dece9684f9915e921e87985dea1d4adf0fe83d70c43d93ea0e8  kondo_opt.txt
27c9d316fcb7bc1667c71cb90eb9ce45b8443e9619f12f1d22aca6ee7ac3d3db  kondo_physcal.txt
```
