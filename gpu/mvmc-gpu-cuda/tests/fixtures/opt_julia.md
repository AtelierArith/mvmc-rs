# Independent Julia GPU Opt fixtures (#560)

Reference: Julia-mVMC `47452f9061ec3de049d891a7aa7fd9cfeac228a4`, Julia 1.13.1,
CUDA.jl 6.4.2, cuRAND 10.4.4, cuBLAS 13.8.0, cuSOLVER 12.3.4; Linux x86_64,
RTX 3060. Rust verification uses cuRAND 10.3.9.90, cuBLAS 12.8.4.1 and
cuSOLVER 11.7.3.90. The independently generated UInt32 cuRAND fixture is
documented separately in `curand_julia.md`.

These files contain Julia-generated expected values, not Rust receipts. They
exercise initialization from seed 79 with eight walkers, measurement without
sampling, and one direct SR update plus gauge normalization. The L128 input
has 8 quantum-projection planes and 772 real parameters. Input sampling/step
counts do not affect this stage fixture; the seed and walker count are explicit.

Generate from the repository's committed model inputs:

```sh
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  julia +1.13.1 --project=/path/to/Julia-mVMC/test/gpu \
  gpu/mvmc-gpu-cuda/tests/fixtures/gen_opt_julia.jl \
  benchmark/hubbard_chain/inputs/hubbard_chain_L128/namelist.def \
  gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia_l128.txt

CUDA_VISIBLE_DEVICES=1 MVMC_OPT_FIXTURE_OPERATORS=1 OPENBLAS_NUM_THREADS=1 \
  julia +1.13.1 --project=/path/to/Julia-mVMC/test/gpu \
  gpu/mvmc-gpu-cuda/tests/fixtures/gen_opt_julia.jl \
  benchmark/hubbard_chain/inputs/hubbard_chain_L64/namelist.def \
  gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia_operators.txt
```

The operator case assigns nonzero rational Gutzwiller/Jastrow parameters, adds
CoulombInter/Hund/PairHop/Exchange terms and all 324 spin-conserving pairs of
one-body operators on sites 0–2. Individual term values check diagonal, Pauli
zero, same-electron, two-electron, same-spin and opposite-spin overlays without
letting an energy sum hide sign errors. `MVMC_OPT_FIXTURE_FULL=1` additionally
exports Slater and inverse arrays for first-divergence diagnostics.

Configuration integers are exact. Computed FP64 results use explicit bounds:
initial parameters `atol=rtol=1e-14`; Pfaffians/overlaps `atol=1e-12,
rtol=5e-11`; derivatives, local energies and updated parameters
`atol=rtol=1e-10`. These accommodate conditioned factorization and direct SR
with different native libraries, not RNG or proposal differences. Full L128
diagnostics located the first floating difference in factorization: initial
Slater/parameter differences were zero; maximum Pfaffian relative difference
was about 1.25e-12, inverse absolute difference 1.91e-11, derivative absolute
difference 4.37e-14, local-energy difference 1.77e-12 and updated-parameter
difference 4.07e-13. The test also reconstructs the mathematical SR system on
the CPU and requires a normalized solve residual below 1e-11 (observed
1.63e-16 for L128).

The separate 20-step test checks same-implementation repeatability, including
an intervening differently seeded run on another device when available. It does not require following a frozen
historical trajectory after a demonstrated floating-point acceptance change.

Fixture SHA-256:

```text
7d7c249bba161e6098732adce20138e53c099db7da8e59c1cbdce6fc46536724  opt_julia_l128.txt
6beeca4f8ac4098431da2472e506db654b20b953ac19e389a09427f023c03a04  opt_julia_operators.txt
```
