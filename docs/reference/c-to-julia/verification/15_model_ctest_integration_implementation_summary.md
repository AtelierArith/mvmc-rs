# 15-model ctest integration implementation summary

## Current branch and commit status

- Branch: `feature/ctest-equivalent-integration`
- Commit status: not committed yet
- Scope implemented in this pass: Milestone 1, the 12 standard real/cmp/fsz
  C-mVMC ctest fixtures

## What was implemented

### C ctest-equivalent runner

Added `test/integration/ctest_equivalent.jl`.

This runner mirrors C-mVMC's standard `test/python/runtest.py` acceptance
criterion:

```text
abs(calc - ref_mean) >= 3 * ref_std && abs(calc - ref_mean) >= 1e-8
```

The runner compares only the first two final optimisation summary values, as
C-mVMC ctest does. It intentionally does not compare every SR step and is
therefore weaker than the existing strict first-10-step integration test.

The runner supports filtering with:

```bash
JULIA_MVMC_CTEST_MODELS=HeisenbergChain,hubbard_chain_cmp \
  julia --project=@. test/integration/ctest_equivalent.jl
```

### Shared model table

Added `test/integration/ctest_models.jl`.

The implemented standard models are:

- `HeisenbergChain`
- `HubbardChain`
- `HubbardTetragonal`
- `HubbardTetragonal_MomentumProjection`
- `KondoChain`
- `HeisenbergChain_cmp`
- `HubbardChain_cmp`
- `KondoChain_cmp`
- `KondoChain_Stot1_cmp`
- `HeisenbergChain_fsz`
- `HubbardChain_fsz`
- `KondoChain_fsz`

The remaining private 15-model set entries are documented as deferred:

- `GeneralRBM_cmp`: blocked by RBM-bearing `initial.def` support
- `SpinChainLanczos`: blocked by mode1/Lanczos driver and output path
- `HubbardChainLanczos`: blocked by mode1/Lanczos driver and output path

### `run_para_opt_from_namelist` support

Updated `MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`.

Changes:

- Added `nsmp` keyword argument.
- Added an API-level guard requiring `nsteps >= nsmp`.
- Added returned fields:
  - `ctest_values`
  - `effective_nsteps`
  - `effective_nsmp`
- `ctest_values` is computed as the mean of columns 1 and 2 over the final
  `nsmp` rows of Julia `zvo_out.dat`.

This avoids comparing Julia's current `zqp_opt.dat` to C's `zqp_opt.dat`.
Those files are not format-compatible: C's ctest reads final optimisation
summary values, while Julia's current file starts with variational parameters.

### Existing strict integration runner

Updated `test/integration/runtests.jl`.

The existing strict first-10-step runner now calls:

```julia
run_para_opt_from_namelist(...; nsteps = N_STEPS, nsmp = N_STEPS, ...)
```

This preserves the existing first-10-step behavior while satisfying the new
`nsteps >= nsmp` invariant.

### Fixture generation

Added `test/integration/tools/generate_ctest_fixtures.jl`.

The script copies:

- generated expert-mode `.def` files from C-mVMC `build/test/python/work/<Model>`
- optional `initial.def` / `zqp_opt.dat` from C-mVMC `build/test/python/data/<Model>`
- `ref/ref_mean.dat` and `ref/ref_std.dat` into `ctest_ref/`

Generation command used:

```bash
julia --project=@. test/integration/tools/generate_ctest_fixtures.jl \
  --c-test-dir ../private-mVMC/mVMC/build/test/python
```

### Public fixtures

Added C ctest-equivalent fixture data under `test/integration/reference`.

Each of the 12 standard models now has:

- `inputs/*.def`
- `ctest_ref/ref_mean.dat`
- `ctest_ref/ref_std.dat`
- optional `inputs/initial.def`
- optional `inputs/zqp_opt.dat`

The total reference directory size after this work is about 908 KiB.

### CI

Updated `.github/workflows/ci.yml`.

CI now runs both:

```bash
julia --project=@. test/integration/runtests.jl
julia --project=@. test/integration/ctest_equivalent.jl
```

### Julia 1.11 Manifest

Added `Manifest-v1.11.toml`.

Reason:

- Existing `Manifest.toml` was generated with Julia 1.12.6.
- Julia 1.11 tried to use 1.12-only package versions from that manifest.
- The failure happened before the new tests ran, during precompile.

With `Manifest-v1.11.toml`, Julia 1.11 picks a compatible dependency set
automatically. In particular, `PrecompileTools` resolves to `v1.2.1` instead
of the 1.12-only `v1.3.4`.

The existing `Manifest.toml` was left unchanged for Julia 1.12.

## C-mVMC work generation

C-side work directories were generated from:

```bash
cd ../private-mVMC/mVMC/build/test/python
python3 runtest.py <Model>
```

Missing work directories were generated for:

- `HubbardTetragonal`
- `HubbardTetragonal_MomentumProjection`
- `KondoChain`
- `HubbardChain_cmp`
- `KondoChain_cmp`
- `KondoChain_Stot1_cmp`
- `HubbardChain_fsz`
- `KondoChain_fsz`

All completed with status 0.

Existing work directories were reused for:

- `HeisenbergChain`
- `HubbardChain`
- `HeisenbergChain_cmp`
- `HeisenbergChain_fsz`

## Post-review fixes

After the implementation review in
`private_docs/15_model_ctest_integration_implementation_review.md`, the
following fixes were applied:

- Added `nsmp = NSTEPS` to all four `examples/*.jl` scripts so the CI smoke
  run with `JULIA_MVMC_EXAMPLE_STEPS=5` satisfies the `nsteps >= nsmp`
  invariant.
- Updated `MVMCOptimizers.jl/README.md` to document the new `nsmp` keyword in
  the quick-start snippet and public API table.
- Verified all four examples with `JULIA_MVMC_EXAMPLE_STEPS=5` on both
  Julia 1.11 and Julia 1.12.

## Verification

### Julia 1.12

Strict first-10 integration:

```bash
julia +1.12 --project=@. test/integration/runtests.jl
```

Result:

```text
Julia-mVMC integration vs C reference | 120 / 120 pass
```

C ctest-equivalent integration:

```bash
julia +1.12 --project=@. test/integration/ctest_equivalent.jl
```

Result:

```text
Julia-mVMC C ctest-equivalent integration | 61 / 61 pass
```

### Julia 1.11

Precompile/load check:

```bash
julia +1.11 --project=@. -e 'using MVMCOptimizers'
```

Result: passed.

Strict first-10 integration:

```bash
julia +1.11 --project=@. test/integration/runtests.jl
```

Result:

```text
Julia-mVMC integration vs C reference | 120 / 120 pass
```

C ctest-equivalent integration:

```bash
julia +1.11 --project=@. test/integration/ctest_equivalent.jl
```

Result:

```text
Julia-mVMC C ctest-equivalent integration | 61 / 61 pass
```

## Remaining work

The following are intentionally not implemented in this pass:

- `GeneralRBM_cmp`
  - Need production support for RBM-bearing `initial.def`.
- `SpinChainLanczos`
  - Need a real `NVMCCalMode=1` / Lanczos driver path and `zvo_ls_out`
    equivalent output.
- `HubbardChainLanczos`
  - Same Lanczos/mode1 blocker as `SpinChainLanczos`.

Those three should be handled after the standard 12-model ctest-equivalent
coverage is reviewed and accepted.
