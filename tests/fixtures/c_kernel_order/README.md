# Mixed C-kernel/Julia runner regressions

These fixtures use the Julia 1.13.1 runner, sampling and SR implementation with
two explicit C contracts substituted in an independent Julia reference:

- `MakeRBMCnt` adds the sum of a neuron's couplings to its bias once. Julia's
  original implementation adds each coupling directly to the bias. General
  canonical inputs additionally use C's spin-major spatial traversal.
- Normal Slater construction retains every coefficient. Julia's original
  `build_orbital_idx_sgn_matrices` drops amplitudes at or below `1e-14`.

The translated `MakeRBMCnt` must pass **all 4,994 checked-in native C counter
cases bitwise before any fixture is written**. Source SHA-256 verification
guards the C provenance. The script records the Julia/BLAS version, seed and
single-thread setting in each reference file. It never reads Rust results.

This is a mixed reference, **not unmodified Julia parity or full C executable
parity**. The original Julia goldens remain in `sr_direct/` and `sr_cg/`.
Sparse historical models are internal kernel controls; they do not establish
C acceptance of their original incomplete definition files. Native C reader,
initialization and counter fixtures separately test the production contracts.
Cargo consumes only checked-in data and never runs these scripts or C/Julia.

Optional regeneration (using the pinned reference workspace):

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_c_kernel_runner_fixtures.jl
```

Individual verification (omit `--write` to compare existing fixtures):

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=rbm_reference_cmp --store=1 --c-kernel-order
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=rbm_reference_cmp --c-kernel-order
```
