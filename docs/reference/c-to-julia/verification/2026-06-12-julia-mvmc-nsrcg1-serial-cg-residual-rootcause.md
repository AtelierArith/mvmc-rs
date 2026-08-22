---
date: 2026-06-12
datetime: 2026-06-12 21:35 JST
model: GPT-5 Codex
status: report
topic: Julia-mVMC NSRCG=1 serial CG 10-step residual root cause
updated:
  - datetime: 2026-06-12 22:20 JST
    model: GPT-5 Codex
    note: Follow-up review corrected the residual attribution: DGEMV/mul! alone is not established as the remaining source.
related:
  - docs/reviews/2026-06-12-julia-mvmc-v0.4-r2-nsrcg1-serial-parity-review.md
  - docs/reviews/2026-06-12-julia-mvmc-nsrcg1-commit-and-rootcause-verification-review.md
  - docs/specs/2026-06-10-julia-mvmc-v0.4-mpi-design.md
---

# Julia-mVMC NSRCG=1 serial CG residual root cause

## Summary

`NSRCG=1` serial parity work showed:

- step 1: C-mVMC vs Julia-mVMC `zvo_out` agrees to about `1e-15`;
- 10 steps: max residual about `2.72e-3`.

This residual is **not MPI-related**. Julia currently rejects `NSRCG != 0`
under MPI before collectives, so this comparison is serial-only.

The root cause is the numerical sensitivity of truncated SR-CG to BLAS /
reduction order differences:

1. C-mVMC built against Accelerate and C-mVMC built against OpenBLAS diverge by
   the same order (`~3.2e-3`) after 10 steps.
2. Julia's CG `xdot` had used `LinearAlgebra.dot`, while C source uses a
   simple sequential loop. This was an additional C-source-parity mismatch.
3. Changing Julia `xdot` to sequential accumulation aligns the 1-step CG
   solution with C(OpenBLAS) and reduces C(OpenBLAS) vs Julia 10-step residual
   to `~6.8e-5`.
4. The remaining `~1e-5..1e-4` residual is not yet pinned to one operation.
   A follow-up review found no difference in an isolated DGEMV-vs-`mul!`
   comparison, so the earlier attribution to `operate_by_S` DGEMV
   accumulation order alone is too narrow. A more plausible working hypothesis
   is that ulp-scale differences across the `NSRCG=1` store / `operate_by_S` /
   CG update path, including GCC `-O3` FMA contraction in the C binary, are
   amplified by the ill-conditioned truncated SR-CG solve.

## Reproduction Setup

Base fixture:

- `Julia-mVMC/test/integration/reference/heisenberg_chain_real/inputs`
- modified `modpara.def`:
  - `NSRCG 1`
  - `NSplitSize 1`
  - `NStore 1` and `NStore 0` both tested; results were identical for this
    issue
  - `DSROptCGTol 1.0e-10`
  - `NSROptCGMaxIter 0`

Runs used:

- C-mVMC existing mac build: `mVMC/build/src/mVMC/vmc.out`
  - linked against Apple Accelerate
  - `OMP_NUM_THREADS=1`
- C-mVMC scratch OpenBLAS build: `mVMC/build-openblas-debug/src/mVMC/vmc.out`
  - linked against `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`
  - `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
- Julia-mVMC:
  - Julia BLAS config: `LBTConfig([ILP64] libopenblas64_.dylib)`
  - `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1`

Scratch artifacts were created under:

- `tmp/nsrcg-debug-20260612-212217/`
- `tmp/nsrcg_probe.jl`

## Observations

### 10-step `zvo_out` comparison

Before changing Julia `xdot`:

| Comparison | Max residual over first 10 steps |
|---|---:|
| C(Accelerate) vs Julia | `2.723855078423e-3` |
| C(Accelerate) vs C(OpenBLAS) | `3.218423380030e-3` |
| C(OpenBLAS) vs Julia | `5.942278458453e-3` |

After changing Julia `xdot` to sequential accumulation:

| Comparison | Max residual over first 10 steps |
|---|---:|
| C(OpenBLAS) vs Julia | `6.757191404638e-5` |
| C(Accelerate) vs Julia | `3.285995294076e-3` |

The C(Accelerate) vs C(OpenBLAS) difference being the same scale as the original
C(Accelerate) vs Julia residual shows that the large residual is dominated by
BLAS backend differences, not MPI.

### 1-step SR-CG solution

`zvo_SRinfo.dat` / probe result for the first step:

| Build / implementation | `rmax` |
|---|---:|
| C(Accelerate) | `-0.514540` |
| C(OpenBLAS) | `-0.516934` |
| Julia old `xdot = dot(...)` | `-0.5137870054285514` |
| Julia sequential `xdot` | `-0.5169806958145439` |

C source defines `xdot` as:

```c
double z=0;
for(i=0;i<n;i++) {
  z += p[i]*q[i];
}
```

Therefore Julia's `LinearAlgebra.dot` was not C-source equivalent for this
truncated CG path.

## Code Change

Julia `xdot` in `MVMCOptimizers.jl/src/stochastic_opt.jl` was changed from
BLAS-backed `dot(p, q)` to explicit sequential accumulation:

```julia
function xdot(p::Vector{Float64}, q::Vector{Float64})::Float64
    z = 0.0
    @inbounds for i in eachindex(p, q)
        z += p[i] * q[i]
    end
    return z
end
```

No `@simd` is used here because the purpose is C-source parity, and SIMD
reduction would allow a different summation order.

## Interpretation

### Corrected residual attribution

The earlier version of this report attributed the post-`xdot` residual mainly
to `operate_by_S` `DGEMV` / Julia `mul!` accumulation-order differences. That
attribution should be treated as **unconfirmed**.

The follow-up review
`docs/reviews/2026-06-12-julia-mvmc-nsrcg1-commit-and-rootcause-verification-review.md`
records an isolated comparison between Homebrew OpenBLAS `dgemv_` and Julia
`mul!` with the same small problem shape (`nSmat=14`, `NVMCSample=100`), where
the two outputs were bit-identical. That does not prove `operate_by_S` has zero
contribution: the comparison script was a temporary artifact and `operate_by_S`
also includes the `xdot(stcO, x)` coefficient and scalar correction
`invW * z - coef * stcO + DSROptStaDel * sdiag * x`.

Therefore the stable conclusion is narrower:

- the large `~1e-3` residual was dominated by BLAS / reduction-order sensitivity
  and by Julia's previous BLAS-backed `xdot`;
- the remaining `~1e-5..1e-4` residual reflects ulp-scale differences somewhere
  in the `NSRCG=1` store / `operate_by_S` / CG update path;
- GCC `-O3` FMA contraction in the C binary is a strong candidate source, but a
  reproducible saved `operate_by_S`-level comparison is needed before assigning
  the remaining residual to a single operation.

The first `zvo_out` row is produced before parameter optimization, so it matches
C to `~1e-15`. The step 1 SR-CG parameter update is sensitive to small
summation-order changes in:

- CG dot products (`xdot`);
- the broader `operate_by_S` path, including matrix-vector products and scalar
  correction terms.

Those small differences alter the optimized parameters, then propagate into the
next sampling / energy step. This explains why the discrepancy appears from
step 2 onward and reaches `1e-3` scale within 10 steps.

For an e2e gate, a strict 10-step comparison against C(Accelerate) is not a
stable C-source-parity test when Julia uses OpenBLAS. A useful gate should be
one of:

- 1-step `NSRCG=1` e2e against committed C reference with tight tolerance;
- 10-step gate generated from a C build using the same BLAS family as Julia;
- 10-step gate with a documented tolerance reflecting BLAS backend sensitivity.

## Verification

After the `xdot` change:

- `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_stochastic_opt.jl")'` passed.
- `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. MVMCOptimizers.jl/test/runtests.jl` passed: `584/584`.
- `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=@. test/integration/runtests.jl` passed: `120/120`.
- `git diff --check` passed.
