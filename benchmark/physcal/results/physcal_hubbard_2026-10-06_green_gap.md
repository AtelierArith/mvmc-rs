# Hubbard-chain PhysCal benchmark: Green gap fix (2026-10-06)

Second fixed-parameter PhysCal measurement on the same machine as
`physcal_hubbard_2026-10-06.md`, after the issue #442 work. One thread,
warm-up-excluded medians over 3 repetitions, `NDataQtySmp=100`. Linux x86_64,
rustc 1.99.0, Julia 1.13.1, system OpenBLAS 0.3.26 (Rust) and Julia's bundled
OpenBLAS.

| model | Rust before (s) | Julia before (s) | before | Rust after (s) | Julia after (s) | after |
|---|---:|---:|---:|---:|---:|---:|
| hubbard_chain_L16 | 1.930 | 1.429 | 0.740x | 1.270 | 1.426 | **1.12x** |
| hubbard_chain_L24 | 4.660 | 2.841 | 0.610x | 2.700 | 2.850 | **1.06x** |
| hubbard_chain_L32 | 9.350 | 5.073 | 0.543x | 5.170 | 5.084 | 0.98x |

`speedup = julia / rust`; values above `1.0x` mean Rust is faster. Rust now wins
L16 and L24 and is within ~2% at L32. `|ΔE| = 0` for every size and every
observable family matched (`max|Δ| = 0`; see below).

## What changed (issue #442)

The section timer located the remaining gap in the un-timed
`VMCMainCal` remainder, dominated by the one/two-body Green measurement. Three
parity-preserving fixes:

1. **Hardware FMA dispatch.** `two_hop_bilinear_real` reproduces Julia's
   `LoopVectorization` reduction with `f64::mul_add`. On the default
   `x86-64` baseline target every `mul_add` lowers to a software FMA, several
   times slower than the hardware instruction. A runtime-dispatched
   `#[target_feature(enable = "fma,avx2")]` variant restores it; hardware and
   software FMA are both correctly rounded, so the results are bit-identical.
   The scalar path remains for CPUs without FMA/AVX2.
2. **Skip the SR `O` vector in PhysCal.** C `VMCMainCal` builds the SR `O`
   vector, `SlaterElmDiff` and the OO/HO accumulators only for `NVMCCalMode==0`
   (`vmccal.c:195`); PhysCal computes the Green functions only. Rust now guards
   that block with `vmc_calc_mode == 0`, removing ~0.24 s of `SlaterElmDiff` per
   L32 run.
3. **Skip the real→complex inverse copy in PhysCal.** In real mode C/Rust copy
   every `InvM_real` plane into `InvM` for `SlaterElmDiff` (and Lanczos/FSZ).
   PhysCal without SR and without Lanczos reads only the real tables, so the
   copy is skipped there (~1.1 s at L32).

## PhysCal observables

All 100 per-sample files of each family match with `max|Δ| = 0` (L32
`zvo_cisajscktalt` has 38400 compared values):

| model | family | files | values | max abs | max rel | status |
|---|---|---:|---:|---:|---:|---|
| hubbard_chain_L16 | zvo_cisajs | 100 | 6400 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L16 | zvo_cisajscktalt | 100 | 19200 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L24 | zvo_cisajs | 100 | 9600 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L24 | zvo_cisajscktalt | 100 | 28800 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L32 | zvo_cisajs | 100 | 12800 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L32 | zvo_cisajscktalt | 100 | 38400 | 0.000e0 | 0.000e0 | ok |

`bench-physcal` on the four `physcal_ref` fixtures also stays `ok` for every
real/complex/FSZ and Lanczos family.

Command:

```sh
cargo run -p xtask -- bench-physcal-hubbard --reps 3 --warmups 1 --threads 1
```

## Caveats

- The L32 gap (~2%) is on one machine and within a few percent of the Julia
  measurement; a quiet-host rerun may cross `1.0x`.
- The comparison uses Rust process wall clock (startup negligible at these
  sizes) against Julia's `run_phys_cal_from_namelist` wall clock excluding JIT.
- The Julia reference is the pinned `extern/Julia-mVMC`.

Related to issue #442.
