# pulp and LoopVectorization on the L128 hot kernels (#641)

This evaluates pulp (Rust) and LoopVectorization `@turbo` (Julia) on the CPU kernels that the
current Hubbard L128 Opt and PhysCal workloads run. It continues the #503 request on current
`main` (`b1c0870d`). The #649 hand AVX2 kernels are already in place (PRs #651, #652, #661).

**Decision: nothing is adopted.** pulp does not beat the hand AVX2 kernels, and a pulp
replacement makes the whole workload 1.3–3.2 % slower. LoopVectorization gives Julia 7–14 % on
two kernels. It does this by fusing multiply-adds (FMA) and reordering the arithmetic, which
breaks the C operation order. Even so, Julia `@turbo` only reaches the speed that the exact Rust
AVX2 kernels already have. The production code, the scalar reference paths and the pinned Julia
manifest are unchanged.

## Environment

- Xeon E5-2699 v3 (Haswell). It has AVX2 and FMA but **no AVX-512**. Native Linux, not
  containerized, on a shared host. Load averages are recorded with every run in `load.txt` and
  `workload/ab-load.txt`; they were mostly 1–3 during the kernel runs.
- rustc 1.99.0 (LLVM 23.1.1); pulp 0.22.2; OpenBLAS (`OPENBLAS_NUM_THREADS=1`).
- Julia 1.13.1 with the pinned `extern/Julia-mVMC/Manifest-v1.13.toml` (sha256 `09ebd06d…`,
  unchanged). It uses LoopVectorization 0.12.174 and VectorizationBase 0.21.74. LoopVectorization
  is reached through `PfaPack.LTLDecomposition.LoopVectorization`, so the manifest did not need
  editing.
- Selected SIMD. pulp `Arch::new()` returns `V3` (AVX2+FMA, 4 × f64). VectorizationBase reports
  `register_size=32`, `pick_vector_width(Float64)=4`, `fma=true`, `avx512f=false`. Neither
  library can pick AVX-512 on this host, so V4 / AVX-512 was **not measured**.

## 1. Profile: which kernels actually run

Command: `perf record -F 499 -g`, run through the explicit
`MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=c-order` route. The release `mvmc` binary
was built with line tables. Each run was pinned to CPUs `0..N-1` with BLAS set to 1 thread. The
table shows self time as % of samples; the raw tables are in `results/20261105/profile/`.

| symbol | Opt t1 | Opt t8 | PhysCal t1 | PhysCal t8 |
|---|---:|---:|---:|---:|
| K1 `update_one_real_avx2` (UpdateMAll one hop) | 37.4 | 19.8 | 33.7 | 18.8 |
| K2 `update_upper_rank2_f64_c_order_avx2` (dsktf2 rank-2) | 11.8 | 7.8 | 9.4 | 7.4 |
| OpenBLAS `daxpy_k`/`dtrmv_NUU`/`dtrmm` (utu2inv trtri/trmm) | ~9.6 | ~5.9 | ~9.0 | ~6.2 |
| `dsktf2` remainder (pivot search, swaps, scale) | 5.8 | 4.0 | 5.6 | 4.3 |
| `utu2inv_real` (permute/skew fill) | 3.8 | 2.6 | 3.7 | 1.9 |
| K4 `solve_sktd_c_real_f64_avx2` | 3.4 | 2.1 | 2.6 | 2.3 |
| `assemble_inv_m_real_plane` | 3.3 | 2.0 | 2.7 | 1.6 |
| K3 `two_hop_bilinear_c_order` (CalculateNewPfMTwo) | — | — | 5.8 | 4.9 |
| thread pool wait/barrier (spindle) | — | 34.8 | — | 29.2 |

Wall times of the profiled runs: Opt t1 70.3 s, Opt t8 16.7 s, PhysCal t1 3.90 s, PhysCal t8
1.02 s.

Notes on the profile:

- K1, K2, K3 and K4 together account for about 52 % of single-thread time (Opt 52.6 %,
  PhysCal 51.5 %).
- The OpenBLAS trtri/trmm time is not Rust code, so it was not a pulp/LV target.
- K3 runs only in PhysCal, for the `greentwo` measurement.
- At 8 threads the largest single item is time spent waiting in the thread pool, not compute.

## 2. Kernel comparison (n = 128, one QP plane, one core)

`run_kernels.sh` runs the comparison:

- It builds `rust/` twice: with default code placement and with `-C llvm-args=-align-loops=64`.
- It alternates five process runs of each Rust build and of `julia/lv_kernels.jl`.
- Each process runs 15 round-robin rounds per variant.
- The table gives the median over processes of each process's median, in µs per call.
- The K1 and K2 calls copy a pristine 128 KB matrix first. "Net" subtracts that copy.

Variants:

- `scalar`: the production scalar reference path, copied verbatim.
- `avx2`: the production hand AVX2 kernel, copied verbatim.
- `pulp-scalar`: the pulp generic code running on `pulp::Scalar`.
- `pulp-v3`: a pulp port using separate multiplies and adds, in the same order as the C code.
- `pulp-v3-fma`: the same port, but with fused multiply-adds.

Julia variants:

- `julia-simd`: the `@inbounds @simd ivdep` loop form used by Julia-mVMC and PfaPack.jl.
- `julia-turbo`: the same loops under `@turbo`.

Raw outputs are in `results/20261105/kernels/` (`summary.md`, plus one file per process).

| kernel | Rust scalar | Rust hand AVX2 | pulp-scalar | pulp-v3 (exact) | pulp-v3-fma | Julia @simd | Julia @turbo |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 one-hop update, net | 10.93 | **6.92** | 10.52 | 6.68 (align64: 7.05) | 8.30 | 8.17 | 7.64 |
| K2 rank-2 sweep, net | 79.7 | **57.8** | 79.1 | 61.2 (align64: 62.3) | 56.6 | 67.7 | 58.0 |
| K3 bilinear | 4.45 | (none; exact transposed AVX2 rows: 4.21) | 7.74 | n/a | 3.30 | 12.09 (C order) | 1.89 |
| K4 sktd solve | 56.6 | **35.5** | 103.3 | 137.2 | n/a | 126.6 | 62.5 |

Setup cost:

- `pulp::Arch::new()` takes 0.70 ns, and `is_x86_feature_detected!("avx2")` takes 0.67 ns.
- Calling `Arch::new()` before every K1 call costs about 0.3–0.6 µs against the cached-arch
  run (13.6 vs 13.0 µs). That is within the run-to-run range.
- There is no measurable setup cost for either library beyond Julia's one-time JIT, which the
  warmup excludes.

### Correctness and numerics

Updated for #697. The archived 2026-11-05 run compared bit patterns; both harnesses now accept a
variant only if every value is finite and within an explicit bound derived from the operation
count, scale and conditioning below. Every decision prints a `check ... status=PASS|FAIL` line.
The process prints `correctness PASS|FAIL` last and exits with status 1 on any failure, and
`summarize.py` fails when a process output has a failing or missing verdict, so `run_kernels.sh`
stops instead of producing a report. A self-test first shows that NaN, infinity, an
out-of-bound value and a nonzero difference under a zero bound are rejected.

Notation: `u = eps/2`; `gamma(k) = k u / (1 - k u)` bounds `k` rounded operations along a
dependency chain. A comparison of two evaluation orders uses `2 gamma(k) S`, with `S` a
componentwise magnitude bound of every partial result. The bound is absolute in `S`, so it
covers cancellation and exact zeros without a relative floor. The reference for the Rust
variants is the production scalar path; for Julia `@turbo` it is the Julia-mVMC loop form.

| kernel | comparison bound | independent check |
|---|---|---|
| K1 (64 chained hops, n = 128) | `2 steps gamma(n+4) kappa max\|X\|` per entry, `2 steps gamma(n+4) kappa` for the Pfaffian ratio; `kappa = \|M\|_inf \|X\|_inf` (1.2e4 here). One hop rounds an n-term dot product, a scale and the three-operation rank-2 update; a relative perturbation of an inverse moves it by up to `kappa` times. This is a conditioning-scaled envelope for this fixed input, not a forward-error guarantee for arbitrary chains | residual `max\|M X - I\| <= r0 + (steps+1) gamma(n+4) \|M\|_inf \|X\|_inf` from the measured initial residual `r0`; every variant against a fresh `dsktf2`+`utu2inv` inverse of the final matrix (Julia: LAPACK `inv`) |
| K2 (`dsktf2` rank-2 sweep) | `2 gamma(5n/2 + 1) S_ij`: n/2 sweeps of `(c + x t1) - y t2` with `t = alpha col`, at most five roundings per sweep along a chain; `S` is the same sweep on magnitudes | — |
| K3 (`b^T M a`) | `2 gamma(2n + 1) S`, `S = sum_i \|b_i\| sum_j \|M_ij a_j\|`: any order or blocking of either summation level, with or without FMA | cancellation case `a^T M a` against the exact value 0 within `gamma(2n + 1) S` (the inverse is exactly skew; Julia uses `(X - X^T)/2`, exactly skew because `fl(x - y) = -fl(y - x)`) |
| K4 (skew-tridiagonal solve) | `2 gamma(2n + 1) S_ij`: two recurrences of n/2 steps with three roundings each (four with a reciprocal); `S` is the recurrence on magnitudes | relative residual of every solved relation, divided by the magnitude sum of its terms, `<= 4 eps` |

Measured on 2026-11-05 (`results/20261105/correctness-697/`, harness revision of this change on
`main` `787c1b15`, same host and toolchain as below, one process per build). All 35 Rust and
13 Julia checks pass in both Rust builds:

| check | max abs difference | max ratio to bound |
|---|---:|---:|
| Rust `avx2`, `pulp-scalar`, `pulp-v3` vs scalar, K1/K2/K4; `avx2-rows` K3 | 0 (identical values on this input) | 0 |
| Rust `pulp-v3-fma` vs scalar: K1 / Pfaffian ratio / K2 / K3 | 1.3e-11 / 3.4e-12 rel. / 3.3e-16 / 1.8e-15 | 1.5e-4 / 1.5e-4 / 6.3e-3 / 6.3e-5 |
| Rust K1 residual (exact order / FMA); vs fresh inverse | 7.6e-12 / 7.1e-12; 6.9e-12 | 6.6e-4; 7.6e-5 |
| Rust K3 cancellation (scalar / FMA) | 5.8e-15 / 1.8e-15 | 4.2e-4 / 1.3e-4 |
| Rust K4 residual (all variants) | 1.6e-16 relative | 0.18 |
| Julia `@turbo` vs `@simd`/plain: K1 / Pfaffian ratio / K2 / K3 / K4 | 3.4e-12 / 1.7e-13 rel. / 6.7e-16 / 3.6e-15 / 3.6e-14 | 3.8e-5 / 7.7e-6 / 8.5e-3 / 1.3e-4 / 1.4e-2 |
| Julia K1 residual (simd / turbo); K3 cancellation; K4 residual (plain / turbo) | 8.1e-12 / 6.6e-12; 7.8e-16 / 6.7e-16; 1.6e-16 / 2.0e-16 | 7.1e-4 / 5.8e-4; 5.7e-5 / 4.9e-5; 0.18 / 0.22 |

The first numerical divergence from the reference is reported per K1 variant
(`first_divergent_hop`): none for the exact-order Rust variants, hop 0 for `pulp-v3-fma` and
Julia `@turbo`. The `differing_values` counts are informational and show where FMA or
reordering changes values: K1 all 16384 entries (FMA, `@turbo`), K2 1040 (FMA) and 1414
(`@turbo`), K4 13263 (`@turbo`, even though the loop has no reduction; consistent with LV
replacing the loop-invariant division by a reciprocal multiplication, not inspected).

Numerical agreement within these bounds is not the C operation-order contract. `@turbo` and
the pulp FMA variants contract to FMA and/or reorder the arithmetic; under #190 that changes the
C numerical algorithm, so they remain excluded regardless of the bounds. The exact-order
variants produce the same values as the scalar path on these inputs, which is consistent with,
but not required by, the bounds.

### Kernel findings

- **K1:** pulp-v3 (exact) and the hand AVX2 kernel are equal within code-placement and
  run-to-run variation. The process ranges overlap: 12.6–15.0 µs vs 12.4–13.7 µs. With FMA,
  pulp is *slower*, because the FMA chain lengthens the accumulator dependency.
- **K2:** this kernel is sensitive to code placement.
  - In the first exploratory build, pulp-v3 beat hand AVX2 by 16 % (60.9 vs 71.5 µs).
  - After an unrelated rebuild the order flipped, and in this final run pulp-v3 is 6 % slower.
  - Replacing the AVX2 masked tail with a scalar tail (pulp's partial stores avoid
    `vmaskmovpd`) made no difference (62.5 vs 62.9 µs).
  - FMA is within about 2 % of hand AVX2 and changes 1040 entries. No variant gives a robust
    gain.
- **K3:** this is the only kernel where a reassociating SIMD form is much faster: −26 % for
  pulp-v3-fma, and 2.4× for Julia `@turbo` against the Rust exact kernel.
  - It is also the one kernel where the C order (a serial left-to-right sum per row) is the
    documented contract (#449). Julia-mVMC already uses `@turbo` here, so this is a known,
    documented C/Julia difference.
  - An exact SIMD form that keeps rows in lanes is shuffle-bound, so it gains only 4 %.
  - K3 is 5.8 % of PhysCal t1 and does not run in Opt, so the most a C-order-preserving
    rewrite could save is about 4 % of PhysCal. See the gaps section.
- **K4:** pulp's generic `Simd` trait has no gather and no transpose, so lanes-as-columns has to
  stage through a stack buffer, which makes it 3.9× slower than the hand `_mm256_set_pd` kernel.
  Julia `@turbo` across columns halves Julia's per-column loop, but remains 1.8× slower than the
  Rust AVX2 kernel.

## 3. Whole-workload check of the exact pulp candidate

The C-order pulp ports (identical values to the scalar path on the kernel inputs) looked like
plausible drop-in replacements for K1 and K2, so the
experiment was carried through to the whole workload. A temporary build replaced both production
AVX2 kernels with the pulp-v3 code. Its diff is `results/20261105/workload/pulp-exact.patch`,
using pulp with default features, which the CLI's dependency graph already enables. The
experiment was not committed.

- **Parity:** the full-state `cpu_scaling_contract_586` probe compared `main` t1 against this
  build at t1, t8 and a t8 repeat, for both Opt and PhysCal. It found an **exact RNG/discrete
  match** and **0 differing numerical fields** out of 732,490 (Opt) and 732,990 (PhysCal). See
  `workload/contract-verification.json`.
- **Timing:** a native A/B on the L128 inputs, with one warmup plus N measured repetitions,
  alternating the binary order. Medians of process wall time, in seconds:

| case | main | pulp-exact | change | C timer `UpdateMAll` main → pulp |
|---|---:|---:|---:|---:|
| PhysCal t1 (N=5) | 3.747 | 3.852 | +2.8 % | 1.339 → 1.415 |
| PhysCal t8 (N=5) | 0.960 | 0.972 | +1.3 % | 0.228 → 0.241 |
| Opt t1 (N=3) | 69.42 | 71.04 | +2.3 % | 27.13 → 28.07 |
| Opt t8 (N=3) | 16.15 | 16.68 | +3.2 % | 4.05 → 4.40 |

Every pulp repetition was slower than every `main` repetition, except at PhysCal t8, where the
ranges overlap (0.938–0.965 vs 0.957–0.996 s). Raw data is in `workload/ab-*.log`.
The slowdown sits in `UpdateMAll` (K1). On the real eight-plane working set, the pulp port
loses about 3–9 % in that timer, while the isolated single-plane microbenchmark showed a tie. The
`CalculateMAll` change (K2) is within ±2 %.

No candidate showed a whole-workload gain worth pursuing, so `scripts/benchmark_cpu.sh` (the
1+3 C/Julia/Rust round) was not rerun. The acceptance list reserves that run for promising
candidates.

## 4. Other findings

- **The workspace pulp configuration is effectively scalar.** The workspace declares
  `pulp = { default-features = false, features = ["std"] }`. That compiles pulp *without* its
  `x86-v3` backend, so `Arch::new()` returns `Scalar` in a standalone `pfapack --features
  simd-backend` build.
  - In the `mvmc` CLI, Cargo feature unification with tenferro / t4a-faer enables `x86-v3` and
    `x86-v4`. So the existing complex `zsktf2` pulp path changes ISA depending on the leaf
    binary.
  - Today this is harmless, because that path falls back to scalar for `kk0 < 512`.
  - The measured `pulp-scalar` rows show what code on `pulp::Scalar` costs: about the same as
    the plain scalar loops, and much worse on K4.
- **NEON (macOS / aarch64) was not measured.** No aarch64 host is available here. On aarch64,
  128-bit NEON is the compilation baseline, so LLVM already auto-vectorizes the scalar reference
  loops at the same width pulp's `Neon` backend would use. A pulp port would not add a wider ISA
  there. This is an inference, not a measurement.

## Remaining gaps / follow-ups

- AVX-512 (pulp `V4`, LV with 8-wide registers) and NEON were not measured. That needs another
  host.
- K3, C-order preserving: `invM` is maintained as an exactly skew-symmetric array. `utu2inv`
  fills the lower triangle by exact negation, and the Sherman–Morrison updates are antisymmetric
  in IEEE arithmetic. If that invariant holds, `t_i = Σ_j M[i][j] a_j` can be read column-wise as
  `-Σ_j M[j][i] a_j`, the same way K1 pass 1 (and C `UpdateMAll`) already read it. That would
  give a contiguous, register-blocked AVX2 form with the same per-row serial order. This is a
  hand-AVX2 idea, not pulp/LV, and it is worth at most about 4 % of PhysCal. It would need its
  own issue and the full-state parity probe.
- Thread-pool wait time dominates the 8-thread profiles. That is outside this kernel-level
  evaluation.

## Reproduce

```bash
# kernels (Rust builds + Julia), 5 processes x 15 rounds on CPU 2; fails on any numerical check
benchmark/pulp_lv_641/run_kernels.sh benchmark/pulp_lv_641/results/<date>/kernels 5 15 2
# correctness only (one process, one timing round; the timings are not evidence)
benchmark/pulp_lv_641/run_kernels.sh benchmark/pulp_lv_641/results/<date>/correctness 1 1 2
# Julia only
julia +1.13.1 --project=extern/Julia-mVMC -t 1 benchmark/pulp_lv_641/julia/lv_kernels.jl 128 15
```

Sections 1 and 3 used the native (no Docker) drivers in `tools/`, which follow
`benchmark/cpu_scaling_l128/{profile.py,contract.py}`:

```bash
tools/profile.sh "$PWD/target/release/mvmc" phys 1 /tmp/pp-phys-t1 perf    # perf profile
uv run --no-project python tools/contract_native.py OUT PROBE_A PROBE_B --threads 1 8
uv run --no-project python tools/ab.py phys 1 0 5 main=BIN_A pulp=BIN_B    # A/B, warmup + 5
```

`PROBE_*` are `cpu_scaling_contract_586` example builds. The pulp build is `main` with
`results/20261105/workload/pulp-exact.patch` applied. The `rust/` crate has its own
`[workspace]` and is not built by the main workspace or CI.
