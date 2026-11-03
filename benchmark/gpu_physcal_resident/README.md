# Resident GPU PhysCal on L128 (#636)

The explicit resident PhysCal API measures **0.554 s**, versus **0.717 s** for
Julia GPU on the same RTX 3060. The existing single-SFMT API measures 2.488 s.
The new API and `mvmc-cuda resident-physcal` use Julia-compatible XORWOW walker
ensembles; the ordinary CLI and single-SFMT API retain their original contract.

## Workload and results

Half-filled real Hubbard chain L=128, U=4, t=1, seed 79, eight QP planes,
320 samples/walkers, one fixed-parameter PhysCal window, no Lanczos. All use
the committed `../cpu_scaling_l128/inputs/phys` definitions and parameters
(SHA-256 `9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90`).

Each process runs once for warmup, then three measured calls. Calls include
parsing, fresh state initialization, sampling, observables and final host means.
Startup/JIT, repetition zero, explicit Julia GC and serialization are excluded;
Rust's API includes its internal state teardown. The implementations run
sequentially.

| Warm public API | Measured calls (seconds) | Median |
|---|---|---:|
| Rust, legacy single SFMT | 2.496164, 2.487675, 2.487960 | 2.487960 s |
| Rust, resident XORWOW | 0.552157, 0.553533, 0.554715 | 0.553533 s |
| Julia, XORWOW | 0.722295, 0.717060, 0.712242 | 0.717060 s |

Rust resident is about 1.30 times faster than Julia in this cohort. The 4.49
legacy/resident wall ratio includes a sampling-contract change: legacy uses
one SFMT chain and 41,088 proposals; the resident implementations use 320
XORWOW walkers and 81,920 proposals (128 warmup + 128 interval per walker).
This is not a matched-kernel speedup or an SFMT trajectory-preservation claim.

Runtime source: Rust `03b339986770c2f07c38f8a7acc405e19ba55382`; legacy binary
`67b0a2c8d418876d8a40e4d35a9cb341567590e9`, hash-verified against #635's
retained executable. The final runtime commit preserves C PhysCal input-file
priority. The original `7955a882` cohort (0.570727 s) is retained in
`performance.tar.gz`; the latest cohort is in `final-performance.tar.gz`. All
4,100 resident values agree numerically between these cohorts (observed difference
zero), and their no-`In*` inputs are byte-identical. Julia source
`611031147f1bad9704c55084f6817feeba481b13`, Juliaup `+1.13` resolving to
**1.13.3**, byte-identical GPU manifest SHA-256
`86c69ea3a0c04c2b6af6f48a3f36ec2a08fa2bc564ac4845f5799e53244fc573`.

Linux x86_64, Xeon E5-2699 v3, Rust 1.99.0, physical GPU 0 RTX 3060 12 GiB,
UUID `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`, driver 580.178.04.
All cases use CPU17 and one host/BLAS thread on a shared host; frequency/host
isolation is not claimed. Per-case timestamps and host load are retained.
A 4.66 s Clippy check on CPUs 16/34/35 overlapped Julia setup, before its warmup
and measured calls. The earlier cohort overlapped separate CPU benchmarks.
Rust reports NVRTC 12.8/cuBLAS 12.8.4/cuSOLVER 11.7.3; Julia uses CUDA runtime
13.4.0/cuBLAS 13.8.0/cuSOLVER 12.3.4. Native/Julia linked libraries are recorded
separately, including Julia's ILP64 OpenBLAS.

## Why it improves

The legacy #587 trace found 41,205 event synchronizations and tens of thousands
of tiny transfers, driven by host decisions for each single-chain proposal.
The new path reuses the resident Opt initialization/sampler, applies Julia
PhysCal's immediate updates, evaluates operators in parallel, and accumulates
means on device. It omits derivative-basis/SR storage and constructs no host
optimizer state per walker. The sampling/measurement scope has **zero explicit
uploads/downloads and one final stream synchronization**, tested independently.

Every complete API call records 26 raw uploads / 6,942,308 bytes, three
downloads / 8,208 bytes, and 11 raw stream barriers. The separate warmed
Nsight capture confirms those transfer counts/bytes and records one device and
one context synchronization in addition to the 11 stream calls. The capture
includes profiler context binding at its edges. Raw counters cover only
explicit raw-execution operations; Nsight records CUDA library operations too.

| Separate Nsight capture | Calls | Device time |
|---|---:|---:|
| Resident sampling | 4 | 333.33 ms |
| Full factorization (including initialization) | 5 | 154.68 ms |
| Operator evaluation | 1 | 32.37 ms |
| All captured kernels, including cuRAND | 33 | 523.47 ms interval union |

All 33 kernel correlation IDs match 29 driver and four runtime launches.
Sampler grids have 320 walker blocks; factorization grids have 2,560 planes.
Kernel activity covers 99.72% of its 524.91 ms first-to-last-kernel window;
this is this process's activity window, not SM occupancy. Instrumented times
are excluded from the performance table. The analyzer also reproduces all
72,110 launches in the legacy trace, without mixing the two timing scopes.

## Numerical and RNG checks

All four full-size calls (including warmup) compare 1,025 returned values with
independent Julia results using the existing `1e-10 + 1e-10*max(abs(a),abs(b))`
bound. Maximum absolute differences: energy `5.681e-11`, one-body `5.409e-12`,
direct two-body `5.756e-12`. Each implementation repeats with observed numerical
difference zero under a separate abs/rel `1e-12` check. Legacy's different
SFMT mean is retained rather than explained away as noise.

The committed [independent fixture](../../gpu/mvmc-gpu-cuda/tests/fixtures/physcal_julia.md)
checks exact initialization, configurations and accepts after a supplied tape,
fixed-state energy and all Green families (including product-side terms), and
that supplying a tape consumes no resident XORWOW words. The first numerical
difference is in factorization: maximum Pfaffian relative difference `5.446e-13`.
Fixed-state mean differences are at most `3.411e-13`. No tolerance was increased.
Twenty-window repeatability with 128 proposals per window (crossing the periodic
rebuild boundary), transfer invariants, resident Opt regressions and
legacy CUDA sampler/measurement/SR routing checks are recorded in validation.
Input parsing retains C's canonical one-body list when `TwoBodyGEx` enables
the indirect reader; this can differ from Julia GPU's raw one-body list and
has a focused regression. Fixed parameters load before explicit `In*.def`
overlays, preserving C PhysCal priority even though Julia GPU loads them in the
opposite order. A Gutzwiller/orbital collision regression verifies this and
retention of other fixed coefficients. The timing input has neither `In*` nor
`TwoBodyGEx` entries, so these C input rules do not change this comparison.

## Reproduction and CLI

Prepare CUDA 12 libraries and the pinned Julia GPU project first, following
`scripts/benchmark/README.md`. The standard full resident/Julia benchmark is:

```bash
bash scripts/benchmark_gpu.sh --workloads phys
```

To repeat the three-way comparison, retain a legacy executable built in a
separate checkout at the source above, then use fresh target/output directories:

```bash
bench_target=/tmp/mvmc-resident-physcal-target
CARGO_TARGET_DIR="$bench_target" cargo build --release --locked \
  --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --example resident_physcal_worker --bin mvmc-cuda
uv run --no-project python benchmark/gpu_physcal_resident/measure.py \
  --resident "$bench_target/release/examples/resident_physcal_worker" \
  --legacy /path/to/legacy/physcal_warm_587 \
  --legacy-source 67b0a2c8d418876d8a40e4d35a9cb341567590e9 \
  --julia-project /path/to/Julia-mVMC/test/gpu \
  --device 0 --cpu 17 --output /tmp/mvmc-resident-physcal-results
CUDA_VISIBLE_DEVICES=0 "$bench_target/release/mvmc-cuda" resident-physcal \
  benchmark/cpu_scaling_l128/inputs/phys/namelist.def \
  benchmark/cpu_scaling_l128/inputs/phys/zqp_opt.dat /tmp/mvmc-resident-physcal-output
```

Adapt CPU/GPU IDs and `LD_LIBRARY_PATH` to the host. The subcommand exports one
mean per line in parsed operator order (`energy.dat`, `green_one.dat`, `green_two.dat`,
`green_two_ex.dat`) plus sampling metadata; it does not produce the ordinary
CLI's full C-format moments/parameters. One first-process CLI verification
took **8.54 s**, including startup, NVRTC, computation, file output and exit.
That single cold-process observation is separate from warmed API timing; OS
and CUDA driver caches were not purged.

`results/20261103` retains all repetitions, outputs, input/binary hashes,
commands, environment, validation logs and the separate trace. Archives have
original-file SHA-256/size/mtime ledgers; executables, Cargo caches and profiler
binaries are excluded. Verify without CUDA/Julia:

```bash
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py \
  benchmark/gpu_physcal_resident/results/20261103
```
