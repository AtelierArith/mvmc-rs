# L128 PhysCal on RTX 3060 (#587)

The legacy single-SFMT-chain CUDA PhysCal path spends about three seconds
compiling the same sampler/measurement kernels in each new process. Reusing
validated PTX removes that repeated startup work. It does not change the shaders,
proposal/acceptance contract, samples, observables or kernel throughput.

## Workload and sources

Half-filled real Hubbard chain L=128, eight QP planes, seed 79, 320 saved samples,
warmup=1, interval=1, one PhysCal window, no Lanczos. Measure energy, 256 one-body
and 768 direct two-body Green terms; there are no factored Green terms or Slater
derivatives in this case. Definitions and fixed parameters are the same as
`../cpu_scaling_l128/inputs/phys`; parameter SHA-256 is
`9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90`.

Before source: `54c20abb28528eafd4d25e26052797d09045e4f9`.
Final runtime/probe source: `e760ad79c08efb9e8342409a6d12b2f67935bc69`
(cache implementation `1c5d01ee`, version-report correction #626).
Binaries were built with `cargo build --locked --release` in the standalone
`gpu/mvmc-gpu-cuda` workspace, whose release profile differs from the root CPU
workspace. Source/build/binary/input hashes and actual environment are retained.
The final binaries were built from the corresponding bytes before committing;
the exact file fingerprints supplement the source identifier.

Linux x86_64, Xeon E5-2699 v3 (18 physical cores/36 logical CPUs), Rust 1.99.0,
RTX3060 12 GiB physical GPU 1, UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`,
driver 580.178.04. BLAS/OMP/MKL/BLIS=1. GPU runs use CPU0 and one host inner/Rayon
thread. CPU API sweeps pin first N physical cores (0..N-1), with inner/Rayon=N.
No frequency control or exclusive host ownership is claimed.

Julia 1.13.1, reference `22bc829d3607e4aff860d5b0d1244932fac585fe`,
`test/gpu/Manifest-v1.13.toml`, ILP64 OpenBLAS, one Julia thread and CPU0.
Native Rust BLAS resolves to the system OpenBLAS pthread library. Julia uses artifact runtime 13.4.0/cuBLAS 13.8.0; native Rust reports NVRTC
12.8/cuBLAS 12.8.4. Driver and library versions are recorded separately; these
are not identical CUDA software stacks.

## Process-wall results

Each condition has one warmup and three measured repetitions. CLI times include
process/CUDA setup, parsing, sampling, measurement, file output and exit.
Diagnostics are separate and excluded. No Docker startup is included.

| RTX3060 / Rust CLI condition | Median |
|---|---:|
| Before change | 5.866 s |
| Final, cache disabled | 5.860 s |
| Final, empty cache for every process | 5.846 s |
| Final, cache reused across processes | 2.852 s |

The final disabled/hit comparison is about 2.05x. Cold executions retain the
compilation cost; a cache hit is not a cold-start speed claim. An intermediate
cohort measured 5.917/2.848 s; retain it as a separate cohort, not a selected
replacement for the final result. All ranges/repetitions and inclusive section
medians are in `results/20261103/summary.json`.

The initial attribution run observed sampler NVRTC 1.594 s and measurement
NVRTC 1.456 s. These diagnostic segments explain most of the startup difference,
without claiming they cover all startup/exit work. Process wall minus internal
All includes all untimed work, not just CUDA initialization. Sampling and
measurement durations remain separately visible in the raw C-compatible timers.


| Internal phase / process boundary | Cache disabled | Cache hit |
|---|---:|---:|
| All | 2.54926 s | 2.55313 s |
| Input/parameter initialization | 0.01067 s | 0.01115 s |
| Sampling | 2.05494 s | 2.05968 s |
| Measurement | 0.44313 s | 0.44190 s |
| Weight averaging | 0.00000 s | 0.00000 s |
| File output | 0.00188 s | 0.00186 s |
| Process work outside All | 3.30610 s | 0.29881 s |

Inclusive sections overlap; do not sum them. Outside All is the median of each
repetition’s process wall minus its own All timer, not a pure startup timer.
Cold cache means empty mvmc PTX storage; OS and CUDA driver caches were not purged.

## Warm public APIs

These calls include input/parameter parsing, fresh state initialization,
sampling, observables and the returned host results. Process/pool setup and
repetition zero are excluded. The output-free Rust API uses `vmc_phys_cal`;
Julia uses its synchronized public GPU API. Julia serializes results only after
stopping the API timer. This is closer timing scope, but the sampling work still
differs as described below.

| CPU threads | Rust CPU, output-free API |
|---:|---:|
| 1 | 4.254 s |
| 4 | 3.004 s |
| 8 | 2.264 s |
| 16 | 1.803 s |

| RTX3060 API | Median |
|---|---:|
| Rust, output-free | 2.563 s |
| Rust, with file output | 2.515 s |
| Julia, output-free | 0.735 s |
| C | No GPU version |

The file/output-free results are separate sequential cohorts with uncontrolled
frequency/load; their subtraction is not an I/O cost measurement. The maintained
Python API driver was additionally exercised on GPU (median 2.516 s), recorded
as a driver verification cohort. Primary rows above remain the predeclared final
sequential sweep. For C/Julia/Rust CPU CLI comparisons and their different build
and timing scopes, see the CPU scaling report and the historical Wiki tables.

Rust uses one SFMT chain, 41,088 configured proposals, and 320 serial saved
samples. Julia uses 320 XORWOW walkers with 81,920 configured proposals. Both
compute the input observables, but these are different trajectories, ensemble
layouts and proposal counts. The Rust/Julia wall-time ratio is not a matched
kernel speedup. Adding walkers or replacing SFMT would change the legacy
execution contract; the accepted resident Opt ensemble is not automatically
an equivalent PhysCal implementation.

## CUDA profile (separate run)

Nsight Systems 2022.4.2.50 was extracted locally from Ubuntu packages, without
system installation or perf/driver setting changes. Sampling/context-switch
collection was disabled; CUDA/OSRT tracing was enabled. The raw QDSTRM,
converted report, SQLite export, tool/package fingerprints and conversion log
are retained. All 72,110 captured driver launches have corresponding kernel
activity records. API and device durations overlap; do not sum them.

| Observed operation | Calls | Device bytes / time |
|---|---:|---:|
| Host to device | 44,295 | 11,614,480 bytes |
| Device to host | 41,944 | 3,199,160 bytes |
| Event synchronization | 41,205 | 1.617 s host API time |
| Stream synchronization | 721 | 0.351 s host API time |
| `k_accept` | 14,966 | 0.708 s device time |
| `k_propose` | 41,088 | 0.273 s device time |
| Sampling Pfaffian rebuild | 117 | 0.318 s device time |
| Measurement batched Pfaffian | 20 | 0.284 s device time |

Kernel-interval union is 1.700 s in the 3.239 s first-to-last-kernel window
(52.5%). This is activity of this process, not SM occupancy or whole-process
GPU utilization. Proposal/update launches have eight blocks; the recorded GPU
has 28 SMs. This gives limited independent work per launch, not a measurement
of achieved occupancy. Measurement already batches 128 planes per call.
Many small transfers and synchronization points, rather than bulk link volume,
remain visible. No allocation or bandwidth cause is inferred from total wall
alone. `analyze_trace.py` reproduces exact counts, kernel groups and interval
union from SQLite; instrumented times are excluded from speed tables.

The resident counters confirm inverse-table downloads are zero, 2,560 planes
are factorized, Slater upload is 4,194,304 bytes and PF/status downloads are
30,720 bytes per window. These counters are only a subset of the full CUPTI
transfer totals above. GPU Green measurement is already resident. Future
ensemble work must declare its RNG/output contract; caching alone does not
resolve every remaining single-chain throughput limitation.

## Numerical and RNG validation

- CPU/GPU four runs each match the previously captured CPU reference exactly
  for all 624 SFMT words, cursor, 206,174 consumed words, all saved configuration
  arrays/counters, and 41,088 actual decisions/draws. The nested trace observer
  was repaired so the inline device runner preserves its caller's capture.
  This is a cross-implementation/Rust-revision probe, not a newly captured
  complete C-executable SFMT-state oracle. Independent C fixtures remain intact.
- 33 saved CLI/API/profile output sets pass shape, exact integer-index and finite
  numerical checks against the captured CPU cohort. Each has 2,054 computed
  values and 7,168 integer indices; maximum absolute difference is
  `3.638e-12`. Comparison uses explicit `1e-11 + 1e-11 max(|a|,|b|)` bounds.
- To locate the Julia/CPU difference before proposals, an initialization-only
  probe changes only sample count to one and warmup/interval to zero. Its first
  sampling word is SFMT 2884474187 versus XORWOW 3704967289; first electron site
  is 75 versus 1. The differing initialization/RNG contracts precede burn-in.
- For an independent operator check, the exact CPU initial configuration is
  supplied to Julia GPU with zero hops. Rust energy 102.72950887694306 and Julia
  102.72950887694073 differ by `2.331e-12`; all 256 one-body and 768 direct
  two-body values agree (max `1.493e-13` / `2.558e-13`). This is a fixed-state
  check, not equality of long trajectories. The original average energies
  116.54573126779692 and 116.29136644085115 must not be dismissed as noise.
- Cache tests cover changed source/options, truncation/corruption, concurrent
  writers, invalid PTX rejection, disabled/unwritable cache and cold/hit paths.
  Three cache tests, five trace tests, nine device-sampler tests and eight
  CUDA measurement/production-sampler/SR regression tests passed. Standalone
  all-target Clippy, core all-target Clippy and formatting passed.

An early supplemental warm-probe log divided an already normalized energy by
`wc` when displaying it. Production output was unaffected; the final example
and all final cohorts print the correct value. Historical logs are preserved,
with this limitation rather than silently rewritten.

## Reproduction

Use a Linux checkout at the source above (or the merged commit), initialized
Julia submodule, Juliaup 1.13.1, and available NVIDIA driver/NVRTC/cuBLAS/cuSOLVER/
cuTENSOR libraries. The native standalone build has no Docker dependency.
Set `LD_LIBRARY_PATH` for your library installation; exact measured paths and
commands are retained in the raw local-command record. Check CPU topology
before using the scripts on another host. All output directories must be new.

```bash
bench_repo=$PWD
bench_target=/tmp/mvmc-587-reproduce-target
bench_input="$bench_repo/benchmark/cpu_scaling_l128/inputs/phys"
export CARGO_TARGET_DIR="$bench_target"
cargo build --locked --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --bin mvmc-cuda --example physcal_warm_587
export MVMC_RS_CUDA_KERNEL_CACHE=off
uv run --no-project python benchmark/gpu_physcal_l128/measure.py \
  --binary "$bench_target/release/mvmc-cuda" --input "$bench_input" \
  --out /tmp/mvmc-587-disabled --device 1
uv run --no-project python benchmark/gpu_physcal_l128/measure.py \
  --binary "$bench_target/release/mvmc-cuda" --input "$bench_input" \
  --out /tmp/mvmc-587-cold --cold-cache --device 1
export MVMC_RS_CUDA_KERNEL_CACHE=/tmp/mvmc-587-reproduce-cache
uv run --no-project python benchmark/gpu_physcal_l128/measure.py \
  --binary "$bench_target/release/mvmc-cuda" --input "$bench_input" \
  --out /tmp/mvmc-587-hit --device 1
uv run --no-project python benchmark/gpu_physcal_l128/api.py \
  --binary "$bench_target/release/examples/physcal_warm_587" --input "$bench_input" \
  --out /tmp/mvmc-587-cpu-api --backend cpu --threads 1,4,8,16
uv run --no-project python benchmark/gpu_physcal_l128/api.py \
  --binary "$bench_target/release/examples/physcal_warm_587" --input "$bench_input" \
  --out /tmp/mvmc-587-gpu-api --backend gpu --device 1
julia +1.13.1 --startup-file=no --project=extern/Julia-mVMC/test/gpu \
  -e 'using Pkg; Pkg.instantiate()'
CUDA_VISIBLE_DEVICES=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 julia +1.13.1 --startup-file=no --project=extern/Julia-mVMC/test/gpu \
  benchmark/gpu_physcal_l128/julia-runner.jl gpu "$bench_input" /tmp/mvmc-587-julia
```

For API file output add `--output-files`. For correctness, invoke the example
with `--contract` and selectors `c-order` or `cuda`; these diagnostic runs must
not be used as speed measurements. The initial-only probe and Julia fixed-state
commands are recorded separately with their modified definitions. Cache controls
and initialization profiling are documented in manual 8.5.

To inspect the published raw archives, verify it first, extract into a fresh
folder, then run `summarize.py`, `compare_outputs.py`, `compare_contract.py`,
`compare_julia_initial.py` or `analyze_trace.py` on the retained paths. The exact
local commands and generic script `--help` options explain their arguments.
Per-archive/file SHA-256 ledgers retain the measured bytes and original mtimes;
executables and third-party profiler binaries are excluded. Third-party profiler
package versions/hashes and extraction/conversion commands are retained instead.

```bash
bench_results="$bench_repo/benchmark/gpu_physcal_l128/results/20261103"
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py "$bench_results"
bench_extract=/tmp/mvmc-587-extracted
mkdir "$bench_extract"
for bench_part in measurements nsight-report nsight-stream; do
  tar -xzf "$bench_results/$bench_part.tar.gz" -C "$bench_extract"
  mv "$bench_extract/raw/original-files.json" "$bench_extract/$bench_part-files.json"
done
uv run --no-project python benchmark/gpu_physcal_l128/summarize.py \
  "$bench_extract/raw" "$bench_extract/summary.json"
uv run --no-project python benchmark/gpu_physcal_l128/compare_outputs.py \
  "$bench_extract/raw" "$bench_extract/raw/reference-cpu-output"
uv run --no-project python benchmark/gpu_physcal_l128/compare_contract.py \
  "$bench_extract/raw" "$bench_results/before-cpu-exact.json.gz"
uv run --no-project python benchmark/gpu_physcal_l128/compare_julia_initial.py "$bench_extract/raw"
uv run --no-project python benchmark/gpu_physcal_l128/validate_julia.py "$bench_extract/raw"
uv run --no-project python benchmark/gpu_physcal_l128/analyze_trace.py \
  "$bench_extract/raw/nsight-probe/trace.sqlite" "$bench_extract/nsight-summary.json"
```

`profile.sh INPUT BINARY FRESH_OUTPUT NSYS [QDSTRM_IMPORTER]` repeats the optional
CUDA trace with sampling/context-switch collection disabled. This helper is
syntax-checked; the recorded run used the equivalent explicit CLI commands in
`raw/nsight-reproduction.sh` and the capture/conversion logs. The SQLite analyzer
expects the recorded Nsight 2022 schema. Profiling is not required to run PhysCal.
