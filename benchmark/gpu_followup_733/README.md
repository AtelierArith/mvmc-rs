# Resident CUDA follow-up (#733)

This follow-up measures additional GPU work after #732. The baseline is
`f59910852fef41a1141d8b40d6229d6d106e9011`; the final numerical implementation is
`bb100b08e6ef11ee1542774c8a31b7ca4aa6ed55`. These comparisons do not reuse the
older CPU/faer baseline.

## Final matched measurements

| L128 workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| complex-opt | 63.155095 s | 57.678730 s | 1.0949x |
| complex-phys | 5.228113 s | 4.780612 s | 1.0936x |
| real-opt | 4.996253 s | 4.986950 s | 1.0019x |
| real-phys | 0.566065 s | 0.567798 s | 0.9969x |

Times are medians of three alternating before/after repetitions, after one
excluded warmup repetition per implementation. Both workers use the same source
for `pfaffian_e2e_worker.rs`, the same CUDA lock file, one RTX 3060 (physical
GPU 1), CPU affinity 17 and one host thread. There are 320 walkers, seed 79,
128 warmup and interval proposals, 20 optimization steps or one PhysCal call.
Only one GPU call runs at a time; no validation runs overlap these measurements.
The scope includes the public API's input parsing, fresh initialization,
sampling, optimization/observables, final downloads and internal teardown.
Worker startup, CUDA JIT warmup and output serialization are excluded.

All recorded output sections have zero observed absolute difference across
revisions and repetitions. Discrete outputs match. Same-implementation
repeatability is checked with the existing explicit absolute/relative bounds,
not floating-point bitwise comparisons. Every explicit transfer call, byte and
synchronization counter matches across revisions. Real L128 is effectively
unchanged; its +/-0.3% variation is not a speedup claim.

Metadata in each `final-*/metadata.json` records source and input SHA-256 hashes,
Git revisions, image ID, GPU UUID and driver version. `timings.json` contains
every repetition, execution order, timestamps, host load and explicit transfer
counters. Counters cover tenferro-ext raw execution calls; library-internal
copies or synchronization are outside that instrumentation.

## Changes and profiling evidence

* Four warps update independent complex QP planes concurrently for matrix sizes
  64 through 128, at least four QPs, and at most 32 KiB dynamic shared memory.
  Each warp retains the original scalar dot-product order. Its identical Smith
  reciprocal is computed once and shared. Large QP counts and other dimensions
  retain the original block-wide update route.
* Separate template-specialized entry points keep the legacy kernel's register
  profile. On this sm_86 device, both final entries use 168 registers/thread,
  zero local spill memory and 480 static shared bytes, permitting three
  128-thread blocks/SM for the measured shared-memory geometry. An intermediate
  runtime branch used 234 registers and allowed two blocks/SM.
* Panel triangular inversion skips complete warp-width groups of structural
  zero products, retaining ascending order for all remaining terms and
  coalesced loads. A finite-factor check preserves the original-input fallback
  for nonfinite factors. This affects the existing complex panel and small-real
  panel routes; the real L128 Julia-paired factorization remains in use.

No RNG, proposal, acceptance or transfer scheduling code changes. There is no
additional per-hop host work, launch, allocation or synchronization.

The baseline three-step Nsight trace attributes 54.9% of GPU kernel time to
`interval_complex_hop`, 34.7% to `interval_complex_factor`, 7.5% to energy terms
and 2.6% to derivatives (`profile/baseline-kernels.csv`). Its wall time includes
cold JIT and profiler overhead and is **not** an E2E timing comparison. These
are kernel-time shares, not measurements of SM occupancy.

Exploratory three-step cohorts, two measured repetitions each, gave 1.037x for
warp updates alone, 1.101x with naive zero skipping, 1.107x with coalesced zero
skipping, and 1.091x with specialized entries. They overlapped validation on the
other GPU and are not final speed claims. Specialization was selected to retain
the legacy register profile outside eligible geometries; the narrow timing
differences do not establish that occupancy caused the E2E gain. Intermediate
patches and source hashes preserve which implementation each cohort measured.

## Validation

The final implementation passed all 134 tests in the CUDA gate: 120 library,
three CLI, one interval gate, three interval parity and seven resident interval
tests. This includes the actual two-device ownership check, supplied-tape
CPU replay, stage and solver residual checks, and existing 20-step repeatability.
The new ten-QP regression checks a partial final warp group. Its acceptance
counts, configuration and SR/gauge checks pass with the existing tolerances.
No numerical tolerance or independent expected result was changed.

`profile/final-gate-coverage.json` audits every expected test name against the
sharded and retry logs; there are no missing or unexpected names. Some initial
runs failed or were interrupted by host disk exhaustion, including a fixture
copy returning `StorageFull`. These logs are retained. Every unfinished or
failed case was rerun successfully, using a tmpfs for test fixtures; the coverage
audit includes only successful test records. `partial-debug.log` contains the
successful focused partial-QP regression. The earlier exploratory 21-test run
is not substituted for final-source coverage.

`final-cuda-gate.log` is an incomplete redundant run affected by the same host
storage problem, not a passing full-suite record. The name-by-name audit uses
the explicit shard/retry logs listed in `final-gate-coverage.json` instead.

Final CUDA `cargo clippy --all-targets --locked --offline -- -D warnings`,
formatting and test compilation passed (`profile/final-gate-build.log`).
The ordinary Cargo tests do not invoke these benchmark or reference helpers.

Physical GPU 0 emitted corrected PCIe RxErr messages at BDF `0000:03:00.0`,
filling host system logs during validation. Final timings use physical GPU 1
(`0000:04:00.0`) alone. Recovery removed only owned build/compiler-cache
artifacts; system logs, driver settings and PCIe registers were not changed.
The existing kache 0.28.1 configuration was retained and an actual cache hit
verified after cleanup (`profile/cache-recovery-verification.log`).

## Reproduction

Run Python helpers through `uv run --no-project python` from the repository root.
Create a detached baseline checkout at the SHA above and build the worker in
separate target volumes (Cargo `--manifest-path gpu/mvmc-gpu-cuda/Cargo.toml
--locked --offline --release --example pfaffian_e2e_worker`). Mount the desired
checkout at `/work`, host Rust toolchains/cargo cache, and the corresponding
volume at `/cuda-target`; set `CARGO_TARGET_DIR=/cuda-target` and one BLAS/OMP
thread. The final source uses the same lock and worker as baseline.

```sh
uv run --no-project python benchmark/gpu_followup_733/prepare_inputs.py /tmp/complex-phys
uv run --no-project python benchmark/pfaffian_faer/run_gpu_e2e.py \
  --before /tmp/mvmc733-baseline --after "$PWD" \
  --before-volume mvmc-733-before-cuda-target --after-volume mvmc-731-cuda-target \
  --input "$PWD/gpu/mvmc-gpu-cuda/tests/fixtures/complex/hubbard_complex_L128/namelist.def" \
  --output /tmp/reproduce-complex-opt --mode opt --steps 20 \
  --image mvmc731-cuda-toolchain:local --driver-cache-volume mvmc-733-driver-cache \
  --device 1 --cpu 17 --walkers 320 --warmup 128 --interval 128 --reps 3
```

For complex PhysCal select `/tmp/complex-phys/namelist.def`, `--mode phys
--steps 1`, and a fresh output directory. Real inputs are
`benchmark/cpu_scaling_l128/inputs/{opt,phys}/namelist.def` with the corresponding
mode and step count. The exact four-cohort command is `run_final_bench.sh`.
Use fresh output directories; the runner refuses to overwrite an existing one.

`assemble_module.py CHECKOUT OUTPUT.cu` reconstructs the concatenated NVRTC
source for the resource observer from that checkout's `INTERVAL_MODULE`.
Build the observer in the CUDA image with
`g++ -O2 cuda_resources.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64
-lnvrtc -lcuda -o /tmp/cuda-resources`, then run it against the assembled source
with only the desired physical device visible. The baseline and specialized
source snapshots and Nsight capture are archived in `profile/captures.tar.gz`.
Raw worker output and logs are archived in each cohort's `raw-results.tar.gz`,
with per-file SHA-256 manifests. Metadata and numerical reports remain directly
readable. `profile/final-test-artifacts.json` records the exact test executables;
`run_cuda_shard.py` and `audit_cuda_gate.py` preserve the validation procedure.
The separate two-device case must be run with both GPUs visible.


CUDA toolchain: Rust 1.99.1, CUDA/NVRTC 12.9, driver 580.178.04,
cuBLAS 12.9.2 and cuSOLVER 11.7.5, Linux x86_64. The Docker image ID is recorded
in every cohort. Both worker binaries must be built from their recorded source
revisions with their locked CUDA workspace before measurement. The optional
`cuda_resources.cpp` observer and Nsight capture are developer profiling tools,
not build or test dependencies.
