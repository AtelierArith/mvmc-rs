# Pfaffian migration probes (#731)

These are preliminary kernel measurements, not the final adoption/E2E report.
CPU: Linux x86_64, Xeon E5-2699 v3, rustc 1.99.1
(`58cf8f9c3`, LLVM 23.1.1), system OpenBLAS 0.3.26. GPU: RTX 3060 sm_86,
CUDA 12.9.2 container image `b129bd583bbe243671d4d4b9ec18cf39b00d0037cadd098197530d2d33750918`.
Sources: the implementation following `00f7106c` on `perf/faer-pfaffian`;
t4a-faer is the registry release 0.32.0.

## CPU complete rebuild probe

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release -p pfapack \
  --example faer_rebuild_bench --features 'faer-backend blas-backend'
```

`cpu-rebuild-probe-20261111.csv` compares the existing BLAS-enabled
`dsktf2`/`zsktf2` + `utu2pfa` + `utu2inv` with sequential faer Pfaffian plus
blocked LU/inverse. Both reuse preallocated workspaces; the reference includes
its necessary input copy. Each row is one adaptive 250 ms timing interval
after a warmup, **not** a multi-sample median. The matrix is a deterministic
bounded skew matrix with strengthened 2x2 diagonal blocks (see the example).
It is not the random/graded workload of the linked upstream benchmark.

This probe predates adding the final inverse skew projection to the adapter;
rerun with final code for adoption decisions. The complex candidate gains
1.17–1.69x for n=64–512. Real full rebuilds do not show a reliable gain; small
sizes regress. Do not infer a real sampler speedup from the standalone faer
Pfaffian benchmark. Full rebuild integration is currently experimental behind
`mvmc-core/faer-pfaffian`; its default is not enabled.

`cpu-rebuild-samples-20261111.csv` repeats the probe five times with the final
skew projection included, on the same shared host while the GPU gate was also
running. Across the arithmetic means of those five samples, complex n=64–512
gains 1.14–1.69x. Real rebuilds remain slower (0.57–0.95x over n=16–512).
These data confirm that adopting the combined real rebuild by default needs
more optimization or a different inverse strategy; a standalone Pfaffian
benchmark is insufficient. Raw per-sample timings are retained rather than
presenting these shared-host probes as final E2E evidence.

## GPU panel probe

```sh
docker run --rm --gpus '"device=0"' \
  -v "$PWD:/work:ro" -w /work/benchmark/pfaffian_faer b129bd583bbe \
  bash -lc 'nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 gpu_probe.cu \
    -o /tmp/mvmc-pf-panel-probe && /tmp/mvmc-pf-panel-probe'
```

The optional CUDA probe compiles the actual production helper source. It is
independent of Cargo builds/tests and adds no C reference runtime requirement.
`gpu-probe-20261111.csv` contains 32-plane batches and CUDA event timing for 40
launches, including device input copies but excluding allocation and host
transfers. It compares the old and panelized standalone-Pfaffian helpers. It
checks 30 bounded dense planes against the old helper, plus analytical
block-diagonal and zero planes. All observed differences were zero; the
assertion uses an explicit numerical bound, not bitwise equality.

Panel updates gain about 1.44–1.94x for real n=128–512 and 1.07–1.68x for
complex n=64–512 in this probe. Production dispatch retains the unblocked
helper below these respective sizes. The new helper uses the existing two
n-vectors and matrix scratch, with no added allocation, transfer or launch.
This is a CUDA port of the panel-update organization, not a faer GPU API call.

**Not yet established:** repeated medians, broad conditioning/nonfinite
coverage, resident Opt/PhysCal E2E performance and transfer counters, full
Julia GPU fixture gates, and the complete Pfaffian/inverse rebuild kernels.
At the time of that standalone probe, the complete rebuild kernels were not
changed. The additional complete-rebuild work below is validated separately.

## GPU complete-rebuild panel probe

`gpu_inverse_probe.cu` builds the actual complete-rebuild helpers. Compile it
with the same nvcc flags as the standalone probe above (substitute its source
and binary name). `gpu-rebuild-probe-20261111.csv` records RTX 3060 device 1,
16-plane batches, CUDA event times for ten launches, including device assembly
copies and excluding host transfers/allocation. Fifteen planes are bounded
dense matrices; one zero plane verifies the reference status/failure fallback.
Inverse differences and Pfaffians are checked numerically against the old
upper-LTL implementation before timing.

The complete shared panel kernel gains about 1.9x for real n=64 and 2.0–2.4x
for complex n=64–512 versus the old LTL path. The existing **real resident**
n>=128 path already uses a Julia paired factorization and is substantially
faster than the new candidate (rows `f64-current-resident`); it is retained,
along with its independent C-order cancellation fallback. Production now uses
the new kernel for real n=64..126 and complex n>=64. Complex launch geometry
reserves `44n` shared bytes and stays within 48 KiB at the supported n<=1024.
The same three global matrices are reused, with no new allocation, launch or
host round trip. The original input backup is retained through inversion;
numerical/pivot failures, overflowing reciprocals and nonfinite inverse entries
restore it and take the reference path on device. `scaled_fallback_probe.cu`
provides a negative control: with the `38f2849b` inverse header, a connected
tridiagonal n=64 input incorrectly changes reference INFO=0 to -1. The fixed
kernel preserves INFO, Pfaffian and reference nonfinite-output classes for both
real and complex storage. This preserves the existing extreme-scale contract;
it does not claim accurate inverse entries where the reference produces NaNs.

Rust tests of the production dispatch pass for real/complex n=24/26/64/128/130/256,
covering an independent CPU Pfaffian/inverse, exact zero-pivot status, zero
failed output, normwise backward residual and conditioning-scaled inverse
comparison. The production complex L128 Opt stage test passes too. Full final
GPU gates, repeated timings, transfer counters and Opt/PhysCal E2E measurements
for the final source remain pending. The earlier standalone-panel gate passed
128 tests; `validation/cuda-full-rebuild-gate.log` passed 131 tests before the
later original-input preservation fix. Targeted inverse residual and scaled
fallback tests and all-target CUDA Clippy pass after that fix; the final broad
gate passed 131 tests at the original-input-preservation source (118 library,
three CLI, one interval gate, three parity and six resident interval tests).
The separate two-device integration test passes too, for 132 total. Its first
follow-up command accidentally selected the library target and ran zero tests;
`validation/cuda-two-device-context.log` records the corrected integration
target. Validation logs identify these distinct source stages. Later performance
experiments changing inlining/dispatch must not inherit this full-source claim.

## End-to-end evidence and current adoption limits

See [e2e/README.md](e2e/README.md) for matched public API measurements, raw
outputs, source manifests and transfer counters. The fixed-input-backup CUDA
kernel improves complex L128 Opt by 1.1593x and PhysCal by 1.1260x. The same
source regresses real L128 Opt and short PhysCal by roughly 7% and 5%; this is
being investigated and prevents declaring adoption complete. CPU default
adoption and its full-factor INFO boundary also remain unresolved.

## CUDA compilation resource diagnostics

`cuda_module_resources.cpp` compiles a concatenated interval module with NVRTC
and queries the driver without launching kernels. Build inside the recorded
CUDA 12.9.2 image with:

```sh
g++ -O2 -I/usr/local/cuda/include cuda_module_resources.cpp \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
  -Wl,-rpath,/usr/local/cuda/lib64 -lnvrtc -lcuda -o resources
./resources interval.cu
# Separate diagnostic, not RawExec::module's production contraction setting:
./resources interval.cu --no-fmad
```

Construct `interval.cu` by joining the files named by `include_str!` in
`gpu/mvmc-gpu-cuda/src/interval.rs` in their listed order, separated by newlines.
The probe uses `compute_86`, C++14 and ordinary contraction by default, matching
`RawExec::module`; `*-no-fmad.csv` disables contraction and is labelled separately.
Recorded files compare baseline `fae8eb14`, combined panel source `4fef0a87`,
that source with `separate-entry-experiment.patch`, and the noinline experiment
(only `resident_block_pfaffian` gains `__noinline__`). The separated entry lowers
its register count but does not cure the measured real E2E regression, so that
patch is retained as a reproducible experiment and is not applied to production.
The noinline experiment also does not cure the regression and is not retained
in production. Resource counts alone do not establish a performance explanation.
