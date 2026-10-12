# L128 resident GPU Opt: Rust versus Julia, 2026-10-13

The new Rust GPU Opt runner takes **7.687900 s** versus **5.143719 s** for
Julia GPU: **1.4946× Julia's time**, down from the historical Rust GPU
**47.858827 s**. The new public runner is **6.2252× faster than that baseline**.
It performs no explicit H2D/D2H copies inside the complete optimization loop.
Related to #560 and #550; this is an actual 20-step E2E measurement.

## Workload and results

Half-filled Hubbard chain, **L=128, 320 independent walkers, 20 SR steps,
FP64, seed 79**, warmup 128 and interval 128 proposals per walker. Both GPU
runners use Julia GPU's initialization, translation normalization and ensemble
contract: 8 QP planes, 772 parameters and 860,160 total proposals. One host/BLAS
thread, RTX 3060 GPU 1, same input files as the earlier report.

| Runner | Warm median | Warm min–max | First public call |
| --- | ---: | ---: | ---: |
| Rust resident GPU | 7.687900 s | 7.676474–7.703933 s | 15.574669 s |
| Julia GPU | 5.143719 s | 5.139102–5.158336 s | 54.857550 s |

| Warm repetition | Rust (s) | Julia (s) |
| --- | ---: | ---: |
| 1 | 7.676474286 | 5.142062286 |
| 2 | 7.685824738 | 5.143719325 |
| 3 | 7.701114443 | 5.150877494 |
| 4 | 7.687900229 | 5.139101701 |
| 5 | 7.703933181 | 5.158336207 |

The boundary includes parsing, parameter/model setup, device allocation and
uploads, GPU configuration initialization, warmup, all sampling/measurement/SR
steps, final downloads and destruction of the run's arrays. Receipt writing is
outside the timer. Each process stays alive; warm calls retain the execution
runtime and compiled kernels but create fresh model arrays and RNG state. The
first call includes library/kernel startup and is a single observation, not a
warm timing. Process launch-to-ready was 0.002741 s Rust and 5.518632 s Julia.

The same alternating harness as the previous report was used: one first-call
warmup, five measured calls per language, reversed language order on even
repetitions, exclusive GPU execution. No other benchmark, test or build ran
concurrently.

The [historical Rust baseline](../L128-opt-20261012/README.md) uses a sequential
C/SFMT chain and its C numerical contract. The new GPU API follows the owner's
Julia GPU reference direction. Therefore the 6.2252× ratio compares complete
public implementations; it is not a same-trajectory speedup against the old
C-chain runner. The current Rust-versus-Julia comparison uses matching GPU
model/setup and proposal budgets. CPU/C-order execution is unchanged.

## Transfers and ownership

| Measured Rust scope | H2D calls / bytes | D2H calls / bytes | Explicit synchronizations | Raw scopes |
| --- | ---: | ---: | ---: | ---: |
| Entire public call | 27 / 27,621,476 | 4 / 6,344 | 11 | 7 |
| Warmup + all 20 Opt steps | **0 / 0** | **0 / 0** | **1** | **1** |

The whole-call counters are identical in all six measured calls. Uploads are
setup model/configuration data. The four downloads are initialization's pending
count, final error status, 20 energies and 772 parameters. Sampling RNG/tapes,
walker configurations, inverse/Pfaffian tables, observables, derivatives, SR
moments/system/solution and parameter/Slater refresh remain on device. The
loop counters are independently asserted in the implementation tests; the
measured whole-call totals include setup and finalization.

These are `RawExec` counters, **not a PCIe trace**. They exclude provider/library
internal transfers or barriers, kernel argument traffic and allocation/free
internals. They directly establish removal of the explicit intermediate array
copies and host decision reads targeted by #560.

One run owns its entire ensemble, without 320 host optimization states. A
thread-local per-device runtime cache preserves compiled-kernel/native-handle
identity between calls; only that execution context is cached. Independent runs
retain separate model arrays and RNG state.

## Numerical verification

All six runs of both implementations produced finite results and passed a
full comparison of all 20 energy entries and 772 final parameters at
`atol=rtol=1e-10`:

| Quantity | Maximum Rust–Julia absolute difference |
| --- | ---: |
| Energy history (total energy) | 4.32735e-10 |
| Final parameter vector | 3.63083e-11 |

Final energy per site is **0.7795418080427156** in Rust and
**0.7795418080426915** in Julia. Same-implementation repeatability passed at
`atol=rtol=1e-12`; observed energy/parameter differences across repetitions
were zero. Julia's saved final configurations and acceptance counters also
matched exactly across its repetitions. Rust's production API returns only
energies/parameters, so no claim of final cross-language configuration equality
is made from these receipts.

Independent Julia fixtures, generated outside Rust, verify exact initialized
configurations/RNG behavior, initial model/Slater values, inverse/Pfaffian
tables, local energies, derivatives and the first SR update. The first floating
difference was located in factorization; see the
[fixture provenance and bounds](../../../../gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia.md).
A second fixture covers nonzero correlations and all 324 spin-conserving
two-pair overlays on three sites, including coincident/same-electron cases.
Final SR normalized residuals against a CPU reconstruction were 1.872e-16
(L128) and 6.546e-17 (operator fixture).

The covariance code preserves Julia's separately rounded outer product and
subtraction. A fused operation was found to activate spurious variance for one
walker; the zero-update regression now passes. Log-space acceptance also
covers overflow, underflow and a zero uniform draw. Repeated 20-step tests
interleave another seed/device; failed native solves cannot publish parameters.

## Source and validation

- Measured Rust implementation: **`6c80cbc0a100a5741148840dc73eb4877e490382`**,
  release build of `gpu/mvmc-gpu-cuda/examples/resident_opt_worker.rs`.
- Binary SHA-256:
  `01509e6ce1cb817487756c8bd566e3c61759d5dd159eff8f322c981f81e32962`.
- Julia: **`47452f9061ec3de049d891a7aa7fd9cfeac228a4`**, Julia 1.13.1,
  CUDA.jl 6.4.2, `test/gpu` environment.
- GPU 1: RTX 3060 12 GiB,
  `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`, driver 580.178.04.
- Actual Rust libraries: cuRAND 10.3.9.90, cuBLAS 12.8.4.1,
  cuSOLVER 11.7.3.90, NVRTC 12.8.93. Actual Julia libraries:
  cuRAND 10.4.4.72, cuBLAS 13.8.0.4, cuSOLVER 12.3.4.7, NVRTC 13.4.92.
  `/proc` mapping paths are recorded in `environment.json`; the inherited
  `LD_LIBRARY_PATH` does not replace Julia's artifact selection.
- Full CPU workspace gate: **1,609 passed, 63 skipped**, `test-fast`.
- CUDA related lane: **20 passed, 2 manual benchmark/diagnostic tests excluded**.
  After final covariance/bounds changes: **5 Opt tests passed**, followed by a
  final general-operator/index check. Root/CUDA fmt and clippy passed; root
  documentation tests passed (no doctests defined). Raw logs are included.

Remaining work under #560/#550 includes delayed/paired inverse optimization and
singular-QP cofactor fallback. This runner currently rejects unsupported modes
and reports singular/nonfinite numerical failure. Its scope and API are
documented in [GPU-resident Opt](../../../../docs/design/gpu-resident-opt.md).
The existing C-order CLI entry point is not automatically redirected to this
new API.

## Reproduce

Build the pinned Rust revision:

```sh
cargo build --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --example resident_opt_worker
```

Use the input reconstruction and pinned Julia checkout described in the
[previous report](../L128-opt-20261012/README.md), including its archived
`modpara.def` and `namelist.def`. Run the unchanged alternating harness with
the new Rust worker:

```sh
uv run --no-project python \
  benchmark/host_gpu_comparison/results/L128-opt-20261012/mvmc-opt-gpu-comparison.py \
  /absolute/path/to/gpu/mvmc-gpu-cuda/target/release/examples/resident_opt_worker \
  /absolute/path/to/Julia-mVMC /absolute/input/namelist.def /absolute/new-results

uv run --no-project python \
  benchmark/host_gpu_comparison/results/L128-opt-resident-20261013/verify_receipts.py \
  /absolute/new-results
```

The harness explicitly selects this host's GPU and CUDA artifact paths; adjust
and record them on another machine. `timings.json`, `summary.json`,
`numerical-verification.json`, a full numerical receipt from each language,
input/source/binary/receipt hashes, and raw validation/timing logs accompany
this report. Original receipts remain locally in
`/tmp/mvmc-560-opt-e2e-final/{rust-output,julia-output}`.
