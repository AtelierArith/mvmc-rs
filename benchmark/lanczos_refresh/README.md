# Lanczos PhysCal refresh (#630)

This is the Julia/Rust-only refresh of the three CPU Lanczos workloads in
[Benchmark-2026-11-01](https://github.com/AtelierArith/mvmc-rs/wiki/Benchmark-2026-11-01#cpulanczos-physcal).
C is not rerun. Retained C Wiki values remain historical measurements.

The completed sweep is in [results/20261103/report.md](results/20261103/report.md):
24 warmups and 72 measured runs, with raw inputs, all outputs, timings and hashes.
The [historical Wiki snapshot](historical-wiki.md) preserves the previous table.

## Work and timing scope

- Real, six-site Hubbard chain, fixed parameters, seed 1.
- Ordinary Lanczos mode 1 and spin-conserving InterAll modes 1 and 2.
- `NVMCSample=20000`, `NDataQtySmp=20` for every invocation.
- Computational thread counts 1, 4, 8, 16; BLAS/OMP/MKL/BLIS one thread.
- One warmup invocation followed by three measured invocations per condition.
- Rust: release CLI wall time from process launch through exit, including output.
- Julia: public `run_phys_cal_from_namelist` API in one process, including input
  parsing and output; first invocation/JIT and explicit pre-invocation GC excluded.
- All workloads run sequentially. As in the original Lanczos procedure, no CPU
  affinity or frequency isolation. Timings have different scopes across languages.
- Rust explicitly selects C-order SR/measurement Pfaffian paths. Julia enables
  its existing inner/PfaPack threading flags. A configured thread count does not
  establish that every kernel uses that many workers.

Julia source is upstream `611031147f1bad9704c55084f6817feeba481b13`
(`Speed up Lanczos intermediate overlaps`, tmisawa/Julia-mVMC#90).
`Manifest-v1.13.toml` is unchanged from the preceding `22bc829` pin.
The Rust runtime source is main `835cbad931e9303a465ee6845e96755dfe5031a1`;
the driver commit also includes the Julia gitlink update, with no Rust runtime change.
Historical measurements have not been regenerated or relabelled.

## Reproduce

Use Linux x86_64 with the normal Rust BLAS/LAPACK dependencies, Juliaup Julia
1.13.1 and `uv`. On another host record the CPU, actual BLAS and compiler versions.
Use a new output directory; the driver refuses to overwrite an existing one.

```bash
git submodule update --init --recursive extern/Julia-mVMC
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia +1.13.1 --project=extern/Julia-mVMC --startup-file=no \
  -e 'using Pkg; Pkg.instantiate(); Pkg.build("PfaPack"); Pkg.build("SFMT")'
CARGO_TARGET_DIR=/tmp/mvmc-lanczos-refresh-target \
  cargo build --locked --release -p mvmc-cli
uv run --no-project python benchmark/lanczos_refresh/measure.py \
  --rust /tmp/mvmc-lanczos-refresh-target/release/mvmc \
  --output /tmp/mvmc-lanczos-refresh-results
uv run --no-project python benchmark/lanczos_refresh/verify.py \
  /tmp/mvmc-lanczos-refresh-results
```

The driver records input and fixed-parameter SHA-256 hashes, source identifiers,
binary hash, `rustc -Vv`, Julia version, `lscpu`, Rust dynamic libraries, host load,
individual times including warmups, medians and all output sets. Each Julia log
identifies its actual BLAS backend and thread count. No C program is invoked.

`verify.py` checks every output set for finite values and shapes, compares all
repetitions/thread counts against that implementation's first 1-thread measured
run, records fixed-thread repeat differences separately, and reports Julia/Rust
differences separately. Optional `--historical PATH`
compares against the original saved Julia/Rust outputs. These comparisons do not
assert cross-implementation RNG trajectory equality. Historical differences are
reported, not excused as Monte Carlo noise or hidden with a larger tolerance.

## Upstream validation

Julia 1.13.1, ILP64 OpenBLAS with one thread:

- `MVMCOptimizers.jl/test_unit/test_unit_lanczos_pfaffians.jl`: 328 passed.
- `test/integration/lanczos_equivalent.jl`: 65 passed.

The upstream benchmark README separately documents a pre-existing Julia/C
discrepancy for the ordinary real InterAll fixtures used here. Passing upstream
integration fixtures does not establish agreement for every input in this sweep.
The full refreshed workload retains an InterAll Julia/Rust QQQQ difference of
0.4573802746838851; the updated Julia outputs agree with the historical Julia
outputs within 3.46e-13. The speed refresh does not resolve that existing discrepancy.
