# Device-resident sampler benchmark (issue #434)

> **Historical record (#643).** The legacy C-order GPU `StageBackend` route measured here
> (`MVMC_RS_*_BACKEND=cuda`, `mvmc-cuda` as a CPU-runner wrapper, and the examples and
> harness scripts named below) was removed in #643. The recorded results are kept unchanged as
> historical evidence; the commands below no longer run. Current GPU work uses the
> device-resident Opt/PhysCal runners, measured with `scripts/benchmark_gpu.sh`.

```sh
scripts/run_device_sampler_bench.sh docker                      # all host cores
MVMC_RS_SAMPLER_CORES=0-3 scripts/run_device_sampler_bench.sh docker   # a 4-core host
```

The program is `gpu/mvmc-gpu-cuda/examples/bench_device_sampler.rs`; design, validation and the
analysis are in `docs/design/gpu-readiness.md` section 12, tables and metadata in
`results/device_sampler.md`. Run on a quiet host: the CPU rows and the host side of the device
rows are sensitive to other load.
