# Device-resident SR benchmark (issue #447)

> **Historical record (#643).** The legacy C-order GPU `StageBackend` route measured here
> (`MVMC_RS_*_BACKEND=cuda`, `mvmc-cuda` as a CPU-runner wrapper, and the examples and
> harness scripts named below) was removed in #643. The recorded results are kept unchanged as
> historical evidence; the commands below no longer run. Current GPU work uses the
> device-resident Opt/PhysCal runners, measured with `scripts/benchmark_gpu.sh`.

```sh
scripts/run_sr_device_bench.sh docker
```

Program: `gpu/mvmc-gpu-cuda/examples/bench_sr_device.rs`. Method, validation bounds and opt-in guidance: `docs/design/gpu-readiness.md` section 14.
