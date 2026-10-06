# Device-resident sampler benchmark (issue #434)

```sh
scripts/run_device_sampler_bench.sh docker                      # all host cores
MVMC_RS_SAMPLER_CORES=0-3 scripts/run_device_sampler_bench.sh docker   # a 4-core host
```

The program is `gpu/mvmc-gpu-cuda/examples/bench_device_sampler.rs`; design, validation and the
analysis are in `docs/design/gpu-readiness.md` section 12, tables and metadata in
`results/device_sampler.md`. Run on a quiet host: the CPU rows and the host side of the device
rows are sensitive to other load.
