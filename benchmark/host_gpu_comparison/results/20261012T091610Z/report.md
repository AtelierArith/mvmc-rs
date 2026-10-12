# L=128 host E2E comparison (Opt + PhysCal), 2026-10-12

Three end-to-end configurations, same input and budget, on the same physical
machine (NVIDIA A100 80GB PCIe + AMD EPYC 7713P, 64 host cores):

| case | runner | device |
| --- | --- | --- |
| `rust-mpi16` | `mvmc --features mpi`, `mpiexec -n 16` | CPU (MPICH 4.2.0, Dev Container) |
| `julia-gpu` | `Julia-mVMC/test/gpu/benchmark_end_to_end.jl` | A100 GPU (320 walkers) |
| `rust-gpu` | `mvmc-cuda` (device sampler + resident measurement) | A100 GPU (one chain) |

Configuration: `hubbard_chain_L128`, `NSROptItrStep=20`, `NVMCSample=320`
(`20` per rank for MPI16), `NVMCWarmUp=1`, `NVMCInterval=1`, `RndSeed=79`,
PhysCal `NVMCCalMode=1`, `NDataQtySmp=1`. Warmup 1 + 3 measured repetitions;
table values are medians.

## Results (seconds)

`wall` is the whole process/launch; `All` is the internal C timer `All [0]`
(the public runner, comparable to the Julia runner time). Julia prints only the
runner time.

| case | phase | n | wall | All |
| --- | --- | ---: | ---: | ---: |
| rust-mpi16 | Opt | 3 | 4.9866 | 4.7802 |
| rust-mpi16 | PhysCal | 3 | 0.5022 | 0.3174 |
| julia-gpu | Opt | 3 | 3.0654 | – |
| julia-gpu | PhysCal | 3 | 0.2954 | – |
| rust-gpu | Opt | 3 | 32.6681 | 30.1506 |
| rust-gpu | PhysCal | 3 | 4.0870 | 1.6132 |

### Ratios

| comparison | Opt | PhysCal |
| --- | ---: | ---: |
| Julia GPU / Rust GPU (All) | **9.84x** | **5.46x** |
| Julia GPU / Rust GPU (wall) | 10.66x | 13.84x |
| Julia GPU / Rust MPI16 (All) | 1.56x | 1.07x |
| Rust MPI16 / Rust GPU (All) | 6.31x | 5.08x |

## Reading

* **Julia's GPU E2E is about 10x faster than Rust's for optimization and about
  5x for PhysCal.** The cause is structural, not the language: the Julia GPU
  runner spreads one SR step over **320 independent XORWOW walkers**, filling
  the A100, while the Rust production device sampler drives **one Markov chain
  in lock-step** with the host (the RNG-on-host contract), so every hop pays a
  host/device round trip and the device is mostly idle. At L=128 a single Rust
  chain runs ~27k hops/s of the ~820k-hop budget, versus ~270k hops/s for the
  walker-parallel Julia run.
* **Rust MPI16 beats Rust GPU** on this 64-core host (6.3x Opt, 5.1x PhysCal):
  16 independent CPU chains at one BLAS thread each already fill the host,
  while the single-chain GPU path cannot.
* **PhysCal is the closest Rust/Julia cell**: Julia GPU 0.295 s vs Rust MPI16
  0.317 s (All), essentially tied; both are dominated by one 41k-hop budget.

## Work budget

| case | Opt attempts (all 20 steps) | PhysCal attempts |
| --- | ---: | ---: |
| julia-gpu | 860,160 | 81,920 |
| rust-gpu / rust-mpi16 | 821,760 | 41,088 |

The budgets match within ~5%; equal retained work does not imply equal
trajectories (Julia uses independent walkers, Rust one chain / 16 group chains).

## Caveats

* The Rust GPU path here is the **production** device sampler, which is
  single-chain by design; the Rust walker-batched device sampler exists only as
  a sampling microbenchmark and is not wired into an E2E runner.
* Julia's `samples=320` means 320 walkers; Rust's `NVMCSample=320` is 320
  samples of one chain (20 per rank for MPI16).
* Host load was low; medians of 3, min/max within a few percent.
* `rust-gpu` `wall` includes ~2.5 s of per-process CUDA/tenferro initialization;
  the `All` column excludes it.

## Provenance

`environment-host.txt`, `environment-container.txt`, the per-case
`*/environment.txt`, `*/benchmark.log` and `*/*.csv` in this directory.
Reproduce with `bash benchmark/host_gpu_comparison/run_all.sh`.
