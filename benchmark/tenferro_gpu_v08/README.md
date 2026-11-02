# tenferro v0.8: RTX 3060 comparison

This report covers the GPU Opt migration in #570 and an API microbenchmark for #571. It is not a PhysCal comparison. The L=128 case uses the fixed definitions from the existing `L128-opt-faster-20261013` report: 320 Julia-compatible GPU walkers, 20 SR steps, seed 79, 128 warmup proposals and interval 128. Each worker process is started once, each version receives one untimed warmup run, then five timed runs alternate order. Each timed `run_gpu_para_opt_from_namelist` call includes input loading, allocation and initialization, warmup, sampling, optimization and final downloads; executable startup and the first compilation are excluded. These are end-to-end call times, not kernel timings. The scripts do not measure a per-phase timing breakdown, process CPU time, or peak resident device memory; those remain unverified.

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, GeForce RTX 3060 12 GiB, UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`, NVIDIA driver 580.178.04, Rust 1.99.0 (b940084d7); single-thread OpenBLAS, OMP and Rayon. Before binary: main `5f5a490e3f4a4b77149b147f684ed2bbae252d77`, tenferro 0.7.1; after: `e4240834b0b1ef0271e197931da0b77221662fb6`, tenferro 0.8. The exact binaries and all input hashes are recorded in [results/rtx3060-summary.json](results/rtx3060-summary.json). The runs were sequential, with no affinity or frequency pinning; two worker processes remain alive across the experiment. Five repetitions describe this host and case, and do not establish a general speedup.

| Version | Five timed calls (seconds) | Median |
|---|---|---:|
| 0.7.1 | 4.838939, 4.849528, 4.843106, 4.849659, 4.859443 | 4.849528 s |
| 0.8 | 4.829981, 4.823095, 4.864835, 4.855331, 4.874908 | 4.855331 s |

The archived same-GPU 0.7.1 result was 4.851715 s, from a different source revision (`051f8983…`) and measurement session; it is context, not the matched before value above. Across all five timed calls, the maximum absolute difference was zero for both recorded energy sequences and final parameters, within and between implementations. This does not by itself compare every intermediate walker state. Explicit raw-exec counters for each complete call were 27 uploads / 27,621,476 bytes, 4 downloads / 6,344 bytes, 11 synchronizations and 7 scopes for both versions. These counters only cover instrumented raw-exec operations, include initialization and output, and are not a measurement of all CUDA traffic. Raw worker logs are [before](results/rtx3060-before.log) and [after](results/rtx3060-after.log).

## Reproduction

Build the example separately in clean checkouts at the two source revisions. Keep their `target` directories separate. Set `CUDA_VISIBLE_DEVICES` and `LD_LIBRARY_PATH` for a working CUDA 12 installation with cuTENSOR; use a GPU with no other workload. A100 results have not been obtained here.

```sh
# Run in each checkout, retaining each resulting binary.
cargo build --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --example resident_opt_worker

# From the 0.8 checkout, reproduce the archived definition set.
mkdir -p /tmp/tenferro-v08-l128
cp benchmark/hubbard_chain/inputs/hubbard_chain_L128/*.def /tmp/tenferro-v08-l128/
cp benchmark/host_gpu_comparison/results/L128-opt-20261012/{modpara,namelist}.def /tmp/tenferro-v08-l128/
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  uv run --no-project python benchmark/tenferro_gpu_v08/compare.py \
  /path/to/old/resident_opt_worker /path/to/new/resident_opt_worker \
  /tmp/tenferro-v08-l128/namelist.def /tmp/tenferro-v08-result
```

The script prints a JSON summary and saves the raw worker output, all energies and final parameters. Compare input and binary hashes with the recorded summary before interpreting differences. The recorded host's cuTENSOR and CUDA library paths appear in the summary; they are installation-specific.

## Shared Pfaffian API

On the same RTX 3060 and CPU, release build at `e4240834…` plus the benchmark example, deterministic f64 input with `n=32`, `NQP=8`, `B=8`, one warmup and seven calls (median):

| Operation and timing scope | Median |
|---|---:|
| Existing CPU PfaPack, host input/output | 1.722426 ms |
| CPU extension, host input/output, prepared graph | 1.787294 ms |
| Existing CUDA API, host input/output, transfers included | 0.754287 ms |
| Existing CUDA synchronized kernel phase | 0.390659 ms |
| CUDA extension, device-resident input/output, prepared graph | 0.419736 ms |

The CUDA rows have different scopes: the legacy kernel phase excludes transfers and allocation, whereas the extension call includes runtime execution and output allocation but excludes explicit host upload/download. They cannot be used as an end-to-end speedup ratio. Raw-exec counters reported zero explicit transfer bytes for the extension calls; this is not an independent measurement of all driver traffic. Correctness and failure/status cases are covered separately by the CPU/CUDA gates. Full process time and output: [results/rtx3060-pfaffian.log](results/rtx3060-pfaffian.log).

```sh
cargo build --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --example bench_shared_pfaffian
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  /usr/bin/time -v gpu/mvmc-gpu-cuda/target/release/examples/bench_shared_pfaffian
```
