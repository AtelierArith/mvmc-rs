# C / Julia / Rust MPI rank and thread sweep

Periodic half-filled Hubbard chain, t=1 and U=4, L=32/64. All configurations use 320 total samples, 300 Opt steps and 100 PhysCal groups, one warmup and three measured repetitions. The 1×1 baseline and all five ranks×threads=16 configurations are included. BLAS uses one thread. The sample count changed from earlier 300-sample reports so that it divides all rank counts; those older tables are separate experiments.

`report.md` contains medians in seconds in the requested five-column format. `measurements.json` contains all 216 timed records. Each configuration retains its full-precision timings, environment, source/binary hashes, validation receipt and runtime logs. Full generated inputs, outputs and reconstructed C sources are under `bench-out/rank-thread-sweep-20261010/` and are not committed. The validator checks finite output, shapes and unchanged source/binary hashes; these checks do not establish cross-language numerical parity.

Linux x86_64, AMD Ryzen 9 PRO 8945HS (8 physical cores / 16 logical CPUs), Julia 1.13.1 and the reference Manifest-v1.13.toml. Julia reference: `02afdae0a19732c727e0d09132d9761c57d68fcb`; authoritative C: `d73d06bd529d3b2573f38eb5817c4a5f52971006`. Rust production kernels are unchanged from PR #507. The first cell recorded Rust checkout `4ee11e26fd1c6c50be4f2f069f206ae21c01d187`; later cells recorded its merge `afde6bffe7f60b2ab30d3d0a496961c52c0cad63`. Their tracked trees are identical. Actual versions and BLAS libraries are recorded per cell.

C reports the internal rank-zero All timer; Julia/Rust report warmed maximum-rank production API time. Startup, JIT, build and warmups are excluded; timer boundaries differ. PhysCal shares C-generated optimized parameters between languages within each cell, but different rank configurations generate different parameters and RNG trajectories. This is an end-to-end workload comparison rather than fixed-trajectory scaling.

Reproduce inside a Linux Dev Container with at least 1 GiB `/dev/shm`:

```bash
for pair in '1 1' '1 16' '2 8' '4 4' '8 2' '16 1'; do
  read -r ranks threads <<< "$pair"
  bash bench/run.sh --sites 32 64 --ranks "$ranks" --threads "$threads" \
    --steps 300 --groups 100 --samples 320 --warmups 1 --reps 3 \
    --output "bench-out/new-rank-thread-sweep/r${ranks}-t${threads}"
done
```

Use fresh output directories. Copy `summarize.py` to the output root and run it with `uv run --no-project python` to regenerate the aggregate table. Validate each original cell with `uv run --no-project python validate.py /absolute/path/to/cell`; the original sources and binaries named by sha256.json must still exist.

`run.sh` preserves the original executed host script. It completed five cells, then the 16×1 launch exhausted the original 64 MiB shared memory during MPI_Init. `shared-memory.md` records the isolated retry with the same image, volumes, MPI transport and binaries and 1 GiB shared memory. The failed attempt has no valid timings and is excluded. Dev Container configuration PR #511 is independent of this results report.

Related to #496 and #497. No numerical kernels, SIMD libraries or Julia compiler options changed for this sweep.
