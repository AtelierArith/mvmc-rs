# One-command entrypoint verification (#634 / PR #635)

Clean source `67b0a2c8d418876d8a40e4d35a9cb341567590e9` completed both entrypoints
end to end. The later evidence/preflight commit changes only dry-run wording,
GPU metadata's thread count and the lock's fixed per-user location; it does not
change benchmark execution, inputs, numerical code or timing boundaries.

```bash
bash scripts/benchmark_cpu.sh --smoke --output /tmp/mvmc-634-cpu-final
bash scripts/benchmark_gpu.sh --smoke --output /tmp/mvmc-634-gpu-smoke
```

CPU: C/Julia/Rust at 1 and 4 physical cores, Opt and PhysCal, one warmup and one
measured run: **24 complete timing rows**. GPU: Julia/Rust Opt and PhysCal,
one warmup and one measured run: **8 complete timing rows**. Both reports and
raw groups are retained. Smoke means two SR steps and 16 samples/walkers; these
single measured observations are **not full-workload performance results** and
must not replace the independently measured #629 performance acceptance tables.
The scripts' default full run instead selects 20 steps / 320 samples, three
measured runs, and CPU 1/4/8/16 threads.

Machine: Linux x86_64, Xeon E5-2699 v3, Rust 1.99.0, Julia 1.13.1, reference
`611031147f1bad9704c55084f6817feeba481b13`. CPU C source
`d73d06bd529d3b2573f38eb5817c4a5f52971006`, freshly built and then hash-verified
from the cache on the final CPU rerun. GPU auto-selection from inherited
`CUDA_VISIBLE_DEVICES=0,1` chose physical GPU 0, RTX 3060, UUID
`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`, driver 580.178.04. Native Rust reports
NVRTC 12.8 / cuBLAS 12.8.4 / cuSOLVER 11.7.3; Julia retains its own pinned CUDA
stack. Full binary/input hashes, library paths, manifests and versions are saved.

Warmup/JIT and Julia explicit GC are excluded from reported medians. CPU C/Rust
process wall and Julia warm API have different boundaries; GPU APIs exclude
serialization. GPU PhysCal Rust SFMT versus Julia XORWOW has different sampling
work/trajectories. The runner validates finite complete timing groups and runtime
success/output checks; this smoke is not a new cross-language numerical/RNG
parity proof. Existing #629 numerical/RNG evidence remains separately recorded.

Bounded checks: six orchestration regressions, both shell syntax checks, root
and standalone CUDA formatting. CI adds orchestration checks without needing
C/Julia/Docker/GPU. PR checks are the authoritative final-head Linux/macOS
serial/MPI/SIMD, lint, docs and build-policy results.

An initial GPU preflight rejected inherited `CUDA_VISIBLE_DEVICES=0,1`; commit
67b0a2c8 fixed selection inside the visible set and added a regression. The
retained failure log documents that development issue. A real concurrent CPU
entrypoint was rejected while GPU preparation held the shared lock, with no
output directory created; that diagnostic is also retained. Other preliminary
cohorts are not substituted for the two clean-source cohorts above.

`archives.json` identifies archives by SHA-256 and each `original-files.json`
ledger records every original file's size/hash/mtime. No executable, Cargo cache
or target tree is included. Verify archives and individual files with:

```bash
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py \
  benchmark/entrypoints/results/20261103-smoke
uv run --no-project python scripts/test_benchmark_entrypoints.py
```

See [entrypoint documentation](../../../../scripts/benchmark/README.md) for
prerequisites and all command options. Generated tables are `cpu-report.md` and
`gpu-report.md`; actual timing ranges and raw repetitions are kept in the
respective JSON files/archives. CPU C/Rust alternating order is retained in
`cpu/c-rust/timings.json`, not flattened away by the combined report.
