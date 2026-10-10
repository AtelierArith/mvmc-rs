# Production measurement path comparison

Full isolated run completed on Linux x86_64, 2026-10-10. See `report.md` for
the table, `measurements.csv` for all 36 timings, and the accompanying files
for source, environment, providers and host provenance.

```bash
bash bench/run.sh --sites 32 64 --steps 300 --groups 100 \
  --samples 300 --ranks 4 --threads 4 --warmups 1 --reps 3 \
  --julia-source /home/vscode/.cache/mvmc/issue497-qpv5-benchmark-source \
  --output bench-out/issue502-production-opt-physcal-20261010
```

Choose a new output directory for a repeat. `bash bench/run.sh` uses the same
default counts with the checked-out `extern/Julia-mVMC` source; the custom
source argument above selects the frozen Julia QPv5 experimental optimization
based on upstream 3254752e1acf7cfe6f5069522889814a8cace90d. It does not use
TriangularSolve or QPv9. Its executed source hashes are in `sha256.json`.

Rust was e1943016c968380862fd9fd46ee0ac5e8ba4b1b9 with the working changes
captured in `dirty.log`: an unmerged SR clear/aggregate parallelization
candidate, lifecycle timer instrumentation, optional PhysCal driver timers,
and this runner correction. These results describe that recorded build;
they are not timings of an unmodified main checkout. The primary runs disable
diagnostics and select `calc-m-all`, the normal production measurement path.

The prior full run selected `MVMC_RS_MEASURE_PF_BACKEND=c-order`, which uses
the separate stage interface and incurs additional batch assembly/publication
work. It remains preserved at
`bench-out/issue497-full-opt-physcal-20261010` with its actual configuration.
Its Rust medians (Opt 4.088/18.729 s; PhysCal 1.567/7.079 s) must not be
reported as production-default performance. A separate same-parameter L64
stage-versus-default check observed zero numerical delta in all 300 energy
and correlation output files; this observation is not a RNG parity proof.

All implementations use an averaging window of 300 here. Historical C
comparison inputs retained 30, while Rust and Julia workers overrode the
same input field to 300. The current all-300 comparison aligns this setting.
Each size uses one shared C-generated parameter file for PhysCal.

All 12 workload/site/implementation cells completed with one warmup and
three repetitions. No numerical tolerance was changed. The primary table
records completed workloads, timing scopes and outputs; it does not establish
deterministic cross-language trajectory parity. Rust PhysCal beats C in both
sizes in this batch, while Rust Opt L64 and Julia Opt remain slower.
