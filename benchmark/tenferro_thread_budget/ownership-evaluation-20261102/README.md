# Load-recorded multiwalker and MPI ownership evaluation (#583)

This is a same-binary comparison of explicit resource choices, not a new default
performance claim. [ownership-policy.md](ownership-policy.md) defines owners,
caller/worker accounting, allocation responsibilities and the retained defaults.
The bounded settings are opt-in; automatic quota discovery and an all-library
global scheduler are not implemented or claimed.

## Source and scope

Source `5b6e70c7827d322cf78e7b28aaa741aef642039a`, release with MPI enabled.
CLI SHA-256 `3f93221fd7e9fade7f31b9c8b851f2c34f02bd0a3f2d4bfce75a1f6d3f9a4506`;
walker harness SHA-256 `ffdf944846355123ff39a7bf26c6edbd3ff274228c8313666247e3a279836ff6`.
Both include #620's existing-inner-worker affinity repair. The build log is
retained. Linux x86_64/Xeon E5-2699 v3, Rust 1.99.0, MPICH 4.2.0,
pthread OpenBLAS 0.3.26, tenferro-cpu 0.8.1/threadteam 0.2.0/spindle 0.2.6.
Development image digest:
`sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`.

Fixed Hubbard L32 inputs: seed 1, 300 samples, warmup 10, interval 1,
20 Opt steps/window 1 or 20 fixed-parameter PhysCal windows. Exact inputs
are archived; walker input hashes and configuration are in its raw directory.
All cases use BLAS/OMP=1. Each condition has one excluded warmup and three
measured processes with alternating condition order. Docker startup is excluded;
program startup, initialization, calculation, output and (for MPI) launcher
startup are included. Walker timings include full discrete record printing.
Their absolute times must not be compared directly with CLI timings.

No local compilation ran during the sequence. A separate observer recorded
362 host samples over 361.311 s, with one-minute system load 0.95–9.56.
The workload itself contributes to load. Raw per-CPU accounting/pressure are
retained; these are not isolation or causal attribution. Frequency was not
controlled and the host was shared. `post-run-environment.txt` is labelled as
a later snapshot: its default container reports `cpu.max=max 100000` and
36 allowed CPUs, not a per-run or ancestor-quota measurement.

## One/four in-process walkers

CPU mask 0–3, inner=1, Rayon capacity=4, requested Engine participants=4,
held lifetime/default spin. The process additional-worker budget is unset/0/3.
With four optimizer callers on four CPUs, budget 0 is the conservative bounded
choice; budget 3 deliberately permits extra Engine workers and is a contention
control, not a recommended four-CPU allocation. PhysCal uses C-order measurement
without SR and is a no-Engine control.

Wall/user CPU medians (seconds):

| Workload | Walkers | Budget unset | Budget 0 | Budget 3 |
|---|---:|---:|---:|---:|
| Opt wall | 1 | 2.346 | 2.321 | 2.345 |
| Opt user | 1 | 2.806 | 2.287 | 2.811 |
| Opt wall | 4 | 3.202 | 3.003 | 3.252 |
| Opt user | 4 | 11.725 | 10.705 | 11.676 |
| PhysCal wall | 1 | 2.131 | 2.158 | 2.148 |
| PhysCal wall | 4 | 2.579 | 2.607 | 2.598 |

The four-walker Opt serial-Engine choice again has lower wall/user medians;
the short shared-host samples do not establish a portable speedup. Full
per-walker discrete RNG state/draw-count/configuration records match across
all budgets and repetitions within each mode/walker count. PhysCal's recorded
decision booleans and converted integer draws match too. Checked finite printed
energies have maximum absolute difference 0. Nonroot Opt NaNs are the intentional
unreported-energy sentinel, checked explicitly by the analyzer. This is not a
comparison of all unprinted numeric intermediates or Opt proposal events.

## One/four MPI ranks with bounded managed reservations

Each rank has inner=4 and an Engine requesting four participants, with additional
worker budget=3; count four inner participants (including the driver) plus three
Engine workers. Initial taskset masks are rank 0: 0–6, rank 1: 7–13,
rank 2: 18–24, rank 3: 25–31. The four-rank run additionally enables collective
`MVMC_RS_MPI_CPUS_PER_RANK=7`. The single-rank launch uses its initial mask and
leaves that multi-rank option unset. Thus this MPI4 × threads4 experiment reserves
**28 CPUs**, not 16; it explicitly counts the coexisting held Engine and Rayon
workers rather than treating idle reservations as free. MPI/runtime threads
inherit the initial per-rank mask too; their OS-thread counts are not Engine
reservation counts. Actual affinity/restoration and failure/recovery are checked
separately in [runtime-placement-20261102](../runtime-placement-20261102/README.md).

Rayon/shared dispatch off/on use the same binary, budgets, masks and input within
each rank count. Gate=40 µs, held Engine lifetime, default/zero spin. Wall medians
(seconds):

| Workload | Ranks | Rayon default spin | Shared default spin | Rayon zero spin | Shared zero spin |
|---|---:|---:|---:|---:|---:|
| Opt | 1 | 2.307 | 1.690 | 2.253 | 2.472 |
| Opt | 4 | 3.071 | 2.227 | 2.976 | 3.034 |
| PhysCal | 1 | 2.799 | 1.638 | 2.139 | 4.091 |
| PhysCal | 4 | 3.828 | 2.290 | 3.593 | 4.943 |

At four ranks/default spin, Opt Rayon/shared user CPU medians were 29.334/32.462 s,
system 4.912/0.510 s; PhysCal user 30.682/33.103 s, system 6.448/0.416 s.
Independent medians need not come from the same run. Do not infer a global
scaling ratio from the rank-count rows: MPI partitioning/RNG ownership and
startup differ. The comparison is within each rank count. Zero spin removes
the shared path's wall advantage here, especially PhysCal. Conversely the
four-walker serial-Engine setting above uses less CPU. These results support
explicit workload selection, not a universal default switch.

All finite non-timing output tokens agree within each mode/rank count across
the 12 measured processes: 4 Opt or 80 PhysCal files, maximum absolute difference
0. Four separate counter-enabled diagnostic launches (excluded from timings)
observe exactly four shared-team participant IDs on every rank, with successful
dispatches. Individual attempts/success/busy/unavailable counters are retained
in `mpi-summary.json`. Participant IDs do not prove simultaneous utilization.
Formatted output agreement is not full RNG or C-oracle parity; use the independent
walker/runner and native MPI correctness gates cited below.

## Acceptance evidence and decision

- Ownership/leading-thread/BLAS/Engine/walker budgets and collectives are mapped
  in [ownership-policy.md](ownership-policy.md) and the earlier audit. #616 binds
  reservations to Engine lifetimes; #620 supplies node-local nonoverlap and
  existing-worker placement, plus collective failure/rollback/recovery tests.
- One/multiple caller and rank comparisons, MPI4 × threads4 and relevant budgets
  are measured here and in the earlier [walker](../walkers-20261102/README.md)
  and [shared-team](../shared-team-20261102/README.md) reports. Actual Engine
  wait/lock/dispatch and park-counter observations, including default/zero spin,
  are in [engine-wait-20261102](../engine-wait-20261102/README.md). These separately
  scoped measurements do not claim a causal application spin-duration estimate.
- Exact fixed-case RNG/configuration/proposal/decision and repeatability checks,
  C numerical references, nested/unwind/fallback, reservation lifecycle and
  native MPI collective recovery gates are recorded in the shared-team,
  [process-budget](../process-budget-20261102/README.md),
  [gap](../gap-sites-20261102/README.md) and runtime-placement reports. #620's final
  Linux workspace had 1642 passing tests; final-head cross-platform CI is required.
- Hardware, settings, inputs, source/binaries, repeated samples, load and limitations
  are retained. Keep Rayon/default gates. Use shared held teams only after a
  matched measurement supports the actual workload and allocation; use serial
  Engines when caller concurrency exhausts the budget. Retain solve-scoped CG
  leases; rejected static caps/per-product recreation are not adopted policies.

This completes the explicit resource/waiting evaluation scope of #583 when its
dependent implementation/report PRs are merged and verified. It does not assert
automatic global scheduling, solve #586/#587, or remove separately documented
unsupported operations.

## Reproduce

Build the recorded source in its own development-container target volume:

```sh
cargo build -p mvmc-cli --features mpi --release --locked
cargo build -p mvmc-core --features mpi --example bench_walkers_583 --release --locked
```

`walkers.sh`, `mpi.sh`, `run.sh` and `observe.py` preserve the exact local commands.
Adapt absolute input/output paths, image, volume, masks and observer filename;
use fresh outputs and run `uv run --no-project python observe.py`. The observer
launches both stages and records child status. No compilation should overlap.

For saved-data reanalysis:

```sh
tar -xzf raw.tar.gz -C FRESH_DIRECTORY
uv run --no-project python analyze-walkers.py FRESH_DIRECTORY/walkers
uv run --no-project python analyze-mpi.py FRESH_DIRECTORY/mpi
```

Both summaries must reproduce the saved JSON. Full original output/inputs are
preserved, including all RNG state words and warmups; identical file contents
are tar hardlinks created by `archive.py`, not discarded or regenerated numeric
expectations. The observer archive preserves the successful child exit and all
host samples. Per-rank diagnostic streams have separate files to avoid interleaving.
