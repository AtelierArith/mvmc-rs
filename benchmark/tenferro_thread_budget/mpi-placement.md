# Shared versus disjoint MPI rank affinity (#583)

This is a same-binary Rust CPU comparison of two placements, not a C/Julia or GPU comparison. Four MPICH ranks run on CPUs 0–15: `shared` leaves every rank's calling thread allowed on all 16 CPUs; `pinned` binds rank `r` and inherited workers to CPUs `4r..4r+3`. The separate [affinity diagnostic](affinity-audit.md) verified the calling-thread masks and pairwise overlap before the measurement: all pairs shared 16 CPUs in the first placement and 0 in the second. Its [raw output](results/placement/affinity.txt) does not observe any worker that later changes its own affinity or CPU quotas.

Each rank uses `MVMC_RS_INNER_THREADS=4`, `RAYON_NUM_THREADS=4`, `MVMC_RS_TENFERRO_THREADS=4`, the upstream default wait policy, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and tenferro SR. The inner and tenferro pools have separate workers; requesting 4 of each is not a global 4-thread cap. Both modes use the repository Hubbard L=32 definitions, fixed seed 1, 300 samples, warmup 10, interval 1: Opt has 20 SR steps with output window 1; PhysCal loads the fixed checkpoint and records 20 windows. [Input hashes](results/placement/input-sha256.txt) and [raw logs](results/placement/raw-logs.txt) are preserved.

The same binary and inputs were used for each placement. For each workload, one warmup per placement was followed by three measured runs, with placement order reversed on alternating repetitions. Bash `time` covers `timeout`, the MPI launcher, initialization, computation and output; Docker startup is excluded. The 16 selected logical CPUs map to 16 distinct cores on socket 0 ([topology](results/placement/cpu-topology.txt)). No frequency control was applied. The host was shared and the samples are short, so small differences and the sign of a change should not be generalized.

| Workload | Placement | Wall samples (seconds) | Median (seconds) |
|---|---|---|---:|
| Opt | shared | 2.786, 2.804, 2.753 | 2.786 |
| Opt | pinned | 2.692, 2.742, 2.758 | 2.742 |
| PhysCal | shared | 3.153, 3.000, 3.239 | 3.153 |
| PhysCal | pinned | 2.956, 3.072, 2.992 | 2.992 |

The output comparison found identical file names and finite formatted numerical values across the six measured runs per workload: 4 Opt files and 80 PhysCal files, excluding timing files, had maximum absolute difference 0. This is not a comparison of complete RNG states, individual acceptance decisions or intermediate floating-point values. The logs contain Bash's aggregate child `user` and `sys` CPU times as well as wall time; these are not per-rank utilization or attribution to a particular pool. Three repetitions do not justify changing the default policy or attributing a difference to a specific cause. [Full numeric summary](results/placement/summary.json).

Linux x86_64 6.8.0-147, Xeon E5-2699 v3, development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, MPICH 4.2.0, Rust 1.99.0, pthread OpenBLAS, repository revision `f5324073b756b2491182fa605ea851ac25e30ee2`, release binary SHA-256 `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62`. Run-time dependencies and lockfiles are those of that revision.

## Reproduce

Use the Linux development container with the matching MPI launcher and a dedicated target volume. Build `/target/release/mvmc` with `CARGO_TARGET_DIR=/target cargo build -p mvmc-cli --features mpi --release --locked`. Construct fixed input copies from the repository root and then invoke the host-side Docker script:

```sh
mkdir -p /tmp/placement-input/{opt,phys}
cp benchmark/hubbard_chain/inputs/hubbard_chain_L32/*.def /tmp/placement-input/opt/
cp benchmark/physcal/inputs/hubbard_chain_L32/*.def /tmp/placement-input/phys/
cp benchmark/physcal/inputs/hubbard_chain_L32/zqp_opt.dat /tmp/placement-input/phys/
sed -i -E 's/^NDataQtySmp[[:space:]].*/NDataQtySmp    20/' /tmp/placement-input/phys/modpara.def
benchmark/tenferro_thread_budget/mpi-placement.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/placement-input /tmp/placement-results
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-placement.py \
  /tmp/placement-results > /tmp/placement-summary.json
```

Use fresh output directories. The script requires 16 consecutive allowed CPUs (default 0–15); set `MVMC_BENCH_CPU_BASE` to change the first CPU. For reproducibility, confirm the CPU topology and use the [affinity diagnostic](affinity-audit.md) with the same MPI launcher and CPU set; MPI launchers and placement policies can differ. The MPI wrapper uses MPICH's `PMI_RANK` and `/opt/mpich/bin/mpirun`.

## Engine thread budget with disjoint rank affinity

A separate alternating run used the same binary, inputs, machine, disjoint four-CPU rank masks and `MVMC_RS_INNER_THREADS=4`, comparing `MVMC_RS_TENFERRO_THREADS=1` and `4` per rank. Each mode/budget received one warmup and three measured runs. This is a separate sequence from the table above, so compare thread budgets within this table rather than treating the two sets of samples as paired. All other settings and timing boundaries are as above.

| Workload | tenferro threads/rank | Wall samples (seconds) | Median (seconds) |
|---|---:|---|---:|
| Opt | 1 | 2.552, 2.588, 2.638 | 2.588 |
| Opt | 4 | 2.717, 2.725, 2.750 | 2.725 |
| PhysCal | 1 | 3.134, 3.226, 3.097 | 3.134 |
| PhysCal | 4 | 2.999, 2.771, 3.042 | 2.999 |

The median direction differs between workloads and PhysCal samples vary considerably; this short experiment does not establish an optimal general budget or a measured cause. Fixed-parameter PhysCal does not run an SR optimization solve, so its timings must not be interpreted as SR-solver scaling. The CLI preflights the selected SR backend before determining the mode, and opening the tenferro backend may reserve a CPU team; its initialization and waiting can therefore be present in PhysCal even when no SR solve occurs. These measurements do not isolate or quantify that effect. Across the six measured runs per workload, four Opt and 80 PhysCal files had maximum absolute difference 0 in finite formatted numerical output, excluding timing files; full RNG states and decisions were not compared. [Summary](results/pinned-budget/summary.json), [raw logs](results/pinned-budget/raw-logs.txt), [input hashes](results/pinned-budget/input-sha256.txt).

With the same prepared input directory and dedicated release target volume as above:

```sh
benchmark/tenferro_thread_budget/mpi-pinned-budget.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/placement-input /tmp/pinned-budget-results
uv run --no-project python benchmark/tenferro_wait_policy/analyze-mpi-l32.py \
  /tmp/pinned-budget-results > /tmp/pinned-budget-summary-full.json
```

The analyzer additionally emits per-file SHA-256 hashes; the committed summary retains the timing and numerical-comparison fields.

## PhysCal SR-backend selection

In a third separate sequence, fixed-parameter PhysCal ran with the same binary, inputs and disjoint four-CPU rank masks, comparing `MVMC_RS_SR_BACKEND=c-order` with `tenferro` and `MVMC_RS_TENFERRO_THREADS=4`. Both retained four inner workers/rank and the other settings above. The choices were alternated over one warmup and three measured runs each; timing boundaries are unchanged.

| SR backend selected | Wall samples (seconds) | Median (seconds) |
|---|---|---:|
| C-order | 3.198, 3.193, 2.975 | 3.193 |
| tenferro, 4 threads/rank | 2.919, 2.971, 3.097 | 2.971 |

The 80 finite formatted numeric output files, excluding timing files, had maximum absolute difference 0 across all measured runs. This comparison does not inspect full RNG states or prove which operations were executed; it also does not isolate initialization or idle waiting. The tenferro median was lower in this short sequence, so it provides no evidence that selecting tenferro slows this PhysCal case. The spread and only three runs preclude a general performance or causal conclusion. [Summary](results/physcal-backend/summary.json), [raw logs](results/physcal-backend/raw-logs.txt), [input hashes](results/physcal-backend/input-sha256.txt).

With the same prepared inputs and release binary:

```sh
benchmark/tenferro_thread_budget/mpi-physcal-backend.sh \
  DEVELOPMENT_IMAGE TARGET_VOLUME /tmp/placement-input /tmp/physcal-backend-results
uv run --no-project python benchmark/tenferro_thread_budget/analyze-mpi-placement.py \
  /tmp/physcal-backend-results physcal-backend > /tmp/physcal-backend-summary.json
```
