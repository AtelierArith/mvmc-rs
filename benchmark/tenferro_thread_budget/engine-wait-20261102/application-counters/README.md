# Application Engine wait-event diagnostic (#583)

This uses an isolated diagnostic copy of tenferro-cpu 0.8.1, adding only read access to the existing threadteam Pool counters. Production manifests, provider source and locks are not changed. `mvmc-583-counters-provider.patch` exposes counters through ExecutionResources/CpuEngine; `mvmc-583-counters-application.patch` records the temporary application helper, final harness print and diagnostic lock change. Original/patched provider file SHA-256 values are recorded in `mvmc-583-counters-provider-hashes.json`. Base application source: `cca22f94`. The binary hash is in `binary-sha256.txt`.

The same fixed L32 inputs and harness as [the process-budget report](../../process-budget-20261102/README.md) were used: Opt 20 steps or fixed-parameter PhysCal 20 windows, seed 1, 1/4 walkers, four requested Engine participants, one inner worker, Rayon capacity 4, BLAS/OMP 1, held lifetime, shared dispatch off and process budget unset. CPUs 0–3; Linux x86_64/Xeon E5-2699 v3, Rust 1.99.0, pthread OpenBLAS and dev-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`. Each condition ran three diagnostic processes, in the fixed order recorded by `run.sh`. Host load and frequency were not controlled or sampled. No timing comparison is made from these diagnostic processes.

The final snapshot counts events since the shared Engine's Pool was constructed, including initialization. It is not an active-phase-only delta. Join sleeps mean a leader slept during a section join; barrier sleeps mean participants slept in a barrier; member parks mean reserved members parked awaiting work. These are event counts, not durations, actual wake latency, spin CPU attribution or participant utilization. Individual atomic counters can change near snapshot time; they are not an atomic multi-counter snapshot. A missing published shared Engine is recorded as `None`, not zero events.

| Workload | Walkers | Spin | Join sleeps, 3 observations | Barrier sleeps | Member parks |
|---|---:|---|---|---|---|
| Opt | 1 | default | 1, 0, 1 | 0, 0, 2 | 64, 64, 66 |
| Opt | 1 | 0 | 37, 32, 33 | 0, 0, 0 | 103, 105, 104 |
| Opt | 4 | default | 1, 2, 4 | 0, 0, 0 | 154, 147, 156 |
| Opt | 4 | 0 | 105, 107, 118 | 2, 0, 0 | 391, 397, 393 |
| PhysCal | 1/4 | both | No published shared Engine | unavailable | unavailable |

Zero spin produced more join sleeps and parks in these Opt observations. This directly observes the Engine-owned Pool, complementing the earlier direct-Pool synthetic counters, but does not isolate a performance cause or justify changing the wait default. The C-order PhysCal control has no SR solve; its absent shared-Engine counter must not be used as a tenferro measurement-backend result.

Every complete discrete per-walker RNG state/draw-count/configuration record matched the ordinary release binary's unset-budget reference; PhysCal exact draw/decision records also matched. Finite printed root Opt energy and per-walker PhysCal components had maximum absolute difference 0; nonroot Opt NaN sentinels were checked separately. No unprinted intermediate or Green-function comparison is claimed. Full records were compared before compacting; the archived SHA-256 records reproduce that comparison against the ordinary reference archive.

## Reproduce

Use a separate checkout at `cca22f94`, target directory and copy of the registry's tenferro-cpu 0.8.1 source. Verify the recorded upstream file hashes, apply the provider patch to that copy and the application patch to the separate checkout. For the exact local layout:

```sh
# UPSTREAM_CPU_DIR is the unmodified tenferro-cpu 0.8.1 source directory.
cp -a "$UPSTREAM_CPU_DIR" /tmp/mvmc-tenferro-cpu-counters
chmod -R u+w /tmp/mvmc-tenferro-cpu-counters
patch -d /tmp/mvmc-tenferro-cpu-counters -p1 < mvmc-583-counters-provider.patch
git apply mvmc-583-counters-application.patch
CARGO_TARGET_DIR=/target/counters-diagnostic cargo build -p mvmc-core \
  --example bench_walkers_583 --release \
  --config 'patch.crates-io.tenferro-cpu.path="/tmp/mvmc-tenferro-cpu-counters"'
```

Run the build in the named development container with the copied provider mounted at that exact path. `run.sh` is the exact executed Docker script; adapt its fixed input/output/target paths for another host, retain a fresh output directory and the recorded input hashes. `analyze.py DIAGNOSTIC_OUTPUT ORDINARY_OUTPUT` checks the records and emits the summary. Ordinary output can be extracted from `process-budget-20261102/raw-compact.tar.gz`; diagnostic output from this directory's `raw-compact.tar.gz`. Running the analyzer on those two archives reproduces `mvmc-583-counters-summary.json` exactly. The original full output remains local. The diagnostic build passed type-checking, and all 24 processes and discrete/numeric checks completed successfully. This is optional developer instrumentation, not a new Cargo test dependency or a production provider patch.
