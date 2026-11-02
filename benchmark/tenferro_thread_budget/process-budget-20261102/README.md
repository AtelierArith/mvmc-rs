# Process-local Engine worker budget (#583)

Source: `c26f88ad` (same release binary in every row). This measures an opt-in Engine worker reservation cap, not a global CPU allocation policy or a before/after-main optimization. One shared Engine is used in this workload: budget 3 allows its requested four participants; budget 0 makes it serial. The independent-factory accounting is tested separately.

The L32 inputs, seed, work and multi-walker harness are those in [the earlier walker report](../walkers-20261102/README.md). Opt uses 20 steps; PhysCal uses 20 windows with fixed parameters. Each condition has one excluded warmup and three measured repetitions, alternating budget order. Processes use four physical cores (CPUs 0–3), one inner worker, Rayon capacity 4, requested tenferro participants 4, held lifetime, default spin and BLAS/OMP 1. PhysCal uses C-order measurement and performs no SR, serving as a control. Timing includes process initialization, calculation, output and harness state printing; excludes Docker startup. No additional local build ran during measurements. The host was shared, without frequency control; three short samples cannot establish a universal gain.

| Workload | Walkers | Worker budget | Wall median | User CPU median | System CPU median | Wall range |
|---|---:|---|---:|---:|---:|---|
| opt | 1 | unset | 2.389 s | 2.838 s | 0.048 s | 2.365–2.404 s |
| opt | 1 | 0 | 2.406 s | 2.364 s | 0.042 s | 2.338–2.415 s |
| opt | 1 | 3 | 2.347 s | 2.803 s | 0.045 s | 2.343–2.477 s |
| opt | 4 | unset | 3.221 s | 11.604 s | 0.165 s | 3.206–3.260 s |
| opt | 4 | 0 | 3.020 s | 10.789 s | 0.160 s | 2.982–3.047 s |
| opt | 4 | 3 | 3.222 s | 11.597 s | 0.161 s | 3.197–3.236 s |
| phys | 1 | unset | 2.182 s | 2.165 s | 0.017 s | 2.132–2.271 s |
| phys | 1 | 0 | 2.178 s | 2.164 s | 0.014 s | 2.151–2.206 s |
| phys | 1 | 3 | 2.124 s | 2.111 s | 0.014 s | 2.123–2.186 s |
| phys | 4 | unset | 2.586 s | 9.862 s | 0.059 s | 2.585–2.589 s |
| phys | 4 | 0 | 2.600 s | 9.897 s | 0.060 s | 2.592–2.627 s |
| phys | 4 | 3 | 2.597 s | 9.865 s | 0.064 s | 2.585–2.608 s |

All full discrete per-walker RNG state/draw-count/configuration records matched across budgets and repetitions within each workload/walker count. PhysCal decision booleans and exact integer draws also matched. Finite printed root Opt energies and per-walker PhysCal energy components had maximum absolute difference 0. Nonroot Opt NaNs are the runner's intentional unreported-energy sentinel and were checked separately. This does not compare unprinted floating-point intermediates or Green functions.

Budget 3 reproduces the shared Engine's existing participant count; its timings do not establish a speedup. Budget 0 had lower four-walker Opt wall/user medians in this sequence, but PhysCal controls varied and this does not justify changing defaults. The cap addresses reservations across independently created mvmc Engines; caller/walker threads, Rayon, BLAS, external Engines and other ranks remain outside it.

## Reproduce

Use the existing development-container image and build `cargo build -p mvmc-core --example bench_walkers_583 --release --locked` at the source above in its own target volume. Prepare the exact Opt/PhysCal input copies as in the linked walker report; hashes are in `raw-compact.tar.gz`. Then run from the repository root:

```sh
benchmark/tenferro_thread_budget/process-budget-20261102/run.sh DEVELOPMENT_IMAGE TARGET_VOLUME INPUT_DIRECTORY FRESH_OUTPUT_DIRECTORY
uv run --no-project python benchmark/tenferro_thread_budget/process-budget-20261102/analyze.py FRESH_OUTPUT_DIRECTORY > summary.json
```

`provenance.txt` records the executable SHA-256, compiler and OpenBLAS version. The image is `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Linux x86_64/Xeon E5-2699 v3. `summary.json` retains every measured timing. `raw-compact.tar.gz` contains compacted harness output, stderr, timings, settings and input hashes, including warmups. Complete discrete records are retained for one reference run per workload/walker count; each run retains their per-walker SHA-256 and its original stdout SHA-256, plus printed numeric records. Full records were compared before compaction. Extracting this archive and rerunning the analyzer reproduces `summary.json` exactly. Full original logs are retained locally.

Separate non-timing checks passed: full workspace test-fast 1633 tests; reservation/real-Engine tests 5; multiwalker C/RNG/collective checks 16 at each budget 0/3; scoped lifetime/shared fallback checks 2. MPICH worlds 2/4 with budgets 0/3, tenferro SR and requested Engine participants 4 passed the existing `run::mpi_runtime_tests::negative_clock_and_asymmetric_runner_failures` test on every rank (`mpi-gates.tar.gz`). These gates test existing failure/recovery behavior, not MPI resource placement or performance. CUDA all-target check, doctests, Clippy and formatting passed; no new CUDA performance run was made.
