# Hubbard L128 CPU scaling (#586)

This report compares C mVMC and Rust on the shared-input 20-step optimization
and fixed-parameter PhysCal workloads. It distinguishes the explicit batched
`MVMC_RS_MEASURE_PF_BACKEND=c-order` route from the default `calc-m-all` route.
The historical optimization metadata does not record its inherited backend
selector environment, so it cannot establish that both routes were measured.
The historical PhysCal command explicitly selects `c-order`.

## Workloads and measurement scope

- Half-filled real Hubbard chain, L=128, eight QP planes, seed 79, 320 samples,
  warmup=1 and interval=1. Optimization performs 20 SR steps; PhysCal performs one
  fixed-parameter measurement window, with no Lanczos.
- Fixed parameters have SHA-256
  `9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90`.
  Both implementations consume identical definitions and PhysCal parameters.
- One warmup and three measured repetitions per workload/thread/implementation.
  C and Rust alternate order between repetitions in the paired runs.
- One persistent Linux development container excludes Docker startup on both
  sides. Bash process wall/user/system times include executable startup and exit;
  C additionally includes `mpirun -np 1`. Docker-exec wall is recorded separately.
  Inclusive C-compatible timers retain initialization [1], sampling [3], main
  calculation [4], SR [5] and nested kernels. Process wall minus internal All [0]
  is untimed startup/exit/launcher work, not a pure startup measurement. Do not
  add nested timer sections.
- Xeon E5-2699 v3, 18 physical cores/36 logical CPUs. Each case is pinned to
  CPUs 0 through N-1 for N=1,4,8,16: physical cores without SMT siblings.
  OpenBLAS/MKL/BLIS=1; C OpenMP=N, Rust inner/Rayon=N. No frequency control or
  host isolation is claimed. Host load and CPU counters are sampled throughout.
- Image ID `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`.
  Exact compiler, MPI/BLAS/library versions, binaries, inputs and affinity are
  retained with each run. The persistent container has no CPU quota; the
  recorded `cpu.max` and Docker configuration delimit that observation.

## Source and change

Before: `cf17b0be51e991209875e54b18f86fccb9d078c4`.
Intermediate: `8b861f21ed3982e4a66e4d4fd3cbfa10c9ad4c60`.
Final runtime source: `4ce6a23c378e86544ab76218ec21fa906d1bdc62`.
The intermediate binary was built from the corresponding source before the commit;
its parent/patch and binary hashes identify the exact build. No dependencies,
RNG algorithms, seeds, solver settings or per-plane arithmetic changed.
C is `d73d06bd529d3b2573f38eb5817c4a5f52971006`, built Release with GCC,
`-O3 -DNDEBUG`, MPICH and OpenBLAS. Its 612 tracked source-tree blobs were checked
against that revision, and the executable hash is retained.

The batched C-order Pfaffian implementation formerly processed planes serially.
It now uses disjoint output windows through the existing bounded inner dispatch,
while retaining `dsktf2 -> utu2pfa_real -> utu2inv_real` within each plane.
Ordering, zero-pivot zero fill and non-finite statuses are preserved.
The final change also prepares disjoint sample/QP matrix planes in parallel
and reuses factorization scratch within each dispatch block. The final binary
was built after committing the runtime source.
The default `calc-m-all` implementation already has QP parallelism and is
measured separately; this change does not switch the default selector.

## Correctness evidence

- Analytic dense 4x4 skew matrices, singular and non-finite planes in one batch,
  with exact status/layout and actual worker-entry checks at 1/4/16 workers.
- Relevant 64 tests passed, followed by the complete workspace: 1,643 passed,
  67 ignored helpers/optional gates. Clippy and formatting passed.
- Six affected CUDA measurement/SR routing regressions passed on RTX 3060.
  These are correctness tests, not GPU speed measurements.
- Developer probe `cpu_scaling_contract_586`: before serial, after 1/4/8/16,
  and a repeated after-16 run for each workload. The initial 12 cases passed;
10 final-source cases passed against the same two hash-verified baseline records. Full SFMT
  state/cursor/draw count, saved configurations, counters and every recorded
  acceptance decision/draw agree exactly. Optimization has 821,760 decisions
  and 4,110,359 consumed words; PhysCal has 41,088 and 206,174. Compared numerical
  fields (732,490 / 732,990) have maximum difference 0. The comparison uses an
  explicit abs+rel `1e-11` bound, separate from exact RNG/discrete assertions.
- Existing independent C-derived regression fixtures retain their provenance
  and numerical bounds. The new full-state probes compare Rust revisions;
  they are not a new full C-process SFMT-state oracle for L128.

## Scripts

Run Python through `uv run --no-project python`.

- `measure.py`: fresh output directory, copied/hash-checked inputs, alternating
  C/Rust repetitions, per-process timing, raw outputs and host observation.
  `--pf-backend calc-m-all --only-rust` selects the separate default-route sweep.
  `--rust-binary PATH` mounts a preserved before executable read-only.
- `summarize.py OUTPUT`: provisional per-condition medians/ranges and inclusive
  section times; incomplete groups are explicitly marked.
- `profile.py`: additional diagnostic runs after all 64 paired runs complete;
  it verifies executable hashes before using the measured binaries.
- `thread_profile.py` and `analyze_profile.py`: sampled process-tree thread
  activity. Samples miss short-lived threads and final intervals; use whole
  process-tree resource accounting for CPU totals. Instrumented times are
  excluded from the speed comparison. `perf task-clock` was denied by host
  `perf_event_paranoid=4`; no host settings were changed.
- `contract.py`: version/thread/repeat probes with exact RNG/discrete checks and
  finite numerical comparisons. This is separate from all performance runs.

## Results

Times below are process-wall medians of three measured runs, in seconds.
C controls are from the intermediate paired cohort; final Rust was measured
later without rerunning C. This is a reused control, not a simultaneous final
paired comparison. Before/final identifies the two source revisions above.
Julia was not measured again in this CPU investigation.

### opt

| CPU threads | C control | Rust before | Rust final | Before/final |
|---:|---:|---:|---:|---:|
| 1 | 93.232 s | 78.911 s | 79.786 s | 0.989× |
| 4 | 29.637 s | 57.129 s | 29.226 s | 1.955× |
| 8 | 19.491 s | 48.049 s | 21.645 s | 2.220× |
| 16 | 21.424 s | 49.168 s | 22.667 s | 2.169× |

### phys

| CPU threads | C control | Rust before | Rust final | Before/final |
|---:|---:|---:|---:|---:|
| 1 | 5.833 s | 4.275 s | 4.446 s | 0.962× |
| 4 | 1.909 s | 3.544 s | 2.930 s | 1.210× |
| 8 | 1.265 s | 3.690 s | 2.281 s | 1.618× |
| 16 | 1.278 s | 2.799 s | 1.792 s | 1.562× |

### Default calc-m-all (separate snapshots)

| CPU threads | Opt before | Opt 8b861f21 | PhysCal before | PhysCal 8b861f21 |
|---:|---:|---:|---:|---:|
| 1 | 77.615 s | 77.885 s | 4.221 s | 4.222 s |
| 4 | 28.410 s | 29.240 s | 2.936 s | 2.889 s |
| 8 | 19.610 s | 19.772 s | 2.185 s | 2.229 s |
| 16 | 22.949 s | 22.632 s | 1.347 s | 1.372 s |

At eight threads the explicit batched route improves optimization 2.22x and
PhysCal 1.62x. At one thread it regresses 1.1% and 4.0%, respectively; these are
observed cohort differences without frequency control, not a statistical
attribution to one instruction. C remains faster in several conditions,
particularly PhysCal. Sixteen threads are not uniformly faster than eight.
The default-route table is specifically before/intermediate, not a claim of a
new final-source default-route performance measurement.

## What profiling established

Sampling and SR were already parallel/small enough that they did not explain
the explicit-route optimization plateau. Before the change, the eight-thread
instrumented run spent about 32.4 seconds in the batched Pfaffian timer. After
the first change, the remaining difference between the batched measurement
section and factorization helper was about 3.78/3.89 seconds at 8/16 threads;
matrix preparation and publication surrounded that helper. The final change
parallelizes matrix preparation without reordering publication.

In final PhysCal diagnostics, Green-function evaluation remains prominent:
about 0.795/0.600 seconds at 8/16 threads in the instrumented helper. This is
not proof that allocation, scheduling or one particular synchronization causes
the residual C gap. Per-thread CPU samples, work-item counts and gap/callsite
observations are preserved in the profile archives. Whole-process CPU use,
wall ranges and inclusive phase medians for every count are in `summary.csv`.
The observer adds atomics/mutex overhead, especially to short copy operations;
profile times must not be substituted for the uninstrumented speed table.
CPU time can include spinning, and per-thread gaps are not global idle time.

## Reproduction

Use an initialized checkout and the Linux development image from
`docs/DEV_CONTAINER.md`. Set `bench_image` to that built image. The recorded
image digest above identifies this run; a local image tag is not a portable
registry image. All output paths must be fresh. The driver assumes this host's
CPU numbering (first N physical cores); verify `lscpu -e` before using elsewhere.
The following commands describe a new run; they do not run on opening this report.

```bash
bench_repo=$PWD
bench_image=YOUR_BUILT_DEV_CONTAINER_IMAGE
bench_target=mvmc-586-reproduce-target
bench_c=/tmp/mvmc-586-reproduce-c
bench_out=/tmp/mvmc-586-reproduce-final
bash benchmark/cpu_scaling_l128/build-c-reference.sh "$bench_repo" "$bench_image" "$bench_c"
docker run --rm -v "$bench_repo:/work" -v "$bench_target:/target" \
  -e CARGO_TARGET_DIR=/target -w /work "$bench_image" \
  bash -lc 'cargo build --locked --release -p mvmc-cli && cargo build --locked --release -p mvmc-core --example cpu_scaling_contract_586'
uv run --no-project python benchmark/cpu_scaling_l128/measure.py \
  --repo "$bench_repo" --image "$bench_image" --target-volume "$bench_target" \
  --c-dir "$bench_c" --opt-input "$bench_repo/benchmark/cpu_scaling_l128/inputs/opt" \
  --phys-input "$bench_repo/benchmark/cpu_scaling_l128/inputs/phys" --output "$bench_out"
uv run --no-project python benchmark/cpu_scaling_l128/summarize.py "$bench_out"
```

For the default route, use another fresh output and add
`--pf-backend calc-m-all --only-rust`. For before/after comparison, create a
separate worktree at the before revision, build into a separate target volume,
and run the same driver with that worktree/volume. Copy the committed developer
example into that old worktree only if running the contract probe; the example
is a probe, not a production change to the before executable. Do not share a
Cargo target directory between checkouts.

```bash
uv run --no-project python benchmark/cpu_scaling_l128/contract.py \
  --out /tmp/mvmc-586-reproduce-contract --inputs "$bench_repo/benchmark/cpu_scaling_l128/inputs" \
  --image "$bench_image" --before-volume YOUR_BEFORE_TARGET_VOLUME --after-volume "$bench_target"
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py \
  benchmark/cpu_scaling_l128/results/20261103
```

The C build helper is syntax-checked but was not itself used to rebuild the
recorded executable; the existing build's tracked source, compiler flags and
binary hash were verified. Exact local run/build/check commands and validation
logs are retained in `results/20261103/validation/`. This path-specific record
explains the historical environment; use the commands above for fresh paths.

## Raw evidence

`results/20261103/archives.json` identifies ten compressed cohorts by SHA-256.
Each archive retains inputs, raw outputs, environment, timings or diagnostic
records, and an `original-files.json` ledger with original hashes, sizes and
mtime nanoseconds. Identical contents are stored as hard links; extraction
mtime is not a replacement for the original ledger. Executable binaries are
excluded; their fingerprints are retained. All ten archives passed both
archive and individual-file verification. Summaries and `tables.md` are plain
text for inspection without extracting the larger records.

This milestone improves the explicit CPU route. It does not claim that all
remaining C performance differences or GPU PhysCal issue #587 are resolved.

