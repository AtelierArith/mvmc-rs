# Hubbard L128: Rust faster than C at 8 and 16 CPU threads (#629)

Fresh paired medians meet C/Rust >= 1.1 for both workloads and both requested
thread counts. This follows #586 / PR #625; historical measurements in the
parent README are preserved and are not substituted for the new C controls.

| Workload | CPU threads | C | Rust | C/Rust |
|---|---:|---:|---:|---:|
| Optimization, 20 SR steps | 8 | 19.565 s | 17.660 s | 1.108 |
| Optimization, 20 SR steps | 16 | 21.817 s | 16.417 s | 1.329 |
| PhysCal, fixed parameters, no Lanczos | 8 | 1.261 s | 1.026 s | 1.229 |
| PhysCal, fixed parameters, no Lanczos | 16 | 1.291 s | 0.961 s | 1.343 |

These are point medians of three measurements after one warmup. All repetitions,
including warmups, are retained. The smallest margin is Opt at eight threads;
no confidence interval or guarantee on other hosts is inferred from three runs.
No timing overlapped our correctness probes or builds. Host load is recorded;
CPU frequency was not locked and the host was not exclusively reserved.

## Input, timing and environment

Half-filled real Hubbard chain L=128, seed 79, eight QP planes, 320 samples,
NVMCWarmUp=1 and NVMCInterval=1. Opt performs 20 SR steps; PhysCal measures one
fixed-parameter window. Identical definitions and fixed parameter files are
copied for C and Rust. Fixed parameter SHA-256:
`9130219b4af78438bee2f30171833e7a8f86a01ff11e23276af439652230fd90`.

The explicit `MVMC_RS_MEASURE_PF_BACKEND=c-order` and SR `c-order` paths are
measured, matching the earlier comparison. The default selector remains
`calc-m-all`; this table is not a measurement of every backend. The CPU-only
batch default chooses two samples for eight QP planes and 8/16 workers; no
batch override was used. GPU/device and other backend defaults are preserved.

Both processes run in one persistent development container, pinned to the first
N physical cores (CPU 0..N-1), without SMT siblings. BLAS has one thread; Rust
inner/Rayon and C OpenMP use N. Bash wall time includes executable startup and
exit, and C includes `mpirun -np 1`. Docker startup/exec launch is excluded from
the table; Docker-exec wall is also saved. C/Rust ordering alternates each
repetition. Inclusive internal timers must not be added together.

Linux x86_64, Intel Xeon E5-2699 v3 (18 physical / 36 logical CPUs), Rust 1.99.0,
GCC 13.3.0, MPICH 4.2.0, OpenBLAS 0.3.26. Development image ID:
`sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`.
See `paired/container-environment.txt`, `metadata.json` and host observations
inside `paired.tar.gz` for actual linked libraries, binary hashes and CPU limits.
The local image tag in metadata is not a publicly downloadable image.

Measured Rust runtime source: `f26458230647f9cf3242a22030ada75f3ba76676`.
C source: `d73d06bd529d3b2573f38eb5817c4a5f52971006` (mVMC 1.3.0), unchanged,
Release `-O3 -DNDEBUG`; the same previously source-verified C executable is used.
The later report/verifier commit does not change measured runtime code.

## Changes

- Keep admitted host worker teams across sampler and measurement stages. Only
  the single-process reducer supplies a thread-safe view; thread-affine/MPI
  reducers retain their lifecycle and callbacks.
- Balance Green operators cyclically across workers, analogous in purpose to
  C `calgrn.c` dynamic scheduling. Each operator retains its arithmetic and
  indexed result slot, with original ordered accumulation.
- Reuse per-worker Pfaffian scratch, factor caller-owned batch buffers in place,
  and publish independent inverse planes in parallel. Preserve singular and
  non-finite statuses, operation order, signs and stale-state publication safety.
- Size only the explicit host C-order batch from QP/worker width. Environment
  overrides and separate device backend routes remain available.

No solver, parameter, seed, RNG algorithm or numerical tolerance was changed.

## Before/after controls

A separate fresh baseline cohort used Rust main `835cbad9` runtime code, built
in the submodule-update checkout before any #629 runtime change. Its metadata
source identifies that checkout; the preserved binary hash identifies the actual
executable. It is not the old pre-#586 baseline.

| Workload | Threads | Baseline Rust | Final Rust | Baseline C | Final paired C |
|---|---:|---:|---:|---:|---:|
| Opt | 8 | 20.941 s | 17.660 s | 19.637 s | 19.565 s |
| Opt | 16 | 22.663 s | 16.417 s | 21.188 s | 21.817 s |
| PhysCal | 8 | 2.147 s | 1.026 s | 1.281 s | 1.261 s |
| PhysCal | 16 | 1.677 s | 0.961 s | 1.271 s | 1.291 s |

Only the final freshly paired C values establish the requested speed ratios.
The changing controls illustrate host variation. Earlier diagnostic batch sweeps
were exploratory, not replacements for the final three-repetition cohort.

## Correctness

The full-state developer probe compares the same hash-verified before records
from #586 (Rust `cf17b0be51e991209875e54b18f86fccb9d078c4`) against final Rust
at 1/4/8/16 threads and a repeated 16-thread run, for both workloads: ten checks.
Full SFMT words/cursor, consumed word count, configurations, counters, proposal
and acceptance decision records match exactly. Opt records 821,760 decisions
and 4,110,359 consumed words; PhysCal records 41,088 and 206,174. All 732,490 /
732,990 numerical fields pass the existing `1e-11 + 1e-11*max(abs(x),abs(y))`
bound; observed maximum difference is zero. This is numerical comparison, not
bitwise float comparison, and is a Rust revision comparison rather than a new
full C-process RNG-state oracle. Cached before executable SHA-256 is verified
before loading its reference records; record hashes accompany the archive.

All final paired runs, including warmups, also pass 56 C/Rust output-file-pair
checks: Opt energy/moments, parameters and final optimized parameters; PhysCal
energy/moments, parameters and one-/two-body Green functions. Green index columns
match exactly. Existing Opt writer naming differs (`zvo_out.dat`/`zvo_var.dat`
versus C `_001` suffix); comparison explicitly maps those families.
The numerical bound is the same existing abs/rel bound above. Largest absolute
difference is `6.868503987789154e-9` in Opt moments at their larger scale; the
largest error/bound ratio across all fields is 0.188. This does not establish
full C/Rust RNG-state parity. SR diagnostics and timer files are not compared.

Runtime CI for PR #633 passes all seven checks: Linux and macOS, default and
MPI/SIMD configurations, formatting/Clippy, documentation, and container build
policy. Local optimized workspace: 1,645 passed; documentation, Clippy and formatting
checks succeeded. Logs and exact commands are archived in `validation.md`.
Initial local workspace discovery was successful, but the first run lacked the
Julia submodule's input files; initialization and a complete rerun are recorded,
not hidden by skipping those tests. Ordinary tests do not invoke Julia or C.

## Reproduction

Use a Linux development environment with Rust/BLAS/LAPACK, `uv` and Docker. Build
the development image following `docs/DEV_CONTAINER.md`, and verify physical CPU
numbering before applying this machine's affinity. Use fresh output/target paths.
Do not run measurements alongside builds, tests or other benchmarks.

```bash
git clone https://github.com/AtelierArith/mvmc-rs /tmp/mvmc-629-reproduce
cd /tmp/mvmc-629-reproduce
git fetch origin pull/633/head
git checkout --detach FETCH_HEAD
# PR #633 includes the scripts/report; measured runtime source is f2645823.
git submodule update --init extern/Julia-mVMC
bench_image=YOUR_BUILT_DEV_CONTAINER_IMAGE
bench_repo=$PWD
bench_target=/tmp/mvmc-629-reproduce-target
bench_c=/tmp/mvmc-629-reproduce-c
bench_out=/tmp/mvmc-629-reproduce-results
CARGO_TARGET_DIR="$bench_target" cargo build --locked --release \
  -p mvmc-cli -p mvmc-core --bin mvmc --example cpu_scaling_contract_586
bash benchmark/cpu_scaling_l128/build-c-reference.sh "$bench_repo" "$bench_image" "$bench_c"
uv run --no-project python benchmark/cpu_scaling_l128/measure.py \
  --repo "$bench_repo" --image "$bench_image" --target-volume mvmc-629-unused-target \
  --rust-binary "$bench_target/release/mvmc" --c-dir "$bench_c" \
  --opt-input benchmark/cpu_scaling_l128/inputs/opt \
  --phys-input benchmark/cpu_scaling_l128/inputs/phys \
  --threads 8 16 --reps 3 --container-name mvmc-629-reproduce \
  --output "$bench_out"
uv run --no-project python benchmark/cpu_scaling_l128/summarize.py "$bench_out"
uv run --no-project python benchmark/cpu_scaling_l128/verify_target.py "$bench_out"
```

`verify_target.py` re-derives medians from all raw timing rows, checks all groups
are complete, compares the 56 output pairs, and requires all four C/Rust ratios
>=1.1. Machine-to-machine variation can make this performance assertion fail.

For a fresh full-state comparison, follow the parent README's isolated before
checkout/target build at `cf17b0be` with the committed developer example, then:

```bash
uv run --no-project python benchmark/cpu_scaling_l128/contract.py \
  --out /tmp/mvmc-629-reproduce-contract \
  --inputs benchmark/cpu_scaling_l128/inputs --image "$bench_image" \
  --before-volume YOUR_BEFORE_TARGET_VOLUME --after-dir "$bench_target"
```

The actual run instead reused hash-verified before records with
`--reference /tmp/mvmc-586-contract-20261103`; their contents and hashes are in
`contract.tar.gz`. This skips only repeating the unchanged before execution.

## Evidence archives

`archives.json` gives SHA-256 and original-file counts for `paired.tar.gz`,
`baseline.tar.gz` and `contract.tar.gz`. Each has an `original-files.json` ledger
with sizes, hashes and original mtime nanoseconds. No executable/target files are
included. Identical full contract records use tar hard links; original-file
ledgers retain each logical file. Extract each into a fresh directory and run the relevant verifier:

```bash
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py \
  benchmark/cpu_scaling_l128/results/20261103-issue629
mkdir /tmp/mvmc-629-saved-evidence
tar -xzf benchmark/cpu_scaling_l128/results/20261103-issue629/paired.tar.gz \
  -C /tmp/mvmc-629-saved-evidence
uv run --no-project python benchmark/cpu_scaling_l128/verify_target.py \
  /tmp/mvmc-629-saved-evidence/paired
```
