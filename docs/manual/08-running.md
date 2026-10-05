# 8. Running

[Contents](README.md) · Previous: [7. Input files](07-input-files.md) · Next: [9. Output files](09-output-files.md)

## 8.1 The `mvmc` command

The binary is built from `crates/mvmc-cli/src/main.rs` (`cargo build --release -p mvmc-cli` gives
`target/release/mvmc`; `cargo run -p mvmc-cli -- ...` also works). Synopsis:

```text
mvmc <namelist.def> [options]
```

| Option | Meaning | Default |
|--------|---------|---------|
| `--nsteps <N>` | number of SR steps; overrides `NSROptItrStep` | `NSROptItrStep` from `modpara.def` (must be $>0$ for optimization) |
| `--nsmp <N>` | final averaging window; overrides `NSROptItrSmp` (must be $\le$ the number of steps) | `NSROptItrSmp` |
| `--out-dir <DIR>` | output directory (created if missing) | `<directory of namelist.def>/output` |
| `--seed <N>` | RNG seed; replaces `RndSeed` (the group/rank offset is still added, [4.6](04-theory-sampling.md#46-parallelism-inside-the-sampler)) | `RndSeed` |
| `--mode real\|cmp\|fsz` | *sanity label only*: validated against `{real,cmp,fsz}` but **does not select** the numerical mode, which comes from the input declarations ([3.3](03-theory-wavefunction.md#33-real-and-complex-modes)) | inferred (`fsz` if general orbitals, else `cmp` if any complex declaration, else `real`) |
| `--initial-def auto\|none\|PATH` | initial parameter file ([7.4](07-input-files.md#74-initial-parameter-values)): `auto` loads a neighbouring `initial.def` if present, `none` skips it, `PATH` is required to exist | `auto` |
| `-o`, `--opt-trans` | enable the C OptTrans mode (the C driver's `-o`) | off |
| `--physcal <PATH>` | run fixed-parameter PhysCal using the parameter file `PATH` ([8.2](#82-physical-quantities-with-fixed-parameters)) | off |
| `--physcal-trace <NEW_DIR>` | write non-consuming serial PhysCal diagnostics (stage-by-stage records of the PhysCal run) into the *new* directory `NEW_DIR`; requires `--physcal` and a single-process launch | off |
| `--help`, `-h` | print usage | |

Exit status: `0` success, `1` input/validation/runtime error (the message starts with `error:`), `2` usage error
(unknown flag, missing value, missing `namelist.def`) **(observed)**. The `--help` text says the default output
directory is "namelist parent dir"; the code uses `<namelist parent>/output` **(observed)**.

Unlike the C driver (`getopt` string `"bhm:oF:esv"`, `vmcmain.c:46`), `mvmc` has no `-b` (binary output), `-m` (multi-definition mode),
`-F` (flush interval), `-e`/`-s` (Expert/Standard mode; only Expert input exists), `-v` (version) or positional initial-parameter file (use `--initial-def`).

### Selecting the calculation

The calculation is selected exactly as in the C driver, by `NVMCCalMode` in `modpara.def`
(`vmcmain.c`, `main`: `NVMCCalMode==0` → `VMCParaOpt`, `==1` → `VMCPhysCal`, else error), and the command-line options must agree with it.
After parsing (before any initialization or file creation, and collectively across MPI ranks) the CLI checks

| `NVMCCalMode` | `--physcal` | Result |
|---------------|-------------|--------|
| 0 | absent | parameter optimization |
| 1 | present | fixed-parameter PhysCal |
| 0 | present | error: "`--physcal` requires NVMCCalMode=1 in ModPara" |
| 1 | absent | error: "NVMCCalMode=1 selects fixed-parameter PhysCal; supply the fixed parameter file with `--physcal <PATH>`" |
| other | – | error: "unsupported NVMCCalMode=… the CLI supports 0 (optimization) and 1 (PhysCal)" |

This dispatch rule was introduced by PR #340 (`select_calculation`, `crates/mvmc-cli/src/main.rs`). Because the CLI parses
and validates the input *before* it dispatches, an optimization run with `NVMCCalMode=1` input is rejected with the
`select_calculation` message; the validation message "cannot run parameter optimization; use fixed-parameter PhysCal"
(`validate_para_opt`) is the library-level counterpart.

> **Implementation**
> - C: `main` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:46`
> - C: `VMCParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:331`
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - Rust: `main` — `crates/mvmc-cli/src/main.rs:114`
> - Rust: `select_calculation` — `crates/mvmc-cli/src/main.rs:560`
> - Rust: `run_with_selected_backend` — `crates/mvmc-cli/src/main.rs:758`
> - Rust: `run_physcal_with_selected_backend` — `crates/mvmc-cli/src/main.rs:579`
> - Rust: `run_para_opt_from_namelist` — `crates/mvmc-core/src/run.rs:1333`
> - Parity: the order "read definition files → set memory → initialize parameters (RNG seeded with `RndSeed + group`) → `InitFile` → run → write timers" of `main` is followed by `run_para_opt_from_namelist`; the C driver's `getopt` options other than `-o` are not implemented.

### Console output

Only the output root (world rank 0) prints. A parameter-optimization run prints a model banner, the resolved paths and
a summary; the exact text of the tutorial run is shown in [chapter 10](10-tutorial.md). Summary fields:

- `Completed N SR steps in T s` — wall-clock time of the run phase.
- `Final energy / site` — real part of the **last step's** $\langle H\rangle$ divided by `Nsite` (noisy; one step only).
- `Final-window means (n steps): [a, b]` — the means of the real and imaginary parts of $\langle H\rangle$ over the last `n = NSROptItrSmp` rows of `zvo_out.dat`.

The banner line `mode : NVMCCalMode=...` prints the value read from `modpara.def`.

## 8.2 Physical quantities with fixed parameters

```bash
mvmc namelist.def --physcal zqp_opt.dat --out-dir phys
```

with `NVMCCalMode 1` in `modpara.def`. The parameter file is the `zqp_opt.dat` written by an optimization run (or an
`initial.def`-style file; [7.4](07-input-files.md#74-initial-parameter-values)). Behaviour (`prepare_phys_cal_from_namelist`, `vmc_phys_cal_in_place_timed`):

1. The Expert input is parsed and validated for PhysCal ([7.5](07-input-files.md#75-supported-and-rejected-inputs)); a missing parameter file is an error ("fixed parameter file not found").
2. The fixed parameters are loaded **before** any random number is consumed (test `physcal_preparation_loads_fixed_parameters_before_rng_consumption`), overlays and synchronization follow, then `UpdateSlaterElm`.
3. For each of `NDataQtySmp` samples the chain is sampled, measured, averaged over ranks and written as one numbered file set
   `zvo_*_NNN.dat` with `NNN = NDataIdxStart + sample` (`%03d`; a negative start prints e.g. `-01`).
4. Parameters are **not** modified; the `zvo_var_NNN.dat` of each sample repeats the loaded values.

`--nsteps`/`--nsmp` have no effect on PhysCal. The console summary is `Completed K PhysCal samples in T s`.

### Diagnostics: `--physcal-trace`

`--physcal-trace <NEW_DIR>` creates `NEW_DIR` (it must not exist) with `schema.txt` and `request.txt`, then one sub-directory per
stage in the order `fixed-loaded`, `overlaid`, `synchronized`, `seeded`, `initialized-clone`, `sample-0`, `sample-1`, ...
Each stage directory holds `parameters.txt`, `resolved.txt`, `settings.txt`, the next 624 words the RNG *would* produce
(`next624.txt`, taken from a clone so the run is not disturbed) with `draw-count.txt`, and, for the sampling stages, the electron
configuration arrays (`ele_idx.txt`, `ele_cfg.txt`, `ele_num.txt`, `ele_proj_cnt.txt`, `ele_spn.txt`, `counter.txt`). The run ends with
`stages.txt` and `terminal.txt`. The records follow the Rust boundaries, "not C chronological replay"
(`crates/mvmc-cli/src/physcal_trace.rs`). It is intended for parity debugging, is not a production output and is refused for multi-rank launches.

> **Implementation**
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - C: `InitFilePhysCal` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:72`
> - Rust: `prepare_phys_cal_from_namelist` — `crates/mvmc-core/src/run.rs:512`
> - Rust: `vmc_phys_cal_in_place_timed` — `crates/mvmc-core/src/run.rs:775`
> - Rust: `read_opt_para_file` — `crates/mvmc-core/src/initial_params.rs:141`
> - Parity: `vmc_phys_cal_in_place_timed` forces `vmc_calc_mode = 1` on its working copy (`run.rs:811`), as C's PhysCal branch implies.

## 8.3 Threads and BLAS

`mvmc-rs` does not use OpenMP. The sampler and measurement loops are sequential by default, so a Markov chain consumes
random numbers exactly as the C code does. Optional shared-memory parallelism over independent work items (projection sectors in the `calc_m_all_*` Pfaffian setup,
rows of the `OO`/`HO` accumulation and of the stored Gram product, transfer terms of the local energy for real wave functions, Green-function entries)
is enabled with
`MVMC_RS_INNER_THREADS` ([8.5](#85-environment-variables)); it does not change the chain or the order in which each result is formed.
Dense linear algebra (`dgemv`, `dpotrf`, Pfaffian kernels) runs in OpenBLAS, whose own thread count is controlled by the usual
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` variables (not read by `mvmc-rs`; the project's benchmarks pin them to 1).

## 8.4 MPI and grouped execution

### Build and launch

```bash
cargo build --release -p mvmc-cli --features mpi
mpirun -np 4 target/release/mvmc namelist.def --out-dir out          # 4 independent chains
mpirun -np 8 target/release/mvmc namelist.def --out-dir out          # with NSplitSize 2 in modpara.def: 4 groups of 2 ranks
```

The launch is detected from the environment before MPI is initialized: `MVMC_RS_MPI_RANK`/`MVMC_RS_MPI_SIZE` (explicit test variables),
`OMPI_COMM_WORLD_RANK`/`_SIZE` (Open MPI), `PMI_RANK`/`PMI_SIZE` (MPICH/PMI) and `PMIX_RANK`/`PMIX_SIZE`
(`LaunchContext::from_env`, `crates/mvmc-core/src/parallel.rs:20`). If `world_size > 1` and the binary was built without the `mpi` feature the run stops with
"MPI launcher detected; rebuild mvmc-cli with --features mpi to enable MPI execution". With the feature, every rank
parses and validates the input, and the CLI checks collectively that the options and `ModPara` values agree across ranks
(`agree_controls`: "CLI run controls differ between MPI ranks"). All ranks therefore need read access to the input files.
MPI runs were **not** executed while writing this manual (no MPI launcher available) **(unverified)**; the statements here come from the code and
the `mpi`-feature tests in `crates/mvmc-core/src/run_mpi_tests.rs`.

### Communicators and `NSplitSize`

The world communicator is split as in the C driver (`vmcmain.c:239-256`):

- `comm1`: ranks `0..NSplitSize-1`, `NSplitSize..2*NSplitSize-1`, ... Each group works on **one Markov chain**; `group = rank / NSplitSize`.
  The last group is smaller when `NSplitSize` does not divide the world size (C prints a load-imbalance warning, Rust accepts it).
- `comm2`: ranks with the same position within their group, used only to reduce the six sampler statistics counters.
- All accumulators of [chapters 2](02-theory-vmc-hamiltonian.md) and [5](05-theory-sr.md) are summed over the **whole world** and divided by the total
  weight $W$, so the effective number of samples is $(\text{number of groups})\times\texttt{NVMCSample}$.

With `NSplitSize = 1` (default) every rank is its own group: it samples and measures its own `NVMCSample` configurations with the seed
`RndSeed + rank`. With `NSplitSize = n > 1` the `n` ranks of a group use the same seed and make identical random draws; the projection sectors
$[0,N_{\rm QP})$ are divided among them for the Pfaffian updates (`SplitLoop`, `partition_range`) and the group's saved samples are divided among them for the
measurement; $\mathrm{IP}$ is reduced inside the group. Grouped runs are therefore useful when `NQPFull` is large (spin/momentum projection).

### Restrictions

Rust rejects only the one grouped combination that is undefined in C, before initialization (`validate_grouped_runtime`, [7.5](07-input-files.md#75-supported-and-rejected-inputs)):

- `NSplitSize > 1` with the CG solver (`NSRCG != 0`). `VMCMainCal` stores `sqrt(w) O` at the global sample slot (`vmccal.c:241,248`) but
  `calculateOO_Store` reads the first `sampleEnd - sampleStart` columns from the base pointer (`vmccal.c:314-318`), so every rank after the first reads
  unwritten `malloc` memory (`setmemory.c:419-421`). The same defect corrupts direct SR with `NStore != 0` and `NSplitSize > 1` in C; Rust computes that case correctly and therefore differs from C there.

Supported (checked against native C at 2 and 4 ranks, `tests/fixtures/grouped_nsplit_349/`): direct SR (`NSRCG = 0`) for any `NSplitSize`; SR-CG (`NSRCG = 1`) with `NSplitSize = 1`;
PhysCal and optimization with sz-conserved, FSZ/general orbitals and any `NQPFull`; OptTrans with `NQPOptTrans > 1`; `NLanczosMode = 1, 2` PhysCal.

### Which rank writes what

Every output file is written by **world rank 0 only** (`Reducer::is_output_root`, `crates/mvmc-core/src/reducer.rs:134`; for the grouped communicator it is
group 0, local rank 0 — the same process). Other ranks compute and take part in the reductions but never write; an explicit `--out-dir` may be
rank-local because only the root touches it (`RunConfig` documentation, `run.rs:1206`). The same holds for the console output and for the timer files.
Failures are agreed collectively, so a failing rank makes all ranks stop rather than hang.

> **Implementation**
> - C: `main` (communicator split, `init_gen_rand(RndSeed+group1)`) — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:46`
> - C: `SplitLoop` — `extern/mVMC-1.3.0/src/mVMC/splitloop.c:31`
> - Rust: `MpiContext::split_groups` — `crates/mvmc-core/src/mpi.rs:125`
> - Rust: `MpiContext::initialize` — `crates/mvmc-core/src/mpi.rs:64`
> - Rust: `assign_group` — `crates/mvmc-core/src/parallel.rs:67`
> - Rust: `partition_range` — `crates/mvmc-core/src/parallel.rs:88`
> - Rust: `validate_grouped_runtime` — `crates/mvmc-core/src/validation.rs:23`
> - Rust: `run_para_opt_from_namelist_with_reducer` — `crates/mvmc-core/src/run.rs:1346`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1614`
> - Parity: communicator widths follow `vmcmain.c:239-256` (`NSplitSize` is the communicator *width*, not the number of chains); sample ranges follow `SplitLoop`; the C Green-function reduction goes to rank 0 only whereas Rust reduces the accumulators with an all-reduce and lets the root write, which yields the same file contents.

## 8.5 Environment variables

| Variable | Read by | Effect |
|----------|---------|--------|
| `MVMC_NSTEPS` | CLI, examples | same as `--nsteps` (the flag takes precedence) |
| `MVMC_C_TIMER` | CLI/core (`TimerEnv`, `crates/mvmc-core/src/c_timer.rs:174`) | any value other than `0` enables the C-compatible section timers; writes `zvo_CalcTimer.dat` ([9.5](09-output-files.md#95-timers)) |
| `MVMC_TIMER` | same | deprecated alias; if set without `MVMC_C_TIMER` a warning recommends `MVMC_C_TIMER=1` |
| `MVMC_CALHAM1_DIAG`, `MVMC_SLATER_DIAG`, `MVMC_MAINCAL_DIAG`, `MVMC_WEIGHTAVG_DIAG` | same | enable the corresponding diagnostic timer family (IDs up to 966); any of them also enables the main timer and writes `zvo_CalcTimerDiag.dat` |
| `MVMC_RS_INNER_THREADS` | `inner_thread_config` (`crates/mvmc-core/src/threading.rs:190`) | number of worker threads for independent inner work items; default 1 (sequential); invalid or 0 falls back to 1. Read once per process. |
| `MVMC_RS_INNER_THRESHOLD` | same | minimum number of work items before the worker pool is used; default 32; invalid or 0 falls back to 32 |
| `MVMC_RS_MPI_RANK`, `MVMC_RS_MPI_SIZE`, `OMPI_COMM_WORLD_*`, `PMI_*`, `PMIX_*` | `LaunchContext::from_env` | launcher detection ([8.4](#84-mpi-and-grouped-execution)) |
| `JULIA_MVMC_ROOT`, `JULIA_MVMC_EXAMPLE_STEPS`, `MVMC_OUT_DIR` | `cargo run --example ...` programs only | location of `extern/Julia-mVMC` inputs, step count, output root (default `output/<model>/`) |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ... | OpenBLAS (not read by `mvmc-rs`) | BLAS threading |

The Julia-specific debug dumps `MVMC_DEBUG_*` mentioned in the original issue are **not** implemented in Rust. Variables such as `MVMC_RS_PHASE`,
`MVMC_RS_CTEST_*`, `MVMC_RS_THREADED_*`, `MVMC_CG_DIAGNOSTICS` and `MVMC_GREEN_INDEX_CHILD` appear only in test code and are not user options.

The values of `MVMC_C_TIMER` and the `*_DIAG` variables are compared with the literal string `0`: `MVMC_C_TIMER=0` disables, any other string (including an empty one) enables.

> **Implementation**
> - Rust: `TimerEnv::from_lookup` — `crates/mvmc-core/src/c_timer.rs:174`
> - Rust: `inner_thread_config` — `crates/mvmc-core/src/threading.rs:190`
> - Rust: `LaunchContext::from_env` — `crates/mvmc-core/src/parallel.rs:20`

## 8.6 Reproducibility

Two runs of the same binary with the same inputs, seed and process layout produce byte-identical output files; the tutorial run was repeated with
`--seed 5` and `cmp` reported identical `zvo_out.dat` **(observed)**. Different BLAS providers, platforms or thread layouts may change
floating-point sums in the last bits and, through the Metropolis test, subsequent configurations; this is expected and is why cross-implementation
comparisons use tolerances ([11.4](11-compatibility.md#114-numerical-comparison-policy)).
