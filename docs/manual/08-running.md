# 8. Running

[Contents](README.md) · Previous: [7. Input files](07-input-files.md) · Next: [9. Output files](09-output-files.md)

## 8.1 The `mvmc` command

The binary is built from `crates/mvmc-cli/src/lib.rs` (`cargo build --release -p mvmc-cli` gives
`target/release/mvmc`; `cargo run -p mvmc-cli -- ...` also works). Synopsis:

```text
mvmc [options] <namelist.def> [initpara]
```

The synopsis is the C driver's `vmc.out [option] NameListFile [OptParaFile]` (`vmcmain.c:714-723`). Options may follow the positional arguments; short options can be clustered (`-bo`) and take their value attached or as the next word (`-F2`, `-F 2`), as with C `getopt`.

| Option | Meaning | Default |
|--------|---------|---------|
| `--nsteps <N>` | number of SR steps; overrides `NSROptItrStep` | `NSROptItrStep` from `modpara.def` (must be $>0$ for optimization) |
| `--nsmp <N>` | final averaging window; overrides `NSROptItrSmp` (must be $\le$ the number of steps) | `NSROptItrSmp` |
| `--out-dir <DIR>` | output directory (created if missing) | `<directory of namelist.def>/output` |
| `--seed <N>` | RNG seed; replaces `RndSeed` (the group/rank offset is still added, [4.6](04-theory-sampling.md#46-parallelism-inside-the-sampler)) | `RndSeed` |
| `--mode real\|cmp\|fsz` | *checked label*: it does **not select** the numerical mode, which comes from the input declarations ([3.3](03-theory-wavefunction.md#33-real-and-complex-modes)); a label that contradicts the inferred mode is an error (exit 2, before any file is created). Redundant with the inference, kept for scripts | inferred (`fsz` if general orbitals, else `cmp` if any complex declaration, else `real`) |
| `--initial-def auto\|none\|PATH` | initial parameter file ([7.4](07-input-files.md#74-initial-parameter-values)): `auto` loads a neighbouring `initial.def` if present, `none` skips it, `PATH` is required to exist | `auto` |
| `-o`, `--opt-trans` | enable the C OptTrans mode (the C driver's `-o`) | off |
| `-b` | binary parameter output: `zvo_varbin_NNN.dat` replaces the `zvo_var` text file ([binary output](#binary-output--b)) | off |
| `-F <N>`, `--flush-interval <N>` | flush the `_time_`/`_SRinfo` files every `N` steps; `N < 1` is an error ([9.5](09-output-files.md)) | 1 |
| `-e`, `--expert` | Expert mode (default); accepted, a no-op | on |
| `-v`, `--version` | print `mvmc-rs version <crate version> (follows C mVMC 1.3.0)` to stdout and exit `0` | |
| `-h`, `--help` | print usage (C option list plus the Rust extensions) to stderr and exit `0`, as C does | |
| `-s`, `--standard` | Standard mode: generate the Expert files from the StdFace input into `--out-dir` (default: current directory), then run `namelist.def` ([7.6](07-input-files.md#76-standard-mode-stdface)) | off |
| `--dry-run` | generate the Expert files from the StdFace input and stop (C `vmcdry.out`) | off |
| `-m <N>` | MultiDef mode: `mvmc -m N DirListFile NameListFile [OptParaFile]` runs `N` independent calculations, one per MPI group and directory ([MultiDef mode](#multidef-mode--m)) | off |
| `--physcal <PATH>` | alias of the positional `initpara` for `NVMCCalMode=1` (the fixed parameter file, [8.2](#82-physical-quantities-with-fixed-parameters)); an error for `NVMCCalMode=0`, and an error together with a positional file | – |
| `--physcal-trace <NEW_DIR>` | write non-consuming serial PhysCal diagnostics (stage-by-stage records of the PhysCal run) into the *new* directory `NEW_DIR`; requires `NVMCCalMode=1` with an explicit parameter file and a single-process launch | off |

Exit status: `0` success, `1` input/validation/runtime error (the message starts with `error:`), `2` usage error
(unknown flag, missing value, argument-count mismatch) **(observed)**. The default output
directory is `<namelist parent>/output` and `--help` now says so (it formerly said "namelist parent dir"). The C driver always writes `output/` relative to its working directory; the Rust default is relative to the namelist.

The C `getopt` string is `"bhm:oF:esv"` (`vmcmain.c:83`). Every option is implemented. `-F` follows C `strtol` rules: no digits or a value outside `int` is an error, trailing characters after the number only warn, and `N < 1` is an error.

### The positional `initpara` file

As in C (`vmcmain.c:177-182`, `:252-260`) the optional second positional argument is a parameter file in the `zqp_opt.dat` layout, and `NVMCCalMode` decides its role:

- `NVMCCalMode=0`: the **initial** parameters. They are loaded after the initialization draws and before the `In*` overlays (C `InitParameter` → `ReadInitParameter` → `ReadInputParameters`). Rust loads it as `--initial-def <path>`; giving both is an error, and a missing file is an error (C prints a message and continues).
- `NVMCCalMode=1`: the **fixed** parameters ([8.2](#82-physical-quantities-with-fixed-parameters)); `--physcal <PATH>` is an alias.

### MultiDef mode (`-m`)

`mvmc -m N DirListFile NameListFile [OptParaFile]` ports the C `vmc.out -m N` option (`initMultiDefMode`, `vmcmain.c:727-800`). It runs `N` independent calculations inside one MPI launch, each in its own directory with its own definition files.

- **Split.** The world of `size` ranks is split into `N` groups: with `div = size / N`, `mod = size % N`, `threshold = (div+1)*mod`, rank `r` belongs to group `r / (div+1)` if `r < threshold`, else `mod + (r - threshold) / div`. The first `mod` groups hold `div+1` ranks, the others `div` ranks. The group communicator is the whole world (`comm0`) of that run: its rank 0 is the output root and prints the console banner, and `NSplitSize` (and the seed offset `RndSeed + comm1 group`, [4.6](04-theory-sampling.md#46-parallelism-inside-the-sampler)) act inside the group. The group index itself does not enter the seed, so two groups that run the same input with the same width produce the same chain.
- **Directories.** Rank 0 reads the first `N` whitespace-separated names of `DirListFile` (C `fscanf("%s")`; the paths are relative to the launch directory); group `g` enters the `g`-th directory. `NameListFile` and `OptParaFile` are resolved **inside** the group directory, and so are the default output directory and a relative `--out-dir`. C uses one process per rank, so it changes the process working directory with `chdir`; `mvmc` does the same (also one process per rank).
- **`-e`/`-s`.** As in C, `-e` and `-s` clear the multi-definition flag, so `-m 2 -e a b` reads `a` as the namelist and `-e -m 2 dirs a` keeps `-m`.
- **Messages and status** (exit status `1`, like C's `exit(EXIT_FAILURE)`/`MPI_Abort`): `error: -m: N should be smaller than MPI size.` (world smaller than `N`; a serial launch is a one-rank world, so only `-m 1` runs), `warning: load imbalance. MPI_size=<size> nMultiDef=<N>` (rank 0, `size % N != 0`), `error: DirListFile does not exist.`, `error: <file> is incomplete.` (fewer than `N` names), `error: chdir(): <dir>: <strerror>`. Argument-count errors use the Rust usage status `2` (C prints the usage and exits with `1`). With the optional `mpi` feature every rank leaves collectively on a failure.

Differences from C, all defects or undefined behaviour there: `N <= 0` is rejected with `error: -m: N should be a positive integer.` (C divides by `N`: `SIGFPE` for `0`, an invalid communicator colour for `N < 0`); every rank of a group changes directory, whereas C changes it only on the group's rank 0 (`group2 == 0`), which is enough there because only that rank reads files; and a failure on rank 0 (list file, directory) stops all ranks before any further work, whereas in C `MPI_Abort` is asynchronous and the other ranks continue with uninitialized directory names (`tests/fixtures/multidef_348/README.md`).

References: native C `vmc.out -m 2` at 2, 3 and 4 ranks (two groups with different inputs; `tests/fixtures/multidef_348/`, regenerated by `c_toolbox/multidef_348/generate_c_runs.sh`). The split arithmetic is checked against the verbatim C formula for every `1 <= N <= size <= 48`. The explicit MPI gate `multidef_groups_match_own_single_runs_and_c_fixture` (`scripts/run_explicit_mpi_gates.sh`) checks that each group's outputs equal a standalone run of its directory at the group's width (scaled difference `<= 1e-12`) and the C fixture (`<= 1e-9`, the PhysCal bound of #349).

### Binary output (`-b`)

With `-b` (C `FlagBinary`, `vmcmain.c:654-660`, `initfile.c:58-66`, `:82-90`) **no** `zvo_var` text file is written; every other output file is unchanged. The parameters go to `zvo_varbin_NNN.dat` (`NNN` = `NDataIdxStart`, one file for the optimizer; one file per sample for PhysCal):

| Bytes | Content |
|-------|---------|
| 0–3 | native-endian `int32` `NPara` |
| 4–7 | native-endian `int32` `NSROptItrStep` (PhysCal: `1`) |
| then, once per step | `2*NPara` native-endian `f64` values: every parameter as (re, im) pairs |

**Intentional difference from C (C defect).** C calls `fwrite(Para, sizeof(double), NPara, ...)` on a `double complex` array, so its block holds only the first `NPara` doubles of the interleaved storage: the first `ceil(NPara/2)` parameters, the last one cut to its real part when `NPara` is odd. The remaining parameters are never written. `mvmc` keeps C's header (`NPara`, step count) but writes the complete `2*NPara`-double block, so the file is `8 + steps*2*NPara*8` bytes (C: `8 + steps*NPara*8`) and every parameter is present; C's block equals the leading `NPara` doubles of the `mvmc` block. A reader written for C files that uses `NPara` as the block length will misparse `mvmc` files; read `2*NPara` doubles per step. Verified against the unmodified C `vmc.out` (`tests/fixtures/issue347_varbin/PROVENANCE.md`, tolerance policy in [`docs/NUMERICAL_COMPARISONS.md`](../NUMERICAL_COMPARISONS.md)): headers are byte-identical, PhysCal parameters and the optimizer's step-0 block match exactly on the leading `NPara` doubles, later optimizer blocks agree to about `5e-10` (SR solve operation order; test bound `1e-8`).

### Selecting the calculation

The calculation is selected exactly as in the C driver, by `NVMCCalMode` in `modpara.def`
(`vmcmain.c`, `main`: `NVMCCalMode==0` → `VMCParaOpt`, `==1` → `VMCPhysCal`, else error), and the command-line options must agree with it.
After parsing (before any initialization or file creation, and collectively across MPI ranks) the CLI checks

| `NVMCCalMode` | `--physcal` | Result |
|---------------|-------------|--------|
| 0 | absent | parameter optimization (positional file = initial parameters) |
| 1 | present or absent | PhysCal (positional file or `--physcal` = fixed parameters; none: C `InitParameter` draws, then `In*` overlays and synchronization) |
| 0 | `--physcal` given | error: "`--physcal` requires NVMCCalMode=1 in ModPara" |
| other | – | error: "unsupported NVMCCalMode=… the CLI supports 0 (optimization) and 1 (PhysCal)" |

This dispatch rule was introduced by PR #340 (`select_calculation`, `crates/mvmc-cli/src/lib.rs`). Because the CLI parses
and validates the input *before* it dispatches, an optimization run with `NVMCCalMode=1` input is rejected with the
`select_calculation` message; the validation message "cannot run parameter optimization; use fixed-parameter PhysCal"
(`validate_para_opt`) is the library-level counterpart.

> **Implementation**
> - C: `main` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:46`
> - C: `VMCParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:331`
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - Rust: `main` — `crates/mvmc-cli/src/lib.rs:201`
> - Rust: `parse_c_int` — `crates/mvmc-cli/src/lib.rs:149`
> - Rust: `select_calculation` — `crates/mvmc-cli/src/lib.rs:969`
> - Rust: `run_with_selected_backend` — `crates/mvmc-cli/src/lib.rs:1201`
> - Rust: `run_physcal_with_selected_backend` — `crates/mvmc-cli/src/lib.rs:985`
> - Rust: `prepare_physcal` — `crates/mvmc-cli/src/lib.rs:1154`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:142`
> - Rust: `run_para_opt_from_namelist` — `crates/mvmc-core/src/run.rs:1512`
> - C: `initMultiDefMode` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:727`
> - Rust: `init_multi_def` — `crates/mvmc-cli/src/lib.rs:847`
> - Rust: `group_of_rank` — `crates/mvmc-core/src/multidef.rs:16`
> - Rust: `split_multi_def` — `crates/mvmc-core/src/mpi.rs:137`
> - Parity: the order "read definition files → set memory → initialize parameters (RNG seeded with `RndSeed + group`) → `InitFile` → run → write timers" of `main` is followed by `run_para_opt_from_namelist`; the C driver's `-m` option is ported as [MultiDef mode](#multidef-mode--m) (#348).

### Console output

Only the output root (world rank 0) prints. A parameter-optimization run prints a model banner, the resolved paths and
a summary; the exact text of the tutorial run is shown in [chapter 10](10-tutorial.md). Summary fields:

- `Completed N SR steps in T s` — wall-clock time of the run phase.
- `Final energy / site` — real part of the **last step's** $\langle H\rangle$ divided by `Nsite` (noisy; one step only).
- `Final-window means (n steps): [a, b]` — the means of the real and imaginary parts of $\langle H\rangle$ over the last `n = NSROptItrSmp` rows of `zvo_out.dat`.

The banner line `mode : NVMCCalMode=...` prints the value read from `modpara.def`.

## 8.2 Physical quantities with fixed parameters

```bash
mvmc namelist.def zqp_opt.dat --out-dir phys    # same as: --physcal zqp_opt.dat
```

with `NVMCCalMode 1` in `modpara.def`. The parameter file is the `zqp_opt.dat` written by an optimization run (or an
`initial.def`-style file; [7.4](07-input-files.md#74-initial-parameter-values)). As in C it is optional: `mvmc namelist.def` with `NVMCCalMode 1` measures the parameters from the initialization draws (verified against C; `tests/fixtures/issue347_varbin`). Behaviour (`prepare_phys_cal_from_namelist`, `vmc_phys_cal_in_place_timed`):

1. The Expert input is parsed and validated for PhysCal ([7.5](07-input-files.md#75-supported-and-rejected-inputs)); a parameter file that is given but missing is an error ("fixed parameter file not found").
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
> - Rust: `prepare_phys_cal_from_namelist` — `crates/mvmc-core/src/run.rs:518`
> - Rust: `vmc_phys_cal_in_place_timed` — `crates/mvmc-core/src/run.rs:908`
> - Rust: `read_opt_para_file` — `crates/mvmc-core/src/initial_params.rs:141`
> - Parity: `vmc_phys_cal_in_place_timed` forces `vmc_calc_mode = 1` on its working copy (`run.rs:811`), as C's PhysCal branch implies.

## 8.3 Threads and BLAS

`mvmc-rs` does not use OpenMP. The sampler and measurement loops are sequential by default, so a Markov chain consumes
random numbers exactly as the C code does. Optional shared-memory parallelism over independent work items (projection sectors in the `calc_m_all_*` Pfaffian setup,
rows of the `OO`/`HO` accumulation and of the stored Gram product, transfer terms of the local energy for real wave functions, the diagonal/PairHop/Exchange/InterAll energy terms, the rank-one `update_m_all_*`/`calculate_new_pf_m*` QP loops, Slater-element planes, doublon-holon counters, RBM hidden units and their derivatives, the Lanczos Hamiltonian/Green terms, the SR matrix assembly and the CG vector updates, Green-function entries)
is enabled with
`MVMC_RS_INNER_THREADS` ([8.5](#85-environment-variables)); it does not change the chain or the order in which each result is formed.
The workers form a low-latency spin pool (`crates/mvmc-core/src/spin_pool.rs`): `threads - 1` threads that spin for
about 300 us after each region (and park afterwards, so serial phases do not burn cores) and a calling thread that runs
block 0, like the OpenMP master. A region costs about 1 to 3 us to dispatch and join at 8 workers (the Rayon pool it
replaced cost 15 to 30 us), the blocks are the static `omp for` partition of the items, and every output element has one
producer, so results are bit-identical for every worker count. By default (no `MVMC_RS_INNER_THRESHOLD`) a region uses the
pool when its estimated serial work is at least 20 us; the many sub-microsecond per-proposal regions stay serial. On the
Hubbard chain with 8 workers this gives 1.9x at 32 sites, 2.7x at 64 sites for optimization and 1.5x for PhysCal at 32 sites
(about the speed of C's OpenMP regions); inputs whose regions are below the gate (Heisenberg, Kondo, FSZ, RBM) run as before
([benchmark results](../../benchmark/cpu_round/README.md)).
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
> - Rust: `MpiContext::split_groups` — `crates/mvmc-core/src/mpi.rs:154`
> - Rust: `MpiContext::initialize` — `crates/mvmc-core/src/mpi.rs:70`
> - Rust: `assign_group` — `crates/mvmc-core/src/parallel.rs:67`
> - Rust: `partition_range` — `crates/mvmc-core/src/parallel.rs:88`
> - Rust: `validate_grouped_runtime` — `crates/mvmc-core/src/validation.rs:23`
> - Rust: `run_para_opt_from_namelist_with_reducer` — `crates/mvmc-core/src/run.rs:1525`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1818`
> - Parity: communicator widths follow `vmcmain.c:239-256` (`NSplitSize` is the communicator *width*, not the number of chains); sample ranges follow `SplitLoop`; the C Green-function reduction goes to rank 0 only whereas Rust reduces the accumulators with an all-reduce and lets the root write, which yields the same file contents.

## 8.5 Environment variables

| Variable | Read by | Effect |
|----------|---------|--------|
| `MVMC_NSTEPS` | CLI, examples | same as `--nsteps` (the flag takes precedence) |
| `MVMC_C_TIMER` | CLI/core (`TimerEnv`, `crates/mvmc-core/src/c_timer.rs:174`) | any value other than `0` enables the C-compatible section timers; writes `zvo_CalcTimer.dat` ([9.5](09-output-files.md#95-timers)) |
| `MVMC_TIMER` | same | deprecated alias; if set without `MVMC_C_TIMER` a warning recommends `MVMC_C_TIMER=1` |
| `MVMC_CALHAM1_DIAG`, `MVMC_SLATER_DIAG`, `MVMC_MAINCAL_DIAG`, `MVMC_WEIGHTAVG_DIAG` | same | enable the corresponding diagnostic timer family (IDs up to 966); any of them also enables the main timer and writes `zvo_CalcTimerDiag.dat` |
| `MVMC_RS_INNER_THREADS` | `inner_thread_config` (`crates/mvmc-core/src/threading.rs:260`) | number of worker threads for independent inner work items; default 1 (sequential); invalid or 0 falls back to 1. Read once per process. |
| `MVMC_RS_MEASURE_BATCH` | `measurement_batch::resolve_batch_size` | samples per measurement batch (stage A tables for B samples, then per-sample consumers in order); default 4; output bytes do not depend on it. |
| `MVMC_RS_SR_BACKEND` | `stage_backend::validate_selected_stage_backend` | `c-order` (default, BLAS/LAPACK parity oracle), `tenferro` or `cuda[:N]`. The opt-in backends run the SR stages (Gram product, S/g assembly, Cholesky solve, CG product) through tenferro `dot_general`/`cholesky`/`triangular_solve`, validated with explicit tolerances and not byte-identical. `cuda` needs the `mvmc-cuda` binary of `gpu/mvmc-gpu-cuda`, which registers the CUDA provider and then runs this same CLI; the stock `mvmc` rejects it. The selector is validated once at startup before any IO: an invalid value or an unavailable backend prints `error: MVMC_RS_SR_BACKEND: ...` and exits with status 2 (collectively under MPI), never a fallback. See [chapter 12](12-accelerated-backends.md#124-selecting-a-backend). |
| `MVMC_RS_MEASURE_PF_BACKEND` | `measurement_batch::selected_measurement_pfaffian` | source of the Pfaffian/inverse tables of the measurement: `calc-m-all` (default, C-order kernels), or `c-order`, `tenferro`, `cuda[:N]` to build them with one batched `PfaffianStages` call per batch (real, non-FSZ mode only). `c-order` is byte-identical to the default; a backend without the stage or an unsupported mode is an error, never a fallback. Like `MVMC_RS_SR_BACKEND` it is validated once at startup before any IO (invalid value or unavailable backend: `error: MVMC_RS_MEASURE_PF_BACKEND: ...`, exit status 2). |
| `MVMC_RS_INNER_THRESHOLD` | same | a positive value selects the plain item-count gate: a region uses the worker pool when it has at least this many items (used by the worker-invariance tests; small values force pooled execution of tiny inputs). Unset, empty, invalid or 0 selects the automatic gate below (reported `threshold` stays 32) |
| `MVMC_RS_INNER_MIN_WORK_NS` | same | automatic gate: minimum estimated serial work of one region in nanoseconds; default 20000 |
| `MVMC_RS_INNER_MIN_SIZE` | same | automatic gate: minimum electron-matrix dimension `n_size` (number of electrons) for regions that scale with the matrices; default 1 (off, the work estimate decides alone; it was `120*w/(w-1)` with the former Rayon pool) |
| `MVMC_RS_INNER_PROFILE` | `threading::dispatch_profile` | `1` makes the CLI print, on stderr after the run, the calls, items and time of every inner-kernel call site, serial or pooled |
| `MVMC_RS_MPI_RANK`, `MVMC_RS_MPI_SIZE`, `OMPI_COMM_WORLD_*`, `PMI_*`, `PMIX_*` | `LaunchContext::from_env` | launcher detection ([8.4](#84-mpi-and-grouped-execution)) |
| `JULIA_MVMC_ROOT`, `JULIA_MVMC_EXAMPLE_STEPS`, `MVMC_OUT_DIR` | `cargo run --example ...` programs only | location of `extern/Julia-mVMC` inputs, step count, output root (default `output/<model>/`) |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ... | OpenBLAS (not read by `mvmc-rs`) | BLAS threading |

The GPU tools use further variables that are not options of `mvmc`: `MVMC_RS_CUDA_GATE` (`1`/`require` makes a missing CUDA device a hard failure of the gate), `MVMC_RS_CUDA_IMAGE` (docker image of `scripts/run_cuda_gate.sh`), `MVMC_RS_CUDA_GATE_OUT` and its siblings (report paths) and `MVMC_RS_SAMPLER_CORES` (host cores of the device-sampler benchmark); see [12.3](12-accelerated-backends.md#123-building-and-running).

The Julia-specific debug dumps `MVMC_DEBUG_*` mentioned in the original issue are **not** implemented in Rust. Variables such as `MVMC_RS_PHASE`,
`MVMC_RS_CTEST_*`, `MVMC_RS_THREADED_*`, `MVMC_CG_DIAGNOSTICS` and `MVMC_GREEN_INDEX_CHILD` appear only in test code and are not user options.

The values of `MVMC_C_TIMER` and the `*_DIAG` variables are compared with the literal string `0`: `MVMC_C_TIMER=0` disables, any other string (including an empty one) enables.

> **Implementation**
> - Rust: `TimerEnv::from_lookup` — `crates/mvmc-core/src/c_timer.rs:174`
> - Rust: `inner_thread_config` — `crates/mvmc-core/src/threading.rs:260`
> - Rust: `LaunchContext::from_env` — `crates/mvmc-core/src/parallel.rs:20`

## 8.6 Reproducibility

Two runs of the same binary with the same inputs, seed and process layout produce byte-identical output files; the tutorial run was repeated with
`--seed 5` and `cmp` reported identical `zvo_out.dat` **(observed)**. Different BLAS providers, platforms or thread layouts may change
floating-point sums in the last bits and, through the Metropolis test, subsequent configurations; this is expected and is why cross-implementation
comparisons use tolerances ([11.4](11-compatibility.md#114-numerical-comparison-policy)).

An accelerated SR backend (`MVMC_RS_SR_BACKEND=tenferro`, [chapter 12](12-accelerated-backends.md)) is also repeatable for a fixed implementation, device and library version (two `tenferro` runs of three steps gave byte-identical `zvo_out.dat` and `zqp_opt.dat` **(observed)**), but it is **not** byte-identical to the default `c-order` path: the RNG draws and Metropolis protocol are the same, the SR matrices and solutions agree within derived bounds, and later steps can differ in the last digits **(observed)**. Several independent chains in one process (`run_para_opt_multichain`, library API only, no `mvmc` flag) follow the C group-seed rule `RndSeed + group_base + walker` ([12.7](12-accelerated-backends.md#127-multi-walker-runs)).
