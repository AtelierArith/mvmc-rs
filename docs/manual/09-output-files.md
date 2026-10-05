# 9. Output files

[Contents](README.md) · Previous: [8. Running](08-running.md) · Next: [10. Tutorial](10-tutorial.md)

All files are written into the output directory (`--out-dir`) by **world rank 0 only**
([8.4](08-running.md#84-mpi-and-grouped-execution)). In the names below `zvo` is `CDataFileHead`
and `zqp` is `CParaFileHead` from `modpara.def` (defaults `zvo`, `zqp`), and `NNN` is
`NDataIdxStart + k` printed with `%03d` (a negative index prints as e.g. `-01`).

Numbers are written with the C format `"% .18e"` (a leading space instead of `+`, 18 digits,
at least two exponent digits; `format_c_double`, `crates/mvmc-core/src/io.rs:50`). The C reference of every format
is [`extern/mVMC-1.3.0/doc/en/source/output.rst`](../../extern/mVMC-1.3.0/doc/en/source/output.rst) and `initfile.c`, `vmcmain.c:outputData`, `avevar.c`.

Summary of what is written ("opt" = parameter optimization, "phys" = PhysCal):

| File | opt | phys | Condition |
|------|-----|------|-----------|
| `zvo_out.dat` | yes | – | always |
| `zvo_var.dat` | yes | – | always |
| `zqp_opt.dat` | yes | – | always (at the end) |
| `zqp_<block>_opt.dat` | yes | – | `NSROptItrSmp > 1` and the block is not empty |
| `zvo_SRinfo.dat` | CG only | – | `NSRCG != 0` |
| `zvo_out_NNN.dat`, `zvo_var_NNN.dat` | – | yes | always |
| `zvo_cisajs_NNN.dat` | – | yes | `OneBodyG` has entries |
| `zvo_cisajscktalt_NNN.dat` | – | yes | `TwoBodyG` has entries |
| `zvo_cisajscktaltex_NNN.dat` | – | yes | `TwoBodyGEx` has entries |
| `zvo_ls_out_NNN.dat`, `zvo_ls_qqqq_NNN.dat` | – | yes | `NLanczosMode > 0` |
| `zvo_ls_cisajs_NNN.dat`, `zvo_ls_cisajscktalt_NNN.dat`, `zvo_ls_cisajscktaltex_NNN.dat` | – | yes | `NLanczosMode = 2` |
| `zvo_CalcTimer.dat`, `zvo_CalcTimerDiag.dat` | yes | yes | timer environment variables ([8.5](08-running.md#85-environment-variables)) |

## 9.1 Energy per step: `zvo_out.dat` (opt) and `zvo_out_NNN.dat` (phys)

One row per SR step (opt; the file is truncated at step 0 and appended afterwards) or a single row (phys). Six columns:

| # | Quantity |
|---|----------|
| 1 | $\operatorname{Re}\langle H\rangle$ |
| 2 | $\operatorname{Im}\langle H\rangle$ |
| 3 | $\operatorname{Re}\langle H^2\rangle_{\rm est}=\langle\lvert E_{\rm loc}\rvert^2\rangle$ |
| 4 | relative variance $\operatorname{Re}[(\langle H^2\rangle-\langle H\rangle^2)/\langle H\rangle^2]$ |
| 5 | $\langle S_z\rangle$ |
| 6 | $\langle S_z^2\rangle$ |

The C format is `"% .18e % .18e  % .18e % .18e %.18e %.18e\n"` — two spaces before column 3 and no leading space before
columns 5 and 6 (`vmcmain.c:640`); Rust reproduces it byte for byte. Example (first row of the tutorial run):

```text
 1.719867648718613040e+01  0.000000000000000000e+00   3.255315646316159359e+02  1.005329525872698249e-01 0.000000000000000000e+00 0.000000000000000000e+00
```

> **Implementation**
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - C: `InitFile` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:33`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:89`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Parity (**naming difference**): for optimization C writes the files with the data index, `zvo_out_001.dat` and `zvo_var_001.dat` (`initfile.c:55,59`, `NDataIdxStart`); Rust writes **`zvo_out.dat` and `zvo_var.dat` without the index** (Julia convention, `io.rs:113,128`). PhysCal files carry the index in both. The variance column is `0.0` in the optimization output when $\lvert\langle H\rangle\rvert\le10^{-14}$ (`io.rs:103`).

## 9.2 Parameters per step: `zvo_var.dat` and `zvo_var_NNN.dat`

One line per step. It starts with $\operatorname{Re}\langle H\rangle,\ \operatorname{Im}\langle H\rangle,\ 0.0,\ \operatorname{Re}\langle H^2\rangle,\ \operatorname{Im}\langle H^2\rangle,\ 0.0$ followed by one triple
`Re Im 0.0` for **every** entry of the parameter vector in the layout of
[3.1](03-theory-wavefunction.md#parameter-layout) (projection, RBM, Slater, OptTrans; unmapped and reserved slots are written too). A line therefore has
$6+3N_{\rm para}$ numbers (306 for the 100-parameter tutorial model). The parameters in line $t$ are those *used* for the samples of step $t$, before the SR update.
The line format is `"% .18e % .18e 0.0 ..."` (`vmcmain.c:655-657`).

## 9.3 Optimized parameters

**`zqp_opt.dat`** is written once at the end of an optimization (`OutputOptData`) and is the input of PhysCal. It is a single line of
triples. For $\texttt{NSROptItrSmp}>1$, for the quantities $\langle H\rangle$, $\langle H^2\rangle$ and each parameter in parameter-vector order:
the window mean (real, imaginary) and the window standard deviation (real). For `NSROptItrSmp = 1` it holds pairs (value, `0.0`) and no per-block files are written.

**Per-block files** `zqp_gutzwiller_opt.dat`, `zqp_jastrow_opt.dat`, `zqp_doublonHolon2site_opt.dat`, `zqp_doublonHolon4site_opt.dat`,
`zqp_{charge,spin,general}RBM_{physlayer,hiddenlayer,physhidden}_opt.dat`, `zqp_orbital_opt.dat` (normal orbitals) or
`zqp_orbitalAntiParallel_opt.dat` + `zqp_orbitalParallel_opt.dat` or `zqp_orbital_general_opt.dat` (FSZ), and `zqp_trans_opt.dat` (OptTrans) have the layout

```text
======================
NGutzwillerIdx  4
======================
======================
======================
0 -1.556663106607272473e+00  0.000000000000000000e+00 
1 -1.337680080798275606e+00  0.000000000000000000e+00 
...
```

i.e. five header lines (the second names the block and its declared count) and rows `index mean_re mean_im`. This is exactly the format of the
`In*` initial-value files ([7.4](07-input-files.md#74-initial-parameter-values)), so a block file can be reused as an initial-value file. The declared count in the header
of the DH blocks is the number of patterns, while the row count is $6\times$ (DH2) or $10\times$ (DH4) that number.

> **Implementation**
> - C: `StoreOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:82`
> - C: `OutputOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:94`
> - Rust: `store_opt_data` — `crates/mvmc-core/src/io.rs:21`
> - Rust: `output_opt_data` — `crates/mvmc-core/src/io.rs:441`
> - Rust: `output_parameter_block` — `crates/mvmc-core/src/io.rs:580`
> - Parity: block order, file names (`RBM_OUTPUT_BLOCKS`), header text and the "pairs, no auxiliary files" rule for `NSROptItrSmp = 1` follow `OutputOptData` literally. The window statistic is $\sqrt{\sum|x-\bar x|^2/(n-1)}$ in both.

## 9.4 Solver information: `zvo_SRinfo.dat`

Written by the **CG** solver only (`NSRCG != 0`), one row per SR step after a one-line header:

```text
#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax
  100   100     0     0  1.22546e+00  1.14682e-02 -6.31767e-02    76, 100
```

Columns: number of parameters `Npara` (complex count), size of the solved system `Msize` ($n_S$), parameters held fixed by their optimization flag `optCut`,
parameters removed by `DSROptRedCut` `diagCut`, the largest and smallest diagonal element of $S$, the solution component with the largest modulus `absRmax`, its
parameter index `imax`, and, after a comma, the number of CG iterations. Format `"%5d %5d %5d %5d % .5e % .5e % .5e %5d, %d"`.
The C manual notes that imaginary components of real-mode parameters are counted in `optCut`.

> **Implementation**
> - C: `fn_StochasticOptCG` (writes the CG row) — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:73`
> - C: `StochasticOpt` (writes the direct-solver row) — `extern/mVMC-1.3.0/src/mVMC/stcopt.c:33`
> - C: `InitFile` (header, opened for every optimization) — `extern/mVMC-1.3.0/src/mVMC/initfile.c:33`
> - Rust: `stochastic_opt_cg_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:101`
> - Parity (**difference**): C also writes `zvo_SRinfo.dat` for the *direct* solver (`stcopt.c:157`, columns without the iteration count); Rust writes it only in the CG path (`crates/mvmc-core/src/sr_cg.rs:208-241`, prefix from `CDataFileHead`). A direct-solver run produces no `zvo_SRinfo.dat` **(observed)**. In Rust the header line is written only if the file is new or empty.

## 9.5 Timers

`zvo_CalcTimer.dat` has one line per instrumented section: a label that includes the C timer id in brackets and the accumulated seconds
(`"%12.5f"`), for example

```text
All                         [0]      4.02645
Initialization              [1]      0.00102
  read options             [10]      0.00000
  ReadDefFile              [11]      0.00094
  SetMemory                [12]      0.00000
  InitParameter            [13]      0.00005
VMCParaOpt                  [2]      4.02351
  VMCMakeSample             [3]      1.46436
    makeInitialSample      [30]      0.00858
    make candidate         [31]      0.03867
    hopping update         [32]      0.95590
      UpdateProjCnt        [60]      0.05897
```

(51 lines for an optimization run, from [0] to the SR sub-timers **(observed)**; PhysCal uses `VMCPhysCal [2]`). `zvo_CalcTimerDiag.dat` additionally lists the diagnostic
slots (ids up to 966; uninstrumented slots read 0). The file name is **always `zvo_...`** in Rust — the timer writers are called with the literal prefix
`"zvo"` (`run.rs`, `crates/mvmc-cli/src/main.rs`), whereas C uses `CDataFileHead` (`vmcclock.c:79`). Timers are inclusive: a parent section keeps running while its children are timed.

> **Implementation**
> - C: `OutputTimerParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcclock.c:79`
> - C: `OutputTimerPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcclock.c:140`
> - Rust: `TimerEnv::from_lookup` — `crates/mvmc-core/src/c_timer.rs:174`

## 9.6 Green functions: `zvo_cisajs_NNN.dat`, `zvo_cisajscktalt_NNN.dat`, `zvo_cisajscktaltex_NNN.dat`

Written once per PhysCal sample, files are re-created (`"w"`) for each index:

- **`zvo_cisajs_NNN.dat`** — one row per `OneBodyG` entry: `i s j s' Re Im` with the C format `"%d %d %d %d % .18e  % .18e \n"`, followed by one empty line.
  The value is $\langle c^\dagger_{is}c_{js'}\rangle$.
- **`zvo_cisajscktalt_NNN.dat`** — one row per `TwoBodyG` entry: `i s j s' k t l t' Re Im` (`"%d ... % .18e % .18e\n"`), followed by an empty line.
  The value is $\langle c^\dagger_{is}c_{js'}c^\dagger_{kt}c_{lt'}\rangle$.
- **`zvo_cisajscktaltex_NNN.dat`** — a **single line** holding `Re  Im ` for each `TwoBodyGEx` entry in order (the factored product, [6.1](06-theory-observables-lanczos.md#61-green-functions)), terminated by one newline.

Examples (tutorial run, `NVMCSample = 3000`):

```text
0 0 0 0  4.156666666666666288e-01   0.000000000000000000e+00 
0 0 1 0  3.677613437662305418e-01   0.000000000000000000e+00 
0 0 0 0 0 0 0 0  4.156666666666666288e-01  0.000000000000000000e+00
0 0 0 0 0 1 0 1  1.019999999999999934e-01  0.000000000000000000e+00
```

A file is not created when its count is zero (the tutorial has no `TwoBodyGEx`, hence no `zvo_cisajscktaltex_001.dat`).

> **Implementation**
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - C: `InitFilePhysCal` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:72`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Parity: row text and the trailing blank line / single-line layout of the `ex` file are reproduced (`io.rs:244-301`, comment "C vmcmain.c:671-675 emits pairs in term order, newline after the loop").

## 9.7 Lanczos files

For `NLanczosMode > 0` ([6.2](06-theory-observables-lanczos.md#62-the-single-step-lanczos-wave-function)):

- **`zvo_ls_out_NNN.dat`** — three numbers $E_{\rm LS},\ \sigma^2_{\rm LS}/E^2_{\rm LS},\ \alpha$ with format `"% .18e  "` each and **no trailing newline** (as in C). Example: `-8.893351117533208949e+00   6.611265253057749952e-03   3.885060397405710741e-01`.
- **`zvo_ls_qqqq_NNN.dat`** — the 16 real parts of `QQQQ` (`"% .18e  "` each) and a newline.
- `NLanczosMode = 2` adds **`zvo_ls_cisajs_NNN.dat`** (rows `i s j s' Re Im`, then blank line; real mode writes the imaginary part as `0.0`), **`zvo_ls_cisajscktalt_NNN.dat`** (8 indices, `Re Im`, then blank line) and
  **`zvo_ls_cisajscktaltex_NNN.dat`** (one line of `Re Im` pairs). The `ex` file is created even when `TwoBodyGEx` is empty (it then contains one newline).

> **Implementation**
> - C: `PhysCalLanczos_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:149`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Rust: `lanczos_energy` — `crates/mvmc-core/src/lanczos.rs:61`
> - Parity: the debug-only C files `zvo_ls_qcisajsq_NNN.dat` and `zvo_ls_qcisajscktaltq_NNN.dat` (`#ifdef _DEBUG`) are not produced. On a failed $\alpha$ determination C writes nothing; Rust writes `NaN` ([6.2](06-theory-observables-lanczos.md#62-the-single-step-lanczos-wave-function)).

## 9.8 C files that Rust does not write

| C file | Content | Rust |
|--------|---------|------|
| `zvo_time_NNN.dat` | per-step sampling progress, acceptance ratios and time stamps (`OutputTime`) | not written |
| `zvo_varbin_NNN.dat` | binary variant of `zvo_var` (`-b`) | not written |
| `zvo_SRinfo.dat` for the direct solver | see [9.4](#94-solver-information-zvo_srinfodat) | not written |
| `zvo_ls_qcisajsq_*`, `zvo_ls_qcisajscktaltq_*` | `_DEBUG` builds only | not written |
| `zvo_out_NNN.dat`/`zvo_var_NNN.dat` during optimization | index suffix | written as `zvo_out.dat`/`zvo_var.dat` |

## 9.9 Post-processing: `greenr2k` (Fourier transform of the Green functions)

C ships the Fortran utility `tool/greenr2k.F90` (installed as `bin/greenr2k`). The workspace provides a pure-Rust port, the binary `greenr2k` of the crate `mvmc-greenr2k`, with the same command line, input files and output files:

```bash
cargo run --release -p mvmc-greenr2k -- namelist.def geometry.dat   # or: greenr2k namelist.def geometry.dat
```

It works in the **current directory**, like the Fortran program: `namelist.def` and `geometry.dat` are the two arguments, and the correlation files are read from `output/` (the default `--out-dir` of `mvmc` is `<namelist directory>/output`, so run `greenr2k` in the directory of the NameList file). A missing argument prints a usage message and exits with status 2.

Inputs:

- **NameList** — keywords `OneBodyG`, `TwoBodyG`, `ModPara` and `CalcMod` (case-insensitive; other keywords are skipped). Without `CalcMod` the data are taken as mVMC (`calctype = 4`); a `CalcMod` file selects the HPhi modes by its `CalcType` (0 Lanczos, 1 TPQ, 2 FullDiag, 3 LOBCG).
- **ModPara** — `NSite`, `CDataFileHead`, `NDataIdxStart`, `NDataQtySmp` (mVMC); `NumAve`, `Lanczos_max`, `ExpecInterval`, `Exct` (HPhi). The mVMC data files are `output/<CDataFileHead>_cisajs_NNN.dat` and `..._cisajscktalt_NNN.dat` for `NNN = NDataIdxStart .. NDataIdxStart+NDataQtySmp-1`, exactly the files of [9.6](#96-green-functions-zvo_cisajs_nnndat-zvo_cisajscktalt_nnndat-zvo_cisajscktaltex_nnndat) (names, `%03d` index and row layout are identical to C's).
- **Geometry** — the `geometry.dat` of C StdFace (lattice vectors, boundary phase in degrees, supercell, one `R orbital` row per site), followed by the k path (`nnode nk_line`, then `label k1 k2 k3` for each node) and the k grid `nk1 nk2 nk3` that the user appends (`echo "..." >> geometry.dat`, [`format.rst`](../../extern/mVMC-1.3.0/doc/en/source/fourier/format.rst)). **Rust does not write `geometry.dat`**: Standard-mode input expansion is not part of the Rust CLI, so use the file produced by C StdFace (`mvmc_dry.out`/`vmc.out`) or write it by hand.

Outputs (Fortran formats reproduced exactly, including the gfortran list-directed spacing):

| File | Content |
|------|---------|
| `output/<head>_corr.dat` (mVMC, Lanczos), `_corr_stepN.dat` (TPQ), `_corr_eigenN.dat` (FullDiag/LOBCG) | four header lines per orbital pair, then per k point: `k-length` and, for mVMC/TPQ, the average and standard error over `NDataQtySmp` (resp. `NumAve`) samples of the real and imaginary parts of the up-up, down-down, density, $S^zS^z$, $S^+S^-$ and $\mathbf S\cdot\mathbf S$ correlations (`E15.5`); for HPhi modes the values without errors |
| `output/<head>_corr<tail>.frmsf` | FermiSurfer file of the momentum distribution on the `nk1 x nk2 x nk3` grid (list-directed numbers) |
| `kpath.gp` | gnuplot `xtics` labels of the k nodes |
| standard output | the same progress messages as Fortran |

The Fourier convention is that of the Fortran program: $\tilde C(\mathbf k)=\frac1{N_R}\sum_{\mathbf R}\big(\frac1{n_R}\sum e^{-i\mathbf k\cdot\mathbf R+i\phi}\big)C(\mathbf R)$, where the inner sum runs over the $n_R$ equivalent nearest images of the supercell vector $\mathbf R$ with boundary phase $\phi$; the one-body parts (up-up, down-down) are not divided by $N_R$. For mVMC the $S^+S^-$ terms are rebuilt from the two-body entries with the exchange of operators described in `greenr2k.F90`.

Differences from the Fortran program (the Fortran behaviours are defects or compiler limitations and are not reproduced):

- An unquoted file name containing `/` (e.g. `./modpara.def`) is read whole; gfortran's list-directed input stops at `/` and reads `.`.
- After the error "Missing indices for the Green function." (printed after the list of missing indices) the Rust tool exits with status 1; the Fortran `STOP "..."` exits with status 0.
- `CalcType 2` (FullDiag): gfortran cannot run the input format `("  MAX DIMENSION idim_max=1", i16)` (constant string in an input format). Rust skips the 26 characters of that string as other compilers do, reads the next 16 columns and then behaves like LOBCG. There is no Fortran reference for this branch.
- Signed zeros of the reciprocal lattice vectors in the console output (`-0.0000000000` from LAPACK) are not reproduced.

Other tools in `extern/mVMC-1.3.0/tool/` are not ported; they are independent of the Rust binaries and are used unchanged with Rust outputs:

| Tool | Decision |
|------|----------|
| `wout2geom.sh` | Shell/`bc` converter from a Wannier90 `.wout` file to the lattice part of `geometry.dat` (Wannier90 mode). Not ported: it does not read any mVMC output and its result is an input of C StdFace and `greenr2k`. Use as `wout2geom.sh seed.wout > geometry.dat`. |
| `respack2wan90.py` | Python/numpy converter from RESPACK output (`dir-wan`, `dir-intW`, `dir-intJ`) to the Wannier90-mode files `seed_hr.dat`, `seed_ur.dat`, `seed_jr.dat`, `seed_geom.dat`. Not ported: pre-processing for C StdFace. Use `uv run --with numpy extern/mVMC-1.3.0/tool/respack2wan90.py [seed]` in the RESPACK directory. |
| `gen_frmsf.sh` | `awk` script that cuts FermiSurfer files out of a momentum-distribution table. Not ported: it only post-processes text columns (and expects a different column layout than the current `greenr2k` output; `greenr2k` already writes `.frmsf` files itself). |

> **Implementation**
> - C: `read_filename` — `extern/mVMC-1.3.0/tool/greenr2k.F90:95`
> - C: `read_geometry` — `extern/mVMC-1.3.0/tool/greenr2k.F90:294`
> - C: `read_corrindx` — `extern/mVMC-1.3.0/tool/greenr2k.F90:467`
> - C: `fourier_cor` — `extern/mVMC-1.3.0/tool/greenr2k.F90:782`
> - C: `output_cor` — `extern/mVMC-1.3.0/tool/greenr2k.F90:819`
> - Rust: `run` — `crates/mvmc-greenr2k/src/lib.rs:1339`
> - Rust: `fourier_cor` — `crates/mvmc-greenr2k/src/lib.rs:998`
> - Rust: `list_real` — `crates/mvmc-greenr2k/src/fortran_fmt.rs:142`
> - Parity: fixtures from the compiled Fortran tool (`tests/fixtures/greenr2k/PROVENANCE.md`); text layout exact, numbers within one unit of the printed digit plus a `1e-13` floor ([NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md)).
