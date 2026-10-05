# 7. Input files

[Contents](README.md) · Previous: [6. Observables and Lanczos](06-theory-observables-lanczos.md) · Next: [8. Running](08-running.md)

`mvmc` reads **Expert-mode** input exactly as the C program does: one list file
(`namelist.def`) names a set of definition files, each introduced by a fixed
keyword. The authoritative description of every file format is the C manual,
[`extern/mVMC-1.3.0/doc/en/source/expert.rst`](../../extern/mVMC-1.3.0/doc/en/source/expert.rst)
(Japanese: `doc/ja`). This chapter summarizes the structure, records where the
Rust implementation differs, and gives the supported/rejected matrix, which is
taken directly from `crates/mvmc-core/src/validation.rs`.

The Standard-mode front end of the C package (`vmcdry`/StdFace, the `-s` option)
is available as `mvmc -s` / `mvmc --dry-run` for the lattices listed in [7.6](#76-standard-mode-stdface);
otherwise generate Expert-mode files with the C tools, or use the committed examples under
`benchmark/hubbard_chain/inputs/` and `tests/fixtures/` (see the [tutorial](10-tutorial.md)).

## 7.1 `namelist.def`

Each non-comment line is `Keyword filename`; keywords may appear in any order, lines starting
with `#` are skipped, and keyword matching is ASCII case-insensitive (`CheckWords`).
The C keyword list is `cKWListOfFileNameList` (`extern/mVMC-1.3.0/src/mVMC/include/readdef.h:33`):

```text
ModPara LocSpin Trans CoulombIntra CoulombInter Hund PairHop Exchange Gutzwiller Jastrow DH2 DH4
{Charge,Spin,General}RBM_{HiddenLayer,PhysLayer,PhysHidden}   (9 keywords)
Orbital OrbitalAntiParallel OrbitalParallel OrbitalGeneral TransSym
InGutzwiller InJastrow InDH2 InDH4  In{Charge,Spin,General}RBM_{HiddenLayer,PhysLayer,PhysHidden}
InOrbital InOrbitalAntiParallel InOrbitalParallel InOrbitalGeneral
OneBodyG TwoBodyG TwoBodyGEx InterAll OptTrans InOptTrans BF BFRange
```

The Rust parser (`C_NAMELIST_KEYWORDS`, `crates/mvmc-expert-parsers/src/lib.rs:322`) uses the same
table and order. It additionally accepts the aliases `DoublonHolon2Site`, `DoublonHolon4Site`
(for `DH2`, `DH4`) and `QPTrans` (for `TransSym`); these are Rust/Julia extensions, not C spellings.

An example (`benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def`):

```text
ModPara  modpara.def
LocSpin  locspn.def
Trans    trans.def
CoulombIntra  coulombintra.def
OneBodyG greenone.def
TwoBodyG greentwo.def
Gutzwiller gutzwilleridx.def
Jastrow  jastrowidx.def
Orbital  orbitalidx.def
TransSym qptransidx.def
```

## 7.2 `modpara.def`

Lines are `Key value` (an optional `Key = value` is also accepted by the Rust parser; lines starting
with `-` are decoration). The parser is `parse_modpara_content`
(`crates/mvmc-expert-parsers/src/parsers/modpara.rs:20`) with the key table in `apply_param`
(`modpara.rs:45`); the C reader is `GetInfoFromModPara` (`readdef.c:1825`).

**Defaults.** Keys that are absent take the values below. The C values come from
`SetDefaultValuesModPara` (`readdef.c:1756`) and **differ from the C manual text** for several keys
(manual: `NSPGaussLeg` 8, `NVMCSample` 1000; code: 1 and 10); the code is authoritative. The Rust values are
`impl Default for ModParaParameters` (`crates/mvmc-expert-parsers/src/types.rs:233`), which follow the Julia
parser and **differ from C**. Always specify every key that matters explicitly.

| Key | Meaning | C default | Rust default |
|-----|---------|-----------|--------------|
| `CDataFileHead`, `CParaFileHead` | prefix of data (`zvo`) / parameter (`zqp`) output files | – (required line) | `zvo`, `zqp` |
| `NVMCCalMode` | 0: optimize, 1: physical quantities (the CLI requires it to agree with `--physcal`, [8.1](08-running.md#selecting-the-calculation)) | 0 | 0 |
| `NLanczosMode` | 0 none, 1 energy, 2 energy + Green functions (PhysCal only) | 0 | 0 |
| `NDataIdxStart`, `NDataQtySmp` | numbering of output files; number of PhysCal samples | 0, 1 | 0, 1 |
| `Nsite` | sites | 16 | 0 |
| C: `Nelectron` or `Ne`; Rust: `NElec` or `Nelec` | electrons per spin ($N_e$). **The two spellings are disjoint**: Rust silently ignores `Nelectron`/`Ne` (observed: $N_e=0$, then a late, unrelated error); C does not read `NElec`/`Nelec`. Prefer `Ncond`, which both read | 8 | 0 |
| `Ncond` | conduction electrons; if set, $N_e=(N_{\rm locspin}+N_{\rm cond})/2$ | -1 | -1 |
| `2Sz` | $2S_z$; -1 means "not fixed" (requires general orbitals) | -1 | -1 |
| `NSPGaussLeg`, `NSPStot` | spin projection mesh and $S$ | 1, 0 | 1, 0 |
| `NMPTrans` | $\lvert N_{\rm MP}\rvert$ translations; negative = anti-periodic; **must be nonzero** (1 = none) | 0 | 0 |
| `NSROptItrStep`, `NSROptItrSmp` | SR steps and final averaging window | 1000, step/10 | 1000, 1000 |
| `DSROptRedCut`, `DSROptStaDel`, `DSROptStepDt` | SR cutoff, diagonal shift, time step | 0.001, 0.02, 0.02 | 1e-6, 0.0, 0.01 |
| `NSRCG`, `DSROptCGTol`, `NSROptCGMaxIter` | CG switch, tolerance, max. iterations | 0, 1e-10, 0 | 0, 1e-10, 0 |
| `NStore` | store per-sample $O$ (0/1) | 1 | 1 |
| `NVMCWarmUp`, `NVMCInterval`, `NVMCSample` | burn-in, interval, samples | 10, 1, 10 | 1000, 1, 10000 |
| `NExUpdatePath` | 0 hop, 1 hop+exchange, 2 exchange (spin), 3 Kondo | 0 | 1 |
| `RndSeed` | SFMT seed; the seed of rank/group $g$ is `RndSeed + g`; negative = clock | 11272 | 11272 |
| `NSplitSize` | MPI group width ([8.4](08-running.md#84-mpi-and-grouped-execution)) | 1 | 1 |
| `Nneuron`, `NneuronCharge`, `NneuronSpin`, `NneuronGeneral`, `NBlockSize_RBMRatio` | RBM hidden neurons and block size | 0, 0, 0, 0, 200 | 0, 0, 0, 0, 200 |
| `NFileFlushInterval` | flush interval (C `-F` option) | 1 | 1 (parsed, **not used**) |
| `ComplexType` | legacy complex flag | – | 0 |
| `NOneBodyG`, `NTwoBodyG`, `NTwoBodyGEx` | counts normally taken from the Green-function files | 0 | 0 |
| `useDiagScale`, `RescaleSmat` | unsupported options of the Julia reference | – | 0 (nonzero rejected) |

`NOrbitalIdx` is not a `modpara.def` key; the number of orbital parameters is derived from the orbital file.
`NSROptFixSmp` is parsed by Rust (default 0) but is not a key of the C `GetInfoFromModPara`.

## 7.3 Definition files

All definition files have the same skeleton: a 5-line header whose second line carries the count
(`NTransfer 64`), optionally a type line (`ComplexType 0`), then a table of rows. In the C manual
the formats are:

| Keyword | Row format | C manual |
|---------|-----------|----------|
| `LocSpin` | `site flag` (flag 1 = localized spin, 0 = itinerant) | `expert.rst:637` |
| `Trans` | `i σ_i j σ_j Re(t) Im(t)` ($-t\,c^\dagger_{i\sigma_i}c_{j\sigma_j}$) | `expert.rst:721` |
| `InterAll` | `i σ_i j σ_j k σ_k l σ_l Re(I) Im(I)` | `expert.rst:837` |
| `CoulombIntra` | `i U_i` | `expert.rst:960` |
| `CoulombInter`, `Hund`, `Exchange` | `i j value` | `expert.rst:1039`, `1118`, `1278` |
| `PairHop` | `i j value` (expanded to $(i,j)$ and $(j,i)$) | `expert.rst:1197` |
| `Gutzwiller` | $N_s$ rows `site idx`, then $N_G$ rows `idx optflag` | `expert.rst:1360` |
| `Jastrow` | rows `i j idx` for site pairs, then `idx optflag` | `expert.rst:1486` |
| `DH2`, `DH4` | partner-site tables and `idx optflag` | `expert.rst:1608`, `1744` |
| `*RBM_*` | index tables and `idx optflag` | `expert.rst:1885-2285` |
| `Orbital`/`OrbitalAntiParallel` | $N_s^2$ rows `i j idx` (sign convention in the manual), then `idx optflag` | `expert.rst:2285` |
| `OrbitalParallel`, `OrbitalGeneral` | spin-resolved variants | `expert.rst:2420`, `2557` |
| `TransSym` | `NQPTrans` lines `idx Re(weight) Im(weight)`, then `idx site mapped_site sign` rows | `expert.rst:2705` |
| `OneBodyG` | `i σ j σ'` | `expert.rst:2921` |
| `TwoBodyG` | `i σ j σ' k τ l τ'` | `expert.rst:3019` |
| `TwoBodyGEx` | rows identifying pairs of one-body entries (C: `GetInfoTwoBodyGEx`) | not described in the C manual |
| `OptTrans`, `InOptTrans` | optimized-translation weights and initial values | not described in the C manual |

Notes:

- Spin indices are 0 (up) and 1 (down). Site indices start at 0. Parameter indices in the `idx optflag` tables
  start at 0 and must cover $0\ldots N-1$.
- `Trans`, `Hund`, `Exchange`, `CoulombInter`, `CoulombIntra` and `PairHop` are parsed by the typed loaders in
  `crates/mvmc-expert-parsers/src/parsers/` (`coulomb.rs`, `hund.rs`, `exchange.rs`, `pairhop.rs`, `trans.rs`); the C
  readers are `GetTransferInfo` (`readdef.c:1973`), `ReadPairHopValue` (`readdef.c:2038`), `ReadPairDValue`
  (`readdef.c:2063`) and `GetInfoInterAll` (`readdef.c:2420`). The `load_hamiltonian_definition` API
  (`crates/mvmc-expert-parsers/src/definition.rs`) loads one of these families into an `ExpertModeData`.
- The `ComplexType` line of an index file selects real (0) or complex (1) parameters of that block and
  therefore the real/complex execution mode ([3.3](03-theory-wavefunction.md#33-real-and-complex-modes)).
- The optimization flag of a parameter (`optflag`) is 1 to optimize, 0 to keep fixed. A fixed parameter whose
  initial value is not supplied is set to 0 and consumes no random number ([3.7](03-theory-wavefunction.md#37-initial-values-and-synchronization)).

## 7.4 Initial parameter values

Initial values come from three sources, applied in this order (C: `InitParameter`, `ReadInitParameter`, `ReadInputParameters`;
Rust: the "init → initial.def → In\*.def → sync" order stated at `crates/mvmc-core/src/run.rs:1358`):

1. **Random initialization** of the optimized RBM and Slater parameters ([3.7](03-theory-wavefunction.md#37-initial-values-and-synchronization)).
2. An **initial parameter file** in the format of `zqp_opt.dat` (a header of 6 floats and 3 floats per parameter; the last
   complete record wins). C passes it as the second positional argument of the driver
   (`vmcmain.c`, `ReadInitParameter`, `parameter.c:95`); Rust reads it with `read_initial_def`
   (`crates/mvmc-core/src/initial_params.rs:117`) and the CLI option `--initial-def auto|none|PATH`
   (`auto` loads a neighbouring `initial.def` if present). The number of floats must match
   $6+3\,(N_{\rm proj}+N_{\rm RBM}+N_{\rm Slater}+N_{\rm OptTrans})$ (`load_para_triples`).
3. The **`In*` files** (`InGutzwiller`, `InJastrow`, `InDH2`, `InDH4`, `In*RBM_*`, `InOrbital*`, `InOptTrans`): a header
   and rows `idx Re Im` — the same format as the per-block files `zqp_*_opt.dat` written by a run
   ([9.3](09-output-files.md#93-optimized-parameters)), so a run's output can be fed back directly.

For a fixed-parameter PhysCal run the parameter file given to `--physcal` is read with `read_opt_para_file`
(`crates/mvmc-core/src/initial_params.rs:141`).

## 7.5 Supported and rejected inputs

The checks below are in `crates/mvmc-core/src/validation.rs` (commit `37348eac`). They run before any
initialization or output; a rejected input exits with `error: ...` and non-zero status.

### Namelist keywords (`validate_para_opt`, `crates/mvmc-core/src/validation.rs:162-214`)

| Keywords | Parameter optimization | PhysCal |
|----------|-----------------------|---------|
| `ModPara`, `LocSpin`, `Trans`, `CoulombIntra`, `CoulombInter`, `Hund`, `Exchange`, `Gutzwiller`, `Jastrow`, `Orbital`, `OrbitalAntiParallel`, `OrbitalParallel`, `OrbitalGeneral`, `OneBodyG`, `TwoBodyG`, `TransSym` (alias `QPTrans`) | accepted | accepted |
| `PairHop`, `InterAll`, `DH2` (`DoublonHolon2Site`), `DH4` (`DoublonHolon4Site`), `OptTrans`, `InOptTrans`, all `{Charge,Spin,General}RBM_*`, `In{Gutzwiller,Jastrow,Orbital,OrbitalAntiParallel,OrbitalParallel,OrbitalGeneral,DH2,DH4}` (and aliases `InDoublonHolon*`), `In{Charge,Spin,General}RBM_*` | accepted | accepted |
| `TwoBodyGEx` | accepted and ignored | accepted **(observed)** |
| any other `In…` keyword | rejected ("unsupported namelist section") | not checked |
| `SpinJastrow` | rejected ("projection layout would be wrong") | not checked |
| `BF`, `BFRange` (BackFlow) and any unknown keyword | rejected ("unsupported namelist section …") | **not checked: a `BF` entry is accepted and BackFlow is not applied (observed on current `main`)** |

The PhysCal row follows from `validate_phys_cal` (`crates/mvmc-core/src/validation.rs:244`), which does not repeat the keyword
check; this asymmetry was observed with a release build and is listed in [11.5](11-compatibility.md#115-open-observations).

### ModPara values (`validate_supported_modpara`, `crates/mvmc-core/src/validation.rs:77-105`; both entry points)

| Condition | Result |
|-----------|--------|
| `NMPTrans == 0` | rejected (use 1 for "no translation projection", the C contract) |
| `NSplitSize < 1` | rejected |
| `NLanczosMode` not in 0..2 | rejected |
| `NSRCG >= 2` | rejected ("not supported by Julia-mVMC") |
| `useDiagScale != 0` | rejected |
| `RescaleSmat != 0` | rejected |

### Parameter optimization (`validate_para_opt`, `crates/mvmc-core/src/validation.rs:108-241`)

| Condition | Result |
|-----------|--------|
| `NLanczosMode > 0` | rejected ("use PhysCal") |
| `NVMCCalMode != 0` | rejected for the optimization entry point ("cannot run parameter optimization; use fixed-parameter PhysCal"); PhysCal is selected with `--physcal` and `NVMCCalMode 1` ([8.1](08-running.md#selecting-the-calculation)) |
| `InterAll` site outside $0\ldots N_s-1$, spin not in {0,1}, or (normal orbitals or fixed `2Sz`) with a spin-changing pair ($\sigma_1\ne\sigma_2$ or $\sigma_3\ne\sigma_4$) | rejected |
| `NSite`, `NElec`, `NVMCWarmUp`, `NSROptItrStep` negative; `NVMCSample` or `NVMCInterval` $\le0$ | rejected |
| incomplete Expert input (`input_errors` non-empty) | rejected (also in PhysCal) |

### Physical quantities (`validate_phys_cal`, `crates/mvmc-core/src/validation.rs:244-291`)

| Condition | Result |
|-----------|--------|
| `NLanczosMode > 0` with general (FSZ) orbitals | rejected |
| `NLanczosMode > 0` with a spin-changing `Trans` row | rejected |
| `NLanczosMode > 0` with any `InterAll` term | rejected |
| `NLanczosMode = 2` without `TwoBodyGEx` and with duplicate `OneBodyG` entries | rejected |

### Grouped execution (`validate_grouped_runtime`, `crates/mvmc-core/src/validation.rs:23-44`; applies when `NSplitSize > 1`)

| Condition | Result |
|-----------|--------|
| Optimization with `NSRCG != 0` (CG) | rejected: undefined in C (`vmccal.c:241,248` writes the stored `O` at the global sample slot, `vmccal.c:314-318` reads it from column 0 with the local count) |
| PhysCal / optimization with general (FSZ) orbitals, any `NQPFull` | accepted (C defines it; checked against native C at 2 and 4 ranks, #349) |
| `NLanczosMode > 0` PhysCal | accepted (C defines it) |
| `NQPOptTrans > 1` / `OptTrans` | accepted (C defines it) |
| `NSplitSize > 1` but the reducer is not a group communicator (`validate_reducer_rank`, `crates/mvmc-core/src/validation.rs:47-66`) | rejected ("requires an MPI group communicator") |

### Command-line level (`crates/mvmc-core/src/run.rs`, `crates/mvmc-cli/src/main.rs`)

| Condition | Result |
|-----------|--------|
| `--physcal` with `NVMCCalMode = 0`, `NVMCCalMode = 1` without `--physcal`, or `NVMCCalMode` other than 0/1 | rejected by `select_calculation` (`crates/mvmc-cli/src/main.rs:429`) right after parsing, before any initialization or output |
| `--nsteps <= 0` and not PhysCal (also `NSROptItrStep = 0`) | rejected ("nothing to run") |
| window `nsmp > nsteps` | rejected (`validate_optimization_window`, `run.rs:1306`) |
| `--physcal-trace` without a serial `--physcal` run | rejected |

### Behaviour of the C program that Rust does not provide

Back-flow (`BF`) and `InterAll` Lanczos are not available (see [11.3](11-compatibility.md#113-known-differences-from-the-c-reference)).

## 7.6 Standard mode (StdFace)

`mvmc -s <stan.in>` ports the C `-s` option: the Standard-mode keyword file is read by the pure-Rust
`mvmc-stdface` crate (a port of `extern/mVMC-1.3.0/src/StdFace/src`, mVMC solver branches only), the Expert
files (`modpara.def`, `namelist.def`, `locspn.def`, `trans.def`, interaction files, `gutzwilleridx.def`,
`jastrowidx.def`, `orbitalidx*.def`, `qptransidx.def`, `greenone.def`, `greentwo.def`, `geometry.dat`,
`lattice.gp`) are written into `--out-dir` (default: the current directory, as in C), and the generated
`namelist.def` is then run. `mvmc --dry-run <stan.in>` only generates the files (C `vmcdry.out`). The C text
that goes to standard output is reproduced, and an error such as an unknown keyword exits with status 255 like
C (`StdFace_exit(-1)`).

The generated files and the console output are byte-identical to the C program for the checked-in fixtures (except for the corrected C defects listed below)
(`tests/fixtures/stdface/`, provenance in `tests/fixtures/stdface/PROVENANCE.md`), including the sign of
zero in `-0.000000000000000`. Keyword parsing follows C (`fgets` chunks of 255 bytes, `TrimSpaceQuote`,
`strtok` on `=`, `sscanf` for numbers, duplicate detection), so for example `2S`, `method` and
`CDataFileHead` are rejected because the mVMC build of StdFace does not know them.

| Lattice (`lattice =`) | Status |
|-----------------------|--------|
| `chain` | supported |
| `ladder` | supported (see the C defect below) |
| `tetragonal`/`square`, `triangular`, `honeycomb`, `kagome` | supported |
| orthorhombic/cubic, face-centered orthorhombic/cubic (fcc), pyrochlore | supported, corrected C defects ([7.6.1](#761-three-dimensional-lattices)) |
| `wannier90` | not yet ported (the run stops with a message) |

Differences from the C program (corrected defects, issue #404). The C StdFace has clear defects in its
lattice routines that the port does **not** reproduce; the C output is kept as the historical fixture
(`tests/fixtures/stdface/<case>/expected/`) and the corrected output is checked separately
(`expected_fixed/`, produced by a patched C build, see `tests/fixtures/stdface/README.md`):

- Ladder: C rejects the isotropic `t`, `V`, `J` (`NotUsed` checks that precede the reads), so a ladder needed
  `t0, t1, t2, t1', t2'` and the Kondo coupling `J` could not be set. They are accepted for the Hubbard and
  Kondo models and rejected for the spin model; `t'`, `V'`, `J'` are rejected everywhere.
- Ladder: C replaces the printed `Wx` by the number of legs (ignoring `a`, `Wx`, `Wy`); the cell vector is now
  the number of legs times the printed `Wx`/`Wy`. All nine `a0W..a2H` entries are rejected, and the spin and
  Hubbard branches use the same `J1'`/`J2'` names in messages.
- Chain, triangular, honeycomb: `J''` messages say `J0''`, `J1''`, `J2''` instead of `J0'`, `J1'`, `J2'`.
- Triangular: `t''` is checked for a spin model (C tested `t'`); `J'`, `J''` are rejected for Hubbard/Kondo.
- Honeycomb: `t''`, `t0''..t2''`, `V''`, `V0''..V2''` are rejected for a spin model, and the complete `J` family
  is rejected for Hubbard/Kondo.
- Kagome: the duplicated `t0` check is removed, and `t''`, `V''`, `J''` (and their bond-specific forms), which
  the kagome Hamiltonian does not use, are rejected instead of being silently ignored.
- Square and honeycomb: the second `V'` console line is not printed.

### 7.6.1 Three-dimensional lattices

`lattice = orthorhombic` (aliases `simpleorthorhombic`, `cubic`, `simplecubic`), `lattice = fcorthorhombic`
(`face-centeredorthorhombic`, `fco`, `face-centeredcubic`, `fccubic`, `fcc`) and `lattice = pyrochlore` (four sites
per cell) port `Orthorhombic.c`, `FCOrtho.c` and `Pyrochlore.c`. The size is `L`, `W`, `Height` (a cuboid of cells) or the
super-cell matrix `a0W` ... `a2H` (not both), the cell edges are `a`, `Wlength`, `Llength`, `Hlength` or the vectors
`Wx` ... `Hz`, and `phase0`, `phase1`, `phase2` are the boundary phases in degrees. Output additionally contains
`lattice.xsf` (XCrySDen; the file is empty if an input check stops the run, as in C).

| Lattice | Bonds | Hubbard/Kondo parameters | Spin parameters |
|---------|-------|--------------------------|-----------------|
| orthorhombic | `t0`, `t1`, `t2` along W, L, H; `t0'`, `t1'`, `t2'` face diagonals (L-H, H-W, W-L); `t''` body diagonals | `t`, `t'` set the defaults of the three components, `t''` and `V''` are single values | `J0`..`J2`, `J0'`..`J2'`, `J''` (and the isotropic `J`, `J'`) |
| fcc / fco | `t0`, `t1`, `t2` (two bonds each, the six nearest neighbours), `t0'`, `t1'`, `t2'` (along W, L, H: second neighbours) | `t`, `t'` as above | `J0`..`J2`, `J0'`..`J2'` |
| pyrochlore | `t0`, `t1`, `t2` and `t0'`, `t1'`, `t2'` are the six bond directions of the tetrahedra (nearest neighbours only) | `t`, `V` only (`t0'` etc. default to `t`) | `J0`..`J2`, `J0'`..`J2'` (default to `J`) |

**C defects that are not reproduced.** The 3D C sources contain clear errors; the Rust port implements the
correct behaviour, keeps the unmodified C output as `c_historical/` in the fixtures and takes the expectation from a
copy of the C source with `c_toolbox/stdface/3d_defects.patch` applied (`tests/fixtures/stdface/README_3d_defects.md`):

* pyrochlore Kondo lattice: C couples every localized spin of a cell to conduction site 3 (`isite + 3`) and leaves
  the conduction sites 0-2 without Kondo coupling; each conduction site now couples to its own localized spin;
* face-centered orthorhombic Kondo lattice: the localized spin ignored `h`, `Gamma`, `Gamma_y` (the other lattices
  apply them); they are applied now;
* unused parameters were accepted silently and are rejected now with the usual `... is not used` error: `V2`, `V0'`,
  `V1'`, `V2'`, `V''` in the spin model; `J'`, `J''` in the Hubbard/Kondo model of the orthorhombic and pyrochlore
  lattices; `t''`, `V''`, `J0''`..`J2''` (and in the spin model `J''`, `J0''`..`J2''`, which C read and dropped) on
  the face-centered lattice; `t'`, `t''`, `V'`, `V''`, `J'` (and in the spin model `J'`, `J''`) on the pyrochlore lattice;
* `ntransMax` omits the `Gamma_y` terms of the itinerant-electron local term (six per site, four counted):
  C overflows the heap (pyrochlore with `Gamma` and `Gamma_y` exits with SIGSEGV); the port needs no fixed bound.
