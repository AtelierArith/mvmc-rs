# StdFace fixtures

Provenance (sources, compiler, flags, case list): [PROVENANCE.md](PROVENANCE.md).

Each case directory holds `StdFace.def` (the input) and

- `expected/`: the output of the **unmodified C** StdFace (`mvmc_dry.out` built from
  `extern/mVMC-1.3.0`): every generated file, `stdout.txt` and `exit_status`. This is kept as the
  historical C reference and is never edited by hand.
- `expected_fixed/` (only where the two differ): the output of the **corrected** build, i.e. the C
  sources with `c_toolbox/stdface/lattice_defects.patch` applied (`build_reference.sh --fixed`;
  the patch is recorded with its SHA-256 in PROVENANCE.md). This is the independent expectation of
  the Rust port, whose policy (issue #404) is to correct clear C defects rather than reproduce them.

`crates/mvmc-stdface/tests/stdface_c_fixtures.rs` compares the Rust output byte for byte with
`expected_fixed/` when it exists and with `expected/` otherwise. Every other case is therefore still
byte-identical to the C program.

## C defects corrected (Julia-mVMC#66, issue #404)

| Where | C behaviour | Corrected behaviour |
| --- | --- | --- |
| `Ladder.c` | `NotUsed` on `t`, `t'`, `V`, `V'`, `J`, `J'` before they are read: the isotropic `t`, `V`, `J` abort, the Kondo `J` can never be set | `t'`, `V'`, `J'` always rejected; `t`, `V`, `J` rejected only for the spin model |
| `Ladder.c` | `direct[0][0] = NsiteUC` overwrites the printed `Wx` (and ignores `a`, `Wy`) | `direct[0][0..1] *= NsiteUC` (identical for the defaults) |
| `Ladder.c` | `NotUsed_J` names `"J1p"`/`"J2p"` in the Hubbard branch; only `a0W,a0L,a1W,a1L` checked | names `"J1'"`/`"J2'"`; all nine `a0W..a2H` checked |
| `ChainLattice.c`, `TriangularLattice.c`, `HoneycombLattice.c` | `J''` calls of `InputSpinNN` labelled `J0'`, `J1'`, `J2'` | labelled `J0''`, `J1''`, `J2''` |
| `TriangularLattice.c` | spin branch `NotUsed_c("t''", tp)` | tests `tpp`; Hubbard/Kondo also reject `J'`, `J''` |
| `HoneycombLattice.c` | no `t''`/`V''` check for the spin model; Hubbard branch checks only `J0,J1,J2,J'` | `t''`, `t0''..t2''`, `V''`, `V0''..V2''` rejected for spin; full `J` family rejected for Hubbard/Kondo |
| `Kagome.c` | duplicate `NotUsed_c("t0", t)`; `t''`, `V''`, `J''` silently ignored | duplicate removed; `t''`, `V''`, `J''` family rejected |
| `StdFace_ModelUtil.c` (every lattice) | each lattice reserves 4 on-site transfers per site (`ntransMax`) but `HubbardLocal` appends 6 when `Gamma` and `Gamma_y` are both non-zero: heap overflow, SIGSEGV for the `gc_all_terms_*` cases (historical `exit_status` -11) | `MallocInteractions` reserves twice as much; the Rust lists grow on demand, so the port needed no change |
| `SquareLattice.c`, `HoneycombLattice.c` | extra `PrintVal_d("V'")` console line | removed |
| `Wannier90.c` | a missing `_hr`/`_ur`/`_jr` file prints "Skip to read" but leaves `tUJ`/`tUJindx` unset and later frees them: SIGSEGV / `free(): invalid pointer` (historical `exit_status` -11 / -6) | the arrays start as `NULL`: no terms, normal run |
| `Wannier90.c` | `PrintUHFinitial`: `IniGuess` is `malloc`ed and partly read uninitialised | zero-initialised |
| `Wannier90.c` | spin model: `Uspin` uninitialised when `_ur` lacks an on-site U | error exit (no C fixture: undefined in C) |

Not corrected (documented only): in `Wannier90.c` the test `cutoff_Rvec[0][0] != NaN_i` compares a
`double` with an integer marker and is always true, so `cutoff_tR`, `cutoff_UR`, `cutoff_JR` are printed
but never used (the box-based `cutoff_*Vec` test always applies); the intended semantics are ambiguous.
Out-of-bounds reads that depend on file contents (more bands than Wannier centres, density-matrix `R`
range smaller than the `hr` range) are undefined in C and read 0 / are ignored in Rust; no fixture
exercises them.

Cases that differ (`expected_fixed/` present) are listed in PROVENANCE.md; cases added for the
corrections are in `c_toolbox/stdface/cases_defects.py`.
