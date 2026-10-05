# greenr2k fixtures (issue #351)

Expected values of `crates/mvmc-greenr2k` come from the **authoritative Fortran
tool `extern/mVMC-1.3.0/tool/greenr2k.F90`** (mVMC v1.3.0), never from Rust.

| Item | Value |
| --- | --- |
| Source | `extern/mVMC-1.3.0/tool/greenr2k.F90`, SHA-256 `cd00c3902002b755cc6434e2f37ba3aa8a8b70099ae50f0316b08daf77570f78` |
| Compiler | GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), `gfortran -O2 greenr2k.F90 -llapack -lblas` |
| LAPACK/BLAS | Ubuntu `libopenblas0-pthread` 0.3.26 (`liblapack.so.3`, `libblas.so.3` alternatives) |
| Platform | Linux 6.8 x86_64 (native Linux reference environment) |
| StdFace driver | `extern/mVMC-1.3.0/src/StdFace/src/dry.c` (SHA-256 `4993120b81953cfc155dd1ac6338e8a39af2739991f04ac047bfb17c03809941`) plus the StdFace sources, `gcc -O2 -D_mVMC -DMEXP=19937` (GCC 13.3.0) |
| Generator | `c_toolbox/greenr2k/generate_fixtures.py` (SHA-256 `9e2bb6127305bb97cf4496e0828525784ce378d0cef60e2a25e75edbd7d7b3ff`) |
| Format probe | `c_toolbox/greenr2k/format_probe.f90` (SHA-256 `127144bfda1f30a8520eb0e10640d29bd8d2071f77c619c083423584cdc58ecd`), inputs `c_toolbox/greenr2k/probe_values.txt` (SHA-256 `67bc766054926913ea6e2aff694019368229880923d511b293657a6d75533ad5`) |

## Reproduction

```bash
c_toolbox/greenr2k/build.sh /tmp/greenr2k-bin
uv run --no-project c_toolbox/greenr2k/generate_fixtures.py /tmp/greenr2k-bin tests/fixtures/greenr2k
gfortran -O2 c_toolbox/greenr2k/format_probe.f90 -o /tmp/greenr2k-bin/probe
/tmp/greenr2k-bin/probe < c_toolbox/greenr2k/probe_values.txt > tests/fixtures/greenr2k/format_probe.txt
```

Normal Rust tests only read the checked-in files; they never build or run C, Fortran or
Python.

## Cases (`<case>/inputs` are the files in the working directory, `<case>/expected` the
Fortran results: `stdout.txt`, `kpath.gp`, `output/zvo_corr*`)

| Case | Mode | Geometry / indices | Correlation values |
| --- | --- | --- | --- |
| `chain6_mvmc` | mVMC, `NDataIdxStart 7`, `NDataQtySmp 2` (average and standard error over two samples) | 6-site Hubbard chain: geometry from C StdFace; `namelist.def`, `modpara.def`, `greenone.def`, `greentwo.def` from `tests/fixtures/physcal_181/two-samples/hubbard_chain_dh_real/inputs` | `zvo_cisajs_00[78].dat`, `zvo_cisajscktalt_00[78].dat` copied from that fixture's `expected/` (independent Julia PhysCal values) |
| `honeycomb_mvmc` | mVMC, three samples, two orbitals, four R vectors, boundary phases 30/45 degrees, `nreq > 1` | C StdFace honeycomb 2x2 (`a0W=2, a1L=2`, `phase0=30`, `phase1=45`): `geometry.dat`, `greenone.def`, `greentwo.def`, `namelist.def` | synthetic (seeded `random.Random(351)`) |
| `honeycomb_lanczos` | HPhi style, `CalcType 0` | same geometry; HPhi-style index files written by the generator | synthetic (seed 7) |
| `chain6_tpq` | `CalcType 1`, `NumAve 2`, `Lanczos_max 3`, `ExpecInterval 2` (four files, average/error per TPQ step) | chain geometry, HPhi-style indices | synthetic (seed 11) |
| `chain6_lobcg` | `CalcType 3`, `Exct 2` | chain geometry, HPhi-style indices | synthetic (seed 17) |
| `chain6_missing_index` | mVMC, last `TwoBodyG` entry removed: Fortran prints the missing index and stops | as `chain6_mvmc` | as `chain6_mvmc` |

`greenr2k` is a linear post-processing tool, so synthetic correlation values exercise
the same code paths; only the file contracts matter. `CalcType 2` (FullDiag) cannot be
generated: gfortran rejects the format `("  MAX DIMENSION idim_max=1", i16)`
(`Fortran runtime error: Constant string in input format`), see
`crates/mvmc-greenr2k/tests/behavior.rs`.

`chain6_missing_index/expected/exit_code.txt` records that the Fortran program exits
with **status 0** after `STOP "Missing indices for the Green function."`.

## Tolerances

See `crates/mvmc-greenr2k/tests/fortran_parity.rs` and `docs/NUMERICAL_COMPARISONS.md`.
The text layout (line count, line length, token count) is exact. Numeric tokens
compare as `|a - e| <= max(floor, one unit of the last printed digit)` with `floor = 1e-13`
for the correlation files (sums of at most ~100 terms of magnitude <= 1, about
100 x 2.2e-16) and `0` for the console. On these fixtures the only observed difference is
the sign of zero in LAPACK's reciprocal-lattice inverse (`-0.0000000000`); the
correlation files agree textually. The same numerical bound applies on macOS, where libm
`cos`/`sin` and BLAS may differ in the last bits.

`format_probe.txt` holds gfortran's output for 596 bit patterns (zero, signed zero,
ties, boundaries 0.1 and 10^d, subnormals, Inf, NaN, random magnitudes) through
`E15.5`, `F15.10`, `F10.5`, `F7.2` and list-directed REAL(4)/REAL(8); the Rust formatters
reproduce it exactly (text, not numerics).
