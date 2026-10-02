# C reference toolbox

Reusable comparison programs and verbatim source excerpts from the authoritative
`extern/mVMC-1.3.0/` snapshot live here. They are optional tools for investigating
contracts and generating expected values. Cargo builds and Rust tests do not
compile, invoke or read this directory. Checked-in expectations live separately
under `tests/fixtures/`; normal Rust tests retain their usual Rust toolchain,
linker and BLAS/LAPACK requirements. Historical Julia input-data dependencies
are unchanged, and no Julia runtime is needed for these C-derived Rust checks.

| Program | What it checks | Reproduction command |
| --- | --- | --- |
| `projection_count.c` | C count/sign conversion and QP kernels, 10 cases | `python3 scripts/check_projection_count_c_parity.py` |
| `orbital_order.c` | Filename registry, fixed keyword order and complete AP/P readers, 480 cases | `python3 scripts/check_orbital_order_c_parity.py` |
| `orbital_contracts.c` | AP/P physical headers, mapping/flag counts and row-order real flags, 90 cases | `python3 scripts/check_orbital_contracts_c_parity.py` |
| `general_orbital.c` | General spin-coordinate reader, counts/signs/flags, 184 cases; actual FSZ Slater and derivative kernels, 104 cases | `python3 scripts/check_general_orbital_c_parity.py` |
| `orbital_initialization.c` | Declared Slater initialization, loading, normalization, native SFMT and shared coefficient matrix, 49 cases | `python3 scripts/check_orbital_initialization_c_parity.py` |
| `initial_records.c` | Successive complete records, final values, C scalar/complex conversion and unchanged native SFMT, 35 cases | `python3 scripts/check_initial_records_c_parity.py` |
| `rbm_header.c` | Declared width 97 with complete flags and sparse mappings | `python3 scripts/check_c_reader_audits.py` |
| `opttrans_activation.c` | Explicit enabled/disabled state and defined flag writes | `python3 scripts/check_c_reader_audits.py` |
| `orbital_flags.c` | Row-order flags and aggregate complex flag value 2 | `python3 scripts/check_c_reader_audits.py` |

The `.inc` files contain extracted function bodies or specifically delimited
reader blocks. Each records the original copyright/license notice and SHA-256
of its upstream source. The scripts verify every stored excerpt against C before
compiling it. Function extraction uses balanced braces (the General extractor
ignores braces in comments and string literals); keyword-loop extraction
ends before the upstream status print, and the AP header switch block ends at
`KWOrbitalGeneral`. Driver files hold only the comparison environment and cases.
The canonical C files are not modified. `.gitattributes` preserves trailing
whitespace in these verbatim `.inc` excerpts; driver files follow normal checks.

The scripts build into temporary directories with Apple clang 17 (`cc`) on
Intel macOS. All use `-O0`; General, projection, initialization and initial-record checks use
`-ffp-contract=off`, and both parameter checks also use `-DMEXP=19937` and the
original `src/sfmt/SFMT.c`. The other reader programs need no BLAS, SFMT or MPI library.
The fixed-width RBM/header and flag audits also work directly, for example:

```sh
cc -O0 c_toolbox/orbital_flags.c -o /tmp/mvmc-orbital-flags
/tmp/mvmc-orbital-flags
```

The default commands verify existing excerpts and fixture contents. Add `--write`
explicitly to refresh extracted source and, for the parity generators, expected
fixtures; review both changes together. Writes preserve files whose contents
have not changed. Generated executables and scratch inputs stay outside Git.

These are standalone function checks, not a build or Monte Carlo run of the full
C project. Drivers supply the dimensions, flags, arrays and disabled correlation
shift plumbing needed for each check; no MPI execution is exercised. The flag
audits use sentinels solely to identify defined writes, not to infer native
initial values for untouched malloc storage. Legacy Julia RBM inputs are not
claimed to satisfy C's complete-row contract. Native C all-zero Slater
normalization and Rust's current zero guard remain a separate #46 discrepancy;
the recorded all-zero normalized rows are not asserted as Rust parity. Reader
audit differences remain tracked by #21/#26/#27 and are not completion claims.

The General probe extracts `GetInfoOrbitalGeneral` from `readdef.c` and both
`UpdateSlaterElm_fsz` and `SlaterElmDiff_fsz` from `slater_fsz.c`. It reuses the
stored common header/flag/site-check functions, supplies two translations and
one identity OptTrans, and serializes fixed binary coefficients, Pfaffians,
inverse matrices and overlap. Matrix allocation is explicitly zeroed, including
untouched diagonals and cells omitted by duplicate mappings: those values are
probe plumbing, not evidence of upstream malloc initialization. Normal and
antiperiodic reader signs, all spin blocks, declared unmapped slots and optional
index/sign carry are checked. Unsafe incomplete coordinate scans and invalid
spins/parameter indices are excluded from C execution; Rust diagnostics have
separate tests. This probe does not establish multi-OptTrans derivative parity
or the C CLI's activation policy (#27).
