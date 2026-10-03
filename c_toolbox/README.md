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
| `orbital_contracts.c` | AP/P physical headers, mapping/flag counts and row-order real flags, 92 cases | `python3 scripts/check_orbital_contracts_c_parity.py` |
| `general_orbital.c` | General spin-coordinate reader, counts/signs/flags, 184 cases; actual FSZ Slater and derivative kernels, 104 cases | `python3 scripts/check_general_orbital_c_parity.py` |
| `orbital_initialization.c` | Declared Slater initialization, loading, normalization, native SFMT and shared coefficient matrix, 49 cases | `python3 scripts/check_orbital_initialization_c_parity.py` |
| `integer_flags.c` | Production AP/P header normalization, raw integer flags, coefficient bits, native SFMT and SR selection, 70 cases | `python3 scripts/check_integer_flags_c_parity.py` |
| `projection_flags.c` | Native DH2/DH4 raw-flag readers, 24 cases; real gauge eligibility, 86 cases | `python3 scripts/check_projection_flags_c_parity.py` |
| `gutzwiller_contracts.c` | Physical headers, complete site mappings and raw ordered flags, 144 cases | `python3 scripts/check_gutzwiller_contracts_c_parity.py` |
| `jastrow_contracts.c` | Directional mappings, physical headers and raw ordered flags, 144 cases; 1,795 native projection workloads | `python3 scripts/check_jastrow_contracts_c_parity.py` |
| `initial_records.c` | Successive complete records, final values, C scalar/complex conversion and unchanged native SFMT, 35 cases | `python3 scripts/check_initial_records_c_parity.py` |
| `interall_real.c` | Actual real Green/Pfaffian/projection/overlap kernels, 4,096 operators and four ordered InterAll sums; MPI_COMM_SELF plumbing, no RBM | `uv run --no-project python scripts/check_interall_real_c_parity.py` |
| `interall_reader.c` | Native physical headers, exact counts, partial scan carry, sites/TwoSz and numeric prefixes, 463 cases (411 accepted / 52 rejected) | `uv run --no-project python scripts/check_interall_reader_c_parity.py` |
| `rbm_header.c` | Declared width 97 with complete flags and sparse mappings | `python3 scripts/check_c_reader_audits.py` |
| `opttrans_activation.c` | Explicit enabled/disabled state and defined flag writes | `python3 scripts/check_c_reader_audits.py` |
| `orbital_flags.c` | Function-level row-order flags with supplied complex argument 2 (production normalizes orbital headers to 1) | `python3 scripts/check_c_reader_audits.py` |

The `.inc` files contain extracted function bodies or specifically delimited
reader blocks. Each records the original copyright/license notice and SHA-256
of its upstream source. The scripts verify every stored excerpt against C before
compiling it. Function extraction uses balanced braces (the General extractor
ignores braces in comments and string literals); keyword-loop extraction
ends before the upstream status print, and the AP header switch block ends at
`KWOrbitalGeneral`. Driver files hold only the comparison environment and cases.
The canonical C files are not modified. `.gitattributes` preserves trailing
whitespace in these verbatim `.inc` excerpts; driver files follow normal checks.

The InterAll reader driver overallocates bounded comparison storage to observe
extra-row count errors without overrunning the declared production allocation.
Unsafe integer overflow and invalid spin ranges are excluded from native
execution and tested separately as Rust diagnostics. It establishes reader
values and acceptance for the documented inputs, not full executable or
sampling parity. See `tests/fixtures/interall/README.md` for the preserved
historical Julia models and the C-valid spin-chain control.

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

The integer-flag probe extracts the actual rank-zero header-normalization block
from `readdef.c`, `InitParameter` from `parameter.c`, and the selection block from
`stcopt.c`. It checks raw flags -2/0/1/2/3, ignored printed labels, real/complex
AP with and without P, positive nonbinary AP headers, both orbital headers set,
and seeds 1/11272. Raw real flags greater than zero consume initialization draws;
SR selects component flags equal to 1. Header sums are collapsed to 1 before
orbital readers; the old argument-2 reader audit never proved production
aggregate-2 SR behavior. P imaginary flags receive this normalized header even
when their real flag is zero or negative. AP/General complex imaginary flags
copy the raw real flag. Real-mode imaginary storage is explicitly zeroed in the
probe and Rust; C leaves some cells untouched, so this is not a malloc claim.

The projection-flag probe extracts actual `GetInfoDH2`, `GetInfoDH4`,
`CheckQuadSite` and `SetFlagShift`. It supplies complete, bounded neighbor rows,
raw flags -2/-1/0/1/2/3 and a nonzero parameter offset. Gauge cases vary each
real flag and include missing-factor and nonbinary imaginary cases. Rust tests
check the parsed DH arrays/flags and which factor gauges change. These establish
gauge eligibility, not C gauge arithmetic or a full SR solver comparison. The
Rust direct/CG solvers use separate controlled identity-covariance workloads to
check exactly the component selection supplied by C. Remaining strict family
readers, RBM widths/storage and OptTrans activation/offsets stay in #21/#26/#27;
this milestone does not close those issues or prove full C sampling parity.

The Gutzwiller probe extracts `ReadBuffIntCmpFlg`, `GetInfoGutzwiller`,
`GetInfoOpt`, `CheckSite` and `ReadDefFileError` from `readdef.c`. It verifies
144 cases (72 accepted / 72 rejected) at one, two, three and six sites with
real and positive complex headers, unused declared slots, shared/reordered/
duplicate site mappings, ignored/reversed printed flag labels, raw flags
-2/-1/0/1/2/3, folded/split C whitespace, nonstandard header labels and missing/
extra mapping or flag pairs. The original reader treats whitespace uniformly;
it does not infer the flag boundary from repeated sites or line widths.

All probe storage is zeroed. Duplicate mappings can leave sites untouched in C;
those cells and real-mode imaginary flags are sentinels, not native malloc
initialization claims. Rust deliberately keeps such unused storage deterministic.
Bad site indices are excluded from native execution because C writes the site
array before checking bounds. Malformed integer scans and out-of-range parameter
indices also have separate bounded Rust diagnostics. These checks establish this
reader's supported contract, not full C validation, initialization, MPI or
sampling parity. RBM complete readers remain separate work in #21/#26.

The Jastrow probe extracts the actual header, `GetInfoJastrow`, `GetInfoOpt`
and site-check functions. Its 144 cases cover directed pairs, count errors,
raw signed flags and a nonzero Gutzwiller offset. `jastrow_projection.c` adds
actual `MakeProjCnt` and `UpdateProjCnt` bodies from `projection.c`, supplying
two Gutzwiller slots, disabled DH factors and eight occupancy patterns. All
1,795 initial/count-update workloads across 60 complete tables are serialized
as checked-in Rust expectations; legal moves in both directions are included.
The reader preserves asymmetric entries, while C's projection kernels always
look up the upper-triangle index. Duplicate tables with unwritten cells are
excluded from kernel execution. Unwritten index=-1 and imaginary=0 sentinels
are documented, and neither driver establishes native malloc initialization,
full sampling, MPI, global complex-header or OptTrans behavior. The script
also reproduces the complete three-site historical replacement without editing
its three original input files. Details: `tests/fixtures/jastrow/README.md`.

The RBM readers preserve the actual header, nine geometry readers, flag and
site-check functions from `readdef.c`. `rbm_parameters.c` adds actual
`InitParameter`/`ReadInitParameter` and the upstream SFMT translation unit.
The reader and parameter checks cover 1,307 inputs and 84 real-initialization/
full-record workloads respectively. Full declared storage includes unused
slots; the 97-slot case is not shrunk to its maximum mapped index. Initialization
uses raw signed neuron totals including zero and negative divisors. Following
Slater values and 624 native SFMT words are serialized with every case.

`rbm_counters.c` adds verbatim `MakeRBMCnt` and `UpdateRBMCnt` from `rbm.c`.
Its 66 completely assigned tables cover 4,994 full-counter/legal-hop/no-op
workloads and verify both separate-output and in-place C updates. Coefficients
are supplied as binary64 bits, with binary and cancellation-sensitive values.
The counters distinguish C's separate coupling sum from adding each coupling
directly to the hidden bias. They also distinguish subtract-then-add hopping
arithmetic from mapping-row traversal. Total neuron count equals the sum of
family dimensions; extra base neurons and FSZ are outside this counter oracle.

```sh
uv run --no-project python scripts/check_rbm_contracts_c_parity.py
uv run --no-project python scripts/check_rbm_parameters_c_parity.py
uv run --no-project python scripts/check_rbm_counters_c_parity.py
```

Each command independently verifies its extracted reader dependency; add
`--write` to regenerate. Counter and parameter commands use
`-O0 -ffp-contract=off`; the initializer also uses `-DMEXP=19937` and native
SFMT. The accompanying Rust tests use checked-in data only. These probes do
not establish complex transcendental arithmetic, parameter normalization,
full executable/MPI execution or production sampling trajectories. The older
Julia RBM definitions and numerical fixture bytes remain historical evidence.
