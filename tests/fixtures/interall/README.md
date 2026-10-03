# InterAll input parity

The historical fixtures extend Julia's `test_parsers.jl` term/parser tests and
`test_parse_expert_mode_files.jl` orchestration tests for the general interaction
parser. They exercise all four independent site/spin pairs, asymmetric complex
coefficients, coincident indices, duplicates, input order, signed zeros, and the
strict `abs(imag(value)) > 1e-14` coefficient classification.

`parser_cases.def` covers comments, extra numeric columns, malformed indices
and coefficients, Unicode whitespace, and a declared header count different
from the actual payload. The historical Julia parser silently skips rows containing
ASCII letters, **including scientific notation**, and retains raw negative or
out-of-range site/spin integers. These are parser semantics, not valid runtime
inputs. `raw.def` verifies that files without headers retain their first five
rows. `kitaev.def` is copied from the reference's
`MVMCOptimizers.jl/test/samples/Standard/Spin/Kitaev/interall.def`.
Only trailing whitespace in the five header lines is trimmed; the Julia script
checks the copy against the reference file.

`parser.txt` records every index, real/imaginary coefficient bit, and complex
classification. `initial.txt` records component flags, initialized orbital bits,
and the next 624 SFMT words after initialization with seed 11272. Comparing a
copy with the interactions removed proves that complex Hamiltonian coefficients
do not change the wavefunction initialization mode or consume additional draws.
Repeated namelist sections replace the previous payload, matching Julia.
Missing InterAll files are recorded as required-input errors in Rust; optional
`In*.def` parameter overlays retain their existing missing-file behavior.

Generate and verify with the pinned Julia 1.13.1 workspace:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_interall_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_interall_parser_parity.jl
cargo test -p mvmc-expert-parsers --locked --test interall
cargo test -p mvmc-core --locked --test runtime_contract
```

The reference is Julia-mVMC `8bb1b9e` (parser/numerical sources `c2ea432`), with
OpenBLAS 0.3.30 ILP64 and one BLAS thread. The fixture headers record Julia's
BLAS configuration; parsing and initialization here perform no BLAS operations.
SFMT.jl wraps a global stream, so the script snapshots each initialization before
reseeding for the independent comparison.

The input-contract milestone initially rejected normal InterAll before RNG
consumption. Normal real and complex optimization now accept spin-conserving
pairs; general-orbital mode also accepts spin-changing pairs when TwoSz=-1.
As in C's `GetInfoInterAll`, a fixed TwoSz requires each pair to conserve spin,
and invalid sites are rejected before initialization. The historical permissive
Julia parser fixtures above do not establish C reader parity. They are now
constructed by `tests/support/historical_interall_model.rs` in tests only;
production uses the native C contract documented below. Their independent
Julia parser, initialization, Green and SR expectations remain unchanged.

## Native C input reader and pure Rust number conversion

`c_reader.txt` records 463 actual C reader cases (411 accepted, 52 rejected).
The optional probe extracts `ReadBuffInt`, `GetInfoInterAll` and their error/site
checks verbatim from `readdef.c`, with source hashes in the stored excerpt.
Rust compares acceptance, input order, all eight indices and both coefficient
bits using checked-in data only. No toolbox or reference runtime is required.

The second physical `fgets` chunk supplies the declared count, independent of
its printed label. Positive counts skip five header chunks and require exactly
the declared number of body chunks; zero counts ignore the body. Each chunk
holds at most 255 bytes. Comments and blank lines are rows, and unsuccessful
`sscanf` conversions retain values from the preceding row, starting from zero.
The tests cover short scans, integer prefixes, extra fields, C whitespace,
fixed/free TwoSz, long rows and missing final newlines. Invalid spin ranges and
integer overflow have separate bounded Rust diagnostics rather than unsafe
native executions.

Numeric conversion is pure Rust, including scientific/hexadecimal notation,
signed zero, infinities, NaN payloads and prefix consumption. Hexadecimal
rounding covers ties, subnormals, overflow and 128 deterministic random inputs.
The former `strtod` FFI used by parameter-record loading is removed; existing
native C record expectations also remain exact. CLI and library regression
tests compare complete output files in real and complex normal mode using
scientific InterAll coefficients, and verify count errors before output.

`c_spin_chain.def` retains all 26 original couplings and their order from
`spin_chain/interall.def`, omitting only five body comment rows that C counts
as extra terms. The complete production namelist uses this C-valid control;
the original Julia input and its 50-step trajectory goldens remain intact.

```sh
uv run --no-project python scripts/check_interall_reader_c_parity.py
cargo nextest run -p mvmc-expert-parsers --test c_interall_reader
cargo nextest run -p mvmc-cli --test runtime_contract -E 'test(normal_interall)'
```

The probe uses Apple clang 17 with `-O0 -ffp-contract=off` and overallocates
comparison storage so extra-row count errors can be observed safely. This
does not establish safe native production allocation, full C initialization,
complex Green arithmetic or sampling/SR parity. Those broader numerical
checks remain under #23; this reader comparison is not a full C runner claim.

The canonical normal-mode `calculate_hamiltonian` currently accesses
`term.sites`, while its `InterAllTerm` type contains `site0` through `site3`
instead. That reference path raises a missing-field error. The FSZ Hamiltonian
uses the four explicit pairs and the `green_func2_fsz` / `green_func2_fsz2`
dispatch. Rust follows the explicit indices and input-order loop in C's
`calham.c` and `calham_real.c`. The original Julia discrepancy is preserved as
reference provenance; the vendored source is unchanged.

## General fixed-Sz Green ratios

`green_normal.txt` records the original `green_func2` results for all 4-site
index combinations and both spins of each operator, on two configurations in
each of real and complex mode (4096 operators). The configurations cover
separate and doubly occupied sites. Every case includes two QP planes with
unequal weights, Gutzwiller and Jastrow projection, and all coincident-index
branches. Input Slater tables, Pfaffians, inverses, overlap, occupations and
projection counts are serialized alongside the outputs.

Rust compares every output bit, including signed zeros, using the supplied
inverse inputs in its padded QP layout. Julia independently applies the four
creation/annihilation operators in a sorted occupation basis, evaluates
analytic 4-by-4 Pfaffians for the new state and checks every ratio with fixed
absolute tolerance `2e-12` (relative tolerance zero). The independent formula
has a different arithmetic order from the update kernels. Equivalent
CoulombIntra, CoulombInter, Hund and Exchange sums (including same-site
Exchange) agree with the original
specialized Hamiltonian with absolute tolerance `2e-14`; Rust also compares
that Hamiltonian's energy bits. Green evaluation leaves the inverse/Pfaffian
state unchanged.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_two_body_green_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_two_body_green_parity.jl
cargo test -p mvmc-core --locked --test two_body_green
```

The general helper replaces the old exchange-only kernel. General one-body
reductions use Julia's projection-count ratio and complex quotient. The real
Transfer calculation retains its specialized direct projection/real quotient
arithmetic; applying that fast arithmetic to a general one-body reduction
produced a one-ULP mismatch in the new exhaustive test. The test was kept exact
and the call-site dispatch was corrected. These historical helper fixtures
continue to preserve Julia arithmetic. Production real InterAll uses the
separately tested C kernel below.
Same-site Exchange now reduces to `2 * J * n_up * n_down`, matching Julia,
instead of being discarded by the previous exchange-only call site.

## General FSZ Green ratios and local energy

`green_fsz.txt` records all 24,576 four-site/four-spin two-body operators and
384 one-body operators for real and complex Slater tables. Six cases cover
balanced and imbalanced spin populations, independent electron-label ordering,
double occupancy, coincident combined spin/site indices, two QP planes and
nonzero Gutzwiller/Jastrow values. Every output bit, including signed zeros,
matches the original `green_func1_fsz`/`green_func1_fsz2` and
`green_func2_fsz`/`green_func2_fsz2` kernels. Rust uses one general FSZ two-body
API and preserves the source's rightmost-hop-first order. The old exchange-only
helper is removed; FSZ Exchange uses the general kernel, including same-site
terms.

Julia independently applies every operator in a sorted Fock basis and checks
the result using analytic four-electron Pfaffians at fixed `atol=2e-12,
rtol=0`. The serialized specialized and full Hamiltonian energies include
Coulomb/Hund/Exchange, arbitrary complex InterAll coefficients, repeated terms,
spin-changing and one-body reduction cases, and an out-of-range site skipped
as in the source. Rust matches both energy bits. The independent specialized
operator expansion and its InterAll extension use `atol=2e-14, rtol=0`.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_fsz_green_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_fsz_green_parity.jl
cargo test -p mvmc-core --locked --test two_body_green
```

The source FSZ Green families read complex Pfaffian/inverse buffers even when
`all_complex=false`. These tests serialize those original buffers for both real
and complex Slater inputs; they do not claim a real FSZ production calculation
or SR dispatch. No vendored source is modified.

## Complex FSZ production gate

`spin_chain/namelist.def` uses the existing six-site General orbital layout and
local-spin input with a 26-term Hermitian InterAll Hamiltonian: alternating
`0.7 Sx_i Sx_j`, `0.5 Sy_i Sy_j`, `0.9 Sz_i Sz_j` bonds and imaginary conjugate
couplings. It exercises both spin-conserving and spin-changing terms, density
reductions, imaginary coefficients and antiperiodic orbital signs. Wavefunction
declarations and initialization select complex FSZ independently of the
Hamiltonian coefficients.

The original optimizer and numerical kernels run with only observation hooks.
`sr_cg/interall_runner`, `sr_direct/interall_runner` and
`sr_direct/interall_store_runner` record independent same-seed prefix runs at
1, 2, 3 and 50 steps for SR-CG and direct SR with both NStore settings. Rust
compares all parameter and energy bits, saved configurations/occupancies/
projection counts/spins, combined burn-in storage, attempted/accepted move
counters and the next 624 SFMT words. SR-CG's complete SR-info output also
matches. The initial boundary additionally checks component flags, initialized
parameters and the next RNG block before sampling. Each independent prefix
is seeded once; observing its RNG happens after the run and cannot change it.
`reference.txt` records Julia/BLAS versions and settings.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=interall
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=interall --store=0
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=interall --store=1
cargo test -p mvmc-core --locked interall_fsz --lib
```

The new complete-state checks exposed two existing complex FSZ differences:
Rust stored burn data in separate shadow arrays and left most of the canonical
combined buffer empty, and it omitted the attempted/accepted statistics.
Both real and complex FSZ sampling now save/restore Julia's combined order
`indices, configuration, occupancy, projection, spins`. Complex FSZ statistics
are reset and updated at the same proposal/acceptance points, including rejected
proposals. Existing real FSZ fixtures now compare the actual combined buffer
directly, strengthening the previous semantic comparison.

Normal and general-orbital InterAll are enabled in library and CLI validation
for real and complex optimization. Invalid spins and sites fail before
initialization/RNG consumption. The low-level historical Green fixtures retain
Julia's skipped-site cases; they do not override the production C input checks.
InterAll Lanczos is still rejected under #31 until its Hamiltonian-overlap path
is implemented.

## Native C normal real InterAll

`c_real_green.txt` contains 4,096 expected `GreenFunc2_real` results and four
ordered InterAll energy sums from actual extracted C function bodies. It uses
the two historical real wavefunction states above with zero and nonzero
Gutzwiller/Jastrow parameters. Projection count updates and ratios, one- and
two-electron Pfaffian updates, and the overlap sum all execute their native C
bodies. All coincident indices, both spins, unequal QP weights and both
configurations are covered. Couplings include imaginary parts, repeated
operators and cancellation-sensitive nonbinary real values. The extracted
`calham_real.c` loop uses a double accumulator and discards the imaginary
coupling contribution as in C. The driver verifies that electron buffers are
restored after every call.

Rust compares every real output bit and the complete ordered energy sum.
The real InterAll kernel uses C's scalar nested reduction, platform `exp`, and
real multiplication followed by division. Historical Julia real PairHop,
Exchange, sampling and Green helpers retain their separately labelled reduction
and quotient paths. Their original fixture bytes and tolerances are unchanged.

The optional probe runs Apple clang 17 with `-O0 -ffp-contract=off`, and stores
SHA-256 hashes and verbatim extraction boundaries in `c_toolbox/`. Its MPI
plumbing models only `MPI_COMM_SELF`; it excludes RBM and does not establish
full C initialization, sampling, complex Green or SR trajectory parity. Cargo
reads only the checked-in expected fixture and never invokes or reads the
toolbox.

```sh
uv run --no-project python scripts/check_interall_real_c_parity.py --write
uv run --no-project python scripts/check_interall_real_c_parity.py
cargo nextest run -p mvmc-core --test two_body_green -E 'test(normal_real_interall)'
cargo nextest run -p mvmc-core --cargo-profile test-fast -E 'test(normal_interall_equivalents)'
```

The optimizer equivalence gate runs three steps in real and complex normal
mode for direct SR with NStore=0/1 and SR-CG with NStore=0. Real density
InterAll terms are compared to CoulombIntra; complex pair transfers are
compared to PairHop on an itinerant Hubbard configuration. It compares every
callback's energy/parameter bits, saved and scratch configurations, projections,
combined burn buffers, move counters and all 624 following SFMT words. This is
an algebraic production-path check, alongside independent kernel fixtures;
it is not an independently generated full C trajectory.

## Native C normal complex InterAll and PairHop

`c_complex_green.txt` contains 4,096 expected complex `GreenFunc2` values,
four ordered InterAll sums and four PairHop sums. The probe executes verbatim
`GreenFunc1/2`, projection updates, one-/two-electron Pfaffian updates and
`CalculateIP_fcmp` bodies, plus the actual serial accumulator loops from
`calham.c`. Two complex wavefunction inputs each use zero and nonzero real
Gutzwiller/Jastrow parameters, with unequal QP weights. All same-/opposite-spin
operators, coincident indices and one-body reductions are covered. The driver
checks that electron buffers are restored after every call. Six bounded
historical PairHop operators cover repeated terms and density reductions; the
historical seventh out-of-range row is not executed or claimed as C-valid input.

The first comparison found 377 bit discrepancies, beginning at a one-body
reduction's quotient. Rust now uses C's platform `exp` for the projection ratio
and a pure Rust port of clang/compiler-rt's scaled complex division. Both
InterAll and complex normal PairHop use this kernel, including PairHop's
Lanczos move weight. The existing three-step direct NStore=0/1 and SR-CG
equivalence test remains exact for every parameter/energy bit, stored
configuration, burn buffer, move counter and next 624 SFMT words.

`c_complex_division.txt` independently covers 373 native compiler quotient
cases, including tiny/huge denominators, signed zero, subnormals and nonfinite
recovery, plus 256 deterministic random finite bit-pattern inputs. All
non-NaN output bits are exact; invalid arithmetic checks NaN classification
because C does not specify its payload/sign. Green and energy fixtures remain
fully bitwise, including zero signs.

The optional compiler probe performs actual C `/` operations. The saved LLVM
17 `divdc3.c` source is a separately licensed reference, not a Cargo/FFI
dependency; its original notice and source hash are retained in `c_toolbox/`,
and the Rust port's full license is in `crates/mvmc-core/LICENSE-llvm.txt`.
Binary exponent scaling is implemented in Rust rather than invoking `scalbn`.

Archived Julia PairHop energy expectations remain unchanged and are checked
using their explicit historical quotient and PairHop-before-Exchange order.
Production complex PairHop energy is checked independently against native C
values. This distinction preserves the original oracle without treating its
different arithmetic as the authoritative C contract.

```sh
uv run --no-project python scripts/check_interall_complex_c_parity.py
uv run --no-project python scripts/check_complex_division_c_parity.py
cargo nextest run -p mvmc-core --test two_body_green -E 'test(normal_complex_interall)'
cargo nextest run -p mvmc-core --lib -E 'test(scaled_complex_quotients)'
cargo nextest run -p mvmc-core --cargo-profile test-fast -E 'test(normal_interall_equivalents)'
```

These probes run Apple clang 17 with `-O0 -ffp-contract=off`, supply one-process
`MPI_COMM_SELF` plumbing, and exclude RBM. They establish these normal Green
operators and ordered energy contributions, not complete C initialization,
FSZ kernels, general Lanczos moments, actual MPI execution or full sampling/SR
trajectories. Those broader numerical checks remain separate under #23/#56.
