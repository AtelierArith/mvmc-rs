# InterAll input parity

These fixtures extend Julia's `test_parsers.jl` term/parser tests and
`test_parse_expert_mode_files.jl` orchestration tests for the general interaction
parser. They exercise all four independent site/spin pairs, asymmetric complex
coefficients, coincident indices, duplicates, input order, signed zeros, and the
strict `abs(imag(value)) > 1e-14` coefficient classification.

`parser_cases.def` covers comments, extra numeric columns, malformed indices
and coefficients, Unicode whitespace, and a declared header count different
from the actual payload. The canonical parser silently skips rows containing
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

The input-contract milestone initially rejected both parsed requests and
programmatically supplied nonempty `inter_all_terms` before initialization or
RNG consumption. The Green and production comparisons below now enable complex
FSZ InterAll; issue #23 remains open for the normal-mode production path.

The canonical normal-mode `calculate_hamiltonian` currently accesses
`term.sites`, while its `InterAllTerm` type contains `site0` through `site3`
instead. That reference path raises a missing-field error. The FSZ Hamiltonian
uses the four explicit pairs and the `green_func2_fsz` / `green_func2_fsz2`
dispatch. The normal-mode reference discrepancy must be resolved explicitly
before claiming a normal-mode production comparison.

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
and the call-site dispatch was corrected. The normal-mode calculation retains
its InterAll restriction pending the reference accumulator correction.
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

Complex FSZ InterAll is enabled in library and CLI validation. Invalid spins
for a term with valid sites fail before initialization/RNG consumption;
out-of-range sites retain Julia's skip rule. Normal fixed-Sz InterAll remains
rejected under #23 because of the documented reference accumulator discrepancy.
Real FSZ InterAll remains rejected under #43 pending its complete calculation/SR
path. A parsed payload or kernel fixture does not override those restrictions.
