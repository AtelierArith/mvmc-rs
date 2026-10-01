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

This milestone implements the input contract only. Both parsed requests and
programmatically supplied nonempty `inter_all_terms` remain rejected by the
optimizer before initialization or RNG consumption under issue #23. General
two-body Green ratios, local-energy integration, and deterministic production
comparisons remain required before enabling or completing that issue.

The canonical normal-mode `calculate_hamiltonian` currently accesses
`term.sites`, while its `InterAllTerm` type contains `site0` through `site3`
instead. That reference path raises a missing-field error. The FSZ Hamiltonian
uses the four explicit pairs and the `green_func2_fsz` / `green_func2_fsz2`
dispatch. The normal-mode reference discrepancy must be resolved explicitly
before claiming a normal-mode production comparison.
