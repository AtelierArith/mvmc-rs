# PairHop input contract

These fixtures port and extend the canonical `test_parsers.jl` PairHop tests.
Each input row `(i,j,value)` becomes `(i,j,value)` then `(j,i,value)`, including
same-site rows and duplicates. Ordering and Float64 bits, including signed zero,
are retained. Recognized five-line headers are skipped; raw inputs keep their
first five rows. Header counts do not truncate the payload. Extra columns and
comments are ignored; scientific notation and Unicode whitespace are accepted.
Nonnegative site indices are retained without checking the lattice size.

Malformed indices/values, short rows and negative sites produce Julia's exact
line errors while retaining all accepted rows in the parser result. A failed
section does not replace an earlier successful Hamiltonian payload. Rust also
records the error for runtime rejection, preventing execution of an incomplete
requested model. Missing required files are recorded similarly.

`parser.txt` serializes original-source results and error strings. `initial.txt`
compares component flags, real orbital initialization bits and the next 624 SFMT
words against Julia with seed 11272. Removing PairHop leaves that boundary
unchanged. SFMT.jl uses a global stream; each result is captured before reseeding
the independent comparison.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_pairhop_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_pairhop_parser_parity.jl
cargo test -p mvmc-expert-parsers --locked --test pairhop
cargo test -p mvmc-core --locked --test runtime_contract
```

Reference: Julia-mVMC `8bb1b9e` (parser sources `c2ea432`), Julia 1.13.1,
OpenBLAS 0.3.30 ILP64, one BLAS thread. The fixture records this configuration;
parsing and parameter initialization perform no BLAS operations. Vendored
sources are unchanged.

This milestone implements the input contract. Library and CLI still reject
PairHop under #22 before parameter initialization or RNG consumption, including
programmatically supplied terms without a namelist. Normal/FSZ local-energy
integration, equivalence to InterAll and deterministic production comparisons
are required before enabling it or completing the issue.
