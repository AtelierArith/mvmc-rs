# PairHop input, energy and production parity

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

Normal and FSZ local energy accumulate the directed terms in source order,
before Exchange. `energy_normal.txt` and `energy_fsz.txt` reuse the independently
verified InterAll Green states and inverse bits, including real and complex
matrices, reordered FSZ electron labels and unequal spin populations. The
Hamiltonian input covers duplicates, self-pairs, invalid sites and nonzero
off-diagonal pair transfers. Pure and combined energies match the original
Julia Hamiltonian bits. Independent Fock-space signs and analytic four-electron
Pfaffians use `atol=2e-12, rtol=0`; corresponding InterAll operators agree at
`atol=2e-14, rtol=0`. These tolerances apply only to the independent numerical
identities; source energy, sampling and RNG comparisons are exact.

The canonical normal InterAll accumulator accesses nonexistent `term.sites`
fields. Its equivalence check therefore evaluates the parsed InterAll operators
through the original `green_func2`, as documented in the script. FSZ equivalence
uses the original InterAll Hamiltonian directly. Rust now tests complex normal
equivalence through its production InterAll accumulator. The historical real
equivalence retains Julia's Green quotient; native real InterAll has separate
C bitwise fixtures documented in `../interall/README.md`. Both real and complex
normal/general-orbital optimization are enabled.

Canonical `hubbard_chain_pairhop_real` and `hubbard_chain_pairhop_fsz` inputs
run through the library and CLI. The original Julia `pairhop_equivalent.jl`
and the Rust port pass the canonical one-step C checks without changing their
tolerances. Same-seed Julia 1.13.1 fixtures under `sr_cg/pairhop_*_runner` and
`sr_direct/pairhop_*_{runner,store_runner}` compare initialization flags,
parameter bits, energy bits, saved configurations, occupations, projections,
FSZ spins, burn buffers, counters and all 624 subsequent SFMT words at
1, 2, 3 and 50 optimization steps. Direct SR covers NStore=0/1; SR-CG covers
its supported NStore=0 path. The canonical sample counts (100 real, 2000 FSZ)
and all proposal/burn settings are retained. Fixed-input Gram matrices, SR
matrix/gradient, Cholesky factor and solution also match Julia bits.

The larger FSZ direct-SR case exposed a BLAS thread-count difference: Rust's
OpenMP OpenBLAS 0.3.34 LP64 defaults to eight threads and ignores
`OPENBLAS_NUM_THREADS=1`. Explicit `openblas_set_num_threads(1)` reproduces
Julia's one-thread OpenBLAS 0.3.30 ILP64 factors. This process-wide serial
setting is shared by direct SR and SR-CG. Additional complex FSZ intermediate
state evidence is documented in `../complex_fsz/README.md`. Vendored sources
remain unchanged.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_pairhop_green_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_pairhop_reference.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=pairhop_real
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=pairhop_fsz
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=pairhop_real --store=0
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=pairhop_real --store=1
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=pairhop_fsz --store=0
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=pairhop_fsz --store=1
cargo test -p mvmc-core --locked pairhop
cargo test -p mvmc-core --locked stored_direct_sr_gram_matches_sampled_julia_bits
cargo test -p mvmc-core --locked sampled_direct_sr_matrix_gradient_factor_and_solution_match_julia
```
