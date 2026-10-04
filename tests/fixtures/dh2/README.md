# DH2 definition and initialization contract

Reference: Julia-mVMC `8bb1b9e` (parser/numerical sources `c2ea432`), Julia
1.13.1 with `Manifest-v1.13.toml`, OpenBLAS 0.3.30 ILP64, one BLAS thread.
The fixtures record version/backend metadata. No vendored source is changed.

`parser.txt` records 42 original-source cases, including unordered multiple
neighbor tables, repeated/self neighbors, positional header labels, nonbinary
ComplexType declarations, empty tables, Unicode whitespace, CRLF, comments,
signed ignored optimization indices, integer limits, invalid headers/sites/
definition indices/flags, duplicate rows and exact row counts. Failed sections
publish no definition. Diagnostic text and last body-line numbers match Julia.
The strict C index-table API is ported; Julia's deprecated value-term overloads
are intentionally omitted. Each optimization row's first column is validated
as an integer and ignored: row order controls the six parameters per table.

Both `DH2` and `DoublonHolon2Site` namelist entries replace a previous valid
definition atomically. Missing or invalid definitions, including a DH2 block
before positive Nsite is established, fail during parsing as in Julia's
required-if-present contract. They cannot return an incomplete model as a
successful parse.

`initial.txt` records six complete input boundaries: orbital-first, DH-first,
alias, real, empty-complex and replacement cases. The final layout reserves
declared Gutzwiller/Jastrow widths and six dense parameters per DH2 table;
orbital component flags are applied after DH flags at their final offsets.
Neighbor tables remain independent of parameter ordering. Nonzero complex DH2
values pack into the projection block, initialization clears them without RNG
draws, and Slater initialization uses the DH2 ComplexType declaration. Parsed
flags, packed parameter bits, initialized bits and all next 624 SFMT words
(seed 11272) are compared exactly. The pre-initialization comparison also
verifies the source Gutzwiller index placeholder values; initialization clears
these before sampling.

`mode.txt` covers 12 original runtime mode decisions: declarations, loaded
imaginary values and explicit authoritative runtime flags. Declaration-based
initialization is kept distinct so overlays do not alter the initial draw count.

The parser verifier also runs the canonical 50-assertion DH parser/layout unit
suite, including DH4 and deprecated-API tests as reference evidence. Rust ports
the strict DH2 API; this does not establish DH4 or deprecated-API support.

`counts.txt` covers all 256 four-site occupations with two DH2 tables, including
self/repeated neighbors and independent table strides after sparse declared
Gutzwiller/Jastrow blocks. `moves.txt` covers all 3,584 allowed ordinary and
spin-changing hops. Original full counters and incremental/alias updates agree
after every move. Rust compares counters, log values/ratios and logarithmic
derivatives against the source fixtures exactly. As in Julia, incremental
Gutzwiller/Jastrow updates are followed by a complete DH2 tail recomputation;
onmoving on-site spin flips retain the previous counters.

`gauge.txt` covers ten synchronization boundaries: all active, disabled shifts,
fixed declared Gutzwiller/DH flags, absent/short flags, partial parameter storage
and missing factors. DH2 averages each three-bin group of real coefficients,
subtracts that real shift without changing imaginary parts, compensates every
Gutzwiller term, then applies the existing Gutzwiller/Jastrow shift and Slater
rescale. Disabling correlation shifts still rescales Slater. `sr-writeback.txt`
records 36 exact original direct-SR solves, independently activating every DH2
real/imaginary component. Their Cholesky results are compared as bits, including
rounding of the solved update. Strict indexed `InDH2` overlays use the number of
definition tables in the header and six rows per table; optional missing files
are skipped and malformed present records commit no values. Initial/fixed
triples loaders reserve DH2 between Jastrow and Slater and validate all tokens
before mutation. The structured validator checks table shape and neighbor
ranges in source order.

`green_normal.txt` and `green_fsz.txt` exercise all four-site two-body operators
and FSZ one-body spin changes with nonzero DH2 real/imaginary coefficients. The
original kernels also pass independent Fock-operator/Pfaffian comparisons using
the existing explicit tolerances. Rust matches every original kernel result and
local-energy result as bits. The real Transfer shortcut is bypassed when DH2 is
present, exactly as in Julia.

The `production_real`, `production_cmp` and `production_fsz` namelists add DH2
and a nonzero `InDH2` overlay to canonical Hubbard inputs. Real uses the exact
one-table definition from the canonical DH measurement input and the unchanged
Hubbard optimization settings (100 samples, 10 warmup sweeps). Complex adds two
neighbor tables and mixed component flags. FSZ uses the canonical complex AP/P
PairHop input settings (2,000 samples, 10 warmup sweeps), removes PairHop and
adds two DH2 tables. The combined canonical DH measurement input also contains
DH4 and TwoBodyGEx and remains outside this DH2-only optimization gate.

The fixtures in `../sr_cg/dh2_*_runner` and `../sr_direct/dh2_*_runner` record
same-seed 1/2/3/50-step runs for SR-CG and direct SR with NStore=0/1. Checks cover
updated parameters including DH2, saved configurations/projection counts, FSZ
spins/burn configuration/counters, energy and every next 624 SFMT words; direct
fixtures also include original first-step derivative Gram/SR matrices and
solves. The source observer requires finite energy and positive weighted sample
count. Original sampling counts, seed and tolerances are preserved.

`loaded-*.txt` additionally checks flags, loaded/synchronized parameter bits and
RNG state immediately before sampling; `history-*.txt` records three post-sync
history points from the original runner. The public Rust namelist runner is
checked against original direct NStore=1 output. All six output files for every
prefix are compared byte for byte. Julia's current history and writers include
Gutzwiller/Jastrow/orbital terms only, omitting DH2 coefficients; Rust preserves
this omission, and tests explicitly check that no DH2 output file is invented.
Initial/fixed loaders still accept the full projection/Slater triples format.

Production DH2 optimization is enabled only with these deterministic gates.
PhysCal, MPI and the source's broken real FSZ runner remain
subject to their own issues and runtime restrictions.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh2_parser_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh2_projection_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh2_runner_boundaries.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_two_body_green_parity.jl --dh2
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_fsz_green_parity.jl --dh2
# Repeat with --case=dh2_cmp and --case=dh2_fsz; direct also with --store=1.
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=dh2_real
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=dh2_real --store=0
cargo test -p mvmc-core --locked --test dh2_projection --test dh2_runtime --test two_body_green
cargo test -p mvmc-core --locked --lib dh2_
```

The numerical/fixture generators accept `--write`; verifier runs regenerate and
compare complete text against committed fixtures. Julia 1.13.1 uses OpenBLAS
0.3.30 ILP64 with one thread. Rust's LP64 SR/Gram backend uses OpenBLAS 0.3.34
and explicitly pins one thread through the existing serial BLAS initializer.
