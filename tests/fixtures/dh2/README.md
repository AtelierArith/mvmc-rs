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

The verifier also runs the canonical 50-assertion DH parser/layout unit suite,
including its DH4 and deprecated-API tests as reference evidence; Rust support
is limited to the implemented DH2 definition API in this milestone. Production
DH2 remains rejected under #24 before initialization/RNG/output, including
programmatic data without a namelist and isolated nonempty DH2 fields. Loading,
gauge shifts, projection counts/ratios, SR and output integration require the
subsequent deterministic production milestone before enabling the runner.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh2_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh2_parser_parity.jl
cargo test -p mvmc-expert-parsers --locked --test dh2
cargo test -p mvmc-core --locked --test dh2_runtime --test runtime_contract
```
