# DH4 definition and initialization parity

Issue #25, first milestone: strict four-neighbor definitions, ten complex
parameters per definition index, final DH2/DH4 projection layout, component
optimization flags, declaration-based initialization and runtime-mode inference.
Serial DH4 execution remains rejected under #25 until loading, counts/ratios,
derivatives, SR write-back and gauge compensation pass their production gates.

Reference: unchanged Julia-mVMC `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
parser/numerical sources `c2ea432785bc14364a3cd5e9eef44db464289cc9`, Julia 1.13.1,
`extern/Julia-mVMC/Manifest-v1.13.toml`, OpenBLAS 0.3.30 ILP64, one thread.
Generated text records the Julia/BLAS metadata. No historical Julia 1.11 results
are used for these fixtures.

`parser.txt` records 56 original strict-parser cases: status, exact diagnostic,
last body line, indexed neighbor tables, row-ordered flags and ComplexType.
Cover shuffled definitions, repeated/self neighbors, comments/blank lines/CRLF,
ignored signed flag indices, arbitrary positional header labels, empty and
nonbinary complex declarations, zero/negative/maximum site counts, truncation/extra rows,
invalid flags and every neighbor/index integer or bounds failure. Definitions
listed by either alias are required; missing, invalid and pre-ModPara files fail.

`initial.txt` records 13 parsed models, including both DH factors, both namelist
orders, aliases, replacement definitions, empty real/complex declarations and
real-header AP/P and General orbitals with fixed components. It compares all
layout widths/offsets, flags, modes, nonzero packed coefficients, original zero
projection initialization, exact Slater bits and the next 624 SFMT words at
seed 11272. DH4 occupies `dh4_offset .. dh4_offset + 10*n_dh4`, after the six-component
DH2 block and before Slater. DH projection initialization consumes no RNG draws.

`mode.txt` covers 12 original runtime-mode boundaries: declaration, loaded
imaginary value and an absent/zero/nonzero authoritative flag vector.
Initialization uses declarations; runtime inference also considers loaded values.

The script runs the canonical original `test_doublon_holon_parser.jl` (50
assertions, including reference-only legacy shim tests), then 159 additional
assertions when writing or 162 when verifying the three files. Rust does not add
legacy value-term APIs. Runner tests retain the DH4 gate for parsed/programmatic
models and ensure rejection preserves parameters/RNG and creates no output.
DH2 and Gutzwiller/Jastrow deterministic regressions remain enabled.

Regenerate and verify from the repository root:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_parser_parity.jl
cargo test -p mvmc-expert-parsers --test dh4 --test dh2
cargo test -p mvmc-core --test dh4_runtime --test runtime_contract
```
