# DH4 definition and production parity

Issue #25: strict four-neighbor definitions, ten complex parameters per index,
combined DH2/DH4 layout, initialization/loading, counters/ratios/derivatives,
gauge compensation and serial direct/CG SR. The normal real/complex and complex
FSZ production gates cover DH4 alone and together with DH2. Real FSZ, RBM,
OptTrans, PhysCal and MPI remain subject to their own implementation issues.

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
legacy value-term APIs. DH2 and Gutzwiller/Jastrow deterministic regressions
remain enabled.

Regenerate and verify from the repository root:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_parser_parity.jl
cargo test -p mvmc-expert-parsers --test dh4 --test dh2
cargo test -p mvmc-core --test dh4_runtime --test runtime_contract
```

`kernels/` and `combined/` contain separate original-source fixtures for DH4
alone and DH2+DH4. Each records all 256 four-site occupations and 3,584 valid
normal/spin-changing moves, exact counts/log values/ratios/derivatives, 60 original
real/complex direct-SR component solves and 12 gauge boundaries. Two independent
four-column tables include repeated/self neighbors. Gauge boundaries cover
declared-but-unused Gutzwiller slots, fixed DH2/DH4 flags, disabled shifts,
empty/short flags, partial storage, absent factors and cancellation across five
bins. Form each factor's compensation separately, add DH2 then DH4, compensate
Gutzwiller, then apply the original Gutzwiller/Jastrow shift and Slater rescale.
Imaginary coefficients remain unchanged by real correlation shifts.

The Green fixtures in both directories cover exhaustive normal two-body and FSZ
one-/two-body spin-changing operators, local energy and InterAll. Original Fock
checks retain their existing tolerances; Rust kernel results match source bits.
The real Transfer shortcut falls back to counter recomputation for DH4 models.
Structured validation reports shape/neighbor errors in source order and remains
an explicit API. Strict InDH4 overlays use the definition count in the header
and ten times that count in the body; each rejected record is atomic. Initial
and fixed loaders consume both DH blocks between Jastrow and Slater triples.

`production_dh4_{real,cmp,fsz}` and `production_dh24_{real,cmp,fsz}` hold six
nonzero-coefficient models. Normal modes preserve canonical Hubbard sampling
(sample=100, warmup=10); FSZ preserves canonical PairHop sampling (sample=2000,
warmup=10), removing PairHop before adding DH factors. Seed is 1. Real DH4 uses
the original six-site DH definition; complex modes add an independent table
with repeated/self neighbors. Complex normal models include fixed components.
Shuffled InDH4 values use exact dyadic coefficients; combined models also retain
the DH2 definitions/overlays already verified under #24.

`loaded-*.txt` records loaded flags, all coefficient bits and the next 624 SFMT
words. `history-*.txt` records three post-sync points. The current original
writers/history omit both DH blocks, retaining Gutzwiller/Jastrow/orbital terms;
Rust preserves this actual source contract and checks the omission. No DH
output file is invented. Initial/fixed loading still expects the full layout.

The corresponding `../sr_cg/dh4_*_runner`, `../sr_cg/dh24_*_runner` and direct
`*_runner`/`*_store_runner` directories record same-seed prefixes 1/2/3/50:
updated coefficients/energy, complete saved configurations/projection counts,
FSZ spins/burn/counters and all next 624 SFMT words. Every prefix also compares
six output files byte-for-byte. Source observation hooks require finite energy
and positive sample weight. Direct fixtures include sampled matrices/gradients,
Cholesky factors/solutions and stored derivative Gram matrices. Rust's serial
LP64 OpenBLAS backend is 0.3.34; the original source backend is ILP64 0.3.30.
No performance claim is made.

Verify production fixtures from the repository root (add `--write` only to
regenerate):

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_projection_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_projection_parity.jl --combined
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_dh4_runner_boundaries.jl
for factor in dh4 dh24; do
  julia +1.13.1 --project=extern/Julia-mVMC scripts/check_two_body_green_parity.jl --$factor
  julia +1.13.1 --project=extern/Julia-mVMC scripts/check_fsz_green_parity.jl --$factor
  for mode in real cmp fsz; do
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=${factor}_${mode}
    for store in 0 1; do
      julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=${factor}_${mode} --store=$store
    done
  done
done
cargo test -p mvmc-core --test dh4_projection --test dh4_runtime --test two_body_green
cargo test -p mvmc-core --lib dh4
cargo test -p mvmc-core --lib dh24
```
