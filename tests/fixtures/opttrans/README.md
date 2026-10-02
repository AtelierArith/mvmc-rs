# Canonical OptTrans input parity

These fixtures come from unmodified Julia-mVMC HEAD
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, with parser source
`c2ea432785bc14364a3cd5e9eef44db464289cc9`. Generation and verification use
Julia 1.13.1, `extern/Julia-mVMC/Manifest-v1.13.toml`, OpenBLAS 0.3.30
ILP64 and one BLAS thread.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_parser_parity.jl
cargo test -p mvmc-expert-parsers --test opttrans
```

The source script also runs the canonical `test_read_input_parameters.jl`
unit contracts, including OptTrans definition and complex overlay tests.

`parser.txt` covers 44 definitions with `Nsite` 0 or 2 and `NMPTrans` -1,
0 or 1 (264 cases). Each case starts with an active valid definition and
distinct runtime weights, then compares the exact error and all retained
or replaced fields. Cases include incomplete and duplicate mappings,
non-permutation maps accepted by Julia, extra tokens, comments, missing
terminal newline, signed zero, finite subnormals, hexadecimal floating
values, base-prefixed integers, nonfinite values and numeric range errors.
For periodic boundaries Julia forces every parsed sign to +1.

`initial.txt` covers eleven namelists and four initialization modes (44
cases). The mixed layout includes Gutzwiller, Jastrow, DH2, DH4, all nine
RBM sections and Slater before OptTrans. It compares component flags,
parameter counts, mappings, every initial parameter bit and the next 624
SFMT UInt32 words for seed 11272. OptTrans initialization copies definition
weights after Slater and consumes no random draws, including when its
optimization flags are inactive. An empty initial-weight vector preserves
existing runtime weights.
Both namelist orders preserve active OptTrans maps when reading QPTrans;
QPTrans installs identity OptTrans maps when no valid definition exists.

`overlays.txt` covers eleven optional `InOptTrans` overlays, including
complex and hexadecimal values, missing files and atomic rejection of
malformed dense records. Overlays replace runtime weights and preserve
definition weights. Diagnostics use basenames so fixtures are independent
of checkout location.

Julia records OptTrans definition read/format failures as input warnings,
retains a preceding valid definition, and continues parsing. `NQPOptTrans`
defaults to 1 while the active parameter count is the runtime vector length.
The declared header keyword and `ComplexType` line do not alter this parser.
Real optimization components are always enabled and imaginary components
always disabled by the final OptTrans flag pass.

This milestone validates the input and initialization contracts. Production
OptTrans execution remains gated by issue #27 until full-record loading,
nonidentity Slater projections, QP-weight refresh, derivatives, SR and
normalization have passed deterministic serial Julia parity.
