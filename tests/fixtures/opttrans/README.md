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
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_load_weights_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_load_weights_parity.jl
cargo test -p mvmc-core --test opttrans --test initial_params
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_projection_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_projection_parity.jl
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

`loaders.txt` covers six models and eleven records through both recoverable
initial loading and strict optimized loading (132 cases). Models include
OptTrans-only and all-factor layouts, no active OptTrans, and runtime widths
that differ from the declared sector count. All diagnostics and gradient
fields are validated before mutation. Exact results and errors are compared,
including hexadecimal values, range errors, extra whole triples when active
versus inactive, and preservation of all preceding factors on rejection.
The full record places OptTrans after Slater and preserves definition weights.

`weights.txt` covers four quadrature sizes, three total-spin projections,
both translation boundary signs, four initial parameter vectors and six
initialization/update phases (576 cases). Every complex component bit in the
seven weight/trigonometric arrays is compared. Data-level initialization and
refresh include active OptTrans sectors; updates replace, grow, shrink and
clear the runtime vector and rebuild full weights from fixed weights.
The source script also runs canonical loader and QP-weight unit tests.

`projection.txt` covers 120 Slater-table and derivative cases: normal real
and complex orbitals with one or three quadrature nodes, FSZ complex
orbitals, periodic and anti-periodic signs, six complete/partial/absent
mapping arrangements and two parameter-update phases. Matrices compose
OptTrans before QPTrans, applying both sets of signs. Julia caches orbital
boundary signs before `vmc_para_opt!` normalizes negative NMPTrans; the
fixture reproduces that phase while Rust retains the boundary marker and
uses its absolute value for dimensions. Normal Slater sign multiplication
uses Julia's real-scalar operation, preserving signed zero.

Supplied deterministic Pfaffians and inverse matrices isolate derivative
mapping and reduction order from factorization and sampling. The fixtures
compare every Slater derivative and the separate real/imaginary OptTrans
derivative pair. They preserve canonical quirks: normal derivatives include
all sectors on the complete cached-map path but only fixed sectors on the
generic fallback; FSZ Slater derivatives use only the first fixed sectors.
The separate OptTrans derivative sums fixed weights and Pfaffians for each
sector. Public Slater derivative calls share the runtime scratch kernels.

`opt_derivatives.txt` covers 162 cases with zero/one/three active parameters,
missing/empty/two fixed weights, empty/partial/full Pfaffian vectors and
six output view sizes. Each real/imaginary pair is committed only when
both slots exist; skipped slots retain their original values. The projection
script also runs the canonical Slater-update unit contracts.

These milestones validate inputs, initialization, full-record loading,
QP-weight refresh, nonidentity Slater tables, derivative kernels and the
SR/synchronization contracts below. Production OptTrans execution remains
gated by issue #27 until main-calculation derivative placement and
deterministic serial trajectories have passed Julia parity.

`sr.txt` covers six active/inactive/all-factor layouts, real and complex
SR buffers, direct and CG solvers, and real/imaginary/both/fixed component
flags (96 cases). Deterministic orthogonal sample columns and covariance
inputs isolate parameter enumeration from sampling. Every physical term,
OptTrans component, definition weight and full QP weight is compared by
Float64 bits. Two nonzero complex fixed translation weights ensure missing
weight refreshes are detected. SR updates include OptTrans after projection, all nine RBM
sections and the declared Slater width; each OptTrans increment immediately
refreshes existing QP weights. State allocation uses the same active count.

`sync.txt` covers three layouts, six complex vectors including zero and
tiny/large amplitudes, both optimizer correlation-shift settings and the
parser/optimizer synchronization paths (72 cases). Optimizer synchronization
rescales OptTrans to maximum amplitude one after Slater rescaling. Parser
synchronization only rescales Slater; the parser has no correlation-shift
keyword. Both definition weights and fixed QP weights are preserved.
Local normalization leaves cached full weights until the next refresh.

Regenerate or verify these fixtures with Julia 1.13.1 using:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_sr_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_opttrans_sr_parity.jl
```

The script also runs unmodified canonical stochastic-optimization and
parameter-synchronization unit tests. Main-calculation derivative placement
and deterministic nonidentity serial trajectories remain before production
can be enabled.
