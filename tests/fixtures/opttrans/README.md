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
These are historical Julia expectations. The current Rust test replaces the
48 optional/fixed NaN, overflow and underflow rejection expectations with
actual C conversion results from [the C initial-record fixtures](../initial_records/README.md).
The other 84 compatible cases retain their historical assertions. C accepts
successive complete records and leaves the final values; broader C OptTrans
activation/flag and RBM declared-width contracts remain #27/#26 work.

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

These fixtures validate inputs, initialization, full-record loading, QP-weight
refresh, nonidentity Slater tables, derivative kernels and the SR/synchronization
contracts below. Production serial OptTrans is enabled after the trajectory
gates described below.

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
parameter-synchronization unit tests.


`slater_threshold.txt` covers 54 cases across real normal, complex normal and
complex FSZ modes, both translation boundary signs and nine orbital vectors.
Cases include zero/signed zero, values immediately below/equal/above `1e-14`,
complex amplitudes and duplicate orbital indices with later zero/tiny values.
Canonical normal Slater construction ignores amplitudes at or below `1e-14`
and preserves earlier nonzero indexed values; FSZ construction does not apply
that cutoff and overwrites duplicates. Each table component is compared by bits.

`grouped.txt` covers 18 combinations of declared sectors, active parameter width
and mapping-vector count for `NSplitSize=2`. Julia rejects any of these counts
above one. Rust preserves this permanent OptTrans restriction before its
remaining general split implementation gate; single-sector serial inputs pass.

Production fixtures use `run_opt_real`, `run_opt_cmp`, `run_opt_fsz` and
`run_opt_dh24_rbm_cmp`. Each has three nonidentity OptTrans sectors and preserves
its canonical base sampling settings (normal sample=100, FSZ sample=2000,
warmup=10). The combined model includes nonzero DH2, DH4 and all nine RBM
sections. No changes to reference source, reseeding or tolerance relaxation
were made.

Direct (`NStore=0/1`) and CG runs are stored under `sr_direct/opt*_runner`,
`sr_direct/opt*_store_runner` and `sr_cg/opt*_runner`. Prefixes 1/2/3/50 compare
all parameter and energy component bits, saved indices/configurations/occupations/
projection counts, burn configuration, move counters, FSZ spins and the next
624 SFMT words. The CG directories also record initial flags, parameters and
RNG blocks. First-step sampled OO/HO, stored sample columns, covariance matrix,
gradient, Cholesky factor and solved update are separate exact-bit gates.
Indexed output files and CG diagnostics are byte-for-byte comparisons. The
canonical writer includes OptTrans in the full record and omits a dedicated
OptTrans file.

Eleven model/solver combinations complete all 50 steps. The real `NStore=0`
Direct run returns native SR status 1 at zero-based step 29; prefixes 27/28/29
also isolate the last successful updates. Rust reproduces the failure step,
unchanged failed-step parameters, saved configurations, RNG block and output
boundary. Julia's sampler may return early while its optimization loop still
accumulates saved configurations; Rust follows that behavior. Normal initial
Pfaffian retry/regeneration count and nonfinite-log remaking follow the source.
Real projected logarithms use the Julia 1.13.1 arithmetic port.

This run also exposed canonical LAPACK behavior: `potrf!` returns positive INFO
without throwing, and the source optimizer ignores it before `potrs!`. Rust
continues with that partial factor and rejects nonfinite solved updates before
mutation. `sr_failure/potrf_status.txt` checks indefinite and singular covariances
in both real and complex modes against the original optimizer.

To verify all production paths:

```sh
for case in opt_real opt_cmp opt_fsz opt_dh24_rbm_cmp; do
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case="$case" --store=0
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case="$case" --store=1
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case="$case"
done
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=opt_real --store=0 --steps=27,28,29
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_failure_parity.jl
cargo test -p mvmc-core --release opttrans
cargo test -p mvmc-cli --test runtime_contract --release opttrans
```
