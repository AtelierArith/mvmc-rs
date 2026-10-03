# Real FSZ setup parity (#43)

`setup.txt` was generated with Julia 1.13.1 and the reference workspace's
`Manifest-v1.13.toml`. Julia used OpenBLAS 0.3.30 (ILP64), one thread; the
Rust `mvmc-core` path uses the required `pfapack/blas-backend` (Accelerate
on macOS). Numerical source revision: Julia-mVMC
`c2ea432785bc14364a3cd5e9eef44db464289cc9`.

Generate or verify without modifying vendored sources:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_real_fsz_setup_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_real_fsz_setup_parity.jl
cargo test -p mvmc-core --locked --test real_fsz_setup
```

The oracle calls the original `calculate_m_all_fsz_real!` and
`make_initial_sample_fsz_real!`. Range and sentinel checks draw on
`test_unit_vmc_sampling_qp_split.jl`; Slater construction follows
`test_unit_slater_update.jl`.

The 16 matrix cases cover 2–8 electrons, interleaved explicit spins and
full polarization, real and complex Slater inputs, interior QP ranges,
untouched outside planes, and empty ranges. Complex inputs test that the
real wrapper still uses the authoritative complex table before extracting
real parts. Pfaffians and inverses match bit for bit, including signed
zero. Rust's per-plane inverse scratch pad is excluded from Julia's
contiguous matrices and must remain untouched.

Initial-sample cases cover successful retries, the 101-attempt failure
limit, a zero matrix, a local-spin site, and fixed magnetization. They
compare electron indices, configurations, occupations, projection counts,
spins, and the complete next 624-word SFMT block. A later-QP nonfinite
Pfaffian must publish neither earlier complex results nor real shadows.

Julia's FSZ kernel ignores the factorization zero-pivot status and rejects
only nonfinite Pfaffians. A zero Pfaffian and nonfinite inverse therefore
return success at this kernel boundary; the regression explicitly records
that upstream behavior. Normal real/complex kernels keep their own guards.

## Issue #176 audit (HEAD `30d8d69`)

The historical failure is resolved on the recorded Linux reference path; no
algorithm or tolerance change is needed. The current Rust gate passes all 96
matrix cases, the later-QP publication transaction, and the retry, 101-attempt
failure, zero, local-spin, and fixed-magnetization cases. Each initialization
case also checks the complete next 624-word SFMT block.

The first historical numerical divergence is reproducible by comparing the
preserved macOS fixture with the independently generated Linux fixture. It is
setup case `ns=2, ne=1, polarized=1, complex_input=0`, inverse QP 1, matrix
element `(row=1, col=0)` in column-major storage: macOS
`bffce739ce739ce6`, Linux `bffce739ce739ce7`; the conjugate entry differs in
the same one-ULP direction. This is the tridiagonal solve's complex division,
not an RNG, indexing, or control-flow divergence: the macOS/LLVM path uses
the exponent-scaled norm-squared quotient while Linux GNU uses the
Smith-scaled quotient. The Linux fixture is therefore the platform-specific
exact-bit expectation; no cross-platform bit requirement is inferred for this
computed value.

Recorded Linux audit environment: Ubuntu 24.04 x86_64, kernel 6.8,
`rustc 1.99.0`/Cargo 1.99.0, GCC/G++ 13.3.0, OpenBLAS 0.3.26 LP64 for
Rust, Julia 1.13.1 with OpenBLAS 0.3.30 ILP64 and one BLAS thread, Julia
source checkout `8bb1b9e8`, and manifest SHA-256
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.
Rust uses the default `mvmc-core` BLAS/LAPACK features and
`pfapack/blas-backend`; the setup fixture SHA-256 is
`5d1c0bb40fd89796252c3b91e11b641d827cf66d25755a4249a3c8a5eefa47d1`.
The checked-out Julia executable reports 1.13.1, but juliaup does not have an
explicit `+1.13.1` channel installed; invoking `julia --project=...` produced
the recorded fixture exactly.

This milestone supplies full real FSZ setup and initialization. Issue #43
remains open until the real sampler, observation/derivative dispatch, and
multi-step deterministic trajectories (including spin flips) are verified.

## Real FSZ sampler milestone

`sampling.txt` covers conduction electrons, all local spins, mixed local
and conduction electrons, and fixed magnetization, each through 1, 2, 3,
and 50 sampling calls. Inputs include Gutzwiller/Jastrow factors and two
QP planes. It compares all saved configurations/spins/projection counts,
burn buffers, attempt/accept counters, real Pfaffians/inverses (exact bits),
and the next 624 SFMT words. Rust's counter representation omits Julia's
reserved slot 10 and puts Julia's counter[11] burn marker in Rust slot 9.

`moves.txt` compares two-electron proposal Pfaffians and accepted inverse
updates for 2–8 electrons, with both calculated inverses and synthetic
arithmetic inputs. `bilinear.txt` evaluates Julia's original scalar FSZ
bilinear loop on the inputs in `pfaffian_cg/two_hop_bilinear.txt`. These
inputs distinguish its sequential operation order from the normal real
path's vectorized/FMA reduction. `proposals.txt` checks the full FSZ proposal kernel on those same arithmetic inputs, covering the dispatch to the scalar reduction. Both paths have exact-bit assertions.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_real_fsz_sampling_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_real_fsz_moves_parity.jl
cargo test -p mvmc-core --locked --test real_fsz_sampling
cargo test -p mvmc-core --locked fsz_bilinear_matches_julia_scalar_reduction_bits --lib
```

Add `--write` to either Julia command to regenerate its fixtures. The
version, manifest and BLAS settings are the same as the setup milestone.

### Reference limitations

The checked-out original `vmc_make_sample_fsz_real!` throws `MethodError`
at its first saved sample: it passes `log_ip_old::Float64` to
`save_ele_config_fsz!(..., log_ip::ComplexF64, ...)`. That argument is
unused in the saving function. The sampler oracle copies the source
function and converts only that unused save argument to `ComplexF64`.
It preserves numerical kernels, proposal/acceptance control flow, and
every RNG draw; it does not edit the reference files. This is source-level
sampler parity with a documented boundary adaptation, rather than an
unmodified Julia production-run gate.

The reference `vmc_main_cal_fsz!` also sets `all_complex = true`
unconditionally and accumulates complex SR buffers even when the outer
optimizer selects real mode. Real optimization dispatch and multi-step
energies still require resolution of these reference behaviors. The Rust
runner continues to reject real FSZ, and #43 remains open.
