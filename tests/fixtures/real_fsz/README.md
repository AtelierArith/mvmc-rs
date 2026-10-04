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
real parts. Current computed Pfaffian/inverse comparisons use explicit absolute
and relative budgets of `512 * f64::EPSILON`, not bitwise equality. Rust's
per-plane inverse scratch pad is excluded from Julia's
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

## Historical issue #176 audit (HEAD `30d8d69`)

The historical failure is resolved on the recorded Linux reference path; no
algorithm or tolerance change was recorded then. That report claimed 96 matrix
cases; the current inspected test actually loops over 16 matrix cases. The
current selected test also verifies the later-QP publication transaction and
retry, 101-attempt failure, zero, local-spin, and fixed-magnetization cases.
Each initialization case checks the complete next 624-word SFMT block.

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
burn buffers, attempt/accept counters, real Pfaffians/inverses (computed values),
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
runner rejection described that historical checkpoint, not current support.
Current selected runner tests execute real FSZ.

## Current #176 criterion audit (2026-10-03)

This source review and selected-test run is not closure or a fresh C/Julia
acquisition. Shared source was mutable; no before/after frozen snapshot was
captured. Observed HEAD was `822c35e221a2ccac0b12ea9215473bf7e90709fb`.
Host Rust was 1.99.0, LLVM23.1.1, x86_64-unknown-linux-gnu. Historical reference
environment above remains labelled historical; it was not re-acquired here.
The historical division diagnosis and exact-bit language above are provenance,
not a current computed-float acceptance requirement.

| Criterion | Current evidence | Remaining gap |
| --- | --- | --- |
| First differing QP/element/operation/environment | Historical QP1 `(1,0)` division report retained | Independent retained-operand C replay and current reference environment closure |
| C contract/fixture/environment distinction | Primary `matrix.c:182–278`: spin-indexed assembly, DSKTRF, finite Pfaffian guard, `utu2inv_d`, sign reversal | This real C path alone does not establish the historical complex-division diagnosis |
| Independent fix without weakened expectations | No kernel, fixture, or tolerance changed in this audit | C replay needed before closure proposal |
| Setup/retry/failure/zero/localspin/magnetized | 16 matrix cases, later-QP transaction, five initialization cases with exact next624 passed | No fresh reference runtime execution |
| Sampling/SR/PhysCal rerun | Selected runs below passed | Serial Linux only; PhysCal observer identity is not independent numerical parity; native Mac FSZ failure remains separate |

```sh
cargo nextest run -p mvmc-core --test real_fsz_setup --test real_fsz_sampling --test fsz_measurements --cargo-profile test-fast --locked --no-fail-fast --retries 0
cargo nextest run -p mvmc-core --lib --cargo-profile test-fast --locked --no-fail-fast --retries 0 -E 'test(fsz_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng) | test(fsz_measurements_preserve_real_and_complex_optimization_sampling_and_rng) | test(physcal_green_observer_fsz_runner_identity_and_raw_boundary)'
```

Run `4e115108-f9bb-49e4-8977-0b76f21cf24d`: 7/7 PASS, 0 skipped,
exit0, 0.252s. Run `862fbbbf-9f37-4ca0-8459-6fc7a3ac2bbd`: 3/3 PASS,
214 filter-excluded, exit0, 1.030s. Excluded tests are not coverage. #176 remains
open; these diagnostics are not full C executable, MPI, or native-Mac proof.

Independent parent reruns used the same two commands: integration run
`bc35a0c9` passed 7/7, exit0, 0.215s; library run
`5c20bf0d-bceb-4053-937d-e4cd0b19e90f` passed 3/3, exit0, 0.980s,
214 filter-excluded. Observed HEAD was `380c8110` in the mutable shared
workspace, not a frozen-snapshot assertion. These fresh parent results are
separate from the original owner runs above.

Original owner terminal output was not captured with tee. Its retained tool
output transcription is `/tmp/mvmc-issue176-selected-runs-retained-tool-output.txt`,
SHA-256 `2fbe0ee2286cce0528518841e2da9c2d8c7c6d5963cabebe723a812d968dc4f0`.
It identifies both commands, run IDs, terminal sessions/chunks and captured
post-run source hashes; it is not a native logfile or before/after source freeze.
