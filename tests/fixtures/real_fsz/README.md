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

This milestone supplies full real FSZ setup and initialization. Issue #43
remains open until the real sampler, observation/derivative dispatch, and
multi-step deterministic trajectories (including spin flips) are verified.
