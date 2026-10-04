# Complex FSZ intermediate sampling parity

`sampling.txt` records the original `make_initial_sample_fsz!` and
`vmc_make_sample_fsz!` implementations, without source adapters, for a
four-site, four-electron, two-QP state. Reference: Julia-mVMC `8bb1b9e`,
numerical sources `c2ea432`, Julia 1.13.1, OpenBLAS 0.3.30 ILP64, one thread,
seed 11272. Each prefix is independently seeded; the global Julia SFMT stream
is captured before any subsequent run is seeded.

Conduction, fixed TwoSz, local-spin and exactly cancelled QP overlap cases
cover 1/2/3/10 sampler calls. An all-NaN Slater table exercises all 101 failed
initialization attempts and the driver's early return. The source checks only
Pfaffian finiteness: an all-zero Slater table is not this failure case. Failed
attempts retain the last attempted configuration without publishing partial
matrix planes. The source early return leaves a ten-element counter array;
the fixture maps the absent FSZ burn slot to zero in Rust's compact layout.

The Rust test compares each initial boundary and complete prefix: saved and
scratch electron labels/configurations/occupations/projections/spins, packed
burn state, all counters, Pfaffians, inverse entries and the next 624 SFMT
words. All comparisons are exact bits, including signed zeros. Cancelled
overlap tests the original nonfinite log-IP remake and its additional draws
both on the first call and after restoring a burn configuration.

These gates exposed three production differences: the complex driver's
conditional conduction-spin proposal (including its fixed-TwoSz behavior),
accepted one-electron updates that must use the source's incremental inverse
update, and Julia's robust complex division arithmetic. Two-electron Exchange
continues to use the original full recomputation. Complex log-IP now takes
`log(ip)` directly, preserving the source's nonfinite result at zero overlap.

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_complex_fsz_sampling_parity.jl
cargo test -p mvmc-core --locked --test complex_fsz_sampling
```

The reference emits expected warnings for cancelled overlap and errors for the
intentional initialization-failure case; its test assertions must all pass.
