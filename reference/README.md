# Reference Data

This directory stores repository-level golden data that is consumed by
integration tests.

## 50-step `zvo_out.dat` regression

The Rust Phase-5 regression test reads:

```text
reference/<model>/zvo_out_first50.dat
```

These files are Julia-mVMC 50 SR-step `zvo_out.dat` slices generated with
`RndSeed=1` for the four upstream models:

```text
heisenberg_chain_real
heisenberg_chain_cmp
heisenberg_chain_fsz
hubbard_chain_real
```

They are produced from the bundled Julia-mVMC checkout (v0.5.0, Julia 1.11) by
`scripts/dump_zvo_50step_reference.jl`:

```bash
julia +1.11 --project=extern/Julia-mVMC scripts/dump_zvo_50step_reference.jl
```

and compared by `crates/mvmc-core/tests/phase5_regression_50step.rs`.
