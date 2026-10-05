# Real overlap logarithm: zero remains nonfinite

Related to #274, #176 and #185; this milestone closes none of them.

## Authoritative operation

`extern/mVMC-1.3.0/src/mVMC/qp_real.c`, `CalculateLogIP_real`, lines
34–50 in the pinned reference, accumulates the owned-QP real overlap,
performs the comm1 `MPI_SUM` when required, then returns `clog(ip)` from
a `double` function. Conversion discards the imaginary part: the returned
real component is ln(abs(ip)), including zero -> negative infinity.
The actual source does not call `log(fabs(ip))`.

Whole-file SHA-256:
`eec104362f5540c368aed7e42403a79eeefc21a509da6c57c668b5c4fe887b71`.
The real sampler's nonfinite-overlap branch in `vmcmake_real.c` performs
one recovery, not a loop until a finite overlap. A `1e-100` floor conceals
that branch and changes subnormal overlaps; it is not a rounding tolerance.

Both Rust public real-log helpers now retain this contract, keeping the
existing Rust log implementation. The sampling helper takes the log only
after its owned-QP comm1 sum. No proposal, draw order, kernel, or tolerance
is changed. The helper is also used by real FSZ; this alone does not prove
the full FSZ sampling trajectory.

## Bounded evidence

Linux x86_64 Dev Container: the focused locked `test-fast` acquisition
selected exactly four new analytic tests, four existing `real_fsz_sampling`
tests and two existing `real_fsz_setup` tests. All ten passed, none skipped,
in 0.020 seconds (nextest UUID
`403c0806-71da-432b-aba1-7b5e32994aeb`; receipt suffix `OspwfL`).
Execution, source/membership/symlink, runtime, tools/providers, selected
ELF postchecks, cleanup and lock release all returned zero.

Validated source base was `1b0e9d90de7ffb90c84d098c181cefa470042021`.
Publication base is `5b0874eb70b72e2a7993662af757d50b25dadc17`:
the two production files and the existing six FSZ tests are byte-identical
between these bases. Existing FSZ assertions and reference expectations
remain unchanged. The new test is only formatted and has its source-only
heading updated for publication; its assertions are unchanged.

The analytic checks cover signed zero, negative overlap, the smallest
subnormal, and reduction-before-log. They use exact branch expectations
or analytic inequalities, not a new floating-point tolerance, and run
without C, Julia or toolbox inputs.

This evidence is not full initializer/recovery, native prefix, full-model,
20-step trajectory, or 13-model numerical acceptance. The separate ten
C-derived lifecycle fixtures were later merged in PR #314; see
`issue-274-shared-normal-initializer.md` for the current mapping.

## Focused publication checks

In an isolated checkout/target with jobs=2 and BLAS/OMP workers=1:

```sh
cargo fmt --all --check
cargo clippy -p mvmc-core --test native274_real_log_ip --locked \
  --profile test-fast -- -D warnings
```

These are the prospective formatting/lint gates, not additional results
claimed above. Normal Cargo validation never compiles or invokes C/Julia.
