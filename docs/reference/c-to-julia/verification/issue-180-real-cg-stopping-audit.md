# Real CG stopping versus residual scale

Optional offline diagnostic `c_toolbox/ctest_audit_real_cg_stopping.jl`
reuses the frozen real retained-operand extractor, without running its
cross-backend comparison. Original sampling records and solves are unchanged.
No normal Rust test invokes this program. It produces diagnostics, not
expected increments or a numerical-comparison tolerance.

Actual session90810, terminal0, chunk3ba5b4. Exact stdout is preserved in
`evidence/issue-180-real-cg-stopping-90810.stdout.txt`, SHA-256
`2099aaa7140c23588da8f3eac6b65cb8575c4edeb0e3bc01d4750f6daf221faa`.
Executed script SHA-256
`3c133ab4bdd2ad89a7d69e91c720300f43fe5cc203f86c14d189321d50d413cf`.
Operand extractor SHA-256
`1a9b3f64ffb49450100be422200320f284961c08b50884774a040e60e3e63931`;
helper SHA-256
`440cbcb61b58663775fb7e69c52f71464bdd7ee3f4fddce7ebdeb79fe43eb7c1`.
The exact extractor/helper archives and original C recurrence provenance are
linked from [the retained real audit](issue-180-retained-cg-operator-audit.md).

Reproduction (Linux x86_64, Julia1.13.1, one Julia/OpenBLAS thread):

```sh
docker exec -e JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot -e OPENBLAS_NUM_THREADS=1 -e JULIA_NUM_THREADS=1 73c57e563c61 /home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia --compiled-modules=existing --project=/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8/extern/Julia-mVMC /workspaces/mvmc-rs/c_toolbox/ctest_audit_real_cg_stopping.jl /home/vscode/.cache/mvmc/issue179-cgref.yqrDc0/matrix/r4-real-s1-cg1-store0/prefix3/julia /home/vscode/.cache/mvmc/issue179-C-cg-first.TBh6at/r4-real-s1-cg1-store0/prefix3/w1/rust
```

This scopes step2 of the world4/real/store0/width1/prefix3 capture. Rust uses
the owner's fresh C-faithful kernel; Julia retains its older algorithm.
Settings are the owner's confirmed unchanged CG defaults: tolerance1e-10,
maxiter0 meaning active dimension10, no harness override; shift1e-5.
Thus C's threshold is `(1e-10*1e-10)*10*10 = 1e-18`, an **absolute
squared recursive residual** criterion, not a relative/backward-error target.
The companion uses these explicitly scoped defaults, not generic inferred
settings, and synthesizes no solver status. Residual arithmetic is 256-bit;
condition estimates are Float64 estimates, not certified bounds.

| Quantity | Julia retained | Rust C-faithful |
| --- | --- | --- |
| Iterations / default maximum | 10 / 10 | 10 / 10 |
| Final recursive residual norm squared | 3.76854e-14 | 9.39494e-14 |
| Actual residual 2-norm | 1.94127e-7 | 3.06512e-7 |
| Gradient 2-norm | 6.62951e-3 | 6.62951e-3 |
| Relative residual 2-norm | 2.92823e-5 | 4.62344e-5 |
| Relative residual infinity norm | 3.42030e-5 | 5.37158e-5 |
| Normwise backward error | 7.81660e-6 | 1.22896e-5 |
| Condition estimate × relative residual 2-norm | 243.044 | 383.747 |
| Condition estimate × backward error | 64.8780 | 102.004 |

Both solves exhausted the iteration limit, with squared residuals about
37,685 and 93,949 times the threshold. Neither reached the convergence
criterion. Their natural RHS-relative target corresponding to sqrt(threshold)
would be about 1.51e-7; actual relative residuals are about 194 and 307 times
larger. A small recursive-versus-actual residual gap does not change this.

The products above are **vacuous for certifying small forward error**;
condition-times-backward-error exceeds one, so a denominator of the form
`1-condition*eta` cannot provide a useful positive perturbation bound.
These are scale diagnostics, not a certified theorem using mixed norms or
an independent high-precision exact solution. The 0.019923 aggregate
intermediate discrepancy is not validated by them. C-faithful maximum-iteration
exit is an algorithm contract, not proof of accurate convergence. No solver
limit, input, policy tolerance, RNG contract, or acceptance bound was changed.
