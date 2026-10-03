# Complex retained CG diagnostic

Optional developer-only program:
`c_toolbox/ctest_audit_mpi_cg_complex_operator.jl`. No normal Cargo test
reads this program or invokes a reference runtime. It adapts the independent
retained-operand calculation described in
[the real audit](issue-180-retained-cg-operator-audit.md), not the production
solver. Original C recurrence/source SHA and operation boundaries are in
that report; no vendored source or numerical tolerance was changed here.

Actual session **35520**, terminal **0**, chunk **a7c305**. The earlier
session 19771 exited 1 on the incorrect assumption that all three parameter
headers were complex. It is not coverage. Source inspection verified the
actual input: orbital ComplexType=1; Gutzwiller/Jastrow ComplexType=0.
The successful auditor explicitly checks that combination and requires
both finite real/imaginary sample planes. It checks NSRCG=1, NStore=0,
three samples per rank, world size four, Wc=12, seed=1, shift=1e-5 and
StepDt=0.01. The two captured active mappings are:
`6 7 8 9 10 11 12 13 14 15 16 17 20 21 22 23 24 25 26 27`.
The auditor checks positive dimension 20, unique active indices and
within-backend rank agreement. It rejects duplicate keys and invalid or
nonfinite operand shapes. No solver status is available in these CG records;
the diagnostic does not manufacture a success/status value.

Full actual stdout: `evidence/issue-180-cg-complex-audit-35520.stdout.txt`,
SHA-256 `4189a62cfc8e5c7041608b00255c13b9c595c038122dfc650de5c646f7e0d9c5`.
Exact executed script: `evidence/issue-180-cg-complex-audit-35520.jl`,
SHA-256 `aa4893a92262ff1aa7dea9bef36954c64f4bdbb175b2bf51694abe230f285f2c`.
It uses the unchanged helper archived for 14439, SHA-256
`440cbcb61b58663775fb7e69c52f71464bdd7ee3f4fddce7ebdeb79fe43eb7c1`.
When reproducing from archived files, place that helper under its original
include name `ctest_direct_sr_capture.jl` beside the script.

Input hashes:

| File | SHA-256 |
| --- | --- |
| modpara.def | 53a029d196e551ee188bd66f637932ba7b7426482efa695d5c1281f1362610b3 |
| orbitalidx.def | 3d77dce00bdc1cc6e364862edc9e776e7a8b6cb0b894df55289701785bbd238d |
| gutzwilleridx.def | c1362e2dcf537556dc4db465322ff4126962c77e28dc52e18c0ac363f4e82ae4 |
| jastrowidx.def | cc5206fe01738c353efdbbcbf1bb6f6730a532c04466a39efd9307c04c7f9fe0 |

Actual command (Linux x86_64, Julia 1.13.1, one Julia/OpenBLAS thread):

```sh
docker exec -e JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot -e OPENBLAS_NUM_THREADS=1 -e JULIA_NUM_THREADS=1 73c57e563c61 /home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia --compiled-modules=existing --project=/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8/extern/Julia-mVMC /workspaces/mvmc-rs/c_toolbox/ctest_audit_mpi_cg_complex_operator.jl /home/vscode/.cache/mvmc/issue179-cgref.yqrDc0/matrix/r4-cmp-s1-cg1-store0/prefix2/julia /home/vscode/.cache/mvmc/issue179-C-cg-first.TBh6at/r4-cmp-s1-cg1-store0/prefix2/w1/rust /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/r4-cmp-s1-cg1-store0/inputs/modpara.def
```

The Rust operands are from the owner's fresh C-faithful-kernel capture
`issue179-C-cg-first.TBh6at`; Julia operands are retained older reference
captures. Thus cross-backend solution differences remain diagnostics, not
an independent full C executable/MPI oracle or acceptance fixture.
No C compiler is invoked by this offline Julia audit. Covariance and residual
arithmetic use 256-bit BigFloat; spectral/rank estimates use Float64 and
are not certified bounds. The comparison is of final captured CG solution
vectors, not parameter increments after StepDt/sign conversion.

| Step | Iterations J/R | Covariance rank estimate | Condition estimate | Solution difference | Actual backward error J/R |
| --- | --- | --- | --- | --- | --- |
| 1 | 12/12 | 7 | 1.04703e7 | 4.01155e-10 | 4.84393e-8 / 4.82626e-8 |
| 2 | 12/12 | 5 | 1.64362e7 | 1.57485e-9 | 2.89188e-8 / 3.31359e-8 |

Step 1 reconstructed operators/gradients agree. Step 2 operator difference
is 6.12924e-12, gradient difference 7.96458e-13. Step 2 actual residual
infinity norms are 4.21932e-10 / 4.83459e-10, while recursive residual gaps
are about 9.72e-19 / 6.65e-19. Minimum absolute recorded denominator is
8.46208e-24; maximum absolute alpha about 2.29970e6.

These diagnostics do not certify a forward error bound, validate all workers,
resolve the aggregate 0.019923 intermediate discrepancy, or justify a new
tolerance. CG iteration counts are numerical-dependent and need not be
bitwise identical under the portable policy; RNG and sampling discrete
contracts remain exact. Fresh 56-prefix/13-long milestone validation has
not been executed by this audit.
