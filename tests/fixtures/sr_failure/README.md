# SR failure references

`potrf_status.txt` preserves historical Julia 1.13.1 results with ILP64
OpenBLAS 0.3.30 on macOS, one thread. Its original numerical payload is
unchanged. That implementation ignored positive POTRF INFO and proceeded to
POTRS: an indefinite matrix could produce a finite invalid parameter update
and report success. These rows document an unsafe historical behavior, not
C-compatible production expectations. The published Julia patch head
`62b0f97f076fb55c71c3ab0caa041a9adff94e04` also contains this defect; its repair
is being coordinated through the existing upstream `julia-patch` PR54.

The authoritative C path is `extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c`,
`stcOptMain`: DPOSV returns positive INFO on failed factorization without
substituting the RHS. `stcopt.c` updates parameters only when INFO is zero.
Rust tests independently exercise zero covariance and indefinite
`[[1,2],[2,1]]` systems, requiring failed status, unchanged parameters/RHS,
positive factor INFO, and absent substitution INFO. Successful SPD and
nonfinite-substitution coverage remain separate.

The historical fixture test checks all four archived real/complex cases and
distinguishes their historical status/update from the supported C failure
contract. It does not generate replacement expectations from Rust.
