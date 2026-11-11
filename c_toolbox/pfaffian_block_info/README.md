# Full versus paired factorization INFO (#731)

Optional standalone library probe; Cargo builds/tests never run or read it.
It compiles the complete, unmodified vendored Fortran routines rather than
extracted source. This is not a full mVMC/MPI/sampling validation.

Origin: mVMC 1.4.0 `e83aa6cb`, nested PfaPack pin `ebaa11b7`:

| Source under `extern/mVMC-1.4.0/src/pfapack/fortran/` | SHA-256 |
| --- | --- |
| `dsktf2.f` | `ea1591aa942377add5cd42597c282d1f282700251fa4b825f026cad1c6c2f65b` |
| `dskr2.f` | `57bf85cfabc22055af6c95d5765200ef5c27c9b5dff074da264fc76bbf3a56f2` |
| `dsktrf.f` | `aeda208bcacdbda11ee2a184a840ee3719e576607a0a5bc25d04ff20e96aa5c9` |
| `dlasktrf.f` | `513b02cee54cfaf9ec9ad1c979407c9727458069f8a91ba500f6a0fae0643771` |
| `dskr2k.f` | `48b4e55e1722d979349763ac711975082fc0006907105f30df95bf649bb1964a` |

Reproduce from the repository root:

```sh
gfortran -O2 c_toolbox/pfaffian_block_info/probe.f90 \
  extern/mVMC-1.4.0/src/pfapack/fortran/dsktf2.f \
  extern/mVMC-1.4.0/src/pfapack/fortran/dskr2.f \
  extern/mVMC-1.4.0/src/pfapack/fortran/dsktrf.f \
  extern/mVMC-1.4.0/src/pfapack/fortran/dlasktrf.f \
  extern/mVMC-1.4.0/src/pfapack/fortran/dskr2k.f \
  -lopenblas -o /tmp/mvmc731-block-info
OPENBLAS_NUM_THREADS=1 /tmp/mvmc731-block-info
```

Recorded on Linux x86_64, Xeon E5-2699 v3, GNU Fortran 13.3.0
(`Ubuntu 13.3.0-6ubuntu2~24.04.1`), system OpenBLAS 0.3.26.

Both the direct `DSKTF2` call and the `DSKTRF` entry used by mVMC are checked.
At these small dimensions the full entry selects the unblocked routine;
this does not validate the large-matrix blocked implementation or MPI caller.
For disconnected 2x2 skew blocks with upper entries 2, the exact Pfaffian
is `2^(n/2)` and the inverse exists. Nevertheless full `DSKTF2("U","N")`
reports INFO=n-2 for n>=4: a middle tridiagonal column is zero, as specified
by the zero-column handling at `dsktf2.f:223`. Paired mode `"P"` returns
INFO=0. Connecting the blocks with upper tridiagonal entries 1 makes both
modes return INFO=0, without changing the analytic Pfaffian.

The CPU mVMC caller uses full `M_DSKTRF("U","N")` and returns nonzero
INFO before inversion (`matrix.c:264`, `matrix.c:265`). A successful paired
provider result therefore does not reproduce that full-factor error boundary.
The experimental adapter accepts these inputs under the existing clear-C-defect
policy: treating a zero even-index separator as unusable rejects a matrix with
an analytic nonzero Pfaffian and finite inverse. The distinction is documented
in the numerical-comparison and compatibility manuals and reported as
[Julia-mVMC#98](https://github.com/tmisawa/Julia-mVMC/issues/98).
This applies an existing owner decision; it does not introduce a new policy or
silently enable the experimental CPU feature by default. Genuine singular
inputs still fail and retain the reference fallback. Do not treat the
structural difference as roundoff or hide it with a tolerance.
