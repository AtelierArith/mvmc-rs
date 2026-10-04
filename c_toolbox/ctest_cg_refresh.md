# Optional fixed-input C CG refresh oracle

This standalone developer program generates checked-in expectations from the
authoritative C recurrence. Cargo neither compiles nor invokes it. It is a
serial fixed-operand kernel check, not C initialization/sampling or MPI proof.

`ctest_cg_main_upstream.inc` is the byte-for-byte concatenation of
`extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c` lines 1–21 (license) and
255–426 (Main and sampled matrix operator). Upstream SHA256:
`41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9`.
Extraction SHA256:
`9f941cc5f9208eed0efc80937f292837c48b7bc7d078f467715ca001d9e9f43f`.
`ctest_cg_dot_upstream.inc` is `stcopt_cg.c` lines 26–34, unchanged.
Upstream SHA256:
`66a36cdff6f22e23367e12c08fee505daff6194447a667f5b89a65eb30130700`;
extraction SHA256:
`f36c3f7d0aa0fc22e7e3cbffde3c95b79808112979e1e0ebc8fdd853d995d214`.

The adapter supplies C's globals, a serial no-op broadcast/barrier and identity
sum, timer no-ops, and the actual LP64 OpenBLAS `dgemv_` symbol. An `inline`
macro gives the extracted dot function static linkage without changing its
arithmetic. The real compile defines `MVMC_SRCG_REAL`; the complex compile
does not. No C numerical function body is patched. OpenMP pragmas are inactive.
The archived input records are means, diagonal, real/imaginary stored samples,
and gradient. The adapter stops reading before any archived expected products
or solver outputs. It resets the solution and restarts the original Main for
every limit 1–41, recording the complete solution/residual/direction and count.
Tolerance is explicitly zero to exercise limits and both refreshes; shift is
`1e-5`, weight is the input sample count. The ordinary public default tolerance
is not changed. The input diagonal is an archived operand, not recomputed by
Rust or by this adapter.

Generation environment: Dev Container `73c57e563c61`, Linux x86_64,
GCC `Ubuntu 13.3.0-6ubuntu2~24.04.1`, `-O0 -ffp-contract=off`, no fast-math,
LP64 OpenBLAS pthread `0.3.26+ds-1ubuntu0.1`, threads=1. Library SHA256:
`bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e`.
Adapter SHA256:
`5e41bb737a018fda5d1ae014672954d5544f4f78fe0882d76be38e81ba77548e`.
Real/complex executable SHA256 respectively:
`c72f45946fab01d6de2b2afa66357b5149439428672a8a29c2add4762102783a`,
`969413856298b78745fcf74c3882fa5d9863bbc844a36bee91e0a31df77f1631`.

Allocation failures are checked before all buffer pointer arithmetic. The
initial operator input is the gradient at `base+n`; its output occupies only
the disjoint solution region `base[0..n]`, which is cleared before each trial.
Mean/diagonal/samples and gradient regions are unchanged by that operator.
The solver's work vectors are separate regions following the sample matrices.
Rebuilding after the allocation-guard correction reproduced all three fixture
SHA256 values unchanged. Hexadecimal records preserve acquisition precision;
Rust compares computed values numerically, never by their serialized bits.

From the repository root in that environment:

```sh
mkdir -p /tmp/mvmc-cg-c-refresh
cc -O0 -ffp-contract=off -DMVMC_SRCG_REAL c_toolbox/ctest_cg_refresh.c -lopenblas -lm -o /tmp/mvmc-cg-c-refresh/real-probe
cc -O0 -ffp-contract=off c_toolbox/ctest_cg_refresh.c -lopenblas -lm -o /tmp/mvmc-cg-c-refresh/complex-probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/mvmc-cg-c-refresh/real-probe tests/fixtures/sr_cg/real.txt /tmp/mvmc-cg-c-refresh/real.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/mvmc-cg-c-refresh/complex-probe tests/fixtures/sr_cg/complex.txt /tmp/mvmc-cg-c-refresh/complex.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/mvmc-cg-c-refresh/complex-probe tests/fixtures/sr_cg/sampled_complex.txt /tmp/mvmc-cg-c-refresh/sampled_complex.txt
```

Generated files are separately imported into `tests/fixtures/sr_cg/c_refresh/`.
Input/output SHA256 pairs:

| Case | Archived input file SHA256 | C output SHA256 |
| --- | --- | --- |
| real | `95765162416342c86bace224de36dd315e5a0efa9ee1dd501556b8074df7c0a4` | `e9c1372e2aa7b1ff12e15c44c74f5875030e622b80c985eacdee35c853122844` |
| complex | `8258518a952cfe34db3a409ee9034a5c25905b7cf87ca0799b71ed4360c64eb5` | `bd48dd44525c79082f6db667ff6dc6103f7c363328198044c8c839df230b80ea` |
| sampled complex | `87829b292761dada69121c4d1fd175c0b4a71e029b5c6ae6dded2dcd2b80894f` | `6036d436653df6c02846827194cfe82f43597e93f392f5dd52b932f2e166b590` |

The input files include historical Julia 1.13.1 expected states, which remain
unchanged and labelled historical: Julia assigns `delta_new` directly and adds
a tiny-denominator early exit. C instead uses `delta=beta*delta` and has no
denominator cutoff. They are not interchangeable recurrence expectations.

Rust validation preserves the preexisting component budgets and independent
explicit Gram residual checks; no tolerance increase or skipped test. Container
command with isolated target `/tmp/mvmc-issue178-target`:

```sh
cargo nextest run -p mvmc-core --locked --cargo-profile test-fast --no-fail-fast --retries 0 -E 'binary_id(mvmc-core::sr_cg) | test(collective_tests)'
```

Run `b9ea303f-f3ca-4e5e-9788-d1eb6f11a32c`: 16 passed, 0 failed, 520 filtered
out, 0.019s. Log `/tmp/mvmc-c-faithful-cg-fixture.log` in the container.
This is a mutable-tree diagnostic, not whole-workspace/frozen/MPI acceptance.
The scalar tiny-SPD regression's forward allowance is two ULP at `1e16`
(ULP=2); its residual bound is about `2*epsilon*|g|`, not an MPI error budget.
