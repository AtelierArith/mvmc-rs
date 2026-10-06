# glibc complex functions for the RBM path (issue #470)

The C RBM code (`extern/mVMC-1.3.0/src/mVMC/rbm.c`) calls the C99 complex functions of the
platform libm:

| C call site | function |
|---|---|
| `rbm.c:41,58` (`WeightRBM`, `LogWeightRBM`) | `clog(ccosh(rbmCnt))` |
| `rbm.c:44,122` (`WeightRBM`, `RBMRatio`) | `cexp(z)` |
| `rbm.c:94,155` (`RBMRatio`, `LogRBMRatio`) | `cexp(-2.0*RbmCnt)` |
| `rbm.c:118,179` (block product) | `clog(zz)` |
| `rbm.c:345,357,369` (`SROptO` derivatives) | `ctanh(rbmCnt)` |

`crates/mvmc-expert-parsers/src/utils/glibc_complex.rs` ports `cexp`, `clog`, `ccosh`, `ctanh`
and `__x2y2m1` line by line from glibc.

## Origin

* GNU C Library **2.39** (upstream tag `glibc-2.39`, tag object
  `9609a435f3f9a07c1cf607ad5821b12f735abd69` in `https://github.com/bminor/glibc`, mirror of
  `https://sourceware.org/git/glibc.git`). The Ubuntu 24.04 package `2.39-0ubuntu8.9` that built
  the reference outputs is upstream 2.39 plus distribution patches that do not touch these files.
* Files read (path in the glibc tree, SHA-256 of the raw file at that tag):

| file | SHA-256 |
|---|---|
| `math/s_cexp_template.c` | `e4e51e4ba0bd037ad53aaec5932c3b2851850b579fad6abd747f3b51a7f5e6e9` |
| `math/s_clog_template.c` | `9c3c2d81eed156f29bb6687c65c63c67bc9c660f2536f4828ee0baf904632f63` |
| `math/s_ccosh_template.c` | `fe0a9e6e5adf76ba368db2b0cebcc1fec58081db0ab204e6764331c8419f0665` |
| `math/s_ctanh_template.c` | `a8e5ade870f9976c44965c6e88d27f9cd6ef9bcf50f696f405154fa7bc18fcdf` |
| `sysdeps/ieee754/dbl-64/x2y2m1.c` | `50020b010f549c1426033428b735faebc453190e54ac9a0a1e70bec85407bf1f` |

* Extraction boundary: the generic `M_DECL_FUNC (__cexp|__clog|__ccosh|__ctanh)` bodies, for
  `double`. Exception-flag side effects (`feraiseexcept`, `math_check_force_underflow*`) have no
  Rust counterpart and are omitted; they do not change returned values. The building blocks
  (`exp`, `log`, `log1p`, `hypot`, `atan2`, `sin`, `cos`, `sinh`, `cosh`) are the platform libm
  through the Rust `f64` methods, i.e. the same `libm.so` functions the templates call
  (`sincos` is evaluated as `sin` and `cos`, bit-identical in glibc).

## Probe and fixture

`probe.c` is a standalone kernel check (not a full `vmc.out` run): it reads hexadecimal inputs
and prints the bits of `cexp`, `clog`, `ccosh`, `ctanh`. `gen_inputs.py` produces the 1600
deterministic inputs: all combinations of signed zeros, +-1, +-inf and NaN; large `|Re z|` around
the overflow thresholds of `cexp`/`ccosh` (709, 3*709) and `ctanh` (354, 708); near the branch cut
of `clog` (negative real part with tiny, subnormal and signed-zero imaginary part); every `clog`
branch (`|x| = 1`, `1 < |x| < 2`, `0.5 <= |x| < 1` with the `x2y2m1` path, scaling regions near
`DBL_MAX` and `DBL_MIN`); and a pseudo-random sample of RBM-like hidden values.

Reproduce (explicit developer command; Rust builds and tests never run C or this script):

```sh
c_toolbox/glibc_complex_470/generate.sh   # rewrites tests/fixtures/glibc_complex_470/probe.txt
```

The header of the fixture records gcc, glibc and CPU of the generating run (gcc 13.3.0, glibc
2.39-0ubuntu8.9, Xeon E5-2699 v3, FMA-capable: glibc selects its FMA `exp`/`log`/`sin`/`cos`
variants through ifunc).

## What the test asserts

`ports_match_the_glibc_probe_fixture` (in `glibc_complex.rs`): on Linux x86_64 every one of the
1600 x 4 results is bit-identical to glibc (NaN matches NaN). On other platforms (macOS) the
system libm differs from glibc in the building blocks, so the same algorithm is compared within
16 eps of the component plus four subnormal quanta. glibc selects different `exp`/`log`/`sin`/`cos`
code paths on CPUs without FMA; those could differ from this fixture in the last bit, in which
case the fixture should be regenerated on that CPU class and the divergence investigated first.
