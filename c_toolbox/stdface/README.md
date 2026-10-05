# StdFace C reference tools (issue #353)

Developer-only tools that produce the C expectations checked into
`tests/fixtures/stdface/`. Cargo builds and Rust tests never compile, invoke or read this
directory; the Rust tests read only the checked-in fixtures.

| Tool | What it does | Scope |
| --- | --- | --- |
| `build_reference.sh <dir>` | Builds the unmodified C StdFace (`dry.c` + all `src/StdFace/src/*.c`) with `-D_mVMC -DMEXP=19937 -O3 -DNDEBUG -ffp-contract=off -w` into `<dir>/mvmc_dry.out` (`OPT=-O0` overrides the level). | Full C StdFace executable (the `mvmc_dry.out` target of `src/StdFace/src/CMakeLists.txt`); no MPI. |
| `generate_fixtures.py <mvmc_dry.out>` | Runs every case (upstream `test/mvmc` chain inputs plus generated feature/error inputs) and stores generated files, `stdout.txt` and `exit_status` under `tests/fixtures/stdface/<case>/expected/`; writes `tests/fixtures/stdface/PROVENANCE.md` (source SHA-256, compiler, flags). | End to end, `uv run --no-project python ...`. |
| `build_reference_3d_fixed.sh <dir>`, `3d_defects.patch`, `cases_3d.py` | 3D lattices (#356): builds a copy of the C StdFace with the 3D defect fixes (`mvmc_dry_3d_fixed.out`); `generate_fixtures.py <mvmc_dry.out> <mvmc_dry_3d_fixed.out>` stores the corrected output in `expected/` and, where it differs, the unmodified C output in `c_historical/`. Defects: `tests/fixtures/stdface/README_3d_defects.md`. | Full C StdFace executable, patched copy only; the vendored sources are not modified. |
| `complex_expr.c`, `check_complex_expr.sh` | Standalone kernel probe: the compound `double complex` expressions of `StdFace_HubbardLocal`, `StdFace_MagField` and `StdFace_GeneralJ` on a grid with `-0.0`, as IEEE bit patterns (per-family FNV-1a digests in `tests/fixtures/stdface/complex_expr.digests`). | Not a full executable. |

`vmcdry.out` (`src/mVMC/vmcdry.c`) only calls `StdFace_main`, as does `dry.c`; the mVMC
executable's `-s` option calls the same `StdFace_main` and then reads `namelist.def`.
The vendored sources in `extern/mVMC-1.3.0/` are not modified.

## Origin and hashes

`complex_expr.c` copies the expressions verbatim from
`extern/mVMC-1.3.0/src/StdFace/src/StdFace_ModelUtil.c` (functions `StdFace_HubbardLocal`,
`StdFace_MagField`, `StdFace_GeneralJ`; extraction boundary: the single return/assignment
expressions of those functions, wrapped in `noinline` helpers with runtime operands).

| File | SHA-256 |
| --- | --- |
| `extern/mVMC-1.3.0/src/StdFace/src/StdFace_ModelUtil.c` | `c2040d3adb10726e77a8881f8d3e54643a4ad2aab0b12f3ca0190844de4295b1` |
| `extern/mVMC-1.3.0/src/StdFace/src/StdFace_main.c` | `9e0dbe5fb2bedb51756f2c565fd30077f6ef38cbe9a5a09e040a87aa7e168c05` |
| `extern/mVMC-1.3.0/src/StdFace/src/ChainLattice.c` | `6d7376370596833d45ccb00d07c47f818f61515f22a3184f75e1a4081202c8ca` |
| `extern/mVMC-1.3.0/src/StdFace/src/dry.c` | `4993120b81953cfc155dd1ac6338e8a39af2739991f04ac047bfb17c03809941` |
| `c_toolbox/stdface/complex_expr.c` | `ebf16d3146b0113bae304d2367014a0263f4c8e3b3a66a853f1bfefcdae0095c` |

(The complete per-source list is in `tests/fixtures/stdface/PROVENANCE.md`.)

## Compiler behaviour verified

The generated Expert files are identical when the reference is built with `-O0` and `-O3`
(checked when the fixtures were created), and the expression probe agrees at `-O0`, `-O2`
and `-O3`. gcc 13.3.0 on Linux x86-64 was used; signed zeros in complex expressions follow
GCC's lowering (`real * complex` scales both components, `real + complex` copies the
imaginary part, `real - complex` negates it, `complex * complex` is the textbook formula).

## Corrected build (issue #404)

`build_reference.sh <dir> --fixed` copies the vendored sources to a temporary directory, applies
`lattice_defects.patch` (corrections of the C lattice defects reported as Julia-mVMC#66; the patch is
a plain unified diff against `extern/mVMC-1.3.0/src/StdFace/src`) and builds `mvmc_dry_fixed.out`.
`generate_fixtures.py <mvmc_dry.out> <mvmc_dry_fixed.out>` stores the C output in `expected/` and, where the
corrected build differs, `expected_fixed/` (see `tests/fixtures/stdface/README.md`).
`cases_defects.py` holds the inputs added for the corrections.

## Reproduction

```
c_toolbox/stdface/build_reference.sh /tmp/stdface-c
c_toolbox/stdface/build_reference.sh /tmp/stdface-c --fixed
c_toolbox/stdface/build_reference_3d_fixed.sh /tmp/stdface-c
uv run --no-project python c_toolbox/stdface/generate_fixtures.py /tmp/stdface-c/mvmc_dry.out /tmp/stdface-c/mvmc_dry_3d_fixed.out /tmp/stdface-c/mvmc_dry_fixed.out
c_toolbox/stdface/check_complex_expr.sh
```
