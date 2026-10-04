# Apple Silicon nextest references (#200)

The default workspace suite on Apple Silicon originally had 26 failures
(Rust 1.98.1, nextest 0.9.140, macOS ARM64, OpenBLAS 0.3.34 OpenMP,
`vortexm4`, `test-fast`). Intel macOS and Linux x86_64 passed.

Two reader tests compared the sign of arithmetic-generated NaNs in
`real + imag * I` with Intel C fixtures. Native ARM C and Rust produced
`7ff8000000000000`, while the archived Intel result was
`fff8000000000000`. Only these computed NaNs compare by classification.
Finite values, infinities, signed zero, parsed/copied representations, record
indices and independently tested RNG initialization/generation retain their exact contracts.

The other failures involved BLAS-dependent SR references. The archived
Julia fixtures used OpenBLAS 0.3.30 ILP64 Haswell; native Rust used 0.3.34
LP64 vortexm4. A fixed-input Cholesky solution differed by `7.71e-13`, and
a fixed CG residual by `1.68e-12`. These discrepancies occur without sampling
and persist with one BLAS thread.

Changing only Julia's solver backend was insufficient. In the real runner's
first 100 samples, weights and all logarithmic derivatives agreed, but 13
local energies differed by a rounding unit. The first was sample 16:
`3fd0c1f366b7ab68` versus `3fd0c1f366b7ab69`. The normal real two-hop
bilinear operation in Rust deliberately retains the archived Julia AVX2
reduction tree: four inner lanes and six outer accumulators. ARM
LoopVectorization chooses a different tree. Small initial discrepancies can
be amplified by the SR solve, and longer optimization runs can change saved
configurations. Long-run differences caused by verified numerical roundoff are permitted; RNG behavior is checked independently.

## Numerical implementation fixes

The FSZ inverse previously used Intel macOS complex division on ARM. Native
ARM libc++ calls compiler-rt `__divdc3`, whose numerator and denominator
contract multiply/add into FMA, even when its caller uses
`-ffp-contract=off`. Disassembly of the native reference bridge confirms
`fmadd`; Rust now reproduces that operation order with `mul_add` on macOS
ARM. An independently generated native C fixture covers 375 divisions,
including cancellation, scaling, subnormals and nonfinite recovery. Linux
retains its libgcc algorithm; Intel macOS retains its verified arithmetic.
Reproduce the native quotient probe with
`uv run --no-project python scripts/check_complex_division_c_parity.py`
(add `--write` to regenerate).
This fixes the RBM FSZ direct/CG discrepancies without relaxing their numerical,
failure-status or sampling/RNG checks.

Complex direct-SR accumulation also multiplied the weight after the complex
product. C `vmccal.c::calculateOO` instead scales the first operand before
multiplication. Rust now follows C. A standalone extraction of that C routine
provides three independent regression cases, including intermediate overflow
and underflow. This change preserves representable weighted products.

The canonical RBM-CG comparison exposed a separate first divergence at
zero-based SR step 13, sample 67, Transfer `(1,2,up)`. Counters, parameters and
Pfaffian inputs were identical. The port forced every Horner `muladd` into
FMA; Julia 1.13.1 ARM vectorizes the sine polynomial branches and leaves the
final `DS2` step unfused. At sine input `bfe5f7d1fcbd2e12`, Rust returned
`bfe4486084862abb`, native Julia `bfe4486084862aba`. The RBM log ratio then
differed by `9.71e-17`, and later CG updates changed the saved configurations.
This is permitted general-math rounding, rather than an algorithm defect.
The sine implementation and reference arithmetic are retained without forcing
one compiler's FMA choices. Twenty-step long prefixes verify reproducibility of this
implementation's discrete state, complete RNG block and numerical output;
short prefixes and fixed-input numerical kernels retain independent references.
Inspect the independent native kernel with
`julia +1.13.1 -e 'using InteractiveUtils; code_native(stdout, Base.Math.sin_kernel, (Float64,); debuginfo=:none)'`.
See [NUMERICAL_COMPARISONS.md](NUMERICAL_COMPARISONS.md) for the distinction
between independent RNG contracts and numerically sensitive long trajectories.

## Independent reference generation

The ARM overlays use Julia 1.13.1 with the reference Manifest-v1.13.toml.
An ABI-only adapter forwards checked ILP64 arguments to LP64 OpenBLAS.
The real bilinear reduction replays the archived Intel AVX2 tree, verified
against 132 independent fixtures. No numerical implementation is copied from
Rust. Existing C FSZ and counter/coefficient adaptations have separate
provenance. These are mixed references, not full C executable or MPI parity.

`tests/fixtures/macos_arm_julia/<OpenBLAS core>/` contains `vortexm4` and
`neoversen1` results. The latter uses `OPENBLAS_CORETYPE=NEOVERSEN1` on the
local M4; it does not establish native M1 verification. Rust tests select the
linked core and decompress checked-in data without invoking an oracle.

Optional reproduction, using a fresh output directory:

```sh
mkdir -p /tmp/mvmc-arm-reference
clang -O0 -ffp-contract=off -dynamiclib \
  -L"$(brew --prefix openblas)/lib" -lopenblas \
  c_toolbox/blas_lp64_reference.c -o /tmp/mvmc-arm-reference/libblas.dylib
uv run --no-project python scripts/check_native_fsz_runner_bridge.py \
  --build-dir /tmp/mvmc-arm-reference/native-fsz
julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_macos_arm_references.jl \
  --blas-bridge=/tmp/mvmc-arm-reference/libblas.dylib \
  --fsz-bridge-dir=/tmp/mvmc-arm-reference/native-fsz \
  --output-root=/tmp/mvmc-arm-reference/fixtures
```

The generator supports `--job=solver/case/store` and `--job=fixed`.
Repeat with `OPENBLAS_CORETYPE=NEOVERSEN1` for that kernel.
`generate_sr_direct_fixed_reference.jl` separately replays 51 archived
systems using the same BLAS bridge; inputs/matrices/gradients stay unchanged.
Provenance and logical SHA-256 records accompany the generated results.
`uv run --no-project python scripts/package_macos_arm_references.py` verifies
hashes and packages checked-in references as deterministic lossless gzip,
recording reused artifacts and omitted duplicate solver dumps.

### Reviewed-CG overlays (BLAS bridge + archived AVX2)

The PR54/`reviewed_cg_62b` CG runner lineage needs the same platform-overlay
treatment. A plain native Julia aarch64 run is not portable enough: the
ill-conditioned SR-CG solve amplifies Julia-vs-Rust GEMV/reduction rounding to
`~1e-6`. The ARM overlay must be generated with the ABI-only system LP64
OpenBLAS bridge (`scripts/reference_lp64_blas.jl`,
`c_toolbox/blas_lp64_reference.c`) and the archived Julia AVX2 two-hop bilinear
replay (`scripts/reference_archived_avx2.jl`), which is what the Rust sampling
kernels retain. `c_toolbox/runner_opt_windows_reviewed_arm.jl` wraps the
reviewed generator with both; `c_toolbox/reviewed_cg_arm.md` documents the exact
build, manifest, generation, import and verification commands. Rust resolves
the reviewed reads through `julia_fixture::fixture_path`, so
`tests/fixtures/macos_arm_julia/<openblas-core>/reviewed_cg_62b/<case>/` overrides
the archived Linux lineage per file. The same core-specific overlays cover the
fixed-input CG reference (`.../sr_cg/c_refresh/`, from `ctest_cg_refresh.c` on
the host) and the PhysCal two-sample references (`.../physcal_181/two-samples/`
and `.../native-c-weighted-green/`).

## CI

The workflow checks Linux x86_64 and macOS ARM64 with the `ci` profile,
following tenferro-rs's matrix and tooling. See [DEVELOPMENT.md](DEVELOPMENT.md)
for the optimized numerical-test profile. Hosted CI uses checked-in references
and system BLAS/MPI; fixture generation remains an optional developer command.
