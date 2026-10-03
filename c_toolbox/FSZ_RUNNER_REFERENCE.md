# Optional native FSZ energy / Julia runner reference

This developer-only bridge uses the original Julia 1.13.1 runner, initialization,
SFMT draws, sampler, SR solvers and output. Only `calculate_hamiltonian_fsz` calls
the serial native C local-energy kernel. It covers the complex, no-RBM FSZ models,
including DH2/DH4; it does not establish full C executable, C sampling, C SR or
MPI parity. Cargo and ordinary Rust tests neither build nor load this library.
Historical Julia fixtures are retained. New fixtures are explicitly labelled
under `tests/fixtures/c_kernel_order/native_fsz/`.

`fsz_runner_reference.c` borrows interleaved Julia complex buffers and returns
two doubles, avoiding assumptions about a complex-valued C return ABI. Native
workspace copies isolate proposed moves. Every call checks Julia's saved
projection counts against the actual C `MakeProjCnt`, including DH type-major
offsets. The bridge does not draw RNG. Its globals require one Julia thread;
only the serial MPI branch runs, and a collective call aborts.

The `.inc` files retain upstream GPL notices, full source SHA-256 values and
function boundaries. `check_native_fsz_runner_bridge.py` reextracts and checks
their exact contents before building. It verifies 72 complex and 36 scalar
native energy fixtures with explicit absolute/relative numerical bounds and
checks every borrowed operand for mutation exactly. The original #186 gate was
bitwise; the current #190 policy is in
[NUMERICAL_COMPARISONS.md](../docs/NUMERICAL_COMPARISONS.md).
The scalar library supports these kernel checks; runner generation uses the
complex library. Recorded provenance includes compiler, platform, source
hashes and `-O0 -ffp-contract=off -fPIC -fvisibility=hidden`; linkage is
`-dynamiclib` on macOS or `-shared` on Linux. Platform-specific native fixture
selection keeps Apple and GNU complex arithmetic distinct.

From the repository root, build and validate independently with uv:

```sh
uv run --no-project python scripts/check_native_fsz_runner_bridge.py --build-dir /tmp/mvmc-native-fsz-reference
```

`--write` explicitly materializes the additional exact `MakeProjCnt` snippet;
it does not regenerate numerical expectations. Use the pinned Julia manifest:

```sh
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_runner_fixtures.jl --bridge-dir=/tmp/mvmc-native-fsz-reference --steps=1,2,3,50 --write
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_runner_fixtures.jl --bridge-dir=/tmp/mvmc-native-fsz-reference --steps=1,2,3,50
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_runner_fixtures.jl --bridge-dir=/tmp/mvmc-native-fsz-reference --steps=1,2,3,50 --general
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_dh_boundaries.jl --bridge-dir=/tmp/mvmc-native-fsz-reference --write
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_dh_boundaries.jl --bridge-dir=/tmp/mvmc-native-fsz-reference
```

For a short job use `--job=sr_direct/dh24_fsz/0 --steps=1`. The generator reuses
the existing runner observation hooks, preserving their controls and inputs.
It prints exact comparisons of saved configurations and complete historical
RNG blocks. `--general` verifies AP/P references rather than writing alternate
expected values. Numerical differences in SR/BLAS must be examined with
explicit tolerances; those tolerances do not apply to configuration or RNG
comparisons. DH boundary generation similarly compares loaded parameters,
flags and RNG against their original source references.

`inheritance.tsv` records explicit native-relative / historical-relative /
SHA-256 mappings for unchanged files. The optional Python utility verifies
historical bytes and any remaining native duplicate before pruning:

```sh
uv run --no-project python scripts/native_fsz_fixture_inheritance.py --write
uv run --no-project python scripts/native_fsz_fixture_inheritance.py
uv run --no-project python scripts/native_fsz_fixture_inheritance.py --prune
```

Only manifest-listed, hash-verified duplicates may be removed. Generator
read-only checks resolve inherited files through that manifest and verify
their hashes; missing changed expectations fail. Mac mapping discovery is
scoped to `sr_direct/` and `sr_cg/`; externally supplied platform mappings are
preserved and verified. `--write` generation creates complete outputs again.

OPT FSZ CG illustrates why the mixed references retain separate labels:
native C energy changes energy and post-SR parameters at step 1, while saved
configurations remain identical to historical Julia through step 47 and first
differ at step 48. The step-50 RNG block therefore differs as well. Both
independent historical/native observation audits reproduce their respective
checked-in 50-step energy, parameters, configurations and RNG exactly. This
trajectory change is retained explicitly; numerical tolerances do not conceal
it. `scripts/audit_native_fsz_opt_cg.jl` records every step without extra RNG
draws and can trace sampler draws/proposals/acceptance at a chosen one-based
`--trace-step`, writing only to an explicit temporary directory.

The step-48 audit locates the first changed acceptance decision at zero-based
sampling `out_step=461`, `in_step=4`, decision 2771: the same conduction-electron
spin-flip proposal moves electron 2 at site 5 from spin 1 to spin 0. Both runs
use exactly `r = 0.1620418371167034`. Historical Julia computes
`w = 0.16202029772339502` and rejects; the native-energy reference computes
`w = 0.16204321917576325` and accepts. The following proposal differs because
the accepted configuration differs. Sampling draw counts and hashes of every
draw agree through step 47; step 48 consumes 60,606 historical draws versus
60,768 native-reference draws. Their common 60,606-value UInt32 stream prefix
is still identical; the changed configuration alters how many draws the
subsequent proposals consume.

The numerical predecessor is visible before any configuration divergence:
step-1 energy differs by at most `2.42861286636753e-17` per component, amplified
by CG into a maximum post-SR parameter component difference of
`1.9290211472622332e-7`. At step 47 the maximum parameter difference is
`0.0001468500716697574` and the energy component difference is
`6.317563877257862e-6`. This is a traced threshold crossing after the intended
C energy change. The observation hooks add no RNG draws; both traced runs
reproduce their untraced 50-step energy, parameters, saved configurations and
624-word RNG blocks exactly. Reproduce the independent audits with:

```sh
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/audit_native_fsz_opt_cg.jl --mode=historical --trace-step=48 --output-dir=/tmp/mvmc-opt-fsz-cg-historical-trace
JULIA_NUM_THREADS=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/audit_native_fsz_opt_cg.jl --mode=native --bridge-dir=/tmp/mvmc-native-fsz-reference --trace-step=48 --output-dir=/tmp/mvmc-opt-fsz-cg-native-trace
```

`step-0-*` through `step-49-*` record zero-based optimizer snapshots; sampler
`--trace-step` uses one-based iteration numbers. `sampling-trace.txt` records
proposals and acceptance values in exact IEEE hexadecimal form;
`sampling-draws.txt` records the chosen iteration's actual UInt32 draws.
The per-step draw-count/SHA files audit earlier sampling without copying the
process-global SFMT generator or perturbing its state.
