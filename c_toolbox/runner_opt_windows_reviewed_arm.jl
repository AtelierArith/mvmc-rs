# Optional Apple Silicon mixed-reference driver for the reviewed-PR54 CG lineage.
#
# Installs the ABI-only system LP64 OpenBLAS bridge and replays the archived
# Julia AVX2 two-hop bilinear reduction, then runs the unchanged reviewed-fork
# generator `runner_opt_windows_reviewed.jl`. Cargo never loads or runs this
# file; it is reference generation only, and it emits no files into the repo.
#
# The bridge makes Julia's BLAS calls use the same system OpenBLAS kernels that
# Rust links, and the archived AVX2 replay restores the reduction tree that the
# Rust sampling kernels retain. Without both, even a native Julia aarch64 run
# differs from Rust by ~1e-6 for these ill-conditioned SR-CG systems, because
# the CG solve amplifies last-bit GEMV/reduction rounding.
#
# Usage:
#   julia +1.13.1 --project=extern/Julia-mVMC \
#     c_toolbox/runner_opt_windows_reviewed_arm.jl STAGE cg \
#     --blas-bridge=/path/to/libblas.dylib [reviewed generator options...]
#
# See c_toolbox/reviewed_cg_arm.md for the full procedure.
using MVMCOptimizers, MVMCExpertModeParsers
const DRIVER_REPO = normpath(joinpath(@__DIR__, ".."))
const BRIDGE_OPTIONS = filter(a -> startswith(a, "--blas-bridge="), ARGS)
length(BRIDGE_OPTIONS) == 1 ||
    error("exactly one --blas-bridge=PATH option is required")
const BLAS_BRIDGE = split(only(BRIDGE_OPTIONS), "="; limit=2)[2]
isempty(BLAS_BRIDGE) && error("empty --blas-bridge path")
isfile(BLAS_BRIDGE) || error("missing BLAS bridge: $BLAS_BRIDGE")
filter!(a -> !startswith(a, "--blas-bridge="), ARGS)
include(joinpath(DRIVER_REPO, "scripts", "reference_lp64_blas.jl"))
const BLAS_BRIDGE_PROVENANCE = ReferenceLP64BLAS.install!(BLAS_BRIDGE)
include(joinpath(DRIVER_REPO, "scripts", "reference_archived_avx2.jl"))
println("Installed LP64 OpenBLAS bridge ($BLAS_BRIDGE_PROVENANCE) and archived AVX2 reduction")
include(joinpath(@__DIR__, "runner_opt_windows_reviewed.jl"))
