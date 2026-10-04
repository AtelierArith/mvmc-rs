#!/usr/bin/env bash
# Run independent numerical oracles in a reviewable copy, never Rust outputs.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
helper="$root/scripts/numerical_reference_stage.py"
reference=c
output=
check=false
instantiate=false
prefixes=1,2,3,50
compiler=${CC:-cc}
julia_binary=${JULIA_BINARY:-}
selected=()

help() {
    cat <<'HELP'
Usage: scripts/generate-numerical-references.sh [options]

By default generate the independent C suite in a fresh reviewable staging tree.
The source checkout and its macOS fixtures are never overwritten by generation.

  --reference c|julia|all  Choose default suites (default: c)
  --workspace DIRECTORY  Source checkout (default: this script's repository)
  --suite NAME           Select a suite; may be repeated (see --list)
  --output DIRECTORY     Use a new/empty staging directory (default: mktemp)
  --check                Run existing generator checks instead of --write
  --compiler EXECUTABLE  C compiler; default $CC or cc (one executable)
  --julia EXECUTABLE     Julia 1.13.1 binary; otherwise Julia binary/juliaup on PATH
  --instantiate          Instantiate/build Julia packages in the staging copy
  --steps 1,2,3,50      Prefixes for explicitly selected Linux Julia runners
  --list                 List supported suites, without running prerequisites
  --apply DIRECTORY      Apply previously reviewed successful staging changes
  --help                 Print this help

Examples:
  scripts/generate-numerical-references.sh --suite c-division
  scripts/generate-numerical-references.sh --suite c-projection --check
  scripts/generate-numerical-references.sh --reference julia --suite julia-fsz-moves
  scripts/generate-numerical-references.sh --apply /tmp/mvmc-references.ABCDEFGH

Python always runs through uv. C needs a compiler and the pinned C source
submodule; Julia needs exactly 1.13.1, its pinned manifest and installed packages.
Ordinary Cargo tests never invoke this script or an oracle runtime.
HELP
}
list() {
    cat <<'LIST'
c-division               Native C complex division
c-projection             Projection/QP count conversion
c-projection-flags       Projection parameter flags
c-integer-flags          Integer flags and RNG initialization
c-gutzwiller             Gutzwiller reader contracts
c-jastrow                Jastrow reader/count contracts
c-orbital                Antiparallel/parallel orbital readers
c-general-orbital        General orbital readers and signs
c-orbital-order          Orbital parameter ordering
c-orbital-initialization Orbital initialization and RNG
c-initial-records        Initial parameter records
c-rbm-reader             RBM reader contracts
c-rbm-counters           RBM counters
c-rbm-parameters         RBM parameters
c-interall-reader        InterAll input reader
c-interall-real          Normal real Green kernels
c-interall-complex       Normal complex Green kernels
c-fsz-real               FSZ real Green kernels
c-fsz-complex            FSZ complex Green kernels
c-fsz-energy-real        Complete scalar FSZ Hamiltonian
c-fsz-energy-complex     Complete complex FSZ Hamiltonian
c-fsz-measurements       Weighted/factored FSZ Green measurements
julia-fsz-setup          Small real FSZ setup/failure/RNG cases
julia-fsz-moves          Small real FSZ proposals/inverse/bilinear cases
julia-fsz-real-sampling  Small real FSZ sampling/RNG cases (explicit selection)
julia-fsz-complex-sampling Small complex FSZ sampling/RNG cases (explicit selection)
julia-linux-all          Full historical Linux overlay (explicit, longer workload)
julia-linux-setup        Linux overlay: fixed setup
julia-linux-real-sampling Linux overlay: real sampling
julia-linux-complex-sampling Linux overlay: complex sampling
julia-linux-dh2-history  Linux overlay: DH2 history
julia-linux-dh4-history  Linux overlay: DH4/DH24 history
julia-linux-direct-CASE  Linux overlay: Direct SR, NStore=0
julia-linux-store-CASE   Linux overlay: Direct SR, NStore=1
julia-linux-cg-CASE      Linux overlay: CG
  CASE: fsz interall pairhop_fsz dh2_fsz dh4_fsz dh24_fsz rbm_fsz opt_fsz
LIST
}
suite_script() {
    suite_flags=()
    case "$1" in
        c-division) suite_file=check_complex_division_c_parity.py ;;
        c-projection) suite_file=check_projection_count_c_parity.py ;;
        c-projection-flags) suite_file=check_projection_flags_c_parity.py ;;
        c-integer-flags) suite_file=check_integer_flags_c_parity.py ;;
        c-gutzwiller) suite_file=check_gutzwiller_contracts_c_parity.py ;;
        c-jastrow) suite_file=check_jastrow_contracts_c_parity.py ;;
        c-orbital) suite_file=check_orbital_contracts_c_parity.py ;;
        c-general-orbital) suite_file=check_general_orbital_c_parity.py ;;
        c-orbital-order) suite_file=check_orbital_order_c_parity.py ;;
        c-orbital-initialization) suite_file=check_orbital_initialization_c_parity.py ;;
        c-initial-records) suite_file=check_initial_records_c_parity.py ;;
        c-rbm-reader) suite_file=check_rbm_contracts_c_parity.py ;;
        c-rbm-counters) suite_file=check_rbm_counters_c_parity.py ;;
        c-rbm-parameters) suite_file=check_rbm_parameters_c_parity.py ;;
        c-interall-reader) suite_file=check_interall_reader_c_parity.py ;;
        c-interall-real) suite_file=check_interall_real_c_parity.py ;;
        c-interall-complex) suite_file=check_interall_complex_c_parity.py ;;
        c-fsz-real) suite_file=check_fsz_green_c_parity.py; suite_flags=(--real) ;;
        c-fsz-complex) suite_file=check_fsz_green_c_parity.py ;;
        c-fsz-energy-real) suite_file=check_fsz_green_c_parity.py; suite_flags=(--real --hamiltonian) ;;
        c-fsz-energy-complex) suite_file=check_fsz_green_c_parity.py; suite_flags=(--hamiltonian) ;;
        c-fsz-measurements) suite_file=check_fsz_green_c_parity.py; suite_flags=(--measurements) ;;
        julia-linux-*) suite_file=generate_linux_julia_references.jl ;;
        julia-fsz-setup) suite_file=check_real_fsz_setup_parity.jl ;;
        julia-fsz-moves) suite_file=check_real_fsz_moves_parity.jl ;;
        julia-fsz-real-sampling) suite_file=check_real_fsz_sampling_parity.jl ;;
        julia-fsz-complex-sampling) suite_file=check_complex_fsz_sampling_parity.jl ;;
        *) echo "Unknown suite: $1 (use --list)" >&2; exit 2 ;;
    esac
}
while (($#)); do
    case "$1" in
        --help|-h) help; exit 0 ;;
        --list) list; exit 0 ;;
        --reference|--workspace|--output|--suite|--compiler|--julia|--apply|--steps)
            (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
            case "$1" in
                --steps) prefixes=$2 ;;
                --reference) reference=$2 ;;
                --workspace) root=$(cd "$2" && pwd -P) ;;
                --output) output=$2 ;;
                --suite) selected+=("$2") ;;
                --compiler) compiler=$2 ;;
                --julia) julia_binary=$2 ;;
                --apply)
                    (($# == 2)) || { echo '--apply must be the only option' >&2; exit 2; }
                    command -v uv >/dev/null || { echo 'uv is required' >&2; exit 2; }
                    exec uv run --no-project python "$helper" apply "$root" "$2"
                    ;;
            esac
            shift 2 ;;
        --check) check=true; shift ;;
        --instantiate) instantiate=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$reference" in c|julia|all) ;; *) echo 'Expected --reference c, julia or all' >&2; exit 2 ;; esac
if ((${#selected[@]} == 0)); then
    if [[ $reference == c || $reference == all ]]; then
        selected=(c-division c-projection c-projection-flags c-integer-flags c-gutzwiller c-jastrow c-orbital c-general-orbital c-orbital-order c-orbital-initialization c-initial-records c-rbm-reader c-rbm-counters c-rbm-parameters c-interall-reader c-interall-real c-interall-complex c-fsz-real c-fsz-complex)
    fi
    if [[ $reference == julia || $reference == all ]]; then
        selected+=(julia-fsz-setup julia-fsz-moves)
    fi
fi
needs_c=false; needs_julia=false
for suite in "${selected[@]}"; do
    suite_script "$suite"
    [[ -f "$root/scripts/$suite_file" ]] || { echo "Missing oracle: $suite_file" >&2; exit 2; }
    case "$suite" in c-*) needs_c=true ;; julia-*) needs_julia=true ;; esac
done
command -v uv >/dev/null || { echo 'uv is required; install it before generating' >&2; exit 2; }
if $needs_c; then
    [[ -f "$root/extern/mVMC-1.3.0/src/mVMC/readdef.c" ]] || { echo 'Initialize extern/mVMC-1.3.0 with git submodule update --init' >&2; exit 2; }
    compiler=$(command -v "$compiler") || { echo 'C compiler must name one executable' >&2; exit 2; }
fi
julia_command=()
if $needs_julia; then
    [[ -f "$root/extern/Julia-mVMC/Manifest-v1.13.toml" ]] || { echo 'Pinned Julia Manifest-v1.13.toml is required' >&2; exit 2; }
    if [[ -n "$julia_binary" ]]; then
        julia_binary=$(command -v "$julia_binary") || { echo 'Julia binary not found' >&2; exit 2; }
        julia_command=("$julia_binary")
    elif command -v juliaup >/dev/null; then
        julia_command=(julia +1.13.1)
    elif command -v julia >/dev/null; then
        julia_command=(julia)
    else
        echo 'Install Julia 1.13.1 or pass --julia /path/to/julia' >&2; exit 2
    fi
    [[ $("${julia_command[@]}" --startup-file=no -e 'print(VERSION)') == 1.13.1 ]] \
        || { echo 'Julia version must be exactly 1.13.1' >&2; exit 2; }
fi
if $instantiate && ! $needs_julia; then echo '--instantiate requires a Julia suite' >&2; exit 2; fi
if [[ -z "$output" ]]; then output=$(mktemp -d "${TMPDIR:-/tmp}/mvmc-references.XXXXXXXX"); fi
mkdir -p "$output"
output=$(cd "$output" && pwd -P)
uv run --no-project python "$helper" prepare "$root" "$output"
export PYTHONDONTWRITEBYTECODE=1
if $needs_c; then
    mkdir -p "$output/bin"
    cat > "$output/bin/cc" <<'COMPILER'
#!/usr/bin/env bash
printf '%q ' "$REFERENCE_CC" "$@" >> "$REFERENCE_COMPILER_LOG"
printf '\n' >> "$REFERENCE_COMPILER_LOG"
exec "$REFERENCE_CC" "$@"
COMPILER
    chmod +x "$output/bin/cc"
    export REFERENCE_CC="$compiler" CC="$compiler" PATH="$output/bin:$PATH"
    export REFERENCE_COMPILER_LOG="$output/compiler-commands.log"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export JULIA_NUM_THREADS=1
trap 'status=$?; trap - EXIT; uv run --no-project python "$helper" finish "$root" "$output" "$status"; echo "Review staging tree: $output"; exit "$status"' EXIT
uv run --no-project python "$helper" environment "$root" "$output"
if $needs_julia; then
    if $instantiate; then
        bootstrap=("${julia_command[@]}" --startup-file=no --project="$output/workspace/extern/Julia-mVMC" -e 'using Pkg; Pkg.instantiate(); Pkg.build("PfaPack"); Pkg.build("SFMT")')
        printf '%s\t' julia-package-setup >> "$output/commands.txt"
        printf '%q ' "${bootstrap[@]}" >> "$output/commands.txt"
        printf '\n' >> "$output/commands.txt"
        "${bootstrap[@]}" 2>&1 | tee "$output/julia-package-setup.log"
        cmp "$root/extern/Julia-mVMC/Manifest-v1.13.toml" "$output/workspace/extern/Julia-mVMC/Manifest-v1.13.toml" \
            || { echo 'Julia package setup changed the pinned manifest; refusing generation' >&2; exit 2; }
    fi
    "${julia_command[@]}" --startup-file=no --project="$output/workspace/extern/Julia-mVMC" \
        -e 'using LinearAlgebra, Pkg, OpenBLAS_jll; @assert VERSION == v"1.13.1"; @assert Pkg.project().path == joinpath(ARGS[1], "Project.toml"); println("Julia ", VERSION, "; ", BLAS.get_config(), "; threads=", BLAS.get_num_threads()); println("Julia bundled OpenBLAS ", unsafe_string(ccall((:openblas_get_config64_, OpenBLAS_jll.libopenblas), Cstring, ()))); using MVMCOptimizers, MVMCExpertModeParsers, SFMT' \
        "$output/workspace/extern/Julia-mVMC" > "$output/julia-environment.log" 2>&1
    cat "$output/julia-environment.log"
fi
cd "$output/workspace"
linux_suites=()
for suite in "${selected[@]}"; do
    suite_script "$suite"
    if [[ $suite == julia-linux-* ]]; then linux_suites+=("$suite"); continue; fi
    case "$suite" in
        c-*) command_args=(uv run --no-project python "scripts/$suite_file") ;;
        julia-*) command_args=("${julia_command[@]}" --startup-file=no --project=extern/Julia-mVMC "scripts/$suite_file") ;;
    esac
    command_args+=("${suite_flags[@]}")
    if ! $check; then command_args+=(--write); fi
    printf '%s\t' "$suite" >> "$output/commands.txt"
    printf '%q ' "${command_args[@]}" >> "$output/commands.txt"
    printf '\n' >> "$output/commands.txt"
    echo "Running $suite"
    "${command_args[@]}" 2>&1 | tee "$output/$suite.log"
done

if ((${#linux_suites[@]})); then
    command_args=("${julia_command[@]}" --startup-file=no --project=extern/Julia-mVMC scripts/generate_linux_julia_references.jl "$prefixes" "$output/julia-generated-paths.txt" "${linux_suites[@]}")
    printf '%s\t' julia-linux-overlay >> "$output/commands.txt"
    printf '%q ' "${command_args[@]}" >> "$output/commands.txt"
    printf '\n' >> "$output/commands.txt"
    "${command_args[@]}" 2>&1 | tee "$output/julia-linux-overlay.log"
    uv run --no-project python "$helper" julia-overlay "$root" "$output" "$check"
fi
