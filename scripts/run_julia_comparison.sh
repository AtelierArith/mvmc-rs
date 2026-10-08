#!/usr/bin/env bash
# Reproduce the 2026-10-08 comparison (issue #488). See benchmark/julia_comparison/README.md.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
output_dir="target/bench/julia-comparison-$(date -u +%Y%m%dT%H%M%SZ)-$$"
dry_run=0
usage() {
    cat <<'EOF'
Usage: scripts/run_julia_comparison.sh [--output DIR] [--dry-run]

Run small optimization (50 steps), Hubbard L16/L24/L32 optimization (300 steps),
small PhysCal including Lanczos, and Hubbard PhysCal (100 samples), sequentially.
Fixed settings: Julia 1.13.1, 3 repetitions, 1 warmup, 1 thread, 1 Rust inner worker.
DIR must be new. CSVs, reports, logs, the Julia lock and environment are saved there.
--dry-run prints benchmark commands without preparing or running anything.
EOF
}
while (($#)); do
    case "$1" in
        --output)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo '--output requires a directory' >&2
                exit 2
            fi
            output_dir=$2
            shift 2
            ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$output_dir" != /* ]]; then
    output_dir="$repo_root/$output_dir"
fi

export MVMC_RS_INNER_THREADS=1
export MVMC_RS_SR_BACKEND=c-order
export MVMC_RS_MEASURE_PF_BACKEND=c-order
export MVMC_BLAS_PROVIDER=openblas
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 JULIA_NUM_THREADS=1
unset MVMC_RS_INNER_THRESHOLD

run_bench() {
    local name=$1
    shift
    if ((dry_run)); then
        printf 'MVMC_RS_INNER_THREADS=1 MVMC_RS_SR_BACKEND=c-order MVMC_BLAS_PROVIDER=openblas '
        printf '%q ' "$@"
        printf '\n'
    else
        printf '\n=== %s ===\n' "$name"
        "$@" 2>&1 | tee "$output_dir/$name.log"
    fi
}

if ((!dry_run)); then
    for command in cargo rustc julia git sha256sum; do
        command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
    done
    if [[ -e "$output_dir" ]]; then
        echo "Output already exists; choose a new directory: $output_dir" >&2
        exit 1
    fi
    julia_root="$repo_root/extern/Julia-mVMC"
    if [[ ! -f "$julia_root/Project.toml" ]]; then
        echo 'Initialize references first: git submodule update --init --recursive' >&2
        exit 1
    fi
    lock_source="$repo_root/benchmark/julia_comparison/Manifest-v1.13.toml"
    lock_target="$julia_root/Manifest-v1.13.toml"
    created_lock=0
    cleanup() {
        if ((created_lock)); then
            rm -f -- "$lock_target"
        fi
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ -e "$lock_target" || -L "$lock_target" ]]; then
        if ! cmp -s "$lock_source" "$lock_target"; then
            echo "Existing Julia lock differs; preserve it elsewhere before running: $lock_target" >&2
            exit 1
        fi
    else
        cp "$lock_source" "$lock_target"
        created_lock=1
    fi
    mkdir -p "$output_dir"
    cp "$lock_source" "$output_dir/Manifest-v1.13.toml"
    {
        date -u
        uname -a
        rustc --version
        cargo --version
        git rev-parse HEAD
        git status --short
        git submodule status --recursive
        sha256sum "$lock_source"
        env | sort | sed -n '/^MVMC_/p; /^JULIA_NUM_THREADS=/p; /_NUM_THREADS=/p; /^RUSTFLAGS=/p; /^CARGO_ENCODED_RUSTFLAGS=/p; /^CARGO_PROFILE_RELEASE_/p'
        if command -v lscpu >/dev/null; then lscpu; fi
    } > "$output_dir/environment.txt"
    # Missing native libraries require make, g++, gfortran and BLAS/LAPACK development packages.
    julia +1.13.1 --project="$julia_root" -e '
        using Pkg, Libdl
        Pkg.instantiate()
        root = dirname(Base.active_project())
        sfmt = joinpath(root, "SFMT.jl", "deps", "sfmt", "libsfmt." * Libdl.dlext)
        ltl = joinpath(root, "PfaPack.jl", "deps", "libltl2inv." * Libdl.dlext)
        if !isfile(sfmt); Pkg.build("SFMT"); end
        if !isfile(ltl); Pkg.build("PfaPack"); end
        using MVMCOptimizers, LinearAlgebra
        BLAS.set_num_threads(1)
        println("Julia: ", VERSION)
        println("BLAS: ", BLAS.get_config())
        println("BLAS threads: ", BLAS.get_num_threads())
    ' 2>&1 | tee "$output_dir/julia-setup.log"
    cmp "$lock_source" "$lock_target" || {
        echo 'Julia changed the pinned lock; aborting before measurement.' >&2
        exit 1
    }
fi

run_bench optimization-small cargo run -p xtask -- bench-julia \
    --steps 50 --reps 3 --warmups 1 --threads 1 --inner-workers 1 \
    --csv "$output_dir/julia_vs_rust.csv"
run_bench hubbard cargo run -p xtask -- bench-hubbard \
    --steps 300 --reps 3 --warmups 1 --threads 1 --inner-workers 1 \
    --model hubbard_chain_L16 --model hubbard_chain_L24 --model hubbard_chain_L32 \
    --csv "$output_dir/hubbard.csv" --report "$output_dir/hubbard.md"
run_bench physcal-small cargo run -p xtask -- bench-physcal \
    --reps 3 --warmups 1 --threads 1 --csv "$output_dir/physcal-small.csv"
run_bench physcal-hubbard cargo run -p xtask -- bench-physcal-hubbard \
    --reps 3 --warmups 1 --threads 1 --csv "$output_dir/physcal-hubbard.csv" \
    --report "$output_dir/physcal-hubbard.md" --keep-output

if ((!dry_run)); then
    if command -v ldd >/dev/null; then
        ldd target/release/mvmc > "$output_dir/rust-libraries.txt"
        ldd extern/Julia-mVMC/PfaPack.jl/deps/libltl2inv.so > "$output_dir/julia-native-libraries.txt"
    fi
    printf '\nResults: %s\n' "$output_dir"
fi
