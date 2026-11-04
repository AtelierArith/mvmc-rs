#!/usr/bin/env bash
# Issue #641 kernel comparison: Rust (scalar / hand AVX2 / pulp) and Julia (@simd / @turbo).
# Builds the standalone Rust crate twice (default code placement and -align-loops=64, to
# expose code-layout sensitivity), then alternates PROCS process runs of each build and of the
# Julia script, pinned to one CPU. Raw outputs go to OUT (default results/<UTC date>).
#
#   benchmark/pulp_lv_641/run_kernels.sh [OUT] [PROCS=5] [ROUNDS=15] [CPU=2]
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
out=${1:-$here/results/$(date -u +%Y%m%d)}
procs=${2:-5}
rounds=${3:-15}
cpu=${4:-2}
target=${CARGO_TARGET_DIR:-$repo/target/pulp_lv_641}
mkdir -p "$out"

(cd "$here/rust" && CARGO_TARGET_DIR="$target/default" cargo build --release --locked)
(cd "$here/rust" && RUSTFLAGS="-C llvm-args=-align-loops=64" \
    CARGO_TARGET_DIR="$target/align64" cargo build --release --locked)

{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) cpu_pin=$cpu"
    lscpu | grep -E 'Model name|^CPU\(s\)'
    rustc -Vv | grep -E '^rustc|^LLVM'
    julia +1.13.1 --version
    git -C "$repo" rev-parse HEAD
    git -C "$repo/extern/Julia-mVMC" rev-parse HEAD
    sha256sum "$repo/extern/Julia-mVMC/Manifest-v1.13.toml"
    uptime
} > "$out/env.txt"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for p in $(seq 1 "$procs"); do
    for build in default align64; do
        taskset -c "$cpu" "$target/$build/release/pulp_lv_641" 128 "$rounds" \
            > "$out/rust-$build-p$p.txt"
    done
    taskset -c "$cpu" julia +1.13.1 --project="$repo/extern/Julia-mVMC" -t 1 \
        "$here/julia/lv_kernels.jl" 128 "$rounds" > "$out/julia-p$p.txt"
    uptime >> "$out/load.txt"
done
uv run --no-project python "$here/summarize.py" "$out" | tee "$out/summary.md"
