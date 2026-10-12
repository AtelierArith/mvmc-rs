#!/usr/bin/env bash
# Julia GPU whole-runner E2E (Opt + PhysCal) on the host (issue #544).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$here/env.sh"
# shellcheck source=lib.sh
source "$here/lib.sh"

case=julia-gpu
dir="$RESULT_DIR/$case"
mkdir -p "$dir"
csv="$dir/$case.csv"
printf 'case,phase,rep,wall_s,all_s\n' >"$csv"
record_env "$dir/environment.txt"

log="$dir/benchmark.log"
echo "running Julia GPU E2E (steps=$STEPS samples=$SAMPLES reps=$REPS)..."
env \
    JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    JULIA_MVMC_E2E_BACKEND=gpu \
    JULIA_MVMC_E2E_STEPS="$STEPS" \
    JULIA_MVMC_E2E_SAMPLES="$SAMPLES" \
    JULIA_MVMC_E2E_WARMUP="$WARMUP_SWEEPS" \
    JULIA_MVMC_E2E_INTERVAL="$INTERVAL_SWEEPS" \
    JULIA_MVMC_E2E_REPS="$REPS" \
    JULIA_MVMC_E2E_PHYSCAL_RUNS="$PHYSCAL_RUNS" \
    JULIA_MVMC_E2E_SEED="$SEED" \
    JULIA_MVMC_GPU_DEVICE="${JULIA_MVMC_GPU_DEVICE:-0}" \
    "$JULIA_BIN" --project="$JULIA_SRC/test/gpu" \
    "$JULIA_SRC/test/gpu/benchmark_end_to_end.jl" "$INPUT_SRC/namelist.def" \
    >"$log" 2>&1

# Extract the GPU runner seconds the benchmark prints for each phase/repetition.
# Only lines that carry both a phase and a rep are per-repetition measurements;
# the trailing "phase=... GPU_..._seconds=[...]" summary lines are skipped.
awk -v label="$case" '
    /rep=[0-9]/ {
        ph="?"; rp="?";
        if (match($0, /phase=[a-z]+/)) ph=substr($0, RSTART+6, RLENGTH-6);
        if (match($0, /rep=[0-9]+/))   rp=substr($0, RSTART+4, RLENGTH-4);
        tag = (ph == "warm") ? "measure" : ph;   # cold includes compilation
        if (match($0, /GPU_optimization_seconds=[0-9.eE+-]+/))
            printf "%s,opt-%s,%s,%s,\n", label, tag, rp, substr($0, RSTART+25, RLENGTH-25);
        if (match($0, /GPU_PhysCal_seconds=[0-9.eE+-]+/))
            printf "%s,physcal-%s,%s,%s,\n", label, tag, rp, substr($0, RSTART+20, RLENGTH-20);
    }
' "$log" >>"$csv"

cat "$csv"
echo "results: $csv"
