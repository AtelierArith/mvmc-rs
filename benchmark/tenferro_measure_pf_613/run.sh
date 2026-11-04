#!/usr/bin/env bash
# Issue #613: matched single-thread timing of the measurement Pfaffian sources on the L32
# Hubbard chain, fixed-parameter PhysCal and 20-step Opt.
# Usage: run.sh PATH_TO_MVMC OUT_DIR [REPS]
set -euo pipefail
binary=$(realpath "$1")
out=$2
reps=${3:-3}
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export MVMC_RS_INNER_THREADS=1 MVMC_RS_TENFERRO_THREADS=1
unset MVMC_RS_SR_BACKEND MVMC_RS_MEASURE_BACKEND MVMC_RS_MEASURE_BATCH
printf 'workload\tsource\trep\twall_s\n' > "$out/timings.tsv"
for workload in physcal opt; do
    for source in calc-m-all c-order tenferro; do
        for ((rep = 1; rep <= reps; rep++)); do
            dir="$out/$workload-$source-$rep"
            rm -rf "$dir"
            if [[ $workload = physcal ]]; then
                args=("$root/benchmark/physcal/inputs/hubbard_chain_L32/namelist.def"
                    --physcal "$root/benchmark/physcal/inputs/hubbard_chain_L32/zqp_opt.dat")
            else
                args=("$root/benchmark/hubbard_chain/inputs/hubbard_chain_L32/namelist.def"
                    --nsteps 20 --nsmp 10 --initial-def none)
            fi
            start=$(date +%s.%N)
            MVMC_RS_MEASURE_PF_BACKEND=$source "$binary" "${args[@]}" --out-dir "$dir" \
                > "$dir.log" 2>&1
            end=$(date +%s.%N)
            printf '%s\t%s\t%s\t%.3f\n' "$workload" "$source" "$rep" \
                "$(echo "$end - $start" | bc)" >> "$out/timings.tsv"
        done
    done
done
cat "$out/timings.tsv"
