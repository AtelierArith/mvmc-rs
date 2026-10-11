#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
output_root=${1:-bench-out/rank-thread-sweep-main-20261011}
for pair in '1 1' '1 16' '2 8' '4 4' '8 2' '16 1'; do
    read -r ranks threads <<< "$pair"
    bash bench/run.sh --sites 32 64 --ranks "$ranks" --threads "$threads" \
        --steps 300 --groups 100 --samples 320 --warmups 1 --reps 3 \
        --output "$output_root/r${ranks}-t${threads}"
done
