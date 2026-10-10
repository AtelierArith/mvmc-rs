#!/usr/bin/env bash
set -euo pipefail
cd /home/terasaki/work/atelierarith/mvmc-rs
for pair in '1 1' '1 16' '2 8' '4 4' '8 2' '16 1'; do
    read -r ranks threads <<< "$pair"
    cell="bench-out/rank-thread-sweep-20261010/r${ranks}-t${threads}"
    printf 'Starting ranks=%s threads=%s\n' "$ranks" "$threads"
    bash bench/run.sh --sites 32 64 --ranks "$ranks" --threads "$threads" \
        --steps 300 --groups 100 --samples 320 --warmups 1 --reps 3 \
        --output "$cell"
done
