#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-582-l64-clean
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
input=/tmp/mvmc-581-profile/inputs/l64-opt
mkdir "$root/results"
sha256sum "$input"/* > "$root/input-sha256.txt"
for workers in 1 4 16; do
 for rep in 0 1 2 3; do
  if ((rep % 2 == 0)); then modes='off on'; else modes='on off'; fi
  for mode in $modes; do
   if [[ "$mode" = on ]]; then enabled=1; else enabled=0; fi
   docker run --rm --cpuset-cpus=0-15 --shm-size=1g -v "$input:/input:ro" -v "$root/results:/results" -v mvmc-582-rebase-target:/target:ro -w /input -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=16 -e "MVMC_RS_INNER_THREADS=$workers" -e "MVMC_RS_TENFERRO_THREADS=$workers" -e MVMC_RS_SR_BACKEND=tenferro -e "MVMC_EXPERIMENT_SHARED_CPU_TEAM=$enabled" -e "BENCH_DEST=/results/${mode}-i${workers}-r${rep}" --entrypoint bash "$image" -lc '
    set -euo pipefail
    unset MVMC_RS_TENFERRO_SPIN_POLLS MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE
    TIMEFORMAT="wall=%R user=%U sys=%S"
    time timeout --kill-after=5s 180s /target/release/mvmc /input/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
   ' > "$root/${mode}-i${workers}-r${rep}.log" 2>&1
  done
 done
done
