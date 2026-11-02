#!/usr/bin/env bash
set -euo pipefail
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
for workers in 4 16; do
 docker run --rm --cpuset-cpus=0-15 --shm-size=1g -v /tmp/mvmc-581-profile/inputs/l64-opt:/input:ro -v /tmp/mvmc-582-l64-diagnostic:/out -v mvmc-582-rebase-target:/target:ro -w /input -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=16 -e "MVMC_RS_INNER_THREADS=$workers" -e "MVMC_RS_TENFERRO_THREADS=$workers" -e MVMC_RS_SR_BACKEND=tenferro -e MVMC_EXPERIMENT_SHARED_CPU_TEAM=1 -e MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1 -e "BENCH_DEST=/out/output-$workers" --entrypoint bash "$image" -lc '
  set -euo pipefail
  unset MVMC_RS_TENFERRO_SPIN_POLLS
  timeout --kill-after=5s 180s /target/release/mvmc /input/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
 ' > "/tmp/mvmc-582-l64-diagnostic/threads-$workers.log" 2>&1
done
