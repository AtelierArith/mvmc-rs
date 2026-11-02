#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-581-profile
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
volume=mvmc-583-solve-lease-target
for case in l32-opt l32-phys l64-opt; do
 for workers in 1 4; do
  docker run --rm --cpuset-cpus=0-3 --shm-size=1g -v "$root/inputs:/inputs:ro" -v "$root:/out" -v "$volume:/target:ro" -w "/inputs/$case" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e "MVMC_RS_INNER_THREADS=$workers" -e MVMC_RS_INNER_PROFILE=1 -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_CASE=$case" -e "BENCH_WORKERS=$workers" --entrypoint bash "$image" -lc '
   set -euo pipefail
   unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS MVMC_RS_INNER_THRESHOLD MVMC_RS_INNER_MIN_WORK_NS
   TIMEFORMAT="wall=%R user=%U sys=%S"
   if [[ "$BENCH_CASE" = l32-phys ]]; then
    time timeout --kill-after=5s 180s /target/release/mvmc namelist.def --physcal zqp_opt.dat --out-dir "/out/${BENCH_CASE}-${BENCH_WORKERS}-output"
   else
    time timeout --kill-after=5s 180s /target/release/mvmc namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "/out/${BENCH_CASE}-${BENCH_WORKERS}-output"
   fi
  ' > "$root/${case}-${workers}.log" 2>&1
 done
done
