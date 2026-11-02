#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-581-gap-sites-clean
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
inputs=/tmp/mvmc-581-profile/inputs
mkdir -p "$root"; [[ ! -e "$root/config.env" ]]
printf 'cpus=0-3 inner=1,4 rayon=4 blas=1 profiler=site binary_sha256=e4924982137bc73c016a142cde345680c4ded42dbb6ae947d1e662eeffc46994\n' > "$root/config.env"
sha256sum "$inputs"/{l32-opt,l32-phys,l64-opt}/* > "$root/input-sha256.txt"
for case in l32-opt l32-phys l64-opt; do
 for workers in 1 4; do
  docker run --rm --cpuset-cpus=0-3 --shm-size=1g -v "$inputs:/inputs:ro" -v "$root:/out" -v mvmc-581-gap-sites-target:/target:ro -w "/inputs/$case" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e "MVMC_RS_INNER_THREADS=$workers" -e MVMC_RS_INNER_GAP_SITE_PROFILE=1 -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_CASE=$case" -e "BENCH_WORKERS=$workers" --entrypoint bash "$image" -lc '
   set -euo pipefail
   unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS MVMC_RS_INNER_THRESHOLD MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_PROFILE MVMC_RS_INNER_GAP_PROFILE
   if [[ "$BENCH_CASE" = l32-phys ]]; then
     timeout --kill-after=5s 180s /target/release/mvmc namelist.def --physcal zqp_opt.dat --out-dir "/out/${BENCH_CASE}-${BENCH_WORKERS}-output"
   else
     timeout --kill-after=5s 180s /target/release/mvmc namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "/out/${BENCH_CASE}-${BENCH_WORKERS}-output"
   fi
  ' > "$root/${case}-${workers}.stdout" 2> "$root/${case}-${workers}.stderr"
 done
done
