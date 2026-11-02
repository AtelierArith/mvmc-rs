#!/usr/bin/env bash
set -euo pipefail
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
inputs=/tmp/mvmc-583-default/inputs
out=/tmp/mvmc-582-spin-comparison
mkdir "$out/results"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
 for rep in 0 1 2 3; do
  if ((rep % 2 == 0)); then conditions='off-default on-zero on-default off-zero'; else conditions='off-zero on-default on-zero off-default'; fi
  for condition in $conditions; do
   case "$condition" in off-*) enabled=0;; on-*) enabled=1;; esac
   case "$condition" in *-default) spin=default;; *-zero) spin=0;; esac
   docker run --rm --cpuset-cpus=0-15 --shm-size=1g -v "$inputs:/inputs:ro" -v "$out/results:/results" -v mvmc-582-shared-team-target:/target:ro -w "/inputs/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e MVMC_RS_INNER_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro -e "MVMC_EXPERIMENT_SHARED_CPU_TEAM=$enabled" -e "BENCH_SPIN=$spin" -e "BENCH_MODE=$mode" -e "BENCH_DEST=/results/${mode}-${condition}-r${rep}" --entrypoint bash "$image" -lc '
    set -euo pipefail
    unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE
    if [[ "$BENCH_SPIN" != default ]]; then export MVMC_RS_TENFERRO_SPIN_POLLS="$BENCH_SPIN"; fi
    TIMEFORMAT="wall=%R user=%U sys=%S"
    launcher=(/opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc)
    if [[ "$BENCH_MODE" = opt ]]; then
     time timeout --kill-after=5s 120s "${launcher[@]}" /inputs/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
    else
     time timeout --kill-after=5s 120s "${launcher[@]}" /inputs/phys/namelist.def --physcal /inputs/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
    fi
   ' > "$out/${mode}-${condition}-r${rep}.log" 2>&1
  done
 done
done
