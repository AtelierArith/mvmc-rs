#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/mvmc-582-spin-diagnostic
for mode in opt phys; do
 docker run --rm --cpuset-cpus=0-15 --shm-size=1g -v /tmp/mvmc-583-default/inputs:/inputs:ro -v /tmp/mvmc-582-spin-diagnostic:/out -v mvmc-582-shared-team-target:/target:ro -w "/inputs/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e MVMC_RS_INNER_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro -e MVMC_EXPERIMENT_SHARED_CPU_TEAM=1 -e MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1 -e MVMC_RS_TENFERRO_SPIN_POLLS=0 -e "BENCH_MODE=$mode" --entrypoint bash vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc '
  set -e; unset MVMC_RS_TENFERRO_THREADS
  launcher=(/opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc)
  if [[ "$BENCH_MODE" = opt ]]; then
   timeout --kill-after=5s 120s "${launcher[@]}" /inputs/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir /out/opt-output
  else
   timeout --kill-after=5s 120s "${launcher[@]}" /inputs/phys/namelist.def --physcal /inputs/phys/zqp_opt.dat --out-dir /out/phys-output
  fi
 ' > "/tmp/mvmc-582-spin-diagnostic/$mode.log" 2>&1
done
