#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-582-classify-diagnostic
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
for layout in mpi4; do
 for mode in opt phys; do
  if [[ "$layout" = mpi4 ]]; then threads=4; rayon=4; else threads=16; rayon=16; fi
  docker run --rm --cpuset-cpus=0-15 --shm-size=1g -v /tmp/mvmc-583-default/inputs:/inputs:ro -v "$root:/out" -v mvmc-582-rebase-target:/target:ro -w "/inputs/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e "RAYON_NUM_THREADS=$rayon" -e "MVMC_RS_INNER_THREADS=$threads" -e "MVMC_RS_TENFERRO_THREADS=$threads" -e MVMC_RS_SR_BACKEND=tenferro -e MVMC_EXPERIMENT_SHARED_CPU_TEAM=1 -e MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1 -e "BENCH_LAYOUT=$layout" -e "BENCH_MODE=$mode" -e "BENCH_DEST=/out/${layout}-${mode}-output" --entrypoint bash "$image" -lc '
   set -euo pipefail
   unset MVMC_RS_TENFERRO_SPIN_POLLS
   if [[ "$BENCH_LAYOUT" = mpi4 ]]; then
    launcher=(/opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc)
   else launcher=(/target/release/mvmc); fi
   if [[ "$BENCH_MODE" = opt ]]; then
    timeout --kill-after=5s 120s "${launcher[@]}" /inputs/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
   else
    timeout --kill-after=5s 120s "${launcher[@]}" /inputs/phys/namelist.def --physcal /inputs/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
   fi
  ' > "$root/${layout}-${mode}.log" 2>&1
 done
done
