#!/usr/bin/env bash
set -euo pipefail
for ranks in 2 4; do
 for mode in opt phys; do
  for cpus in unset 1 2; do
   label="$mode-r$ranks-c$cpus"
   docker run --rm --cpuset-cpus=0-7 -v mvmc-583-budget-mpi-target:/target:ro \
    -v /tmp/mvmc-583-default/inputs:/inputs:ro -v /tmp/mvmc-583-runtime-cli-positive:/out \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_SR_BACKEND=tenferro \
    -e MVMC_RS_INNER_THREADS=1 -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_TENFERRO_WORKER_BUDGET=0 \
    -e "MPI_RANKS=$ranks" -e "MODE=$mode" -e "CPUS=$cpus" -e "LABEL=$label" \
    --entrypoint bash vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc '
     set -euo pipefail
     if [[ $CPUS == unset ]]; then unset MVMC_RS_MPI_CPUS_PER_RANK; else export MVMC_RS_MPI_CPUS_PER_RANK=$CPUS; fi
     if [[ $MODE == opt ]]; then
      timeout 90 /opt/mpich/bin/mpirun -n "$MPI_RANKS" /target/test-fast/mvmc /inputs/opt/namelist.def --nsteps 2 --nsmp 1 --initial-def none --out-dir "/out/$LABEL"
     else
      timeout 90 /opt/mpich/bin/mpirun -n "$MPI_RANKS" /target/test-fast/mvmc /inputs/phys/namelist.def --physcal /inputs/phys/zqp_opt.dat --out-dir "/out/$LABEL"
     fi
    ' > "/tmp/mvmc-583-runtime-cli-positive/$label.log" 2>&1
  done
 done
done
