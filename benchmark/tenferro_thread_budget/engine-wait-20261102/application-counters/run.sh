#!/usr/bin/env bash
set -euo pipefail
out=/tmp/mvmc-583-counters-observed
[[ ! -e $out ]]
mkdir -p "$out"
for mode in opt phys; do
 for walkers in 1 4; do
  for spin in default 0; do
   for rep in 1 2 3; do
    label="$mode-w$walkers-s$spin-r$rep"
    docker run --rm --cpuset-cpus=0-3 --shm-size=1g \
     -v /tmp/mvmc-583-default/inputs:/inputs:ro -v "$out:/out" \
     -v mvmc-583-budget-mpi-target:/target:ro \
     -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 \
     -e MVMC_RS_INNER_THREADS=1 -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro \
     -e "MODE=$mode" -e "WALKERS=$walkers" -e "SPIN=$spin" -e "LABEL=$label" \
     --entrypoint bash vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc '
      set -euo pipefail
      unset MVMC_RS_TENFERRO_WORKER_BUDGET MVMC_RS_MEASURE_PF_BACKEND MVMC_EXPERIMENT_SHARED_CPU_TEAM
      if [[ $SPIN != default ]]; then export MVMC_RS_TENFERRO_SPIN_POLLS=$SPIN; fi
      timeout 120 /target/counters-diagnostic/release/examples/bench_walkers_583 "$MODE" "/inputs/$MODE/namelist.def" /inputs/phys/zqp_opt.dat "$WALKERS" "/out/$LABEL-output"
     ' > "$out/$label.stdout" 2> "$out/$label.stderr"
   done
  done
 done
done
