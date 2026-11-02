#!/usr/bin/env bash
set -euo pipefail
out=/tmp/mvmc-583-final-policy-mpi
mkdir "$out"
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
run_case() {
 local mode=$1 ranks=$2 shared=$3 spin=$4 rep=$5 profile=$6
 local label="$mode-r$ranks-s$shared-spin$spin-rep$rep"
 docker run --rm --cpuset-cpus=0-13,18-31 -v mvmc-583-budget-mpi-target:/target:ro \
  -v /tmp/mvmc-583-default/inputs:/inputs:ro -v "$out:/out" \
  -e "MODE=$mode" -e "RANKS=$ranks" -e "LABEL=$label" -e "SPIN=$spin" -e "PROFILE=$profile" \
  -e "MVMC_EXPERIMENT_SHARED_CPU_TEAM=$shared" -e MVMC_RS_INNER_THREADS=4 \
  -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_TENFERRO_WORKER_BUDGET=3 \
  -e MVMC_RS_INNER_MIN_WORK_NS=40000 -e MVMC_RS_TENFERRO_TEAM_LIFETIME=held \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_SR_BACKEND=tenferro \
  --entrypoint bash "$image" -lc '
   set -euo pipefail
   if [[ $SPIN == 0 ]]; then export MVMC_RS_TENFERRO_SPIN_POLLS=0; else unset MVMC_RS_TENFERRO_SPIN_POLLS; fi
   if [[ $RANKS == 4 ]]; then export MVMC_RS_MPI_CPUS_PER_RANK=7; else unset MVMC_RS_MPI_CPUS_PER_RANK; fi
   if [[ $PROFILE == 1 ]]; then export MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1; fi
   TIMEFORMAT="wall=%R user=%U sys=%S"
   time timeout --kill-after=5s 120 /opt/mpich/bin/mpirun -n "$RANKS" bash -lc '\''
    set -euo pipefail
    rank=${PMI_RANK:?}
    case $rank in 0) mask=0-6;; 1) mask=7-13;; 2) mask=18-24;; 3) mask=25-31;; *) exit 99;; esac
    if [[ $MODE == opt ]]; then
     exec taskset -c "$mask" /target/release/mvmc /inputs/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "/out/$LABEL-output" > "/out/$LABEL-rank$rank.stdout" 2> "/out/$LABEL-rank$rank.stderr"
    else
     exec taskset -c "$mask" /target/release/mvmc /inputs/phys/namelist.def --physcal /inputs/phys/zqp_opt.dat --out-dir "/out/$LABEL-output" > "/out/$LABEL-rank$rank.stdout" 2> "/out/$LABEL-rank$rank.stderr"
    fi
   '\''
  ' > "$out/$label.launcher.stdout" 2> "$out/$label.time"
}
for mode in opt phys; do
 for ranks in 1 4; do
  for rep in 0 1 2 3; do
   if ((rep%2)); then shared_set='1 0'; spins='0 default'; else shared_set='0 1'; spins='default 0'; fi
   for shared in $shared_set; do
    for spin in $spins; do run_case "$mode" "$ranks" "$shared" "$spin" "$rep" 0; done
   done
  done
 done
 for spin in default 0; do run_case "$mode" 4 1 "$spin" diagnostic 1; done
 done
