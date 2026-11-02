#!/usr/bin/env bash
set -euo pipefail
result=/tmp/mvmc-582-crossover-measured
mkdir "$result"
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
volume=mvmc-583-budget-mpi-target
run_case() {
 local case_name=$1 workers=$2 gate=$3 shared=$4 rep=$5 profile=$6
 local label="$case_name-w$workers-g$gate-s$shared-r$rep-p$profile"
 docker run --rm --cpuset-cpus=0-3 -v "$volume:/target:ro" \
  -v /tmp/mvmc-581-profile/inputs:/inputs:ro -v "$result:/out" \
  -e "CASE=$case_name" -e "LABEL=$label" -e "MVMC_RS_INNER_THREADS=$workers" \
  -e "MVMC_RS_INNER_MIN_WORK_NS=$gate" -e "MVMC_EXPERIMENT_SHARED_CPU_TEAM=$shared" \
  -e "PROFILE=$profile" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro \
  --entrypoint bash "$image" -lc '
   set -euo pipefail
   if [[ $PROFILE == 1 ]]; then
    export MVMC_RS_INNER_PROFILE=1 MVMC_RS_INNER_GAP_SITE_PROFILE=1 MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1
   fi
   TIMEFORMAT="wall=%R user=%U sys=%S"
   if [[ $CASE == l32-phys ]]; then
    time timeout 120 /target/release/mvmc /inputs/$CASE/namelist.def --physcal /inputs/$CASE/zqp_opt.dat --out-dir /out/$LABEL-output
   else
    time timeout 120 /target/release/mvmc /inputs/$CASE/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir /out/$LABEL-output
   fi
  ' > "$result/$label.stdout" 2> "$result/$label.stderr"
}
for case_name in l32-opt l32-phys l64-opt; do
 for rep in 0 1 2 3; do
  if (( rep % 2 )); then gates='80000 40000 20000'; workers_set='4 1'; shared_set='1 0';
  else gates='20000 40000 80000'; workers_set='1 4'; shared_set='0 1'; fi
  for workers in $workers_set; do
   for gate in $gates; do
    for shared in $shared_set; do run_case "$case_name" "$workers" "$gate" "$shared" "$rep" 0; done
   done
  done
 done
 for shared in 0 1; do run_case "$case_name" 4 40000 "$shared" diagnostic 1; done
 done
