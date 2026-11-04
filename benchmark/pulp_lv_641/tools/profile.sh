#!/bin/bash
# usage: prof.sh BIN MODE THREADS OUTDIR [perf]
set -eu
BIN=$1; MODE=$2; N=$3; OUT=$4; PERF=${5:-}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
rm -rf "$OUT"; cp -r "$REPO/benchmark/cpu_scaling_l128/inputs/$MODE" "$OUT"; cd "$OUT"
for v in $(env | grep -o '^MVMC_[A-Z_0-9]*'); do unset "$v"; done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  MVMC_RS_INNER_THREADS=$N RAYON_NUM_THREADS=$N \
  MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=c-order MVMC_C_TIMER=1
EXTRA=""; [ "$MODE" = phys ] && EXTRA=zqp_opt.dat
PRE=""; [ -n "$PERF" ] && PRE="perf record -F 499 -g --call-graph=fp -o perf.data --"
/usr/bin/time -f "wall=%e user=%U sys=%S" -o time.txt taskset -c 0-$((N-1)) $PRE "$BIN" namelist.def $EXTRA > run.stdout 2> run.stderr
cat time.txt
