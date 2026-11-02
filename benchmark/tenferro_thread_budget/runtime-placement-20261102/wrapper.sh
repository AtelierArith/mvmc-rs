#!/usr/bin/env bash
set -uo pipefail
rank=${PMI_RANK:?}
export MVMC_RS_MPI_CPUS_PER_RANK=2 MVMC_RS_INNER_THREADS=1 MVMC_RS_TENFERRO_WORKER_BUDGET=0
case $CASE in
 invalid) export MVMC_RS_MPI_CPUS_PER_RANK=bad ;;
 mismatch) if [[ $rank == 0 ]]; then export MVMC_RS_MPI_CPUS_PER_RANK=1; fi ;;
 capacity) export MVMC_RS_MPI_CPUS_PER_RANK=5 ;;
 budget-over) export MVMC_RS_INNER_THREADS=2 MVMC_RS_TENFERRO_WORKER_BUDGET=1 ;;
 budget-missing) unset MVMC_RS_TENFERRO_WORKER_BUDGET ;;
 budget-asym) if ((rank + 1 == MPI_RANKS)); then export MVMC_RS_TENFERRO_WORKER_BUDGET=2; fi ;;
 *) exit 99 ;;
esac
prefix="/evidence/$CASE-r$MPI_RANKS"
/target/test-fast/mvmc /missing/namelist.def --out-dir "$prefix-output" > "$prefix-rank$rank.stdout" 2> "$prefix-rank$rank.stderr"
rc=$?
printf '%s\n' "$rc" > "$prefix-rank$rank.status"
for ((attempt=0;attempt<200;attempt++)); do
 complete=1
 for ((peer=0;peer<MPI_RANKS;peer++)); do [[ -f $prefix-rank$peer.status ]] || complete=0; done
 ((complete)) && break
 sleep 0.01
done
exit "$rc"
