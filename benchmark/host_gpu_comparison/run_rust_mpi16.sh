#!/usr/bin/env bash
# Rust MPI whole-runner E2E (Opt + PhysCal), 16 ranks, run INSIDE the Dev
# Container (MPICH 4.2.0). Issue #544.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$here/env.sh"
# shellcheck source=lib.sh
source "$here/lib.sh"

# One BLAS/OpenMP/inner thread per rank (the MPI comparison policy).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1
export MVMC_C_TIMER=1

if [ ! -x "$MPI_BIN" ]; then
    echo "building MPI CLI ($MPI_BIN)..."
    cargo build --locked --release -p mvmc-cli --features mpi --manifest-path "$ROOT/Cargo.toml"
fi

case=rust-mpi${NPART}
dir="$RESULT_DIR/$case"
mkdir -p "$dir"
csv="$dir/$case.csv"
printf 'case,phase,rep,wall_s,all_s\n' >"$csv"
record_env "$dir/environment.txt"

samples_per_rank=$((SAMPLES / NPART))
command -v mpiexec >/dev/null || { echo "mpiexec not found (run inside the Dev Container)" >&2; exit 1; }

# PhysCal fixed-parameter checkpoint (single rank, two-point window -> triples).
ckpt=$(make_checkpoint "$MPI_BIN" "$dir/checkpoint-input" "$dir/checkpoint-out" mpiexec -n 1)
echo "checkpoint: $ckpt"

opt_in="$dir/opt/input"
mkdir -p "$dir/opt"
prepare_input "$opt_in" "$samples_per_rank" 0 1
for rep in $(seq 1 $((WARMUPS + REPS))); do
    tag=warmup
    [ "$rep" -gt "$WARMUPS" ] && tag=measure
    run_timed "$csv" "$case" "opt-$tag" "$rep" "$dir/opt/run-$rep" \
        bash -c "cd '$opt_in' && mpiexec -n $NPART '$MPI_BIN' namelist.def --nsteps $STEPS --nsmp 1 --seed $SEED --out-dir '$dir/opt/run-$rep'"
done

phys_in="$dir/phys/input"
mkdir -p "$dir/phys"
prepare_input "$phys_in" "$samples_per_rank" 1 1
for rep in $(seq 1 $((WARMUPS + REPS))); do
    tag=warmup
    [ "$rep" -gt "$WARMUPS" ] && tag=measure
    run_timed "$csv" "$case" "physcal-$tag" "$rep" "$dir/phys/run-$rep" \
        bash -c "cd '$phys_in' && mpiexec -n $NPART '$MPI_BIN' namelist.def --physcal '$ckpt' --out-dir '$dir/phys/run-$rep'"
done

echo "results: $csv"
