# Sourced by optional issue179 gates. Never use a workspace/host target tree.
export CARGO_TARGET_DIR=${MPI179_CARGO_TARGET_DIR:-/tmp/mvmc-issue179-target}
case "$CARGO_TARGET_DIR" in
    /tmp/mvmc-issue179-target|/home/vscode/.cache/mvmc/target/issue179-*) ;;
    *) echo "unsupported issue179 target directory: $CARGO_TARGET_DIR" >&2; return 2 ;;
esac
export LIBCLANG_PATH=${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
mpi179_mpirun=(mpirun)
mpi179_version=$(mpirun --version)
if [[ $mpi179_version == *"Open MPI"* ]]; then
    mpi179_mpirun+=(--oversubscribe)
elif [[ $mpi179_version != *HYDRA* && $mpi179_version != *MPICH* && $mpi179_version != *Hydra* ]]; then
    echo "unrecognized MPI launcher; record and configure backend explicitly" >&2
    return 2
fi
