#!/bin/bash
# Run every explicit MPI gate (ignored tests of the `mpi` feature) at 2 and 4 ranks.
#
# Explicit developer / dispatched-CI command (#392, #179); never invoked by Cargo or
# normal tests. Run from the repository root with an MPICH `mpiexec` on PATH (the Dev
# Container, or the PMI1/Hydra provider of optional-gates.yml):
#
#   scripts/run_explicit_mpi_gates.sh /tmp/mvmc-mpi-gates
#
# These gates are invisible to ordinary CI (they are `#[ignore]`d and need a live
# launcher), so run this before changing validation, initialization, output files
# or the grouped (NSplitSize) paths. Each cell prints `rc=0` on success. The script
# fails closed: a nonzero exit if any cell fails, if fewer cells than the expected
# minimum ran, or if provenance cannot be recorded. It writes into <dir>:
#   provenance.txt   revision, toolchain, launcher, binary digests
#   cells.txt        one line per executed cell with its exit status
#   summary.md       Pass/Fail table (status vocabulary of the optional-gates ledger)
set -uo pipefail
out=${1:?new scratch directory for gate outputs}
[[ ! -e $out ]] || { echo "refusing to reuse $out" >&2; exit 2; }
mkdir -p "$out"
export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-HASWELL} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
profile=test-fast
min_cells=${MVMC_MIN_MPI_GATE_CELLS:-86}
cargo test --locked -p mvmc-core --features mpi --profile $profile --no-run || exit 2
target=${CARGO_TARGET_DIR:-target}
mpiexec=${MPIEXEC:-mpiexec}
inputs=extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def
status=0
cells=0

{
  echo "revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -srm)"
  echo "rustc=$(rustc -V)"
  echo "launcher=$("$mpiexec" --version 2>&1 | head -2 | tr '\n' ' ')"
  echo "mpichversion=$(mpichversion 2>/dev/null | head -1 | tr -s ' ')"
  echo "threads=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=$OPENBLAS_CORETYPE"
  echo "profile=$profile features=mpi"
  echo "min_cells=$min_cells"
} > "$out/provenance.txt" || exit 2

# run <binary-name> <ranks> <extra env...>: every ignored test of the binary.
run() {
  local name=$1 n=$2
  shift 2
  local bin tests t base rc
  bin=$(ls -S "$target/$profile/deps/$name"-* | grep -v '\.d$' | head -1)
  tests=$("$bin" --ignored --list 2>/dev/null | sed -n 's/: test$//p')
  if [[ -z $tests ]]; then
    echo "$name n=$n $* rc=NO-TESTS" | tee -a "$out/cells.txt"
    status=1
    return
  fi
  echo "binary_sha256 $name $(sha256sum "$bin" | cut -d' ' -f1)" >> "$out/provenance.txt"
  for t in $tests; do
    base=$out/$name-$n-$(echo "$t$*" | md5sum | cut -c1-8)
    mkdir -p "$base"
    env "MPI_ISSUE178_MATRIX_OUTPUT=$base/o" "MPI_ISSUE178_PREPARATION_OUTPUT=$base/o" \
      "MPI_ISSUE178_SR_OUTPUT=$base/o" "MPI_ISSUE178_OUTPUT=$base/o" \
      "MPI179_SINGLETON_OUTPUT=$base/o" "MPI_PHYSCAL_CALLBACK_OUTPUT=$base/o" \
      "MPI179_PHYSCAL_OUTPUT=$base/o" "MPI177_EVIDENCE_DIR=$base/o" \
      "MPI_ISSUE234_SUMMARY_OUTPUT=$base/o" "MPI_ISSUE349_OUTPUT=$base/o" \
      "MPI_ISSUE179_MATRIX_OUTPUT=$base/o" \
      "MPI179_EXPECT_RANKS=$n" "MPI196_EXPECT_RANKS=$n" MVMC_RS_MPI_PHYSICAL=1 "$@" \
      timeout 300 "$mpiexec" -n "$n" "$bin" --ignored --exact "$t" --nocapture \
      > "$base.log" 2>&1
    rc=$?
    echo "$name $t n=$n $* rc=$rc" | tee -a "$out/cells.txt"
    cells=$((cells + 1))
    [[ $rc -eq 0 ]] || status=1
  done
}

for n in 2 4; do
  for name in mpi_issue178_support_matrix mpi_issue178_preparation_contract \
    mpi_issue178_sr_failure mpi_issue179_collectives mpi_issue179_mapping mpi_issue196_world \
    mpi_issue184_parallel_scalar_contracts mpi_issue234_summary mpi_issue274_normal_initialization \
    mpi_issue177_seed_lifecycle mpi_physcal_callback_contract mpi_physcal grouped_nsplit_349 \
    mpi_issue179_matrix; do
    run $name $n
  done
  for th in 1 2 4; do run mpi_issue179_physcal_singleton $n MVMC_RS_INNER_THREADS=$th; done
  for w in 1 2; do
    mkdir -p "$out/r182-$n-$w"
    run issue182_mpi_inner_physcal $n "ISSUE182_MPI_RECEIPT_ROOT=$(cd "$out/r182-$n-$w" && pwd)" \
      ISSUE182_MPI_GROUP_WIDTH=$w
    for f in mode nsteps nsmp; do
      for r in 0 last; do
        run mpi_issue178_preflight $n MPI_ISSUE178_WIDTH=$w MPI_ISSUE178_FIELD=$f \
          MPI_ISSUE178_FAIL_RANK=$r
      done
    done
    for a in opt physcal; do
      for r in root last-owner; do
        run mpi_issue178_sampling_failure $n MPI_ISSUE178_WIDTH=$w MPI_ISSUE178_API=$a \
          MPI_ISSUE178_FAIL_RANK=$r
      done
    done
  done
  run mpi_issue179_state $n "MPI179_INPUT=$inputs" "MPI179_STATE_DIR=$out/state179-$n"
done
# Gates fixed to one world size.
run mpi_issue179_literal_operator 2
run issue283_mode_preflight 2
run mpi_issue179_parallel_literals 4
# CLI gates launch mpirun themselves.
cargo test --locked -p mvmc-cli --features mpi --profile $profile --test runtime_contract \
  -- --ignored > "$out/cli-runtime-contract.log" 2>&1
rc=$?
echo "mvmc-cli runtime_contract --ignored rc=$rc" | tee -a "$out/cells.txt"
cells=$((cells + 1))
[[ $rc -eq 0 ]] || status=1

if [[ $cells -lt $min_cells ]]; then
  echo "only $cells cells ran (expected at least $min_cells): failing closed" >&2
  status=1
fi
{
  echo "# Explicit MPI gates (2 and 4 ranks)"
  echo
  echo "Cells executed: $cells (minimum $min_cells). Status vocabulary: Pass = exit 0; anything else Fail."
  echo "Julia comparisons are not executed here (Unverified); cells with native-C references are"
  echo "listed in docs/reference/c-to-julia/verification/issue-179-mpi-matrix.md."
  echo
  echo "| Verdict | Count |"
  echo "|---|---|"
  echo "| Pass | $(grep -c ' rc=0$' "$out/cells.txt") |"
  echo "| Fail | $(grep -vc ' rc=0$' "$out/cells.txt") |"
} > "$out/summary.md"
echo "provenance, cells.txt and summary.md written to $out"
exit $status
