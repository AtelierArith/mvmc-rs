# #179 matrix-v2 retained execution evidence

This is a retrospective transcription of the tool command and terminal output,
not a contemporaneous shell trace. Original generation is NOT an overall PASS.
No original generation was rerun to investigate its terminal status.

Container: `73c57e563c61`. All paths below are inside that container.

## Original Rust generation, handle 15476

The following decoded script was passed to `docker exec 73c57e563c61 bash -lc`:

```bash
set -euo pipefail
cd /home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8
binary179=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_state" and .executable!=null)|.executable' /home/vscode/.cache/mvmc/issue179-sr-build.json)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_MVMC_MPI=1 JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot MVMC_RS_INNER_THRESHOLD=1
output179=/home/vscode/.cache/mvmc/issue179-sr-matrix-v2
mkdir -p "$output179"
printf 'cell\tprefix\tworkers\texit\n' > "$output179/rust-execution.tsv"
bad179=0; count179=0
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode <&3; do
[[ $expected == success ]] || continue
export MPI179_INPUT=/home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/$id/inputs/namelist.def
for steps in 1 2 3; do
export MPI179_STEPS=$steps
cell179="$output179/$id/prefix$steps"
for workers in 1 2 4; do
export MVMC_RS_INNER_THREADS=$workers MPI179_STATE_DIR="$cell179/w$workers/rust"
mkdir -p "$MPI179_STATE_DIR"
rc179=0
timeout --kill-after=5s 60s mpiexec -n "$ranks" "$binary179" --ignored --exact issue179_state --nocapture </dev/null > "$MPI179_STATE_DIR/launch.log" 2>&1 || rc179=$?
printf "%s\t%s\t%s\t%s\n" "$id" "$steps" "$workers" "$rc179" >> "$output179/rust-execution.tsv"
count179=$((count179+1))
if ((rc179!=0)); then bad179=1; fi
done
done
done 3< /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/matrix.tsv
echo "RUST cells=$count179 bad=$bad179"
test "$count179" -eq 495 || bad179=1
sha256sum -c snapshot-sources.sha256 > "$output179/rust-source-check.txt" || bad179=1
exit "$bad179"
```

Captured merged terminal output:

```text
RUST cells=495 bad=0
exit status 1
```

Tool-reported terminal exit code: **1**. No `set -x`, last-command/status trap,
or separate wrapper stderr file was installed. Thus the final command in the
submitted script is known (`exit "$bad179"`), but its actually reached command
and variable value cannot be established from a shell trace. The only additional
terminal text was `exit status 1`; it cannot safely be attributed specifically
to Bash, Docker, SSH, or a logout hook. Direct Bash login/non-login exit-0 probes
returned 0, so they did not reproduce a logout failure. Cause remains unresolved.

Retained `rust-execution.tsv` has 495 data rows, all cell exit statuses 0.
Each cell's `launch.log` retains merged MPI test stdout/stderr.
`rust-source-check.txt` contains 9653 `: OK` lines and no non-OK lines; a separate
read-only `sha256sum -c` recheck returned 0, recorded in
`rust-source-recheck.txt`. These independent observations do not overwrite the
original wrapper terminal 1.

## Final frozen-checker independent audit

Directory:
`/home/vscode/.cache/mvmc/issue179-sr-matrix-v2/worker-audit-final.SiBWbk`.

- `results.txt`: per-cell/per-rank actual kernel classification and worker entries;
  final line `RETAINED_EXECUTION_AND_ACTUAL_WORKER_EVIDENCE_495_PASS`.
- `mpi_issue179_worker_evidence.py`: copied final strict checker, not a moving
  shared-worktree script.
- `checker-sha256.txt`: checker SHA-256
  `ab24601d6d842a2e6682d23681ad1da8119eb697b7b0fe7067e0d83ca9bbff3e`.

This independent audit tool command terminated **0**. Captured merged output:

```text
/home/vscode/.cache/mvmc/issue179-sr-matrix-v2/worker-audit-final.SiBWbk/mpi_issue179_worker_evidence.py: OK
Final checker evidence: /home/vscode/.cache/mvmc/issue179-sr-matrix-v2/worker-audit-final.SiBWbk
```

It required exactly 495 retained rows, each original cell rc0, validated actual
worker evidence for every rank, and checked the frozen checker hash after the
audit. It establishes retained execution/worker evidence only, not independent
numerical parity or a successful original generation wrapper.

Frozen generation source manifest:
`/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8/snapshot-sources.sha256`;
SHA-256 `e492fb961e0df28685a9298e460bc1e8867eedac143f2c36abed01c8386c0e4b`.
This is distinct from the later frozen checker hash above.

## Original Julia generation terminal (same handle)

Handle **81717** was polled without restarting and terminated **1**. Actual
merged terminal output:

```text
JULIA cells=165 bad=0
exit status 1
```

`julia-execution.tsv` in the same root has exactly 165 data rows, all recorded
reference cell exits zero. Whole-file inspection of `julia-source-check.txt`
found **9653 OK lines and zero non-OK lines**, not merely an inspected tail.
The check-output file SHA-256 is
`318c788a62bd7da257944b6be8264da2cc19645bd80d4a6180eeada1391af870`.
These are retained cell/source-check observations, **not a successful generation
wrapper**. No command-status trace was captured; the wrapper terminal-1 cause
remains unresolved. Do not assign it to a logout hook or convert it to PASS.
