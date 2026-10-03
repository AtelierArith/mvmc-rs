# Retained #179 callback matrix checkpoint

Container `73c57e563c61`; all paths below are container paths. This retrospective
record does not add missing observations to original captures.

## Fresh expanded error-contract checkpoint

Generation handle **3798**, terminal **0**, retained stdout:

```text
CALLBACK cells=498 expected=498 bad=0
```

Root `/home/vscode/.cache/mvmc/issue179-callback-expanded.9qd4np` retains
`results.tsv` (498 launches and validations, all zero), `source-check.txt`,
`validator-binary-sha256.txt`, and `validator-binary-check.txt` (all checks OK).
Per-cell `w<workers>/failrank<rank>/` directories retain `launch.log`,
`validation.log`, rank states, raw `callback-result-rank-N.txt`, and root outputs.

Frozen snapshot `issue179-cgfix.PtoNII`, source-manifest SHA-256:
`31d62d4dea86309d9c4e2179ae82db93a7024f5436ca9a63a85d7c6c8edc0432`.
Actual binary SHA-256:
`b97917e94ee84b479dceef4c79e6e8113ed3d5e1d4acb748bfb75c5969f1c24a`.
Archived comparator SHA-256:
`72670a44a7f871a51a40fd0b7c685a775be5c128c86da6e1edc197d3cb0cfe23`.
Archived callback validator SHA-256:
`b6da066b207ac0aec424f5e4ff385435a66b740fe6d75e3dfa6de416062aa0d9`.

The reproduction loop below applies with this new snapshot/binary/checker root
and the validator argument `--failure-rank "$failed"`. The actual harness
records injected rank, one callback call per rank, a local error only on the
injected rank, and the returned callback reason. Validation checks those fields,
one actual sampling checkpoint despite three requested steps, one SR callback
record, exactly one finite first-step line in each root output, and absence of
the final optimized-parameter output. This is bounded coordinated callback
failure/error-boundary evidence, **not numerical parity**. The older 65604
checkpoint below remains separately scoped; its missing observations are not
retroactively supplied by these new captures.

## Actual terminal and artifacts

Generation handle **65604**, terminal **0**, merged stdout:

```text
Callback evidence: /home/vscode/.cache/mvmc/issue179-callback-matrix.Eh5bKT
CALLBACK cells=498 expected=498 bad=0
```

Root `/home/vscode/.cache/mvmc/issue179-callback-matrix.Eh5bKT`:

- `results.tsv`: 498 rows, columns cell / workers / failure_rank / process exit /
  retained-validator exit. All recorded exits are zero.
- `source-check.txt`: frozen source SHA-256 check after execution.
- `validator-binary-sha256.txt` and `validator-binary-check.txt`: binary/checker
  hashes and successful post-run checks.
- `<cell>/w<workers>/failrank<rank>/launch.log`: actual per-rank test stdout/stderr.
- Same directories: rank state files, worker snapshots, first-step root outputs,
  `validation.log`.

Generation binary SHA-256:
`c2a908d2b9348d1bb19fc2be8064a8d6434acf57af2a437b1023be606b562336`.
Generation comparator SHA-256:
`79ba8d760c3ca8787411496eb69e37b141337c333d36780a4b295249a5e7edb8`.
Generation callback validator SHA-256:
`4817e2709eca7aa2ff018cc0792a310e8cd59e3b81323a76164b7e3ed995f4b6`.

The source snapshot is `issue179-sr.WbN8B8`, manifest SHA-256
`e492fb961e0df28685a9298e460bc1e8867eedac143f2c36abed01c8386c0e4b`.
Its actual callback closure in `crates/mvmc-core/tests/mpi_issue179_state.rs`
records `sr-step`, returns an injected callback error only when
`failure_rank == Some(world.rank())`, and asserts the returned error contains
`callback`. Each rank runs this actual ignored test; zero tests cannot establish
coverage. The frozen source, not the newer shared draft, defines this checkpoint.

Independent stronger trace-shape revalidation handle **48291**, terminal **0**,
directory `final-validator.b9XZ0a` inside the same root:

- `results.txt`: per-cell results, final line
  `COORDINATED_CALLBACK_498_STRICT_RETAINED_PASS`.
- `source-sha256.txt`: copied comparator
  `72670a44a7f871a51a40fd0b7c685a775be5c128c86da6e1edc197d3cb0cfe23`,
  callback validator
  `688eb7f6d853a86631c7c4c488c351ecbca47771240bc00c449503dfc1e25170`.
- Copied `.py` files: immutable audit implementations, hash checks passed.

This checkpoint proves bounded real execution, all-rank error status, one
callback-stage parameter record and one actual sampler checkpoint instead of
the requested three. It does NOT independently record injected rank identity,
actual callback call/error counts, returned reason, or audit first/second-output
boundaries. Those additional fields and negative tests were subsequently added
to the owned shared harness/validator and require fresh captures. Do not upgrade
this retained checkpoint to the expanded contract without rerunning it.

## Reproduction command shape (original frozen protocol)

Run inside the named-volume devcontainer, using the frozen source/binary and
checker hashes above. The submitted shell used `bash -c`, `set -euo pipefail`,
the JSON-selected `mpi_issue179_state` binary, and this exact launch/validation
loop (matrix reader uses fd3; MPI stdin is isolated):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MVMC_RS_INNER_THRESHOLD=1 MPI179_STEPS=3
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode <&3; do
    [[ $expected == success ]] || continue
    export MPI179_INPUT=/home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/$id/inputs/namelist.def
    for workers in 1 2 4; do
        export MVMC_RS_INNER_THREADS=$workers
        for (( failed=0; failed<ranks; failed++ )); do
            export MPI179_FAIL_RANK=$failed MPI179_STATE_DIR=$output179/$id/w$workers/failrank$failed
            mkdir -p "$MPI179_STATE_DIR"
            rc179=0; vrc179=0
            timeout --kill-after=5s 30s mpiexec -n "$ranks" "$binary179" --ignored --exact issue179_state --nocapture </dev/null > "$MPI179_STATE_DIR/launch.log" 2>&1 || rc179=$?
            uv run --no-project python "$output179/mpi_issue179_callback_evidence.py" "$MPI179_STATE_DIR" --ranks "$ranks" > "$MPI179_STATE_DIR/validation.log" 2>&1 || vrc179=$?
            printf "%s\t%s\t%s\t%s\t%s\n" "$id" "$workers" "$failed" "$rc179" "$vrc179" >> "$output179/results.tsv"
            count179=$((count179+1))
            if ((rc179!=0||vrc179!=0)); then bad179=1; fi
        done
    done
done 3< /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/matrix.tsv
expected179=$(awk -F "\t" '$8=="success" {n+=$2*3} END {print n}' /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/matrix.tsv)
[[ $count179 == "$expected179" ]] || bad179=1
sha256sum -c snapshot-sources.sha256 > "$output179/source-check.txt" || bad179=1
sha256sum -c "$output179/validator-binary-sha256.txt" > "$output179/validator-binary-check.txt" || bad179=1
printf "CALLBACK cells=%s expected=%s bad=%s\n" "$count179" "$expected179" "$bad179"
exit "$bad179"
```

Initialize a new `output179` directory, copy the recorded frozen checker files,
record binary/checker hashes, initialize the TSV header, `count179=0` and
`bad179=0` before that loop. The current expanded validator additionally needs
`--failure-rank "$failed"` and new actual harness fields; it is intentionally
not interchangeable with the original protocol.
