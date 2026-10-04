# Issue 178: public optimization preflight failure agreement

Related to [#178](https://github.com/AtelierArith/mvmc-rs/issues/178).
This fixes one independently reproduced collective-entry defect. It does not
close #178 or prove its entire supported-input/failure matrix.

## Defect and minimal fix

`run_para_opt_from_namelist_with_reducer` previously executed
`validate_run_options(&config)?` locally before collective parse/validation.
One rank with invalid mode, nonpositive nsteps or nonpositive explicit nsmp
returned while its peers proceeded into the runner's failure agreement.
CLI control agreement does not protect callers of this public library API.

The five-line source change wraps that existing validation in
`collective_result(..., reducer, "optimization configuration")`. It changes
neither accepted settings nor numerical/RNG algorithms. Every participating
rank joins the same configuration-failure agreement before parsing, seed
resolution, initialization or output. This is a Rust public-API liveness
contract, not a claim that these Julia feature guards are C-invalid input.

`Reducer::any_failure` reduces integer failure flags. For `MpiGroupContext`,
`allreduce_sum_i64` uses `global_communicator`, not just the local sampling
group. Consequently a failure also stops ranks outside the offending group.
The actual world4/width1 and width2 tests exercise those outside-group peers.

## Regression and observation boundary

New disjoint test: `crates/mvmc-core/tests/mpi_issue178_preflight.rs`,
`one_rank_invalid_public_options_stop_all_ranks_before_seed_and_output`.
The explicit ignored gate uses the public high-level optimization runner,
existing Heisenberg input, seed11272, nsteps1/nsmp1, and no initial overlay.
Exactly one global rank receives the invalid option. A newly and exclusively
created parent directory contains an immutable sentinel; the requested child
output directory must remain absent. No pre-existing directory is deleted.

On success of the regression, each rank returns Err, with the original local
diagnostic on the offending rank and collective peer diagnostic elsewhere.
Each has exactly one observed failure agreement, zero observed i64/c64
broadcasts, and zero explicitly delegated numeric reductions. The test checks
unchanged input bytes, sentinel bytes and directory inventory, then all ranks
reach the final barrier.

The observer does **not** count `broadcast_f64`, and does not assert that every
possible collective is instrumented. Its own failure-agreement implementation
delegates directly to the inner reducer, whose necessary flag reduction is not
counted as a model reduction. Zero seed/parameter broadcasts establishes a
pre-seed reachability boundary for this source path. This high-level API accepts
no caller RNG/state and returns none on Err: **no raw RNG/next624 or caller-state
snapshot is claimed**. Existing in-place rejection tests provide separate
caller-state/RNG evidence; they are not substituted for this process gate.

## Actual old failures retained

Frozen container `73c57e563c61`, checkout `/tmp/mvmc-cli174-gmUVyr`, dedicated
target `/home/vscode/.cache/mvmc/target/cli174-gmUVyr`.
Build handle47152 terminated0 before any production fix. Old `run.rs` SHA:
`3cb175056af1e247da386be8245d4807ed38d44f6357b218c77c47a79a8325b5`.
The test source was already the final `e0e03df...` hash below.

| Old handle | World / width / bad rank / option | Retained host log | Terminal |
| --- | --- | --- | --- |
| 20707 | 2 / 1 / 0 / mode | `178-old-2-0-mode-1.log` | 124 |
| 66861 | 2 / 2 / 1 / nsteps | `178-old-2-last-nsteps-2.log` | 124 |
| 21151 | 4 / 2 / 0 / nsmp | `178-old-4-0-nsmp-2.log` | 124 |
| 48569 | 4 / 1 / 3 / mode | `178-old-4-last-mode-1.log` | 124 |

Every log records only the offending rank's API return with `agreements=0`,
`broadcasts=0`, `reductions=0`. Its assertion requiring one agreement then
fails; peers never report a runner return. The bounded launch times out during
this failing regression's execution/unwind. This proves the rank-local API
return and a failing protocol gate, not an uninstrumented native-C deadlock.
All four original logs remain under the host checkout path above.

## Fixed validation and source scope

Fixed build79088 terminated0. MPI matrix handle40187 terminated0, all
24 launch cases: worlds2/4 × widths1/2 × global root/last rank ×
mode/nsteps/nsmp. Across them there are **72 unique rank-return markers and
72 per-rank test summaries, each 1 PASS / 0 FAIL / 0 ignored**. No case needed
timeout or forced termination. Full rank-marker validation also checks each
rank0..world-1 appears once and the exact case and observer counters match.

Final host and frozen-container source hashes matched after validation. The
test hash also matches the recorded pre-build/old-repro hash; the minimal
run.rs diff was inspected before the fixed build:

```text
run.rs af5204754499721f75667ae20df8e802ef8c220f24a0a515a159d92131563a6e
mpi_issue178_preflight.rs e0e03dfd1f3c9d321310f622b22cfa926e7c7fff66f84e3f1f22718b031ab5de
validation.rs dd170e07f765315900058b10e15e03c0fe8540028ab1ccad5c913c2d663381df
fixed binary 8e0a97045786496a717819646411a1b1d85f9c6455927fee02f512b04c931ff1
```

This reused the previously frozen dirty workspace snapshot for #174, with the
new test and exact run.rs fix overlaid. Relevant old run/validation hashes
matched the shared audit source at HEAD5629a83 before the fix. **No complete
pre/post source digest of the entire checkout/compiler-input closure was
captured for #178**, and the old executable was not separately retained before
rebuild. Do not label this a clean-current-main whole-workspace proof.

Actual Linux x86_64, Rust1.99.0, MPICH4.2/Hydra ch4:ucx, system OpenBLAS,
requested OpenBLAS/OMP threads1. Build:

```sh
cargo test --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue178_preflight --no-run
```

Each case launches the selected binary with environment
`MPI_ISSUE178_FIELD=mode|nsteps|nsmp`, `MPI_ISSUE178_FAIL_RANK=0|last`,
`MPI_ISSUE178_WIDTH=1|2`, and a fresh `MPI_ISSUE178_OUTPUT` directory:

```sh
timeout --kill-after=2s 8s /opt/mpich/bin/mpiexec -disable-auto-cleanup -n N \
  /home/vscode/.cache/mvmc/target/cli174-gmUVyr/test-fast/deps/mpi_issue178_preflight-cfc96d5fff6f9209 \
  --ignored --exact one_rank_invalid_public_options_stop_all_ranks_before_seed_and_output \
  --nocapture
```

Artifacts under `/tmp/mvmc-cli174-gmUVyr/`:

- Host `178-fixed-matrix.tsv`: all24 statuses0.
- Container `178-fixed-N-W-R-F.log`: all24 complete process logs. Host copy
  `178-rank-logs.tar` contains them, with original filenames.
- Host `178-rank-log-validation.json`: validated cases/rank identities/counts.
- Host `178-source-binary-environment.log`: source/binary SHA, all24 log hashes,
  actual compiler, launcher and linked-library information.
- Host `178-clippy.log`: handle77292 **terminal0**, focused MPI test strict
  clippy, test-fast, locked, `-D warnings`, 2.66s.
- Parent fresh normal regression
  `59778011-0c24-4784-910f-3fc0878debad`: public rejection + runner_config,
  **16/16 PASS, 0.113s**, including positive configurations. This is separate
  normal proof, not another MPI matrix. Owned-file fmt/diff-check also pass.

The parent independently validated all24 rank logs and reviewed the entire
test and minimal source patch. No independent numerical reference was generated.

## Remaining issue 178 scope

This closes only the rank-local public optimization option-preflight defect.
The five-criterion audit still distinguishes library/CLI and optimization/
PhysCal, root-only output failures from nonroot failures, genuine errors from
synthetic sampling/SR agreement injections, and current source from frozen
historical protocol executions. Broader matrix/early-sampler failure gaps need
their own authority/proposal and validation; they are not resolved by24 PASS.
Do not close #178 or extend this result to InterAll, native-C sampling parity,
MPI initialization failure recovery, or unobserved raw state.
