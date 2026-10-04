# Issue 179: current user clarification and remaining validation

The latest explicit user clarification supersedes the earlier cross-language
trajectory completion requirement. Same input and fixed seed must reproduce runs
within the same implementation and explicit MPI world, group width, worker count,
storage and numerical configuration. Random algorithms remain fixed across
languages: initialization, conversion, outputs and state must agree for the same
draw order/count. Existing strict primitive-stream tests remain useful and intact.

Numerically dependent branch/acceptance differences require identification of the
first cause and decision threshold; they are neither automatically RNG defects nor
Monte Carlo noise. Mathematical-function implementation differences may use explicit
operation/scale-appropriate absolute/relative bounds. This does not authorize
algorithm, operation-order, sign or input-contract changes, or an arbitrary CG
forward tolerance. Solver and MPI-reduction effects are separate from direct math
function discrepancies and require meaningful residual evidence.

The prospective long-runner baseline is **20 steps**, not 50. Completed 50-step
artifacts retain their original labels and cannot supply 20-step final parameters
by truncation. A fresh 20-step reference must use reviewed published PR54 sources
and a consistent effective parameter window. Prefixes 1/2/3, purposeful failure
boundaries 27/28/29, and CG-kernel refresh checks through 41 (refresh 20/40) remain
separate regressions and are not rewritten indiscriminately.

Current owned MPI state scripts exercise prefixes 1/2/3; they contained no 50-step
long-run baseline to replace. Full scenario coverage and fresh same-configuration
repeatability remain required after the production/reference consumers are ready.
Historical execution, exact-discrete and diagnostic captures are not upgraded into
current full numerical acceptance.

Supporting independent C fixed-operand evidence and its deliberately narrow
first-common-solve scope are documented in
[the C replay record](../../../../c_toolbox/mpi_issue179_cg_replay.md).
It is not a blocker requiring indefinite bitwise trajectory auditing, and it is
not full C sampler parity. Issue 179 remains incomplete.

## Frozen Rust repeatability execution

`scripts/verify_mpi_issue179_repeat.sh` launches two fresh MPI processes per
selected cell/configuration, retains both captures, compares rank state and CG
records plus root output/SR metadata, and independently checks actual worker
activity. Worker scheduling IDs are not required to repeat. Input, executable and
checker hashes are checked; missing records, failed launches and incomplete
selection fail closed. This gate is same-implementation repeatability, not a
cross-language numerical acceptance gate or solver convergence assertion.

First run handle **21017**, terminal **0**, covered six configurations:
world 4, width 1, CG, store 0, real/complex, prefix 3, workers 1/2/4. All twelve
fresh launches and six repeat validations passed. Evidence is in container
`73c57e563c61` at
`/home/vscode/.cache/mvmc/issue179-repeat-current-first-v2`.
The binary is the retained `issue179-62b.qAZUvg` snapshot executable, SHA-256
`cf5c188b1d858464661f50eafcb7099d9bb23ebd49058401e3f10a6fa580deb5`.
The initial `issue179-repeat-current-first` attempt failed during script hash
setup before any MPI launch; it is not counted as a successful execution.

The full prospective repeat inventory is 55 declared-success inputs × prefixes
1/2/3 × workers 1/2/4 = 495 configuration pairs, with world 2/4 and input-declared
capability restrictions. Declared rejection cells are not relabelled numerical
successes. Original full handle 51328 exited 1 (108 CG validations passed;
387 direct-SR validations failed the erroneous SRinfo requirement). Corrected
direct-only handle 83487 terminated 0: all 387 pairs passed, with separate evidence; see
[the exact commands and hashes](issue-179-repeat-command.md). Independent residual diagnostics are
supporting evidence with the narrower scope stated in the C replay record.
