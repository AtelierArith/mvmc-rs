# A201 existing signed scalar max — SOURCE reconciliation only

Related to #184 and #185. Source base main
`2fb451d63d1922797cfbd267f64412c12c931e0f`; no implementation or status promotion.
The canonical A201 top-level absent-API claim is corrected; its original row,
owner `#180/#179`, all row order and status are preserved. Full API/scenario
acceptance remains pending.

Public `ParallelScalarOperations::max_integer` is implemented for serial,
bare MPI world and split MPI group contexts. Serial returns its input in every
domain. Bare MPI world rejects Sampling/CrossGroup; split groups use actual
World/Sampling/CrossGroup communicators, sampling delegates to the existing
signed max operation. Inputs are C-compatible `i32`, not every Julia Integer.
Rust enum selectors do not represent arbitrary Julia Symbols. No wrapper is added.

Original Julia `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
`MVMCOptimizers.jl/src/parallel.jl:269–272`, whole SHA256
`a7d72725f193a121f79267be6f931e2439fecdf104de01a0cb1d3a5ca8d91fc1`;
`test_unit/test_unit_parallel.jl:49–55` literal serial comm1 max7.
Rust serial control additionally covers −7 and i32::MIN in all three domains.
Native control independently calculates actual membership at widths1/2/3/world
and checks negative maxima, i32::MIN and two collective passes. Related sum
controls are not evidence of additional max semantics or model parity.

## Actual bounded CI execution, not combined-main execution

Run [37205217809](https://github.com/AtelierArith/mvmc-rs/actions/runs/37205217809),
head `ef0c89f43f1e2e9e1db1c6da2ce4a3961c095ff5`, tested merge
`487f113dc4d89b37c9cd0719e4c4fc46c164453b`, parents20dab/ef0c.
Main2fb also incorporates PR322: scalar implementation/test paths compare
byte-identical, but no new combined-main execution is claimed.

Serial [job111444893555](https://github.com/AtelierArith/mvmc-rs/actions/runs/37205217809/job/111444893555):
`issue184_parallel_scalar_contracts::original_serial_scalar_literals_and_negative_max_are_identities`
PASS0.006s ordinal448/1177. Command:
`cargo nextest run --workspace --locked --cargo-profile ci --no-fail-fast --retries 0`.

Explicit native [job111444893611](https://github.com/AtelierArith/mvmc-rs/actions/runs/37205217809/job/111444893611):
`mpi_issue184_parallel_scalar_contracts::scalar_domains_use_actual_membership_and_signed_negative_max`.
Inventory: `cargo nextest list --workspace --all-features --cargo-profile ci --locked --message-format json`.
For world2 then4, native command:

```sh
timeout -k 10s 120s "$MVMC_ISSUE234_MPI_PREFIX/bin/mpiexec" -disable-auto-cleanup \
  -outfile-pattern "$out/world$world-rank%r.stdout" \
  -errfile-pattern "$out/world$world-rank%r.stderr" \
  -n "$world" "$binary" --ignored --exact \
  scalar_domains_use_actual_membership_and_signed_negative_max --nocapture
```

Artifact [11304578875](https://github.com/AtelierArith/mvmc-rs/actions/runs/37205217809/artifacts/11304578875),
ZIP SHA256 `682dbb0993853293fa81b7632885a691f6732107e3d1ebe9d977fdedcb5cb687`,
matches upload digest. Six rank captures each1PASS, native prior0/post0,
world2/4 statuses0. Provider startup sixrecords required1/provided1/ok1,
rank sums1/6, prior0/post0, world statuses0; all ten native/provider POST logs
empty. MPICH4.2.0/ch4:ucx/Hydra/PMI1, requested UCX_TLS=self,sm,tcp.
Negotiated topology and arbitrary thread/collective failure schedules are not observed.
Normal workspace skipping this ignored identity is not native PASS evidence.

Raw all-job log SHA256
`a72a165c7a5ca48b4688868a99389ac8a632a5c5b02629a765daaa408fc46c11`;
serial raw SHA256
`9a71a7cc84b9b0502949b322d39303f7d5b18e956f555291d1ddb0c8a34c3175`.
Retained local review root `/tmp/mvmc-pr324-native-review.1NIH3d` is supplementary;
durable run/job/artifact links above are the evidence authority. No host replay
of unavailable runner-absolute libraries or invented owner-emptiness proof is claimed.

Only A201 SOURCE fields change. All2342 keys/order/originals/owners/statuses
and93CI/6LOCAL/1583Missing/80Historical/579Review/1Difference counts remain.
No A200, array/root-only/abort API, MPI model or umbrella completion follows.
