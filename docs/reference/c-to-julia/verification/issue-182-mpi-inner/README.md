# MPI × inner Green workers: bounded #182 evidence

Related to #182 and #185; neither issue is completed by this milestone.
The ignored native test exercises two consecutive public PhysCal frames using
the original real Heisenberg input, seed1, non-RBM/non-Lanczos ordinary route,
two one-body descriptors and **36 original direct two-body descriptors**.
Only the declared run length changes from one to two frames. No synthetic
descriptor duplication, production change, numerical expectation refresh,
new tolerance, runtime C/Julia oracle, or global cache/environment change.

## Recorded Linux result

Scientific base `2fb451d63d1922797cfbd267f64412c12c931e0f`, test source
SHA `222893c7e82b4abc48ebe1da2e4dde086b0b4e98735cf814f2ad44877d586d98`.
Publication changes only its introductory documentation line; executable
test statements are identical. No new publication-head test result is implied.

Twelve configurations: worlds2/4 × group widths1/2 × inner workers1/2/4.
Thirty-six rank processes each reported one native libtest PASS and public
`Ok(2)`. Twelve **same-world/width/rank** worker triples passed exact initial
and final raw624/cursor/canonical-u128 draw count/nonconsuming future624,
callback ordinals/statuses, saved configurations, complete packed burn32,
separate inactive burn buffers, scratch and sampling counters. No cross-world
or different-seed-group RNG equality is asserted. Every proposal/intermediate
RNG state is **not captured**; C/MPI parity and full #182 threading/performance
acceptance are **not established**.

After the discrete gate, each triple compared192 numeric components using the
existing same-implementation `max(1e-12, 1e-12*max(abs(a),abs(b)))` policy.
Observed maximum error was0 in each triple; computed bit equality was not the
acceptance criterion. This is worker invariance, not an independent C oracle.
Actual worker IDs were `[0,1]`/`[0,1,2,3]` for workers2/4; actual per-rank
worker entries64800 at width1 and32400 at width2. Worker1 uses the serial route.
Observation counts and full operands are preserved in the result JSON files.

## Environments, providers, and ownership

Linux x86_64, rustc1.99.0 `b940084d7`, Cargo1.99.0 `5f94df478`, nextest0.9.146
`8af696ddc`; `--locked --cargo-profile test-fast`, default core features plus
`mpi` (core's normal dependency enables PfaPack BLAS), build jobs2.
Actual LP64 OpenBLAS/libgfortran and MPICH4.2.0 ABI16:0:4 ch4:ucx/HydraPMI1
without PMIx were loaded. All BLAS/OMP threads1, inner threshold32;
`UCX_TLS=self,sm,tcp` and `/opt/mpich/lib` loader environment unchanged.
Selected ELF SHA
`df67665c35e7cca1ac6cd7b82e8b893149758a6a13bb817f4c124fe15f412b3d`;
all11 actual nongenerated providers joined to prior canonical SHA authority.
Hashes and join evidence are under `linux-20261004/provider/`.

Both containers used image
`sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`.
World2 ran in73c57e563c61 with **64MiB SHM**. World4 ran in70444e0ff679 with
**256MiB SHM** and shared Cargo/cache volumes READONLY; the same ELF/providers
were verified, private source/tools copied byte-exact into a task-owned volume,
and an external task lease used. These are distinct recorded environments,
not a relabelled identical64MiB experiment. No UCX transport/algorithm change.

Native jobs used180s/k5 STOP-before-CONT ownership, complete source/fixture/
compiler/startup/tool/ELF/provider PRE/POST checks,32MiB proof bound and384MiB
root reserve. SHM guard was world*8MiB initial and8MiB post/continuous reserve.
All successful native owner cleanup/group-empty/statuses0, except original
pilot outer inventory failure described below. Readonly comparisons used
120s/k5,4MiB streams,49 independent transport controls (2positive47negative)
before each world's comparison; source/input/runtime PRE/POST and both phase
owner/writer/group-empty statuses were0. Logical counters are not timings,
allocation measurements or speed claims. Unmeasured performance remains open.

## Preserved failures and corrected harness assumptions

- Originalpilot MPI/test statuses were0, but outer overall1: Hydra did not
  create stderr files. A separately authorized readonly validation succeeded;
  the original outer failure is retained, not rewritten as a successful run.
  MPICH4.2.0 release archive SHA
  `a64a66781b9e5312ad052d32689e23252f745b27ee8818ac2ac0c8209bc0b90e`:
  `proxy/pmip_cb.c:51–84` sends stderr only with received data,
  `mpiexec/pmiserv_cb.c:370–386` invokes the callback,
  `mpiexec/uiu.c:239–336` creates the output file inside that callback.
  ENOENT is `HYDRA_FILE_NOT_CREATED`, bytes:null; it is **not an independent
  zero-byte measurement**, nor retrospective installed-binary source proof.
- First world4 preflight in73c stopped before MPI:29960KiB free SHM vs32768KiB
  required. Guard remained unchanged; no unrelated SHM deletion. The separate
 256MiB environment successor produced the recorded world4 acquisitions.
- First readonly comparator failed before discrete/numeric interpretation:
  `burn.idx.len()==32`, not6. `state.rs` allocates/saves canonical combined
  `idx[6],cfg[12],num[12],proj[2]` storage. Corrected schema preserves all32,
  verifies order/domains/consistency, separately reports active6 and all
  captured inactive fields, and retains complete discrete content. No slicing
  away content, manufactured padding or Rust-generated numerical expectations.
  Independent literal controls reject missing/extra/order/value/inactive
  drift and prove discrete failure yields `numeric:[]`.
- Existing owner exit-observation `/proc/.../stat` ENOENT warnings are retained
  in `provider/world4-orchestration.json`; measured cleanup/group-empty0 is
  reported separately, not inferred from absent stderr. Historical readonly
  schema failure and capacity/native failures are under `failures/`.

## Evidence and optional replay

`linux-20261004/` holds byte-identical36 rank JSONs,12 comparison JSONs,
rank stdout, top-level/owner statuses and identities, SHM/settings metadata,
provider authority and both readonly control receipts. `SHA256SUMS` binds all
durable members with relative paths. These are diagnostics, **not Cargo golden
fixtures**. Ordinary Rust tests do not read them or run Node/C/Julia.

Optional developer transport commands, Node24 (no installs):

```sh
timeout -k 5s 120s node scripts/issue182_mpi_inner/compare-controls.mjs /tmp/ABSENT-CONTROL-ROOT
timeout -k 5s 120s node scripts/issue182_mpi_inner/replay-recorded.mjs \
  docs/reference/c-to-julia/verification/issue-182-mpi-inner/linux-20261004 \
  /tmp/ABSENT-READONLY-OUTPUT
```

Both outputs must be fresh/absent, never inside retained input. Replay verifies
the durable manifest then discrete-first comparator; first failure stops.
These relocated-publication commands have **not yet been run**; actual proof
used the pinned historical harness (schema seal
`35ab1f4d7a1d596f249a4fcdfb35c0230936d3af9e883f3ff7bcb54fceaa8ee1`).

Explicit native developer gate: build/list the ignored test with `mpi`, retain
the actual selected ELF/provider inventory, then run each world/width/workers
setting in a fresh owned bounded process with the same thread/provider guards.
Required env: `MVMC_RS_INNER_THREADS`, `MVMC_RS_INNER_THRESHOLD=32`,
`ISSUE182_MPI_GROUP_WIDTH`, `ISSUE182_MPI_RECEIPT_ROOT` (fresh exclusive dir),
all four BLAS/OMP thread envs1. The exact libtest identity is
`two_frame_public_physcal_keeps_rank_local_rng_and_executes_green_workers`.
Do not launch under global RLIMIT_FSIZE (it can cap UCX internal SHM); bound
owned logs/proof instead. MPI test is opt-in ignored and is **not automatically
exercised by ordinary CI/native matrices** merely by listing it.

Linux exact publication-head serial/all-features/lint/docs gates are pending.
No new native/model rerun is implied or needed for relocating unchanged code;
integration-relevant changes would require separate validation. Full #182
callsite/worker/threshold/FSZ/SR/QQQQ/Lanczos/allocation/timer/benchmark scope,
and broader #185 acceptance, remain open.
