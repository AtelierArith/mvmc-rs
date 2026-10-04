# Issue 178: real sampling failure and bounded frozen859 MPI evidence

Related to [#178](https://github.com/AtelierArith/mvmc-rs/issues/178).
**Authoritative status:** the frozen859 12-case MPI matrix completed and passed
parent independent validation. The parent verified36 rank identities,
72 RETURN/72 CHECKPOINT records, exact repeats, the saved validator (stdout
only, no JSON overwrite), and the artifact manifest: all checks terminated0.
This is bounded evidence for base859d13e2/test69b0ffc5/binarybfd1a950, **not**
current176-kernel validation, whole #178 acceptance, or native-C sampling parity.
No production source change is part of this test extension. The prelaunch and
earlier snapshot sections below are historical, not current pending status.

## Real finite failure through the public runners

`crates/mvmc-core/tests/mpi_issue178_sampling_failure.rs` retains the committed
phase1 direct-kernel/sampler test and adds public serial controls plus an
explicit ignored MPI gate. The public controls use `vmc_para_opt` and
`vmc_phys_cal_in_place`, caller-owned data/state/SFMT, and actual callbacks.

The declared General mapping covers all six upper-triangle pairs of four
combined spin/site coordinates. Every pair shares one real coefficient with
sign+1; the reverse is antisymmetric. One explicit identity translation and
one QP are used. Nsite2/Nelec2/TwoSz0 implies complete occupancy. A zero
Gutzwiller coefficient has a nonempty projection-count plane. Settings are
sample3/warmup1/interval1, optimization step/window1, Lanczos0/CG0.

The public Slater refresh forms the antisymmetric table from that declared
coefficient, rather than a test overwriting the final table. With coefficient
2^600, the table entries remain finite (the antisymmetrized value is 2^601),
while the four-by-four Pfaffian overflows. The real initializer reports
`NonFinitePfaffian { qp: 0 }` after its actual 101 failed attempts. These huge
parameters intentionally exercise an error boundary; they are not presented
as a valid C physical workload. There is no NaN or FaultReducer injection.

The public serial test executes each failure twice from seed1 after one word
has already been consumed. Both APIs report the exact nonfinite-Pfaffian
diagnostic, callback0, unchanged coefficients/flags/SR buffers, no optimization
history, and identical final raw624/cursor/count/nonconsuming-next624 plus
saved/tmp configuration checkpoint. This same-implementation repeatability
is not an independently generated C trajectory oracle.

The coefficient1 control succeeds on each API and calls its callback once.
OPT's healthy control explicitly uses `skip_sr=true`: it proves sampling and
public runner acceptance, not successful solving of this synthetic SR system.
The failure controls use `skip_sr=false`.

Owner handle3989 terminated0: nextest
`ca8a1694-c040-4689-96cf-4037533845e2`, 2 PASS, 0 skipped, 0.013s.
After making identity translation records explicit and compiling the MPI gate,
handle61964 terminated0: nextest
`2ca2454a-71db-4f7a-8060-c7d6457ab2ab`, 2 PASS, 1 explicit MPI gate skipped,
0.054s. Both report failure count617/cursor617 for each repeat and healthy
count35. The earlier compile-error log is retained, not treated as a pass.

## Publication boundary

Failed initialization publishes the last tmp configuration/projection counts
and a real Slater copy, but no saved configurations/counters or failing owned
Pfaffian/inverse result. Kernel results are staged before publication.
PhysCal replaces its caller state before sampling; therefore its checks use
the new entry allocation, not a claim that old caller sentinels survive.
The MPI draft deliberately supplies PhysCal sentinels to expose this boundary.

Normal groups may finish saved sampling before observing a failure in another
group. The test must not demand global saved-state rollback or unchanged RNG.
It instead checks appropriate failing-group versus healthy-group publication,
then checks same-input/rank/group/seed repeatability within each process.

The callback count is observed through the actual public closures. No SR
internal-call instrumentation is added: unchanged public SR buffers/history
and coefficients, together with the runner's sampler-error return before
accumulation/SR, establish the scoped pre-SR boundary.

## Historical prelaunch MPI draft: exact 12 cells, then unexecuted

For each world2/world4:

| API | group width | fault global ranks |
| --- | --- | --- |
| OPT | 1 | 0, world−1 |
| OPT | 2 | 0, world−2 |
| PhysCal | 1 | 0, world−1 |

Only the selected fault rank receives the huge coefficient. With width2 and
one QP, group-local rank0 owns range0..1, its partner owns range1..1. Thus the
last owner is world−2, not the empty global-last rank. Grouped FSZ PhysCal
remains unsupported and is deliberately excluded.

An observational Reducer delegates all operations to the actual
`MpiGroupContext`, including QP assignment, comm1 sums/status, global
reductions/status, broadcasts, barrier and counter reduction. It only records
the local booleans passed to comm1/global agreement. It changes no result.
The initializer performs actual comm1 failure MAX agreement; the owned-QP
rank retains its local error and the empty peer receives comm1 PeerFailure.
The public runner then agrees globally, including ranks in healthy groups.

Each case repeats twice. Required observations are every rank returning the
correct local/comm1/global diagnostic, callback0, fixed coefficients/flags,
unchanged SR/history, stage-appropriate saved/tmp/Pf/inverse publication,
exact repeated raw RNG/cursor/count/next624/configuration, and final barrier.
The process-root-owned parent is created exclusively and keeps a sentinel.
PhysCal may create empty trial output directories before sampling; no measured,
var or zqp files may be written. All test artifacts are retained.

The planned launcher uses an external bounded timeout; MPICH/Hydra additionally
uses its documented `-disable-auto-cleanup` option so each rank's exit status
is retained. Per-rank return/DONE/checkpoint markers and test summaries must be
validated; launcher success alone will not establish a pass.

## Historical initial-snapshot environment and evidence limits

Serial compilation/execution uses container73c57e563c61, copied checkout
`/tmp/mvmc-cli174-gmUVyr`, separate target
`/home/vscode/.cache/mvmc/target/cli174-gmUVyr`, Linux x86_64, Rust1.99.0,
OpenBLAS with OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1. This reuses a dirty
snapshot with reviewed-path overlays, not a clean-main compiler-input closure.

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --features mpi --test mpi_issue178_sampling_failure \
  --success-output immediate --no-fail-fast --retries 0
cargo clippy --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue178_sampling_failure -- -D warnings
```

Retained host logs under `/tmp/mvmc-cli174-gmUVyr`:
`178-public-sampling-serial.log` (compile failure),
`178-public-sampling-serial-corrected.log`,
`178-public-sampling-final-serial.log`, and
`178-public-sampling-clippy.log` (initial lint failure, to be followed by a
separately named final log). Final frozen source hashes and actual MPI launch
results will be recorded only after review and execution.

Final pre-MPI-review source SHA-256:
`17c9b9ac8c7d28c57d0ed911c7d69b957b368002122c6233ac4b83213f86af53`.
Host/container source hashes match after the final serial run. Handle91039
terminated0, nextest `037a957b-399b-405b-b17d-3890cc2aaa4d`: 2 PASS,
1 explicit MPI gate skipped, 0.058s. Handle39147 final strict clippy
terminated0, 1.38s. These runs compile the MPI test but do not execute it.

```text
178-public-sampling-review-serial.log 519b1f8c1c4e15b30be0c2e8969a139a437fba32d37b58f039cf543ccbd55509
178-public-sampling-final-clippy.log aee9513c127dd5a410d8b71d0ef07c77a652c74478c49e6d44bea0623ffef96d
```

## Historical fresh committed prelaunch association (then unexecuted)

The earlier dirty-snapshot results above remain historical. The fresh final
snapshot is `/tmp/mvmc-178-phase2-final.QPzi3r` on host and container73c57e563c61,
exported with `git archive` from committed
`859d13e2c8921b64c4b0f4dd8fe88d3b8df511de` (including dd3fe238). Each reference
submodule was separately archived at the exact committed gitlink:
Julia8bb1b9e8, PfaPack0dcf52c1, C-mVMCd73d06bd. Only this test and this document
were overlaid; no dirty support file or host `.cargo` configuration was copied.
The frozen document at export is d87a8098; this subsequent evidence append is
not retroactively assigned to that immutable snapshot.

A preceding fresh snapshot `/tmp/mvmc-178-phase2-immutable.zMi3CJ` reproduced
2 PASS but revealed that the healthy OPT control's `output=None` wrote
`zvo_out.dat` and `zvo_var.dat` into the crate directory. Its logs, binary and
generated files are retained. The test now uses an exclusively created
temporary output path for each healthy public control and removes only that
owned path. The final snapshot was exported anew, rather than cleaning or
relabeling the preceding snapshot.

Final test source:
`69b0ffc517b4baf5c72bde527bc729c12c500c0ad9c73b8ba2aa685b07d8de63`.
Dedicated target: `/home/vscode/.cache/mvmc/target/issue178-phase2-QPzi3r`.
The committed `.devcontainer/cargo-config.toml` is supplied explicitly, with
that target and OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1 in the environment.

```sh
cargo --config .devcontainer/cargo-config.toml test --locked --profile test-fast \
  -p mvmc-core --features mpi --test mpi_issue178_sampling_failure \
  --no-run --message-format=json
cargo --config .devcontainer/cargo-config.toml nextest run --locked \
  --cargo-profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue178_sampling_failure --success-output immediate \
  --no-fail-fast --retries 0
```

Build handle60409 terminated0, 14.57s. Fresh serial nextest
`5defa82c-3d49-4b52-b50e-11d253c08e1f` terminated0: 2 PASS, the one explicit
MPI gate skipped, 0.053s. Both public failure repeats retain the exact
nonfinite-Pfaffian diagnostic/count617, and both healthy controls succeed
with count35. Final strict clippy handle95262 terminated0, 4.37s.

The build JSON associates this source with executable
`/home/vscode/.cache/mvmc/target/issue178-phase2-QPzi3r/test-fast/deps/mpi_issue178_sampling_failure-e740c1879603890e`,
SHA `bfd1a950743e7e26e7aaeab77c64a53b7d0807cdf8503a9ddedb10f83e2f9536`.
`--list` reports three tests; `--ignored --list` reports only
`mpi_phase2::actual_sampling_failure_reaches_comm1_then_all_global_ranks`.
Listing is not execution of that gate.

All26477 exported snapshot files have identical before/after manifests:
`e043a1b804ea20b5a6b7b51080e1a3ebcaadb05f0a491cc5833b6afc8782e405`.
Additional manifests cover registry package files selected from locked Cargo
metadata, generated build outputs, Rust sysroot/dependency artifacts,
native SDK headers/tools/configuration, and every resolved runtime library
from the executable's ldd output. Actual linkage is system OpenBLAS0.3.26
and MPICH4.2 libmpi.so.12; compiler is Rust1.99.0. These retained file sets,
metadata and selected environment are an explicit association record, not
a claim of a hermetic build against all possible unobserved system inputs.

The host proof files all have prefix `/tmp/mvmc-178-phase2-final.QPzi3r-`:
`association.json`, `proof.sha256`, `source-before.sha256`, `source-after.sha256`,
`registry.sha256`, `generated-build.sha256`, `compiler-artifacts.sha256`,
`native-toolchain.sha256`, `runtime-libraries.sha256`, `metadata.json`,
`environment.log`, `build.jsonl`, `build.log`, `serial.log`, `clippy.log`,
`binary.sha256`, `tests.list` and `ignored.list`.
The proof manifest hashes these evidence files; the association identifies
base/source/binary/run explicitly. No old failed log was overwritten.
The 12 MPI launches require a separate approval after this proof's review.

## Authorized completed 12-case MPI evidence (frozen859 only)

This section supersedes the draft's pending-execution status above. After full
source/document/association review and the parent's corrected hash-check
handle54522 terminated0, the parent authorized the actual launches. The earlier
prelaunch and dirty-snapshot records retain their original scope and identities.
No test/source edit or rebuild occurred after launch started. This is **not**
the PR215/current Ram176 real-kernel proof and does not close #178.

The same owner matrix handle42266 terminated0. All12 launches completed without
timeout or forced termination: worlds2/4 × OPT widths1/2 root/last-QP-owner,
plus PhysCal width1 root/last. Each rank ran two repetitions inside its one
explicit test. Width2 last owner is world−2; the empty world-last rank receives
the comm1 peer error, not an invented local numerical failure.

The exact executable remains the prelaunch bfd1a950 binary, built from
859d13e2/dd3fe238 plus test69b0ffc5. Container73c57e563c61 used Rust1.99.0
(b940084d7, LLVM23.1.1), Cargo1.99.0, MPICH/Hydra4.2.0, system OpenBLAS0.3.26,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Host offline validation used
Node24.16.0; no reference oracle generated numerical expectations.

For every cell the launcher was the following command, with ranks/api/width/
fault and a new exclusively owned case directory selected by the 12-case
matrix. It executes the actual ignored test, not a test double:

```sh
timeout --kill-after=3s 8s mpirun -disable-auto-cleanup -print-all-exitcodes \
  -n "$ranks" -outfile-pattern "$case_dir/rank%r.stdout.log" \
  -errfile-pattern "$case_dir/rank%r.stderr.log" \
  env MPI_ISSUE178_WIDTH="$width" MPI_ISSUE178_API="$api" \
  MPI_ISSUE178_FAIL_RANK="$fault" \
  MPI_ISSUE178_OUTPUT="$case_dir/owned-output" \
  /home/vscode/.cache/mvmc/target/issue178-phase2-QPzi3r/test-fast/deps/mpi_issue178_sampling_failure-e740c1879603890e \
  --exact mpi_phase2::actual_sampling_failure_reaches_comm1_then_all_global_ranks \
  --ignored --nocapture --test-threads=1
```

The matrix enumerates `opt:1:root`, `opt:1:last-owner`, `opt:2:root`,
`opt:2:last-owner`, `physcal:1:root`, `physcal:1:last-owner` for each world2/4.
Hydra's primary help confirms `-print-all-exitcodes`, per-rank output patterns
and `-disable-auto-cleanup`. Its retained launcher logs report all36 individual
native rank exit codes0, in addition to all12 launcher statuses0.

### Actual rank-by-rank contracts

The host validator succeeded, requiring **36 unique rank PASS summaries and
DONE markers, 72 RETURN markers and 72 CHECKPOINT markers**. Every required
rank0..world−1 appears with the correct API/width/fault/QP range and both
trial0/1. RETURN is printed before assertions, preserving actual diagnostics:

- Fault QP owner: `calc_m_all: non-finite Pfaffian at qp=0`.
- Same comm1 empty-QP peer: `sampling initialization failed on another comm1 rank`.
- Other groups: the precise OPT/PhysCal global peer sampling diagnostic.

No generic/unexpected error or successful API result is accepted as a fault
pass. The fault group's101 comm1 calls retain the true local failure boolean
only on its QP owner. Healthy groups report no local numerical failure, then
participate in the runner's global agreement. Therefore the real fault input
survived runner preparation; the typed local failure is not inferred solely
from another rank's generic Err. The coefficient/flag preservation assertions
also passed; an unnoticed broadcast replacing the bad parameter would fail
these checks or eliminate the required local error.

Each CHECKPOINT carries624 raw u32 words, cursor, consumed count,624 next words,
and saved/tmp/burn/counter configuration. All captured fields are identical
between the two repeats on the **same rank, same group width, input and seed**.
The peek is nonconsuming in the test. Counts differ appropriately across group
seeds/failure versus successful sampling; observed counts are
35/42/43/44/580/582/617. They are actual diagnostics, not independent-C golden
counts or a requirement for identical trajectories across different groups.

Both callbacks remain0 on all ranks. Fixed projection/Slater values and flags,
public SR buffers and empty optimization history remain unchanged. The failure
group retains only its final tmp configuration and Slater preparation, with no
saved configurations/counters or failed owned Pfaffian/inverse publication.
PhysCal's original caller sentinels are replaced at entry; its checks refer to
the replacement allocation, not whole-call state rollback. Healthy groups can
publish complete saved sampling before the global Err, as explicitly checked.
No SR-call counter or new production observation API is claimed.

All exclusively owned parent sentinels remain unchanged. OPT writes no trial
output directories/files; PhysCal creates empty trial0/trial1 directories
before sampling but no energy/var/Green/zqp files. All case artifacts remain
available. There were no failing matrix cells to discard or rerun; earlier
compile/lint/healthy-output-preparation artifacts are preserved separately.

### Postlaunch identity and retained proof

Before and after MPI, every26477 snapshot file matches manifest SHA
`e043a1b804ea20b5a6b7b51080e1a3ebcaadb05f0a491cc5833b6afc8782e405`.
The binary remains
`bfd1a950743e7e26e7aaeab77c64a53b7d0807cdf8503a9ddedb10f83e2f9536`.
All ldd-resolved runtime libraries pass their recorded SHA checks before/after.
Canonical backends and hashes:

```text
/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so
bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e
/opt/mpich/lib/libmpi.so.12.4.0
638c51116955894e0a8fd91b3f7091246ad3155c37141780958e39f5e102da11
```

Host and container artifact directory: `/tmp/mvmc-178-phase2-mpi.FQGO3F`.
Each of its12 case directories retains `launcher.log`, individual rank stdout
(stderr if any), sentinel and output inventory. Additional host evidence:

```text
/tmp/mvmc-178-phase2-mpi.FQGO3F-matrix.tsv
7dd4b20d11395ca79f55663d1971187fbc3dac31d8abb43e16ab73c66950018b
/tmp/mvmc-178-phase2-mpi.FQGO3F-validation.json
644788a61610df73c67da01a9e6bbc8d026269df4b23688fe90be7d53dbfa63b
/tmp/mvmc-178-phase2-mpi.FQGO3F-artifacts.sha256
c69be32d3a7f20ec33f0a0303445535632a7d4bcbb5da65a17b616ce5c21b82b
/tmp/mvmc-178-phase2-mpi.FQGO3F-validate.js
8a16731ef551ea2f43c7052d1decc714e7544612acb288e45f8f5afdf413e549
```

The validator originally ran as `node -e '<body>'`, terminating0; its same
body was subsequently saved in the listed script for review. The saved script
has not been rerun and the existing validation JSON has not been overwritten.
To independently validate without overwriting that JSON:

```sh
node /tmp/mvmc-178-phase2-mpi.FQGO3F-validate.js
```

Exact postlaunch identity files:
`/tmp/mvmc-178-phase2-final.QPzi3r-mpi-source-after.sha256`,
`/tmp/mvmc-178-phase2-final.QPzi3r-mpi-binary-after.sha256`,
`/tmp/mvmc-178-phase2-final.QPzi3r-runtime-libraries.sha256` and
`/tmp/mvmc-178-phase2-final.QPzi3r-mpi-libraries-after.log`.
Corresponding before-files are retained. Binary manifest SHA is63193f66;
successful runtime-check log SHA is9b9bb3f4 before/after. These are manifest/log
hashes, distinct from the executable/library hashes above.

This closes the bounded natural-sampling-failure evidence gap on **frozen859**,
not all #178 criteria, every supported model, full native-C sampling parity,
or validation of later real/complex-kernel changes.

## Parent independent acceptance of this bounded evidence

After full review of document SHA6bcde2a7 and the unchanged test/association,
the parent independently checked all36 ranks,72 RETURN and72 CHECKPOINT
records, both trial0/1 records, and exact per-rank repeated checkpoints.
That audit terminated0. The saved validator was independently rerun to stdout
only, terminated0 with `allTwelveValidated=true`, and did not overwrite the
retained validation JSON. The artifact manifest also checked successfully,
terminal0. An earlier parent audit terminal1 was not evidence of missing
trials; the completed audit explicitly verifies both records.

The acceptance retains owner handle42266, frozen base859d13e2/dd3fe238,
source69b0ffc5, binarybfd1a950, fresh serial run5defa82c and all recorded hashes.
It is not relabeled as PR215 squashae1e83d7/current production validation.
The proposed future current176 phase1 preparation of an explicit real Slater
plane is unimplemented and outside this frozen proof. No issue closure is
implied by this bounded parent acceptance.

### Nonpromotion: newly identified C retry-limit boundary

The parent subsequently identified `vmcmake_fsz_real.c` 487–500: the C
101st-attempt limit aborts even if that attempt succeeds, whereas the audited
Rust branch returns Ok on success before testing attempt100. Independent
native boundary evidence and any repair remain pending. No lifecycle/test
change is included here. The frozen859 cases above fail numerically on all101
attempts, so their actual error/publication/agreement records remain valid;
they do not exercise or establish parity for success only on the101st attempt.
Do not promote this bounded old-behavior evidence to current/full #178 or
complete C retry-lifecycle acceptance.
