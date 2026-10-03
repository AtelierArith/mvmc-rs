# Issue 175 callback completion audit

Issue: <https://github.com/AtelierArith/mvmc-rs/issues/175> (**CLOSED**, reason
`COMPLETED`, GitHub-confirmed at `2026-10-03T13:18:03Z`). Parent reviewed the full
current source/evidence and the original five acceptance criteria, verified
callback production/tests identical to `origin/main`, and committed this audit
in `859d13e2` before closing the issue. Closure is limited to #175.
Historical audit baseline: `bff283dab667d7720291ba190546629dd68f44fb`.
Current focused verification is recorded below, separately from the historical
runs. Historical no-closure statements below describe those earlier checkpoints;
they are not current issue status. These are the original five
#175 criteria, not whole-issue-185 or whole-issue-179 acceptance.

## Current focused verification

Serial snapshot: immutable checkout `/tmp/mvmc-post212-verified.hbhcU3` at
`69a53195a97d66aae0fa6cf55d7ee9c41b7557b4`, with its own target and no overlays.
Parent session `30236`, nextest run
`3abbf332-c07d-4f18-9b5b-26d066cf9bf9`, completed with **exit 0, 22/22 passed,
0 skipped, 209.577s**. The exchange-spin callback test took 209.577s.
This is the focused gate's terminal result, not a claim that the separate
full-workspace run passed.

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast --locked \
  --test physcal_callback --test physcal_callback_reference \
  --test physcal_callback_exact_contract --no-fail-fast --retries 0
```

The binaries select 9 serial/error tests, 12 independent-reference tests and
1 exact observation test. Their current source hashes are recorded below.
The diff from `69a53195` to `c8db43bf461fe6fb426b5fee117319849c1b9c21`
contains only three documentation files; production code, callback tests and
fixtures are identical. The measured `run.rs` SHA-256 in both snapshots is
`9e31787961860233c094d810043627e01d321045d6affa52bccf614a594f8968`.

MPI snapshot: container `73c57e563c61`, root
`/tmp/mvmc-175-c8-frozen.t9A8dB`, extracted only from the committed `c8db43bf`
git archive. Reference hydration uses Julia commit
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, nested PfaPack
`0dcf52c15caec63516d0703f36bfc8a4bc0e58d0` and SFMT
`1526553009f318ae78338151460fda78beadddc2` git archives, not dirty files.
No C/Julia oracle is run by this gate. The pinned `Manifest-v1.13.toml` hash is
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.

Compile/list-only session `92500` completed with exit 0 and listed exactly one
ignored test, `mpi_physcal_callback_preserves_two_samples_and_collective_error_boundaries`.
Separate target: `/home/vscode/.cache/mvmc/target/mpi-callback-c8-t9A8dB`.
Actual MPI worlds 2 and 4 subsequently **both completed with exit 0**; each rank
reported **1 passed, 0 failed, 0 ignored**, 0.14s. This is one distinct test
launched twice. Each world exercises eight runner cases per width, widths 1 and
2, hence 16 cases/rank: 32 runner executions for world 2 and 64 for world 4,
not 96 distinct tests.

The eight cases are callback-off prefixes 1/2, callback-on two samples,
last-world-rank callback errors at samples 0/1 with and without a callback on the
root, and a genuine root filesystem write failure. Every rank has ten diagnostic
markers (four callback-error cases and one output-error case per width).
Assertions verify every-rank callback indexes/status/post-average energy,
collective error agreement and retained boundary, full configurations/counters,
actual raw SFMT words/index/draw count/next624, fixed coefficient bits/flags,
QP values and same-run copies where anchored, and ordered root-only output.
Output failure prevents callbacks; callback failure prevents later samples.
Each world retains sixteen case directories, writer 0 only. Root-write failures
retain out/var sample 001 and the blocking cisajs directory, with no sample 002.

```sh
docker exec -u vscode -w /tmp/mvmc-175-c8-frozen.t9A8dB 73c57e563c61 \
  env MPI_PHYSCAL_CALLBACK_OUTPUT=/tmp/NEW-exclusive-output \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  MVMC_RS_INNER_THREADS=1 MVMC_RS_INNER_THRESHOLD=32 \
  timeout --kill-after=15s 120s /opt/mpich/bin/mpiexec -n 2 \
  /home/vscode/.cache/mvmc/target/mpi-callback-c8-t9A8dB/test-fast/deps/mpi_physcal_callback_contract-6c9a483627524c9b \
  --ignored --exact mpi_physcal_callback_preserves_two_samples_and_collective_error_boundaries \
  --test-threads=1 --nocapture
```

Repeat with `-n 4` and another nonexistent output path. Actual outputs and native
tee-captured logs are under container `/tmp/mvmc-175-c8-results.l830c5/`, with
`world2` and `world4` output roots. Linux x86_64, Rust 1.99.0/LLVM 23.1.1,
MPICH 4.2.0 `ch4:ucx` under `/opt/mpich`, OpenBLAS 0.3.26, threads 1.
Image ID: `sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`.
Archive has no local `.cargo/config.toml`; build uses explicit separate target,
`RUSTC_WRAPPER=/opt/devtools/bin/kache`, `KACHE_CONFIG=/opt/devcontainer/kache.toml`,
`MPICC=/opt/mpich/bin/mpicc`, `LIBCLANG_PATH=/usr/lib/llvm-18/lib`,
`LD_LIBRARY_PATH=/opt/mpich/lib`, `PKG_CONFIG_PATH=/opt/mpich/lib/pkgconfig`.
No dependency, fixture, numerical budget or production change was made.

All 25,881 archived/hydrated source files have identical pre/post sorted
`sha256sum` inventory digest
`0e4f1d8e40d1abc04912fb37522d686e64dfd423a36d0d1d66c77627ee9402e6`.
Binary SHA-256 is unchanged before/after launches:
`cfa2fbc448347520876b395e0bf2f82c5464d518e8c1e0b56f3e3fa41aeb5eda`.
Every resolved linked library, loader and launcher was hashed before/after and
remained unchanged; the full post list is retained in `post-hashes.log`.
In particular, OpenBLAS is
`bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e`
and libmpi is
`638c51116955894e0a8fd91b3f7091246ad3155c37141780958e39f5e102da11`.
Parent independently reviewed the fresh MPI logs and hashes.

| Artifact under `/tmp/mvmc-175-c8-results.l830c5/` | SHA-256 |
| --- | --- |
| `world2.log` | `afb848b37b7bf4efa6503c6060055118bfeacdf0b7967e4f3b96f0b056c5d291` |
| `world4.log` | `13e640aaf794be137fadc5d648aaef02364b1f08259611f6fdc2aca481a8f141` |
| `post-hashes.log` | `6a519d226061203de24fb03321082394fb1ab71fa0e5430219ea07e8db32f3f0` |
| `qualified-output-inventory.log` | `d80b95444ef5ca7b36da1691774b645624c27b1d5ed43f5aca230f17575d7102` |

## Original five-criterion acceptance mapping

The current serial 22-test gate and current c8 MPI2/4 gate above execute the
specific checks below. Historical runs are not silently reassigned to these
snapshots. No general full-model or whole-workspace prerequisite is added to #175.

| Criterion | Existing concrete evidence | New bounded gap closure (details below) |
| --- | --- | --- |
| Serial and Reducer callbacks | `run.rs`: serial, output-directory, Reducer wrappers delegate to `vmc_phys_cal_in_place`; callback receives immutable data | Actual multi-rank callback execution, worlds 2/4, widths 1/2 |
| Index/data/energy/status/rank | Julia `vmc_phys_cal.jl` full source: loop starts at zero; averaging and output precede callback; callback outside root guard; `info` remains zero. Rust mirrors that ordering and invokes on every rank | Every rank observes indexes 0/1, status 0 and post-average energy; output root observes completed files |
| No RNG/parameter/weight/output effects | `physcal_callback.rs`: three-sample on/off final configuration, next624, fixed Slater values, QP weights, energy and output comparisons. `physcal_callback_reference.rs`: independent six-model first boundaries, two-sample references, actual final draw counts/next624/configurations | New serial and MPI checks cover raw SFMT state/count/next624, full fixed signed bits, flags and indexed inventory. QP initialization across preparations is numerical; bit-copy checks use only the same execution's first callback anchor |
| Errors without deadlock/false success | Serial callback Err stops at requested boundary. In-place tests retain actual averaged state/RNG. Reducer mocks test remote failure agreement and absent callback participation. Genuine serial filesystem failure prevents callback | Actual MPI last-rank callback Err at indexes 0/1 and root write failure; every rank returns Err and completes subsequent agreement, retains the matching boundary, never writes a later sample |
| Success/failure/multisample fixtures | Offline `physcal_181/two-samples` and `physcal_callbacks_175`; callback energy comes from independent averaged references, not Rust-generated expectations | New dedicated 2/4-rank success/failure/multisample checks use existing offline inputs without creating reference expectations |

## Authority and scope

Callback shape and placement are Julia API behavior, not a C callback feature.
Julia invokes callbacks on every rank but does not implement Rust's collective
`Result` failure agreement. That agreement is a Rust distributed-error contract;
its mock tests alone cannot establish freedom from MPI deadlock.
C remains authoritative for the measurement, fixed-parameter layout and indexed
output contracts. Julia's unindexed out/var extension is not adopted as C parity.

Existing reference tests use their documented `1e-12` absolute / `1e-10` relative
measurement bounds and labelled OptTrans phase-order harness. They are not full
native-C sampling evidence. Callback on/off comparisons are same-input executions,
not replacement independent numerical oracles. No fixture regeneration or numerical
budget change is part of this audit.

Historically, Wegener confirmed existing actual MPI PhysCal repeat gates cover configuration,
SFMT state/count/next624, fixed signed bits and output, but not callbacks. His
optimization callback gates are not PhysCal callback proof. The new disjoint
`mpi_physcal_callback_contract.rs` is assigned separately; no edits to
`mpi_issue179_state.rs`, `run_mpi_tests.rs` or runner production code are planned.
The requested Linux MPI worlds 2 and 4 were subsequently executed against exactly
that baseline plus only the owned MPI test overlay, as recorded below.

## Historical initial validation (QP policy correction supersedes source)

The initial sources below incorrectly compared computed QP initializations
between two preparations bitwise. Parent review identified this policy issue.
Their recorded passing runs are retained as historical executions, not accepted
same-run QP-copy proof. Revised sources separate numerical cross-run initialization
from exact same-run copies; fresh validation is recorded in the next section.

`mpi_physcal_callback_contract.rs` source SHA-256:
`3e2ad1943e96af6a529d705ee0bd098719ab8a923f17afd389197ce6e8c92c52`.
Full source and both launch logs were reviewed. Both MPI launches completed
successfully, with one passing test on every rank and no failure/ignored test.
World sizes 2 and 4 each cover ungrouped width 1 and grouped width 2:

- Callback off/on with two samples, every-rank zero-based calls and status zero,
  callback after post-average energy and complete root output.
- Failure injected only on the last world rank at sample 0 and sample 1; peers
  return the remote-error diagnostic, never success. Repeat with no callback on
  the output rank, proving absent callbacks still participate in agreement.
- Genuine root filesystem write failure prevents all callbacks and further
  samples; completed first-sample data/RNG are retained on every rank.
- All saved/scratch configuration planes and counters, SFMT raw state/cursor,
  primitive draw count and non-consuming next624 agree with the matching
  callback-off boundary. All declared fixed coefficients retain signed bits,
  flags are unchanged. This initial version also checked all seven QP buffers
  bitwise against another preparation; that comparison was a policy error,
  not a valid same-run immutable-copy proof, and is corrected in the revision.
- Root output inventory and ordered index columns agree, nonroot paths remain
  absent, and no later sample is written after failure. Computed energy/Green
  and output values keep the existing same-implementation `1e-12` absolute plus
  `1e-12` relative budget. No new independent expectation is generated.

Snapshot directories: `/tmp/mvmc-mpi-callback-baseline-bff` and
`/tmp/mvmc-mpi-callback-bff`. Recursive comparison reports **only** the new
`crates/mvmc-core/tests/mpi_physcal_callback_contract.rs`. Source hash was unchanged
before/after launches. Container `73c57e563c61`: Linux x86_64, Rust 1.99.0,
MPICH 4.2.0 `ch4:ucx` under `/opt/mpich`, OpenBLAS 0.3.26, threads 1.
Cargo profile `test-fast`, `--locked`; separate target
`/home/vscode/.cache/mvmc/target/mpi-callback-bff`.

Reproduction, with a fresh nonexistent shared output path for each launch:

```sh
docker exec -u vscode -w /tmp/mvmc-mpi-callback-bff 73c57e563c61 \
  env MPI_PHYSCAL_CALLBACK_OUTPUT=/tmp/NEW-exclusive-output \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  timeout 120 /opt/mpich/bin/mpiexec -n 2 \
  /home/vscode/.cache/mvmc/target/mpi-callback-bff/test-fast/deps/mpi_physcal_callback_contract-6c9a483627524c9b \
  --ignored --test-threads=1 --nocapture
```

Repeat with `-n 4` and a different nonexistent output path. Binary SHA-256:
`0bce67fdd621107785ee15e424fc1da42680570f8785223ea26e794187f13b22`.
Retained launch logs:

| Artifact | SHA-256 |
| --- | --- |
| `/tmp/mvmc-mpi-callback-bff-mpi2.log` | `9b30740d4d670d2c73928ada0faa94a104237e864e3c052b3100742c38a3557f` |
| `/tmp/mvmc-mpi-callback-bff-mpi4.log` | `fee9507f373cb4b65978711e74405c1b8d8637b290a29b4b4db3a2c75de7d5bc` |
| `/tmp/mvmc-mpi-callback-bff-container-source-manifest.txt` | `0926b42b9fc07ef509a496abe798a04f773de710291f2d053288b656b9744255` |
| `/tmp/mvmc-mpi-callback-build.log` | `b6af97ea7ea5a88ce00c2d715a86a84994391f89d028b3b74dd7a91ebe22cf10` |
| `/tmp/mvmc-mpi-callback-bff-clippy.log` | `a93afb821465cdb5f1e37e465413a1a63dce0b1de0204f83dcfab7f2f9668444` |
| `/tmp/mvmc-mpi-callback-bff-environment.log` | `61dac101c0b31ffb50516ccc2e9de31c4135db627c3f9d4956889fe72f1499f0` |

The frozen-container MPI focused clippy completed with exit 0, source/binary
hashes unchanged. Full lint and actual environment/linkage logs were reviewed.
Command:

```sh
docker exec -u vscode -w /tmp/mvmc-mpi-callback-bff 73c57e563c61 \
  cargo clippy -p mvmc-core --test mpi_physcal_callback_contract \
  --features mpi --profile test-fast --locked \
  --target-dir /home/vscode/.cache/mvmc/target/mpi-callback-bff -- -D warnings
```

The separate short `physcal_callback_exact_contract.rs` uses the offline standard
real Heisenberg two-sample input, seed 1, sample count 3 and warm-up 1; parsed
interval, QP settings and starting index 7 remain unchanged. It checks the same
serial raw-RNG/full-fixed contracts, ordered output inventory and numeric
content. Untimed indexed PhysCal output only is compared; timer files are not a
deterministic output contract. Source SHA-256:
`8ef3f269841a9f70c6963185b148d37f15461cf2bc5c8aab1fe683aee596a040`.
Current shared-workspace focused nextest run
`772e0fd7-2237-49d5-848b-9dde2bc53d43`: terminal 0, 1 passed, 0 skipped, 0.025s.
It is not relabelled as the frozen MPI snapshot. Command:

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast --locked \
  --test physcal_callback_exact_contract --no-fail-fast --retries 0
```

Focused serial clippy (`--test physcal_callback_exact_contract --profile
test-fast --locked -- -D warnings`) completed with exit 0. `rustfmt --check` on
both new test files completed with exit 0. An existing dependency deprecation
warning in vendored tenferro was emitted; no suppression or unrelated edit was
made.

Full serial/reference helper rerun: handle `47137`, run
`4d16d82e-d80c-48e6-965a-a63924a62f62`, terminal 0, **21/21 passed**, 0 skipped,
218.892s (spin Lanczos was the slow test). This was a current shared-workspace
run, not the frozen MPI snapshot; the two reviewed test source hashes below
remained unchanged. Command:

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast --locked \
  --test physcal_callback --test physcal_callback_reference \
  --no-fail-fast --retries 0
```

Host MPI-enabled focused clippy could not build `mpi-sys` because this host has
no `mpicc`/MPI pkg-config installation. It is not a source lint result; MPI
validation uses the configured container. No dependency or warning suppression
was introduced to bypass that setup failure.
Parent review/current-integration validation remain separate from these bounded
proofs. Existing independent fixtures and their provenance are unchanged.

## Historical revised QP policy and validation

Only QP comparison scope changes; no production code, parameter/RNG algorithms,
independent fixtures or numerical bounds change. Both tests compare independently
initialized QP buffers between runs numerically, preserving each buffer's length.
Serial retains the existing QP bound from `physcal_callback.rs`, `32 * EPSILON`
absolute and relative. MPI retains its existing same-implementation `1e-12`
absolute and relative bound.

Exact prior serial source: `physcal_callback.rs:156` labels the QP
coefficient/quadrature comparison; lines 175–177 use `32.0 * f64::EPSILON` for
both tolerances and label it `callback QP weights`. Its unchanged SHA-256 is
`02f37f7e8e276dd4437909134048c82859e5551bf3c4d44719b8b896c36ace76`.
Thus the new comparison adopts an existing separate-run QP budget, not a budget
previously present in the new test's erroneous bit check. No unverified gamma8
operation-count derivation or new cross-backend admissibility bound is claimed.

The callback-on execution captures its own seven initialized QP buffers at its
first callback. Its subsequent callbacks and returned data must retain those
exact bits, including signed zeros. Callback-error runs similarly compare their
returned retained buffers to their own callback anchor, when a callback ran.
This anchor does not prove exact immutability from initialization through the
first sample; the first observation occurs after the first averaged sample.
An error at callback 0 proves only the first-callback-to-return copy boundary;
it does not establish a pre-sampler anchor.

Filesystem failure happens before any callback. No public pre-sampler initialized
QP checkpoint is substituted or invented. This case checks QP values numerically
against the callback-off boundary and **does not claim exact same-run QP
immutability**. A peer with no callback likewise has no own callback anchor.
Exact configuration/RNG/fixed-copy and distributed failure assertions remain.

Revised serial source SHA-256:
`31093b6cdf7c5023d0f1d4fdb5287351b448b5bcadb996722463e68508444aac`.
Owner fresh short nextest run `c8397a31-b577-4bcc-838a-46005e653930` completed
with exit 0, 1 passed, 0 skipped, 0.029s; focused serial clippy exit 0. Parent
read the full revised serial source and independently reported fresh run
`608e4980-e455-4890-a849-af5cbc6a2d96`, terminal 0, 1 passed, 0.029s. This is
bounded short serial proof, not a re-execution of the long independent references.

Revised MPI source is frozen at SHA-256
`5846bc61ed12bca23fa4adc556b413b063c28f71d9274e1e577df1d5222d2c60`.
Its complete old-to-new QP policy diff was read. Fresh snapshot, baseline bff plus
only that overlay: `/tmp/mvmc-mpi-callback-5846bc61-bff`. Explicit pre-launch source
proof is retained at `/tmp/mvmc-mpi-callback-5846bc61-source-pre.log`; dedicated
target `mpi-callback-5846bc61-bff`. Fresh MPI2 and MPI4 each completed with exit 0,
one test passed on every rank, 0 failed/ignored. Both widths and all success,
callback-error/absent-peer, and genuine filesystem-error cases ran. Fresh focused
container clippy also completed with exit 0. Full new logs, metadata and source
diff were read; historical logs above are not assigned to these revised hashes.

Pre/post source-proof logs are byte-identical (same SHA-256 below) and report
baseline bff plus only the revised MPI test overlay. Binary SHA-256 is
`741d83e1f56777e8cd06c484e7c9b9a87ffc060974abfa36c310e5868d48088f`;
exact executable:
`/home/vscode/.cache/mvmc/target/mpi-callback-5846bc61-bff/test-fast/deps/mpi_physcal_callback_contract-6c9a483627524c9b`.
Fresh Cargo/nextest metadata records one ignored test with binary ID
`mvmc-core::mpi_physcal_callback_contract`. This is one distinct test executed on
2 and 4 ranks, not six distinct test scenarios. Each world executes 16 runner
cases per rank (eight cases per width), and every rank completes the final
collective agreement. Actual environment/linkage is unchanged from the initial
container environment described above and freshly captured below.

| Fresh artifact | SHA-256 |
| --- | --- |
| `/tmp/mvmc-mpi-callback-5846bc61-mpi2.log` | `0580847b37bc87457eab8a2337bdb6bcdbc0053f8b015861e4dceaf641a416d8` |
| `/tmp/mvmc-mpi-callback-5846bc61-mpi4.log` | `da6b94326c25db1e592d59685786d042dce0f63935268dd7d56607d193d2ac36` |
| `/tmp/mvmc-mpi-callback-5846bc61-source-pre.log` | `36b8396f014c653e99284acaf758d14fbb2a7cfbf8ce9ae5a63bbbbccaf788af` |
| `/tmp/mvmc-mpi-callback-5846bc61-source-post.log` | `36b8396f014c653e99284acaf758d14fbb2a7cfbf8ce9ae5a63bbbbccaf788af` |
| `/tmp/mvmc-mpi-callback-5846bc61-build-list.json` | `54a77796b02bf5f6ec955853a6a79a6f65b99205b3f6706ab9e490fe5673534d` |
| `/tmp/mvmc-mpi-callback-5846bc61-clippy.log` | `572bfbc6f9008669d7a533e5a9bf9746eaed9761c2348eee54ee6beb1850937b` |
| `/tmp/mvmc-mpi-callback-5846bc61-environment.log` | `b770fe899fd25ca78c53cf021a7d46f84a9965d991e91c7eb1d6bbf150f8164a` |

Sibling `mpi2.exit`, `mpi4.exit`, `clippy.exit` files each contain `0`.
Retained fresh output roots inside container `73c57e563c61`:
`/tmp/mvmc-mpi-callback-output-2-5846bc61` and
`/tmp/mvmc-mpi-callback-output-4-5846bc61`. Their inventory artifact
`/tmp/mvmc-mpi-callback-5846bc61-output-inventory.txt` has SHA-256
`3d7e5750806867240cbed063c15e9e21aab729c0217c3e2d3cca274547b92697`.
Each root contains sixteen case directories, writer 0 only. Filesystem-error
cases for both widths contain exactly `zvo_out_001.dat`, `zvo_var_001.dat` and
the deliberately blocking directory `zvo_cisajs_001.dat`; no sample 002 output.
Fresh launch reproduction (use a new output path and repeat with `-n 4`):

```sh
docker exec -u vscode -w /tmp/mvmc-mpi-callback-5846bc61-bff 73c57e563c61 \
  env MPI_PHYSCAL_CALLBACK_OUTPUT=/tmp/NEW-exclusive-revised-output \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  timeout 120 /opt/mpich/bin/mpiexec -n 2 \
  /home/vscode/.cache/mvmc/target/mpi-callback-5846bc61-bff/test-fast/deps/mpi_physcal_callback_contract-6c9a483627524c9b \
  --ignored --test-threads=1 --nocapture
```

No exact same-run QP-copy claim is made for filesystem failure or absent-callback
peers. No production API or hook was added. Parent review and current-integration
validation remain distinct from this frozen baseline proof; no issue closure,
commit or push is performed by this task.

## Read-source hashes

Full Julia source, `extern/Julia-mVMC/MVMCOptimizers.jl/src/vmc_phys_cal.jl`:
`6dd6136057f1ed93a9e7d0b33518cd1b1d434b1595e3fb324b307a9bbebe03bf`.

Full serial tests, `physcal_callback.rs`:
`02f37f7e8e276dd4437909134048c82859e5551bf3c4d44719b8b896c36ace76`.

Full independent-reference tests, `physcal_callback_reference.rs`:
`6c6267569280044c1c0dbbc61d8eb2357dea6401c261741974f609e7fb7f1fd9`.

Current exact serial test, `physcal_callback_exact_contract.rs`:
`31093b6cdf7c5023d0f1d4fdb5287351b448b5bcadb996722463e68508444aac`.

Current actual-MPI test, `mpi_physcal_callback_contract.rs`:
`5846bc61ed12bca23fa4adc556b413b063c28f71d9274e1e577df1d5222d2c60`.
