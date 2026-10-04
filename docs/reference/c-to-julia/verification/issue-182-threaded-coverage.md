# Issue #182: threaded coverage checkpoint

This is an implementation and validation checkpoint, not a claim that issue
#182 acceptance is complete. Normal tests use Rust and checked-in reference
fixtures only. No fresh Julia 1.13 reference run or performance claim is recorded.

## Inventory and implementation

Julia reference revision: `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`.
The executable `@threads` inventory has fourteen sites:

| Julia source and lines | Rust counterpart/status |
| --- | --- |
| threading.jl:66,86 | `threading.rs` real/complex copies; independent entries |
| calculate_m_all.jl:440,879 | `pfaffian.rs` complex/real QP workers; FSZ QP workers added |
| vmc_main_cal.jl:1707 | `observables.rs` normal-real transfer ratios; serial ordered energy reduction |
| vmc_main_cal.jl:3406,3423,3462,3481 | complex/real OO and HO independent entries |
| vmc_main_cal.jl:3516,3555 | complex/real stored sample copies and HO |
| vmc_main_cal.jl:3590,3671 | complex/real CG means and diagonals; sample reductions remain serial |
| vmc_main_cal.jl:3630 | complex stored Gram independent columns; ordered scalar sample sums |

Real stored Gram retains the existing BLAS path. Julia workspace thread-local
scratch is also relevant but is not an additional `@threads` loop. Runner copy
helper wiring was coordinated separately: the runner owner wired real-to-complex
copies in `run.rs` FSZ shadows and SR bridges. This agent did not edit that file.
Complex/FSZ transfer loops are not represented by Julia's normal-real
transfer threaded site.

Rust defaults to one worker and threshold 32. Julia inner-loop defaults are 64,
transfer uses 16, and QP fallback depends on worker count. These are intentional
scheduling differences, not altered numerical algorithms. No RNG draws occur in
the new worker closures; each floating reduction retains its scalar order.

## Error order and allocation

Each QP worker returns a result in indexed order. Results are collected into a
`Vec<Result<_, _>>` before a serial ascending-QP error selection. Direct parallel
`Result` collection does not guarantee earliest-error identity: a regression
run reproduced QP 17 instead of QP 1 with four workers.

FSZ publishes no output until every chunk succeeds. Repeated failures at QPs 3
and the final selected QP check lowest-error identity and unchanged complex
outputs/real shadows. Real/complex error identity is tested too; their existing
serial partial-publication contract is not claimed to be transactional.
Boundary tests check the successful serial prefix, assembled-zero failing
inverse, unchanged later planes, and the pre-existing parallel private-buffer
publication boundary. No normal-path rollback behavior was changed. C
`matrix.c` assembles the public inverse before factorization and writes PfM
after its finite check (complex lines 356–384, real child below line 561).
C/OpenMP itself does not promise earliest failure or whole-range rollback;
ascending error identity and FSZ transactionality are explicit Rust contracts.

Worker inverse buffers currently allocate through each chunk's end QP, rather
than only its local length. Their summed capacity is O(workers × QP count), in
addition to FSZ transaction buffers. No allocation improvement or speedup is
claimed. Measured allocation accounting and benchmarks would be required before
making a speed claim; they are not an additional runtime-completion blocker
under this explicit no-performance-claim scope.
For example, a selected 32-QP range starting at 1 with four workers has ends
9/17/25/33: inverse capacity is stride × 84, not stride × 32. At matrix side 4,
stride is 17 including padding. This is source-derived capacity accounting,
not a measured allocator profile.

## Validation

```sh
cargo nextest run --locked -p mvmc-core --cargo-profile test-fast \
  --test threaded_issue182 \
  -E 'test(qp_threshold_workers_scratch_and_lifecycle_match_independent_expectations)' \
  --no-fail-fast --retries 0
```

Handle 68703 completed successfully: one passed, two skipped. The parent test
uses isolated child processes for workers 1/2/4 and sizes 31/32/33, plus forced
serial/parallel threshold cases. Checks include analytic dense 4×4 Pfaffian and
inverse cofactors, historical checked-in Julia QP fixtures, analytic SR/copy
expectations, workspace isolation, pool recovery, and repeated multi-error
selection. Historical fixtures are not newly verified Julia 1.13 references.

Computed values use explicit absolute and relative bounds of 512 × f64 epsilon
for these small kernel checks; discrete outcomes and untouched sentinels are
exact. Worker invariance alone is not an independent numerical oracle.

The opt-in runner test requires explicit selection:

```sh
MVMC_RS_THREADED_182=1 cargo nextest run --locked -p mvmc-core \
  --cargo-profile test-fast --test threaded_issue182 --run-ignored only \
  -E 'test(runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal)' \
  --no-fail-fast --retries 0
```

## Runner evidence and limitations

Handle 62446 passed in 47.505 s, and 30851 passed both tests in 47.648 s.
The latter includes independent one-step canonical real/complex/FSZ fixtures:
`tests/fixtures/ctest_model_prefixes`, whose provenance records Julia 1.13.1,
Linux and actual serial native C FSZ local energy. These are existing checked-in
mixed C/Julia references, not full C executable parity or a new regeneration.
Their saved configurations and 624 RNG words are checked exactly; parameters
and energy use the established canonical-prefix absolute/relative 1e-11 policy.
Provenance SHA-256:
`06540034cb04f0f1d0ce448a85f246992a7fefc131d3940aad85e216fc578a8c`.

The threshold runner matrix is explicitly **synthetic in-memory input coverage**,
not a claim that newly generated Expert definition files were loaded by C/Julia.
It exercises normal real/complex and complex/real FSZ, direct/CG × NStore 0/1,
two optimization steps, 200 samples and warmup 10, at work sizes 31/32/33 and
workers 1/2/4. PhysCal uses four samples and two iterations, preserving valid
TwoBodyGEx descriptors. Full initial/final RNG debug states, saved/burn-in
configurations, counters and optimization statuses are exact across workers.
Numerical state comparisons are tolerance-based, not bitwise. The real-FSZ
variant deliberately declares real mode and strips imaginary orbital values;
it is not independently certified by the complex-FSZ oracle.

FSZ thresholds repeat its original single translation map, split its weight,
retain the AP sign and disabled spin quadrature, and keep `n_qp_trans`, entries
and parameter weights consistent. C distinguishes `NQPTrans` read from the
TransSym header (`readdef.c:509`) from `NMPTrans` read from modpara: its sign
sets AP (`:742`) and its magnitude enters `NQPFix=NSPGaussLeg*NMPTrans` (`:778`).
Only changing the modpara count without corresponding maps would not establish
valid public input coverage. The independent prefix uses original, public-loadable
definitions without these transformations.

The expanded matrix additionally exercises Hubbard real transfer lengths
31/32/33 by splitting its last coefficient into repeated equal entries. The
original Hubbard prefix remains a separate independent fixture comparison.
Handle 32708 passed both tests in 52.696 s, including Hubbard transfer
thresholds and the independent Hubbard prefix. Across workers the runner
compared 1,664 records with maximum observed numerical difference zero;
kernel children compared 128 records under each scheduling configuration,
also with zero observed difference. Assertions remain tolerance-based for
computed values. Focused Pfaffian unit validation handle 91410 passed three
tests; focused rustfmt and owned-source whitespace checks passed.
Six focused transfer-cache/Green/Gram/finalization unit tests also passed
(nextest run `f937259d-604b-4aa9-9d7e-cbefd669ab67`). Cargo check of the integration
test completed successfully (handle 49524). The unrelated third-party deprecated
atomic-method warning remains unchanged.
Default reporting handle 72655 passed the kernel test and skipped both ignored
tests. Explicitly requesting the runner with its selector unset failed as
intended (nextest run `5ab85e81-0690-4b7f-9b8e-7199177e60be`, exit 100), before
starting the numerical matrix. Setting the selector alone does not opt in:
the ignored runner also requires `--run-ignored`.

### Failed draft history

Handle 32865 failed when optimization with `output_dir=None` tried default
working-directory output, reporting permission denied. No chmod or shared
output replacement was performed. Every optimization now passes an explicit
fresh test-owned directory; PhysCal calls `vmc_phys_cal_to_dir`. Retained output
paths are printed for inspection. Production C default output behavior is unchanged.

Handles 40779, 24324 and 72815 then failed at FSZ direct SR step 0. The draft
had replaced disabled FSZ spin quadrature with 31-point spin projection, and
changed its AP marker; its first observed output was zero energy, before SR.
Increasing the sample count did not fix this. That draft was not an independent
oracle. It was corrected to repeated original translation sectors with retained
AP sign and disabled quadrature, without changing algorithms, seeds, solver
thresholds, numerical bounds or independent expectations. A first analytic QP
draft also used a zero-pivot block-diagonal test matrix; dense analytic cofactor
inputs replaced it, with independent expectations preserved analytically.

### Environment and outstanding acceptance

Linux x86_64; Rust 1.99.0 (`b940084d7`, 2026-09-28); test-fast, locked;
loaded `/lib/x86_64-linux-gnu/libopenblas.so.0`, distro pthread OpenBLAS
0.3.26+ds-1ubuntu0.1, LP64. Each child sets OPENBLAS/OMP/MKL/VECLIB threads to 1
and RAYON_NUM_THREADS=1; the explicit inner pool nevertheless verifies workers
1/2/4. Kernel/runner timings are test-harness duration, not performance benchmarks.
Tests use disabled timers; transfer diagnostic timing intentionally retains its
serial path. Benchmark settings and measured allocations/timers remain future
work before any speed claim.

C `matrix.c` SHA-256:
`849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764`;
`readdef.c` SHA-256:
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.

Julia source SHA-256 inventory:

| Source | SHA-256 |
| --- | --- |
| threading.jl | a1a7c13b77611172a831520e4cc36b981361cb96d82630b4fbd9283e84681299 |
| calculate_m_all.jl | 051f04ea7f1956befa73c2377376d27e71a09b0e3a82ce8a6c0bf75b9bd80dbd |
| vmc_main_cal.jl | 16def1b75c5a8b36462884983c18a4422882b62d3bb45543baf699756b8839d0 |
| workspace.jl | 8d918bda0d8c63f07f558fec63e17d2aaaed86f14a46fea84f59e0972be574ee |

No fresh reference generation, per-proposal trace comparison, alternate BLAS or
macOS validation is claimed. MPI/inner-worker interaction is coordinated with
the MPI agent and remains pending here. The public issue remains open until
its remaining acceptance is reviewed; this report does not close it.

### Actual execution observation and explicit artifact wrapper

`threading::start_observation()` binds an opt-in per-run observer to the calling
thread. `finish()` reports actual entered QP, transfer-term and independent
entry jobs, with serial/parallel counts and observed inner-pool worker IDs.
Requested pool capacity is not execution evidence. Counts include attempted
entries that return errors, not only successful jobs. Nested scopes must drop
in reverse order; error and panic exits restore the prior observer. Inactive
item entry performs no observer Arc clone, lock or counter update. No MPI call,
RNG draw or parallel floating-point reduction was added by observation.

Frozen validation handle 85780, nextest run
`e5c3f3ec-834b-44a6-90f0-6c2645823b03`, passed 1 test / 10 skipped in 151.411 s.
The explicit runner compared observation off/on and workers 1/2/4, retaining
exact discrete records and the existing numerical assertions: each of six
comparisons covered 1664 records, with observed maximum numerical difference 0.
Worker 1 recorded 1,300,255 serial QP and 174,822 serial transfer-term entries,
and zero parallel entries. Workers 2 and 4 each recorded 877,794 parallel QP
and 116,839 parallel transfer-term entries. The scheduler happened to use all
requested workers here; tests require only a nonzero distinct count bounded by
requested capacity. This is non-MPI evidence, not MPI protocol validation.
Focused observer/kernel/preflight validation passed 6 tests / 200 skipped;
focused strict Clippy passed after the separately owned SR observer docs repair.

Evidence is retained at `/tmp/mvmc-issue182-observer-evidence-2zopCS`:
`85780-terminal-fragment.txt` explicitly labels the captured terminal fragment
(not a complete initial CLI transcript), `85780-actual-observations.txt` records
actual counts, and `fixture-files.after.sha256` inventories frozen fixture files.
The captured source, fixture and binary archives were rehashed after execution;
their hashes match the original capture:

| Archive | SHA-256 |
| --- | --- |
| source-snapshot.tar | fe53dcea7735bd15f4f07c60ece1f71e59d64abee66052efa758ba72d717dbb1 |
| fixture-snapshot.tar | 803fedcadd2544ae9b027904ff59184442fb2bb9fd3c6e83cfaa9663b7075a2b |
| threaded-tests.tar.zst | 972b1fd0bb53ab325316ffbee25ea4acdf5b48e7d519fafd6aa4e416ce4c9d0b |

The source archive is a scoped capture, not a complete committed repository
revision. File-level fixture hashes before the original run were not separately
captured; the immutable fixture archive hash supplies the before/after evidence.

Run the standalone developer wrapper explicitly:

```sh
bash scripts/verify_threaded_issue182.sh
```

It creates a fresh owned artifact directory, freezes fixtures, archives the
locked `test-fast` binary with default features using a separate target, saves
selection JSON and full run output, and records terminal exit status even on
failure. Metadata, source/fixture before/after manifests, archive hashes and
linkage diagnostics are retained. It requires exactly one selected ignored
runner matrix and validates actual parallel QP/term entries for workers 2/4;
worker 1 must remain serial. It neither executes nor regenerates C/Julia
references. Ordinary Cargo tests do not invoke this wrapper or any oracle.

Optional environment controls are `MVMC_RS_THREADED_TARGET_DIR` (separate build
cache), `MVMC_RS_THREADED_FIXTURE_ROOT` (reference input root), and
`MVMC_RS_THREADED_FILTER` (additional intersected selection). A supplied
`MVMC_RS_THREADED_ARCHIVE` replays a captured binary; metadata explicitly marks
its association with current live sources as unverified. A zero-selection
negative probe passed by failing closed with `NotRun`, exit 1, and terminal
capture (`/tmp/mvmc-threaded-182.oyQbRw`). Shell syntax validation passed.
These records do not assert full shared-workspace validation or close #182.

Final wrapper negatives on the independent source snapshot retained artifacts
`/tmp/mvmc-threaded-182.mFFTwS` (zero selected, exit 1) and
`/tmp/mvmc-threaded-182.NrhOPC` (absent root, `MissingFixture`, exit 1).
The absent-root probe exposed Bash conditional-errexit suppression in an early
draft helper; it now explicitly returns failure if `cd` fails. The wrapper does
not continue hashing the current directory after an invalid fixture-root request.
An earlier positive attempt, handle 6884, completed its numerical test but the
script was edited while Bash was still reading it, producing terminal exit 2.
That attempt is not wrapper success evidence. Subsequent positive validation
uses an independent immutable source checkout and a newly built archive, not
the earlier supplied archive. The cleanup-test immediate closure was replaced
with a named `Result` helper and `?` return boundary; its two focused tests
passed, and later strict all-target Clippy passed (handle 11820).

Final positive wrapper handle 50952 terminated with exit 0. Artifact directory:
`/tmp/mvmc-threaded-182.4M4HTH`; independent source checkout:
`/tmp/mvmc-issue182-wrapper-source.ksngk1/repo`. The wrapper actually built its
locked default-feature test-fast archive in a separate target (4m 51s), then
passed 1 test / 10 skipped in 146.219s, nextest run
`7c2ae797-d717-45cb-a373-b5f0cb7e62c5`. This is a later source capture than the
original 85780 binary; the two archive hashes must not be conflated. Each of six
comparisons again covered 1664 records with observed maximum difference 0;
actual serial/parallel counts matched those listed above.

Exact retained evidence paths, relative to the final artifact directory:

- `run.log`: full stdout/stderr, summary, off/on comparisons and actual entries;
  `actual-execution.txt`: extracted enabled per-worker snapshots. OFF runs do
  not create counters; their result records are compared to enabled runs.
- `terminal.txt`: `exit_status=0`; `metadata.txt`: command, selected=1, source
  HEAD plus dirty-file manifests, requested axes and `archive_origin=built here`.
- `source.before.sha256` / `source.after.sha256`,
  `fixtures.before.sha256` / `fixtures.after.sha256`, and
  `frozen-fixtures.before.sha256` / `frozen-fixtures.after.sha256`: all three
  pairs compared equal after terminal capture.
- `artifacts.sha256`: tests archive
  `0949546cdf371dec3f0edb747f7400253023c67501dec34c88b95dda895fe58a`,
  fixture archive
  `803fedcadd2544ae9b027904ff59184442fb2bb9fd3c6e83cfaa9663b7075a2b`,
  and scoped source archive
  `d60f22359f68b72959d1fc0b1944d4fb5d77bbb721fabc538820630711f6b282`.
- `archived-binary.sha256`: the actual archived integration-test executable,
  `66c9db1aff9d6ecc239db2e1529f02b44fb6bff30fe07d5c967775e02078b59d`
  (computed from the archive member, not the whole compressed archive).

The snapshot's reference submodules were not initialized (metadata's `-`
prefix); its runtime reference fixtures came from the immutable copied fixture
root. No reference executable ran. The initial list-time `linkage.txt` may only
contain a diagnostic because that extraction is temporary. Instead,
`live-binary-linkage.txt` and `live-blas-maps.txt` capture the actual running child
PID 1362136's executable and mapped OpenBLAS library. A separate explicit
`uv run --no-project` identification probe, not a numerical oracle, reports
`OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`
in `openblas-config-probe.txt`. Its thread-count result describes the probe
process, not a measurement of the Rust child's live BLAS thread count.
This evidence remains non-MPI and is not full shared-workspace validation.

## Issue-specific acceptance matrix (2026-10-03 review)

The current public [issue #182](https://github.com/AtelierArith/mvmc-rs/issues/182)
is OPEN; its five acceptance checkboxes remain unchecked. The rows below map
those requirements to actual evidence, not to a global pass/fail label. `E`
means `/tmp/mvmc-threaded-182.4M4HTH`; `S` means the independent checkout
`/tmp/mvmc-issue182-wrapper-source.ksngk1/repo`. S has HEAD
`b521e4ae2093b587d1697fac1d53597c1fe14018` plus captured draft changes; HEAD
alone is not its implementation identity. The scoped source/binary hashes above
and E's before/after manifests identify the tested implementation.

| Requirement | Actual artifact / scope established | Remaining gap or qualification |
| --- | --- | --- |
| AC1: inventory every Julia threaded callsite and classify Rust | Fourteen executable sites at Julia `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`; individual classifications below. Current source hashes match the earlier inventory hashes in this report. | Source audit is not runtime activation proof for every individual site. Observer `Entry` aggregates copy/SR sites rather than reporting fourteen named branch counters. |
| AC2: implement important missing paths or document scope differences | S's `threading.rs`, `pfaffian.rs`, `observables.rs` implement disjoint copies/QPs/transfer/SR entries; runner wiring is in captured `run.rs`. E/source.tar and source manifests preserve their tested versions. | Imported runner/MPI work is separately owned; this checkpoint does not certify subsequently edited shared sources. Threshold differences and backend/serial exceptions are intentional, not numerical-algorithm changes. |
| AC3: workers 1/2/4 and threshold boundaries | E/run.log, terminal.txt, actual-execution.txt: 1 pass / 10 skip, six 1664-record comparisons. E/kernel-gates.log and kernel-gates.terminal.txt: 9 pass / 2 skip, exit 0, 0.728s, run `07800467-0e78-4418-a2e9-f9d5e6615c86`; kernel tests cover 31/32/33 and overrides 1/34. | Finite Linux cases only. QP size 31 correctly has zero parallel QP entries even when larger complex SR entry ranges activate workers; do not equate every entry dimension with QP count. |
| AC3: real/complex/FSZ, direct/CG, NStore, PhysCal, configurations/RNG/results | E/run.log covers normal real/cmp, complex FSZ and synthetic real-FSZ, all four direct/CG × NStore 0/1 optimization combinations, plus PhysCal. Exact discrete/RNG records and existing numerical bounds are checked across workers and observation OFF/ON. | Threshold inputs are synthetic in-memory transformations, not independently loaded C/Julia Expert-file fixtures. Real-FSZ is not independently certified by the complex-FSZ oracle. PhysCal is a worker-invariance check here, not an independent Green/output oracle. |
| AC3: independent numerical/discrete reference evidence | E/fixtures/tests/fixtures/ctest_model_prefixes/provenance.txt and four model step-1 folders. `independent_runner_prefixes` checks original public-loadable inputs, configurations/burn-in/counters, next624, parameters and energy. Provenance identifies mixed Julia 1.13.1 + native C FSZ kernels. | The entire 1664-record stream is **not** independently referenced. This helper does not assert against `sr_oo.txt` or `sr_ho.txt` despite requiring their presence in preflight. No independent full CG/NStore/threshold/PhysCal matrix, full C executable parity, or new reference regeneration is established. Analytic SR kernel expectations cover a smaller independent scope. |
| AC4: scratch isolation / error and lifecycle boundaries | E/kernel-gates.log covers private workspace isolation, panic/pool recovery, indexed earliest-error selection, FSZ rollback, and normal real/cmp serial partial-publication contracts. Earlier focused cleanup run `4d35dfb9-be0a-49ee-a24d-c9fec6ff25f9` passed two TLS isolation/error/panic tests. | No exhaustive race detector or arbitrary workload claim. Normal-path serial failure publication is intentionally not whole-range rollback; FSZ transactionality is an explicit Rust contract, not a claim about C/OpenMP failure identity. |
| AC4: deterministic reduction order | S source audit: independent output entries/QP jobs; per-entry sample sums and final transfer energy accumulation remain serial. E/run.log verifies exact discrete results and bounded computed values for the finite matrix. | No bitwise floating-result requirement or claim. Alternate BLAS and MPI reduction trees are not validated by this non-MPI run. |
| AC4: BLAS settings and backend identity | Child launch explicitly sets OPENBLAS/OMP/MKL/VECLIB threads to 1 and global RAYON_NUM_THREADS=1; actual inner workers are observed separately. E/live-binary-linkage.txt, live-blas-maps.txt, openblas-config-probe.txt identify the mapped pthread OpenBLAS 0.3.26 library/config. | The separate probe's thread count is **not** the Rust child's measured live BLAS thread count. No alternate-provider or oversubscription sweep; list-time ldd failure is not provider evidence. |
| AC4: MPI interaction | This wrapper is explicitly default-feature/non-MPI, world 1. Separate issue-179 evidence and harness are owned by the MPI agent; older pool-capacity probes do not prove actual kernel activation. | Pending separately reviewed terminal artifact connecting genuine multi-rank worlds, per-rank actual QP/term entries, collective ordering/FUNNELED safety and exact reference/discrete comparisons. No MPI success is claimed by E. |
| AC5: allocations, timers and benchmark settings before speed claims | Source-derived inverse capacity formula and one padded 32-QP example are recorded above. E/metadata.txt records profile, architecture, compiler, axes and command. | No measured allocation profile, production timer study, warmed benchmark repetitions or speedup. Test duration is not a benchmark. No speed claim is made. |
| Platform policy: native macOS portability / numerical validation | E is Linux x86_64 only; historical Linux fixture provenance is explicit. | **Native macOS NOT RUN.** Neither correctness/worker activation nor justified numerical portability on native macOS is established by a Linux container or these artifacts. |
| Integration / milestone acceptance | Frozen source and fixture manifests compared equal; explicit missing selection and fixture negatives fail closed. | Not a full-workspace check on final integrated main. Production drafts and acceptance owned by other agents remain outside this artifact's guarantee. Issue #182 should stay open. |

### Per-callsite classification

Paths below are relative to Julia `MVMCOptimizers.jl/src`; there are fourteen
executable `@threads` sites, excluding comment/docstring mentions. No additional
`@spawn` or `Threads.foreach` site was found in this audited Julia tree.
`Equivalent` here means the disjoint-work scheduling counterpart exists, not
bitwise floating equality or a freshly executed Julia threaded reference.

| Julia site | Rust counterpart | Classification / intentional difference |
| --- | --- | --- |
| threading.jl:66 | `threading::copy_real_to_complex` | Equivalent independent prefix entries; destination tail preserved. |
| threading.jl:86 | `threading::copy_complex_realpart` | Equivalent independent prefix entries; destination tail preserved. |
| calculate_m_all.jl:440 | `pfaffian::calc_m_all_complex_with_kernel` | Equivalent normal complex QP jobs; additional FSZ QP path is implemented separately. |
| calculate_m_all.jl:879 | `pfaffian::calc_m_all_real` | Equivalent normal real QP jobs; scalar child arithmetic preserved. |
| vmc_main_cal.jl:1707 | normal-real Transfer branch of `calculate_local_energy_timed` | Equivalent independent Green jobs, followed by serial ordered accumulation; diagnostic timer path remains serial. |
| vmc_main_cal.jl:3406 | `observables::calculate_oo` first-row / HO update | Equivalent disjoint entries. |
| vmc_main_cal.jl:3423 | `observables::calculate_oo` remaining rows | Equivalent disjoint rows; inner arithmetic remains serial. |
| vmc_main_cal.jl:3462 | `observables::calculate_oo_real` OO update | Equivalent disjoint matrix entries, partitioned by Rust columns rather than Julia rows. C layout and entry operation order remain numerical authority. |
| vmc_main_cal.jl:3481 | `observables::calculate_oo_real` HO update | Equivalent disjoint entries. |
| vmc_main_cal.jl:3516 | `observables::calculate_oo_store` | Equivalent complex sample copy + HO entries. |
| vmc_main_cal.jl:3555 | `observables::calculate_oo_store_real` | Equivalent real sample copy + HO entries. |
| vmc_main_cal.jl:3590 | `observables::finalize_oo_store`, diagonal-only | Equivalent complex CG means/diagonals; ordered sample sums. |
| vmc_main_cal.jl:3630 | `observables::sr_store_gram_julia` | Equivalent complex Gram output entries; ordered sample sums. |
| vmc_main_cal.jl:3671 | `observables::finalize_oo_store_real`, diagonal-only | Equivalent real CG means/diagonals; ordered sample sums. |

Real full stored Gram is backend-specific BLAS SYRK/generic-small-matrix work,
not another Julia `@threads` site. Complex/FSZ transfer stays serial where Julia
does not provide the audited normal-real threaded path. MPI collectives must
remain outside Rayon worker closures. InterAll and its Lanczos implementation
are explicitly excluded by issue #182 and must not be counted as missing work
in this matrix.

### Remaining work proposal (no production changes made by this audit)

1. Obtain and review the MPI owner's actual-entry sidecars plus terminal
   rank/group/worker/reference evidence; distinguish supported/rejected cells
   and discrete-only diagnostics from numerical parity. Do not replace this
   with configured capacity or a singleton-world launch.
2. Review whether the finite independent-prefix/analytic reference scope is
   sufficient for acceptance, or commission independently generated fixtures
   for uncovered CG/NStore, real-FSZ and PhysCal numerical intermediates.
   Public-loadable threshold definitions and a fresh Julia 1.13.1 threaded
   reference are **not** supplied by the synthetic matrix. Generation stays an
   explicit developer task; normal Rust tests must remain oracle-independent.
3. Run native macOS portability/worker checks with recorded platform/BLAS and
   existing justified numerical bounds. Resolve the first numerical divergence
   rather than relaxing discrete/RNG contracts.
4. Retain the explicit no-performance-claim scope. If a subsequent change makes
   a speed claim, first measure allocations, production timers and warmed
   benchmark settings/results. This conditional requirement does not block
   runtime acceptance without speed claims; the current buffer capacity formula
   is not a measured allocation result.
5. After active owners finish, integrate and validate the final milestone's
   exact source, full Rust workspace, docs, fmt and strict Clippy. This audit
   changes only this owned report and artifact evidence, not production files.

## Current parent focused checkpoint (2026-10-03, mutable worktree)

Parent reported nextest run `a1df4260-9c81-4ec0-9219-b081b085c73d`, terminal
exit 0: **2 passed, 11 excluded, 1.460s**. The independent PhysCal test took
1.320s and the full normalized pre-SR OO/HO test took 1.459s. This is a focused
parent-run result, not a frozen snapshot or full-workspace verification.

The OO/HO test covers the existing five prefixes: Heisenberg real, complex and
complex FSZ, Hubbard real, and general RBM complex CG. It checks full reference
array shape/order at the normalized pre-solver stage with the existing 1e-12
numerical policy, alongside saved configurations and RNG checks. PhysCal covers
the existing four Heisenberg/Hubbard models, two output frames and workers
1/2/4, retaining Hubbard's supported mode-2 Lanczos input. Parsed output rows
ignore blank lines/whitespace: this is not byte-format proof. Only the final
sample-1 saved configuration is independently checked, not both sample
configurations or every proposal. **Neither test includes the new real-FSZ
oracle.** These newer tests are not part of the older frozen wrapper evidence.

Hashes read after the parent report, while other owners could still edit the
worktree (no before/after stability claim):

| Item | SHA-256 / identity |
| --- | --- |
| Current Git HEAD, not dirty-source identity | `66e496fdbe5947a396a30be3b25ee07fc9a8c370` |
| `crates/mvmc-core/tests/threaded_issue182.rs` | `77433f3952701e51d9bd8ef022a05e9a39a7e6e610a5cadce6d917ecb293cdc3` |
| Five prefix `step-1` path/content manifest digest | `908799a1e1061161f2eb312d0fad05abc038cd10e9abc1acd91383c6b7ac5fa1` |
| Four PhysCal `two-samples` tree path/content manifest digest | `3bddde5d73d0f2653037b69031fe58c4363ac7f7b8ae93796b974d4c32d06be0` |

Manifest digests hash the UTF-8 output of `sha256sum` for files listed by
`rg --files --hidden`, sorted using `LC_ALL=C`, with repository-relative paths.
Prefix trees are `tests/fixtures/ctest_model_prefixes/{heisenberg_chain_real,
heisenberg_chain_cmp,heisenberg_chain_fsz,hubbard_chain_real,general_rbm_cmp_cg}/step-1`;
PhysCal trees are `tests/fixtures/physcal_181/two-samples/{heisenberg_chain_real,
heisenberg_chain_cmp,heisenberg_chain_fsz,hubbard_chain_real}`. These are scoped
fixture-tree digests, not a manifest of all input sources or linked libraries.

Reproduce the two focused tests, using current repository fixtures:

```sh
MVMC_RS_THREADED_182=1 cargo nextest run --locked -p mvmc-core \
  --cargo-profile test-fast --test threaded_issue182 --run-ignored only \
  -E 'test(independent_runner_prefixes_match_full_normalized_pre_sr_arrays) | test(independent_physcal_workers_match_saved_rng_and_ordered_outputs)' \
  --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Parent also reported `bf71c505-8e9e-4421-bdfd-f6a3799324da`, terminal 0:
the QP-threshold focused test passed in 0.727s (12 filtered). Parent's fmt and
diff checks passed; workspace/all-target Clippy handle 51694 exited 0 in 2.26s
with the unchanged nonfatal vendor warning. These are reported mutable-tree
checks, not newly frozen source/fixture evidence.

After Pauli's active OO-weight API change, parent reported handle 46156,
nextest run `998a9b7d-a030-4c79-8b2d-406eeef6256e`, terminal 0: the existing
five-model independent full normalized pre-SR OO/HO prefix test passed in
1.288s (12 filtered). This is a matching focused checkpoint on the mutable
worktree, not frozen-source or full-workspace evidence. The existing real
small-Wc guard remains an explicit Rust policy; its C numerical-contract audit
is pending with Pauli and the pass must not certify that policy as C parity.
No new real-FSZ oracle is included.

### Real-FSZ C shadow lifecycle investigation (not full oracle acceptance)

The suspected missing PfM synchronization was a **superseded investigation,
not a C bug**. `setmemory.c:359–360,379–380` allocates each inverse and PfM
contiguously; PfM aliases its allocation's tail. `vmccal_fsz.c:82` copies the
full `NQPFull*(Nsize*Nsize+1)` allocation, including PfM. Pauli independently
confirmed this. Julia's separate vectors require copying both real inverse
and real PfM into their complex shadows before the original differential.

Developer-only `c_toolbox/check_threaded_182_real_shadow.jl` guards the complete
upstream source hashes and extracts the allocation/alias statements and copy
loop verbatim. It compiled with `cc -std=c11 -O0 -ffp-contract=off -Wall -Wextra
-Werror`, then passed 16 patterned-input cases (QP counts 1–4, matrix sides
2/4/6/8), terminal 0, owner handle 29386. This proves that memory-layout/copy
boundary only: it does not invoke the C matrix factorizer, sampler or
differential, and is not full Mat/IP/energy/OO/HO/RNG parity.

```sh
julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/check_threaded_182_real_shadow.jl /tmp/NEW-real-shadow-artifacts
```

The destination must be fresh and outside the repository. Current artifacts:
`/tmp/mvmc-threaded182-real-shadow-20261003-a/{probe.c,probe,stdout.txt,provenance.txt}`.
Probe-script SHA-256:
`238673fc3004c61c4bc79a9af50de7beaa21bac99c8b71d24d5750492fb32724`;
provenance SHA-256:
`bac55ca8a6af7354da2731a8a05c3db74a39222cd3599bcfbb8246171f318baf`.
The provenance includes upstream hashes, exact compiler command/version and
generated source/binary hashes. No shared adapter or canonical fixture changed.

License/origin follow-up: the revised generator retains both upstream
copyright/GPL-3.0-or-later notices verbatim in generated `probe.c` and its
provenance, names the original files and author, resolves the existing parent
directory before repository exclusion, and uses exclusive `mkdir`. Handle
22623 passed the same 16 cases, terminal 0, in fresh
`/tmp/mvmc-threaded182-real-shadow-20261003-license-b`. Its script SHA-256 is
`e24987bf36f18474931a51916a25e58dffeaa6c7e4414826e60832c5ff51a74d`;
generated C SHA-256 is
`cfe02eadb8d0e6fd0abe1c756b940bb03e8b7c8d2c4eb3b21bcf161795639759`;
provenance SHA-256 is
`ab67b00a5c381a1797f0be9c83d5221585e49b9f332d7763518f1dd403f83979`.
The original `...-a` artifact and hashes above remain unchanged historical
evidence, not hashes of the revised generator. Reusing that existing artifact
directory was rejected before writes (handle 28618, terminal 1).

The fresh native energy bridge separately passed 36 real plus 72 complex ABI
cases with unchanged borrowed operands; this remains energy-ABI scope only.
The new real-FSZ full-stage oracle and fixture test remain pending. Upstream
Julia's real FSZ matrix wrapper computes complex matrices then copies real
parts, while its measurement driver selects complex SR unconditionally. A
source-guarded C-compatible real adaptation must therefore be labelled as an
adaptation, not unmodified Julia 1.13.1 parity. Fixed expectations cannot be
accepted merely by stripping imaginary components from a complex oracle.

### Bounded actual C-real Mat stage (2026-10-03)

Owner handle 86728 finished terminal 0: developer-only
`c_toolbox/check_threaded_182_real_mat.jl` passed twelve fixed-operand cases,
matrix sides 2/4/6/8 with three QP planes each. It guards the complete C
`matrix.c` hash and extracts `calculateMAll_child_fsz_real` verbatim, retaining
the GPL notice/origin. It links actual upstream real Fortran DSKTRF and C++
`utu2pfa_d`/`utu2inv_d`, plus OpenBLAS DSCAL. Julia constructs the identical
real matrix and invokes real PfaPack factorization/inversion—not a complex
oracle followed by real-part stripping.

Existing fixed-matrix absolute+relative 1e-13 comparisons passed. Largest
observed Pfaffian difference was 3.997e-15; largest inverse difference was
4.974e-14 at side 8/QP 1, whose condition number was about 184.3 and inverse
residual was 3.286e-14. This is bounded computed-value comparison, not bitwise
parity. Residuals and conditioning are diagnostics recorded for every case;
the script does not impose a separate residual acceptance bound. The upstream
`diag2char` missing-return warning was nonfatal; no vendored source changed.

```sh
julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/check_threaded_182_real_mat.jl /tmp/NEW-real-mat-artifacts
```

Current artifacts: `/tmp/mvmc-threaded182-real-mat-20261003-a/`, including
`probe.c`, `libreal_mat.so`, `stage.log`, `provenance.txt` and build objects.
Script SHA-256:
`26f4876c3e6beeb737a29d808a5ef94d4a0e5106a36ac0b9a67e74e920c18227`;
library SHA-256:
`3da50d90d026b922a32f6dca1d090a6e64cbd254009b44bc73a1d49e601eb56d`;
provenance SHA-256:
`2f1bc8520e65109310bb6f2be67a8c0161e1b5db1d05204e5ca9a550ef269f5c`.
Provenance records build commands and local upstream/template hashes. Those
transitive hashes were sampled after compilation and verified unchanged across
the numerical probe, not measured as a full build-before/build-after manifest.

This is synthetic kernel-stage evidence only: no public real-FSZ model input,
initialization/QP-weight contract, IP/differential/energy call chain, RNG draws,
saved configurations, normalized OO/HO, SR solve or worker-activation proof is
provided by these twelve cases. Full fixture generation/adaptation remains
pending; neither this nor the sixteen shadow-layout cases closes that gap.

The next bounded checkpoint extended the same optional probe with the
source-guarded original C `CalculateIP_real` and unmodified Julia
`calculate_ip_real`, using the same C-real Pfaffians and fixed real weights
`[1, 0.5, -0.25]`. Handle 29019 exited 0: twelve real matrix cases plus four
real-IP cases passed, with observed IP differences zero. The size-one MPI
stub rejects any collective by aborting: this is not MPI protocol evidence.
Matrix borrowed Slater/index/spin operands were also checked unchanged.
These remain synthetic kernel operands, not public-model QP-weight fixtures.

Fresh artifacts: `/tmp/mvmc-threaded182-real-mat-ip-20261003-b/`.
Revised script SHA-256:
`3749931cbbd75f3742dc655b1ae731dfea98e28cddaa04f023d6d8426499f44e`;
provenance SHA-256:
`a5ee772aeac524cea8b3f5702fbc845db420cc53b3a08ba56db36ccf4890d7ea`;
stage-log SHA-256:
`89d0b8f1c341d7b14b8053281219b2dd810c765bb187d118730d0894a23c1078`.
The earlier Mat-only artifact and script hash above are retained historical
lineage, not replaced by these revised-source hashes. Differential/even-O,
normalized OO/HO, SR and exact full-run RNG/configuration proof remain pending.

Differential/even-O bounded extension: handle 30833 exited 0 using fresh
`/tmp/mvmc-threaded182-real-mat-ip-diff-20261003-d/`. It retained the twelve
matrix checks and four real-IP checks, then compared the original, source-guarded
C `SlaterElmDiff_fsz` against unmodified Julia `slater_elm_diff_fsz!` at the same
C-real inverse/PfM operands. C's complete inverse-plus-PfM copy loop populated
the aliased complex shadow; Julia's separate inverse and PfM vectors were both
explicitly synchronized. Four differential/even-O cases checked full 38/134/
294/518 complex slots (including explicit constant/projection plumbing), then
the original C `creal(srOptO[2*i])` boundary against Julia's even-index real
selection. Maximum observed difference was 5.552e-17 (side 4); the other three
cases had zero observed difference, within the same 1e-13 stage comparison.
This does not assert bitwise floating equality or locate every intermediate
rounding divergence. Real matrix operands were checked unchanged by C.

These are still synthetic identity/repeated-translation kernel tables, not
public-loadable model coverage or calibrated full sampling trajectories.
No complex-run expected result was stripped to create a real fixture: the
complex differential/shadow and even-O conversion are themselves the defined
C real-driver path. Full normalized OO/HO/SR/public-input/RNG remains pending.

The first differential draft (handle 65647) failed C compilation because its
extractor matched a forward declaration and included `UpdateSlaterElm_fsz`;
this was a harness extraction error, not a numerical/model failure. It is
retained at `...-20261003-c`. The corrected extractor requires the unique
complete function-definition signature before extracting its body. No live
process was cancelled/restarted or previous artifact overwritten.

Revised script SHA-256:
`e320af562117e50ca4bc56a3849cc024056dac97463cdb57934d50a385e237da`;
successful provenance SHA-256:
`c412c3425b7020d3d893df7d3493c19dd9d51f17d037291e151138a4cef530a2`.
Earlier Mat/IP artifacts and source hashes remain separate historical lineage.

### Real-FSZ flag guard: current discrete failure

The staged public-derived reference at
`/tmp/mvmc-threaded182-real-public-20261003-f` is not an adopted fixture.
Parent run `a175bf4c-5cc4-4af7-9cec-b94538970332` passed the then-checked
OO/HO, parameters, RNG and saved-configuration fields, before flag assertions.
It is not a current all-contract pass.

Exact initialized/pre-SR/final Rust flag assertions now compare against the
independent initialized flag file; that file does not separately capture the
reference's later-stage flags. Run
`57cdd547-5149-4c2e-ac6e-0a206f2d57ba` exited 100 at initialized flags;
parent independently reproduced this in
`61931e36-a103-437e-873d-8da364ef3c3c`. Rust has real-slot flags
`[0,0,0,0,1,0,1,0,...]`, whereas this Julia reference has
`[0,0,0,0,1,1,1,1,...]`. This is a discrete layout/default distinction,
not numerical error. The new initialized QP-weight comparison uses the
existing `1e-12` bounds, but is not reached in these failed runs.

Julia's `utils/read_input_parameters.jl:395` initializes newly allocated
flags to true; `set_orbital_opt_flags!` writes real slots and writes imaginary
slots only for complex mode. C `readdef.c:2101` (`GetInfoOpt`) likewise only
writes imaginary slots when `iComplxFlag > 0`. However, C's backing region
is allocated with `malloc` (`setmemory.c:43`) and `OptFlag` points into that
region (`setmemory.c:227`); this investigation has not established a
zero-initialization of its unwritten entries. Native C imaginary zeros must
not be inferred from the parser. Rust explicitly defines unwritten
real-mode imaginary entries as zero.

Next reference work must distinguish the parser's written-slot mask, exact
active real flags, raw Julia defaults and explicitly defined inactive-slot
policy. A developer adapter's explicit zeros are not evidence of native C
allocator contents. Real CG uses `OptFlag[2*pi]` in
`stcopt_cg_impl.c:140`; this alone does not establish the direct solver's
full storage contract. Direct-solver authority review remains pending.
The original failed artifact and full exact assertion are preserved; no
fixture has been rewritten, tolerance widened or production parser changed.

Fresh three-way stage `...-20261003-g` (generator handle 48536, exit 0)
preserves those raw Julia defaults and separately emits `julia-raw-flags.txt`,
`c-written-mask.txt`, and `defined-flags.txt` at initialized/pre-SR/final.
Its extracted, hash-guarded original C readers run twice with different
sentinels against the public input's actual flag rows. Equal results identify
written slots; unequal sentinel values identify unwritten slots, not native
allocator values. `GetInfoOptOrbitalParalell` explicitly writes imaginary
zero, unlike real `GetInfoOpt`. The mask reflects this distinction.
Unwritten imaginary zeros in `defined-flags.txt` are the explicit defined
inactive-axis policy, not a claim about native C raw storage.

Rust checks all active real slots are C-written and match both independent
C values and raw Julia real slots exactly; all C-written values match
exactly, and inactive slots must be zero under the explicit policy. Root
`optimization-flags.txt` is also read and compared to the initialized raw
diagnostic. Initialized full QP weights use the existing `1e-12` policy.
No Julia flag is changed by the generator, so the retained original Julia
direct solver still receives its raw defaults. Independent C direct-SR
active-selection validation remains pending with the C-reference owner.

Generator SHA-256 at this stage:
`2a7ab0481d4c09c8f79bf942f34668f9d9dbab9b29c101849839a4f34c551a32`;
provenance:
`0534c28656db75506e6ace755c07fe6ba5aa7fe49926756a5474e7169fc18e4c`.
This stage is review evidence, not adopted fixtures or complete issue #182
acceptance. Comprehensive original loaded numerical-source identities and
direct-SR authority remain separate pending requirements.

Final focused Rust run `5d3fd08d-891f-4b27-bdb9-81cc0dc928de` (handle
62968) exited 0: 1 passed, 13 excluded, 0.425 s, workers 1/2/4. This now
reaches the initialized QP-weight and all three flag-boundary assertions.
Actual QP execution remains serial for this public one-QP input; threaded
entry work is observed separately. Test source SHA-256:
`fb1c00b62b4b5d99c22fe447a3ea62441af624a803c166130d68e0ac598073dc`.
Full output: `...-20261003-g/rust-workers-final.log`, SHA-256
`57a6803ef45b66dc3b225ced8fc830a1df5fb9114369c47b8a46c88c8b9a6cf3`.
Focused strict clippy, owned-file rustfmt check and `git diff --check` passed
after the final test-source edit. The unchanged dependency deprecation
warning is not a new owned-source lint. No whole-workspace or complete
issue acceptance is claimed by this focused result.

Reproduction (ordinary Rust test reads artifacts only, never invokes an oracle):

```sh
MVMC_RS_THREADED_182=1 \
MVMC_RS_THREADED_REAL_FSZ_ROOT=/tmp/mvmc-threaded182-real-public-20261003-g \
cargo nextest run -p mvmc-core --test threaded_issue182 \
  --cargo-profile test-fast --locked --run-ignored only \
  -E 'test(independent_real_fsz_workers_match_public_pre_sr_and_rng)' \
  --no-fail-fast --retries 0 --success-output immediate
```

### Scalar mathematical-function policy clarification

The user's reviewed initial `cexp`/`sincos` first-difference scope
(`8.7e-19`, one-ULP mathematical-function discrepancy) is acceptable without
bitwise equality. It is not evidence of a newly measured first divergence
in this real-FSZ stage. The initialized parameter and full QP-weight checks
now use explicit absolute and relative bounds of `8 * f64::EPSILON`
(approximately `1.78e-15` each), a small scalar-rounding budget, tightened
from the previous `1e-12` bounds. These initialization operations do not
involve an SR solve or MPI reduction. No downstream OO/HO/parameter bound
is increased, and no CG forward-error allowance is inferred from this
mathematical-function provenance. Solver conditioning/residual evidence
and MPI reduction effects require their own analysis.

Flags, RNG state, draw counts and saved configurations remain exact. This
clarification does not authorize changing algorithms, operation order,
signs, inputs, proposals or acceptance decisions. Independent C direct-SR
authority and comprehensive reference identity requirements remain pending,
but a scoped scalar-function discrepancy alone is not an adoption blocker.

### Final reproducibility clarification and scoped repeats

The user's final clarification withdraws the intervening broad
cross-language RNG exemption. The fixed RNG algorithm requires exact
primitive initialization, conversions, outputs and raw state across
languages for the same draw order/count. Same-input/fixed-seed repeatability
within the same implementation and configuration remains required.
Mathematical-function rounding is allowed under justified numeric bounds;
if it changes a proposal/acceptance branch or subsequent trajectory, the
first numerical cause and decision threshold must be identified. Neither
Monte Carlo noise nor an assumed RNG defect explains such drift. Existing
strict invariants were not deleted or weakened during these clarifications.

New normal test `same_worker_configuration_repeats_kernel_records` launches
two fresh processes for each worker configuration 1/2/4, threshold 32,
kernel sizes 31/32/33, with BLAS/OMP/MKL/VECLIB/Rayon ambient thread counts
fixed to one. Numeric records use the existing `512 * EPSILON` bounds;
discrete records remain exact. The opt-in real-FSZ test likewise repeats
each worker configuration in fresh processes with the same input/seed and
existing numeric/discrete checks. Observer worker IDs are scheduler
evidence, not values required to repeat identically. These tests are
single-process checks, not MPI repeatability proof.

Run `ae4d44e4-b446-498b-90ba-ad60c943c811` (handle 97681) exited 0:
2 passed, 13 excluded, 0.711 s. Full output is
`/tmp/mvmc-threaded182-real-public-20261003-g/rust-same-config-repeatability.log`.
Canonical reviewed CG-reference migration is coordinated with its owner;
the previous old-CG golden failure is not hidden or repaired by relaxing a
forward tolerance. C direct-SR authority for real-FSZ remains pending and
does not block the other models' scoped repeatability validation.

The independent PhysCal test now also repeats each unchanged worker
configuration 1/2/4 in fresh processes for all four existing models:
Heisenberg real/complex/complex-FSZ and Hubbard real (actual mode 2,
supported non-InterAll LS). Existing ordered output, saved configuration
and RNG assertions are preserved; numeric repeats are bounded, not a
computed-bitwise claim. Run `6d54013a-aceb-450e-8f94-7af6da03fae5`
(handle 72714) exited 0: 1 passed, 14 excluded, 2.514 s. Full output:
`...-20261003-g/rust-physcal-same-config-repeatability.log`.
This strengthens scenario repeatability but does not by itself establish
actual activation of every inventoried Julia threading callsite.

### Reviewed 62b CG migration: two independent serialization schemas

The approved candidate is
`tests/fixtures/reviewed_cg_62b/canonical_general_rbm`. Its
`step-1-parameters.txt` is a hexadecimal mapped-term serialization (966
complex values, including spatial mapping repetitions), not declared NPara.
The original observer concatenates G/J terms, DH arrays, nine mapped RBM
sections, orbital terms and OptTrans. The independent
`step-1/c-window-input.txt` instead captures the C-declared pack: header
window 1, NPara 102, fifteen declared section widths; one decimal row contains
four E/E2 scalars and 204 parameter scalars. Rust validates this complete
header/row layout and compares both the dense and mapped schemas separately.
Neither serialization is renamed, truncated or silently substituted.

Only this CG case's final status/configurations/RNG/energy/parameters are
redirected, fail-closed, to the reviewed candidate. Hexadecimal storage uses
the existing decoder and is not a bitwise numerical comparison requirement.
The normalized pre-SR full OO/HO arrays remain in the independent canonical
prefix fixture. Its complete input/initial-overlay SHA manifest must match
the reviewed provenance file for every input, with unchanged seed/settings;
declared layout and full array dimensions are asserted. Metadata pins Julia
1.13.1, fork `62b0f97f076fb55c71c3ab0caa041a9adff94e04`, C-faithful
CG recurrence and production `stochastic_opt.jl` SHA-256
`b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3`.
This is explicitly a mixed two-stage reference, not a fresh full-C runner.
OO/HO bounds remain `1e-12`; final numeric bounds remain `1e-11`.

Owner run `d27aabce-4c22-4e9d-8af9-583130dec69d` exited 0 (1.255 s),
all five prefix models across workers 1/2/4, CG dense 102/mapped 966 both
checked. Parent independently passed
`c1985cde-d3c8-4293-8493-05bb3f4a812b` (1.258 s).
Earlier draft failures are retained under `...-20261003-g`: initial status
schema `0 -1` versus old `0`; decimal parsing of hexadecimal energy;
and using the declared-coefficient read-only RBM visitor for a mapped-term
schema. These were test-adapter defects, not numerical tolerance failures.
The corrected mapped helper visits the original nine term sections on a
clone, following the already reviewed canonical consumer's serialization.

### Actual callsite activation checkpoint

Nested per-call observers now check exact actual QP/entry/term work counts,
serial versus parallel classification, and worker IDs bounded by requested
capacity. Numeric records with observation enabled/disabled still match under
the existing bounds. No production kernel or reduction order was changed.

| Julia site | Scoped actual observation |
| --- | --- |
| threading.jl:66 | real-to-complex entry jobs |
| threading.jl:86 | complex-to-real entry jobs |
| calculate_m_all.jl:440 | normal complex QP jobs |
| calculate_m_all.jl:879 | normal real QP jobs |
| vmc_main_cal.jl:1707 | normal real transfer jobs |
| vmc_main_cal.jl:3406 | complex first-row/HO entry jobs |
| vmc_main_cal.jl:3423 | complex remaining-row jobs |
| vmc_main_cal.jl:3462 | real OO column jobs |
| vmc_main_cal.jl:3481 | real HO entry jobs |
| vmc_main_cal.jl:3516 | complex store/HO paired jobs |
| vmc_main_cal.jl:3555 | real store/HO paired jobs |
| vmc_main_cal.jl:3590 | complex CG mean/diagonal jobs |
| vmc_main_cal.jl:3630 | complex full Gram column jobs |
| vmc_main_cal.jl:3671 | real CG mean/diagonal jobs |

Combined OO/HO kernels assert the sum of both original loop lengths and
the resulting serial/parallel job partition. Sizes 31/32/33 exercise the
threshold directly. Complex work lengths are even: added parameter sizes
15/16/17 exercise components 30/32/34 and remaining rows 28/30/32,
including the mixed partition at parameter size 16. Real full Gram is a
BLAS backend path, deliberately observed as no Rust entry jobs, not another
Julia `@threads` site. The transfer test splits the final coefficient into
repeated equal terms retaining the Hamiltonian, uses four saved samples,
and asserts exactly `4 * term_count` jobs. This is explicitly synthetic
kernel-threshold coverage, not a newly public-loadable production fixture.

Current combined run `428fff5f-e4a2-4faf-a3a9-4dedab1e459a` (handle
49186) exited 0: 6 passed, 10 excluded, 2.597 s. It includes actual-site
observation, transfer activation, all five independent prefixes, four
PhysCal models, REALg, and same-configuration kernel repeats. Full output:
`...-20261003-g/rust-six-focused-acceptance.log`. This establishes scoped
activation for all fourteen inventoried sites, not every scenario/error
branch, MPI interaction, native macOS or complete issue acceptance.

### Matched real-FSZ direct SR authority and long-run setting

Pauli's standalone original-C direct-SR audit (handle 15087), reviewed and
independently rerun by the parent, compares retained REALg pre-SR buffers
through original StochasticOpt/DPOSV and original post-update parameter sync.
The unsynchronized-versus-final discrepancy was a lifecycle-stage mismatch;
matched post-sync maximum absolute difference was observed as zero.
Condition estimate 873.704 and independent 256-bit residual estimate
`1.245e-17` are recorded with actual matrix/rhs/increments. Unknown native
imaginary slots were tested under two assignments, not claimed zero.
See `c_toolbox/real_fsz_direct_sr_audit.md` and retained
`evidence/issue-180-real-fsz-direct-15087-*`. This proves the scoped solver
on supplied buffers, not a full native C sampler, executable or MPI run.

The user changed the long-runner acceptance baseline from 50 to **20 steps**.
Existing completed 50-step evidence remains historical. Fresh 20-step
references must come from reviewed 62b with the matching effective parameter
window; no 50-step final parameters may be truncated or relabelled as 20.
Owned #182 test/generator/wrapper audit found no 50-step runner literal:
current prefix proofs are one step, threshold matrix runs two steps, and
PhysCal uses its actual two frames. These short/scoped jobs are not 20-step
long-run acceptance. Purposeful 1/2/3, failure boundaries 27/28/29 and CG
kernel limit/refresh checks remain separate. Long20, MPI and platform scope
are not completed by the focused results above.

### Subsequent canonical CG long20 checkpoint

Fresh reviewed 62b `canonical_general_rbm/step-20` references subsequently
became available. New opt-in test
`reviewed_cg_twenty_step_workers_match_and_repeat` requires them without
old-file fallback, sets steps **20** and effective window **20**, and checks
metadata for twenty successful/selected rows. It repeats workers 1/2/4 in
fresh processes at threshold 32 and ambient BLAS threads one. Complete
mapped parameters, final energy and the final C-declared pack use unchanged
`1e-11` bounds; saved configurations and next624 RNG outputs remain exact.
The twenty-row C-window input is fully shape checked: NPara102, fifteen
section widths, twenty complete E/E2/Para rows. No 50-step values are used.
This checks final state and window schema, not all output formatting or a
new window-average proof; canonical output consumers own that validation.

Fixed test-fast binary archive run
`27a26fa4-e86c-48c4-acd0-5090b8095e9e` (handle 95329) exited 0:
1 passed, 16 excluded, 9.491 s. Repeats compared fourteen records with zero
observed maximum difference under the bounded policy, not a computed-bitwise
requirement. Parent independently passed long20 `1ba4d50b` (9.234 s) and
all ten nonignored tests `a67fba6d` (0.904 s). The worktree was mutable at
archive build: this is a frozen binary snapshot, not a claim that all shared
production source was frozen during compilation. The initial draft compile
failure (integer assertion on a unit-returning API) is retained separately;
no running process was cancelled or restarted.

Artifacts in `/tmp/mvmc-threaded182-real-public-20261003-g/`:

- `rust-cg-long20-archive.log`: complete nextest output.
- `threaded-long20-tests.tar.zst`: archive SHA-256
  `7dd6f68a53781b911666bb060af7362c3908f5dd9e1812f7396dd5fd4e8e5d7e`.
- `threaded-long20-implementation.sha256`: test source
  `fce98252f21e7b7f339697450e3446074b40380c12cde0214198c41bda5db195`
  and archive hash.
- `threaded-long20-fixtures-before.sha256` and
  `threaded-long20-fixtures-after.sha256`: full candidate manifests,
  comparison exited 0 after terminal completion.

```sh
MVMC_RS_THREADED_182=1 cargo nextest run \
  --archive-file /tmp/mvmc-threaded182-real-public-20261003-g/threaded-long20-tests.tar.zst \
  --run-ignored only -E 'test(reviewed_cg_twenty_step_workers_match_and_repeat)' \
  --no-fail-fast --retries 0 --success-output immediate
```

This long20 checkpoint covers canonical general-RBM CG only, not fourteen
model/scenario long20 runs, MPI, macOS or full issue acceptance. Final
milestone verification requires a fresh archive after all owners' final
source/fixture changes; this earlier archive must not be relabelled final.
Focused strict clippy, owned-file formatting and diff checks passed on the
current test source. No shared production source was changed for these tests.

### Exact PR204 snapshot and extended-matrix gap

Commit `02c83f31e0feec8c97105727a4975f48afaa5687` was extracted into
`/tmp/mvmc-pr204-02c83f31.uudrv1` with its own Cargo target and pinned Julia
submodule `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`. Cargo executed no Julia
or C oracle. Handle 51571, run `deb4d95c-3585-43dc-bbc5-8ee9d2bfb192`,
terminated 0: four passed, thirteen excluded, 9.574 s. The selected tests
were QP threshold/lifecycle, observed kernel classification, transfer
activation, and canonical CG long20 repeatability. Three production-kernel
hashes and the test hash verified unchanged; `fixtures.before.sha256` and
`fixtures.after.sha256` compared equal after terminal completion.

This exact-commit result does not turn the broader two-step runner matrix
into twenty-step coverage. A separate external test extension in
`/tmp/mvmc-pr204-augmented-long20.xjhBkN` sets that existing matrix's steps
and window to twenty and repeats every worker configuration twice. Its
five variants and threshold-modified operands remain explicitly synthetic
kernel/runner coverage, not new independent public-input oracles. This
extension is not the exact PR204 test source. At this checkpoint no terminal
pass had been recorded; the subsequent result below keeps its distinct settings.
RBM-FSZ failure-boundary coverage and scoped timing evidence are
also outstanding. Neither activation nor one canonical long20 case proves
the complete issue acceptance matrix.

### Subsequent external and threshold-sample outcome checkpoints

External handle 83227 terminated 0, run
`507c1191-b585-4026-a370-2688af75b0da`: one passed, seventeen excluded,
1767.916 s. Its test SHA was
`5e8cbf3add2b967c385fb9f7578c65d1abd47ce2db355fc67d5220bc62c4eda3`.
This extension used **200 samples**, QP sizes 31/32/33, five variants and
four store/CG combinations: sixty optimization cases per worker child,
workers 1/2/4 each twice. It did not prove the later 45-case matrix with
sample counts 31/32/33. Its source/fixture after-hashes matched their
before-hashes. No native macOS, MPI or performance claim follows.

The new long20 outcome matrix instead uses samples and QP lengths 31/32/33,
five synthetic variants, and direct/no-store, direct/store and CG/no-store:
45 cases per child. The initial success-only run 17558 failed at Hubbard
sample32/store0, step10. Capture 39518 retained eleven original SR solves;
native C DPOSV replay of the same upper-column-major A/b returned INFO2.
Run 24298 then failed at sample32/store1, step10; capture 65626 retained
INFO4 with no substitution. These failed runs remain evidence, not passes.

Explicit diagnostic collection 27704 used frozen binary
`/tmp/mvmc-182-all45-diagnostic.hfiaGD/threaded_issue182`, SHA
`8a149b75e483a18cd28b8a29cff4bd6fdd3a70bf8d39f9e4e3595b9326115153`,
from test source SHA
`b0c750de711f9ff7ac645ff5c1af54b11715d319e34b0448be84a4cd1bcde4eb`.
Each of six worker/repeat children executed all 45 cases: 42 successes and
the same three direct-SR failures, then exited **101**, not acceptance success:

| Hubbard sample/QP count | NStore | first failing zero-based step | original factor INFO |
| --- | --- | --- | --- |
| 32 | 0 | 10 | 2 |
| 32 | 1 | 10 | 4 |
| 33 | 1 | 16 | 2 |

All have dimension five and no POTRS call. All eighteen original failure
operands were independently replayed with the native developer-only probe
documented in `c_toolbox/retained_direct_sr_status.md`, reproducing their
respective INFO values. This is retained-system factorization authority,
not a full native C sampler proof. Actual probe log:
`/tmp/mvmc-182-all45-retained-c-probes.log`, SHA
`9e48cbcf11adea9ab2a597fad041d87c99da354fcdaa15ccfe4df705ed716428`.

Diagnostic logs and per-case artifacts are retained under the frozen directory
and `/tmp/mvmc-issue182-{2662802,2663025,2663195,2663608,2665399,2695793}-0`.
All 45 outcome files compared byte-identical across the five other children
after excluding only the requested-worker metadata line: actual result,
completed steps, parameters/last successful post-sync parameters, full RNG,
draw count, next624 and saved configurations. This is Rust repeatability,
not a replacement for independent numerical oracles.

That byte-identical diagnostic comparison is historical observational evidence,
not the portable computed-float acceptance policy. Computed A/RHS and parameters
across worker configurations use the existing explicit numerical comparison
bound (`512 * f64::EPSILON` absolute and relative), with no tolerance increase.
Only the **same solve's original RHS versus its actual post-return immutable
copy**, and unchanged parameters versus the last successful post-sync copy,
retain bitwise assertions (including signed zero). INFO, first failing step,
shape, flags, active indices, no-POTRS, RNG and saved configurations remain exact.

The supported-outcome gate must assert these three exact case/step/INFO
contracts, parameters and RHS unchanged (including signed zero), no POTRS,
and execution counts 45/three rejections. Other errors remain failures;
the diagnostic-only collector's continuation is not enabled in normal gates.
Source SHA `5c2f7b72644991abcc199706f6fc2bf58efd091c0080526a6e2517210803a79f`
completed handle 94157 with terminal zero, run
`ca1e152d-faf5-4c8e-a49c-473d26c19cb4`: two passed, twenty-one excluded,
228.543 s (focused failure boundary and full supported-outcome matrix).
Each worker configuration ran twice; 3848 records compared with maximum
computed numerical difference zero. Actual worker snapshots were retained
for all six children; worker1 had no parallel entries. This is 42 successful
optimization cases and three verified rejections per child, **not** 45
successful optimizations. PhysCal remains two frames/four samples, not twenty
SR steps. The shared-source result is separate from the clean dependency
snapshot below and is not full issue #182/#185 acceptance.

Parent review artifacts for that diagnostic collection (not Cargo dependencies):

- Historical v1 comparator source `/tmp/mvmc-182-all45-artifact-review.v1.sh`, SHA
  `efc182283e9c5971c573c4453ce317a2c2a8fa0779e1794e56c8d1ac61a5877b`.
- All eighteen input manifests, native C stdout, probe/source/library identities
  and unchanged-parameter/RHS checks:
  `/tmp/mvmc-182-all45-artifact-review.stdout.txt`, SHA
  `e516e0ecf11ffff2f636a89a5f9ad0a84aaa7c74ed229dcef59c08ed3dec1fee`.
  Running the comparator completed exit zero; this reports artifact consistency,
  not a successful 45-case optimization run.
- Strengthened comparator `/tmp/mvmc-182-all45-artifact-review.sh`, SHA
  `958ada315f109c020f9bf1d6a649cd62b785d0ecd9f1bea558cfb0054dfe2b08`,
  captures each probe's stdout and asserts the exact dimension/triangle/NRHS/
  expected INFO line itself. It completed exit zero for all eighteen operands;
  `/tmp/mvmc-182-all45-artifact-review.v2.stdout.txt` has the same stdout SHA
  `e516e0ecf11ffff2f636a89a5f9ad0a84aaa7c74ed229dcef59c08ed3dec1fee`.
  Historical v1 stdout remains unmodified at its original path.
- Reviewed test-only patch capture `/tmp/mvmc-182-three-rejections-source.patch`,
  SHA `db82020d3f0e8d484fed86abbb1e31cfb4f009fb2ed12d3ae8ac8970260c40ea`.

For **each** of the six PID roots above, the corresponding first-failure files
have these identical SHA-256 values (RHS and actual post-return increment also
compare byte-identical, including signed zero):

```text
samples32/store0/step10 matrix bf89c6d1286e714b0fa3825bf000fc0e7b9f4a70bec8fc5d285f69b22d21a36b
samples32/store0/step10 rhs    204814be0e9cf40e2c17961044386ed252c04467e3b0dbaf1beeac558ca9a90a
samples32/store0/step10 meta   6b3316dca7ea5641cd167ac54c0d829aa596398f5a7cff7ada6c9be8020635b2
samples32/store1/step10 matrix 59b3f63c15161b08f33b2dc58951356b6f963d6d40389f2cd8f4a809427e4400
samples32/store1/step10 rhs    f4f7ac898226788c07a8943dd0b10dd724e3b130b207511cca373637f943082e
samples32/store1/step10 meta   bc7942faeb3b7524c27489913a56f9023655d51960d5199fe64882660ee02542
samples33/store1/step16 matrix f0872672e841ec3c09064cefecd3f33e0561eec00bf9e4bc84ec5b8dfcef6fde
samples33/store1/step16 rhs    0ab6fc246f182908c22084325c4850885cecedd737640006c7850c50d8774771
samples33/store1/step16 meta   7e5750209434a95f3cf9bc37f81d1927b42658353cc98f5f719b03ff40b53b30
```

The native probe executable SHA is
`626a1bd68106c7892ca3f9092f49c3220396532dc4eeb550a76331aa4394f6ee`,
adapter source SHA
`e60376b86872c67db282a46c4f67bf2bfe380c061f6b8c1038cef53a6c5aaf0a`;
upstream calling-contract source/compiler/LAPACK versions are documented in
`c_toolbox/retained_direct_sr_status.md`. These hashes describe retained-system
replays and must not be relabelled full-C sampler or native macOS proof.

Independent clean-dependency verification was performed in
`/tmp/mvmc-182-committed-helper.h0Cn0R`, extracted from committed
`b2096d2ae84bcbae58e2144b7ef88db5f9bf8d61` with only the owned thread test
overlaid. Committed `support/ctest_provenance.rs` SHA is
`b957320c502b7331d5476976353fd682ebbc1c11d147b420ddc62666c7012246`;
no dirty helper tests or fresh52 archive were copied. It uses a separate Cargo
target. Handle 32416 terminated 100, run
`79e4b178-9ab2-4f45-9736-59204eedd988`: all three selected tests failed
MissingFixture before numerical execution. The archived submodule placeholder
caused the attempted reference symlink to be nested instead of replacing the
placeholder; this was a snapshot setup defect, not a numerical failure.
The original placeholder was preserved as `extern/Julia-mVMC.unconfigured`.
The correct reference directory was then populated by `git archive` of pinned
Julia commit `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, without running Julia
or C. Source and committed helper hashes did not change. Corrected verification
handle 41033 terminated zero, run
`ca1c813e-70dd-41c8-a0bb-6d738a99c975`: three passed, nineteen excluded,
225.685 s. This selected the independent five-model normalized pre-SR
OO/HO prefix gate, focused failure boundary and all 45 supported-outcome cases
per worker child, workers 1/2/4 each twice. The 3848 runner records compared
with maximum numerical difference zero. The original test/helper hashes above
were unchanged after terminal completion. Actual worker counters and all three
failure payloads remain in `validation-reference-fixed.log` and the fresh
per-child `/tmp/mvmc-issue182-*-0` directories named by that log. This is
same-platform repeatability of 42 successful cases plus three strict supported
rejections, not 45 successful numeric goldens or complete issue acceptance.
Clean-snapshot strict clippy handle 45445 terminated zero in 36.92 s
(existing nonfatal vendor deprecation warning only).

The source `5c2f7b...`/run 41033 checkpoint above is retained as historical:
parent review subsequently found computed A/RHS emitted as discrete bit records.
Those records were replaced with numerical records using the unchanged bound;
self-copy immutability checks were retained. New source SHA
`41452d5fe1c6deb42c8868bda4b5df1f06c5570f7a15de2668100f269ee20024`
required fresh focused and full-45 verification at that review checkpoint;
its completed result is recorded below with the same committed helper.
Native C retained-input replay operands and their historical hashes are unchanged.
Policy-corrected verification handle 29780 terminated zero, run
`eb87b111-8c44-4f76-bc57-bd123b71f0c2`: three passed, nineteen excluded,
230.072 s, with the same committed helper and pinned reference snapshot.
Independent OO/HO passed in 1.390 s, the focused failure boundary in 2.752 s,
and the 45-outcome worker/repeat gate in 230.071 s. It compared 3860 records
with maximum numerical difference zero using the unchanged explicit numerical
policy for computed values; original-RHS immutable-copy checks and parameter
rollback checks remain bitwise within the same run. This proves the scoped
42-success/three-strict-rejection outcome matrix, not 45 successful numerical
goldens or full issue acceptance. After termination the test SHA remained
`41452d5fe1c6deb42c8868bda4b5df1f06c5570f7a15de2668100f269ee20024`
and committed helper SHA remained
`b957320c502b7331d5476976353fd682ebbc1c11d147b420ddc62666c7012246`.
Strict clippy handle 13926 terminated zero in 1.25 s (same vendor warning).
Final policy-corrected logs and SHA-256:

```text
/tmp/mvmc-182-committed-helper.h0Cn0R/validation-numeric-policy.log
54516d2a492b1f07184af0c97a87c702504bafae4ce5fe9ae6a483e2a1c68a70
/tmp/mvmc-182-committed-helper.h0Cn0R/clippy-numeric-policy.log
cc30ef4bd50f11623392cc9f1f42194d9d148a483a1e7ad2d97de46c78de4707
```

Remaining MPI activation scope is separate: grouped QP8 with threshold32
falls back to serial QP kernels even when independent SR entries run in
parallel. That does not prove MPI/QP worker activation or all fourteen
callsite branches. The MPI owner must record the actual local QP range and
entered QP/term workers, while separately checking that collectives stay on
the initiating FUNNELED thread. Threshold1 experiments are labelled distinct
from default-threshold32 boundary checks. With groups of width two, global
QP31/32/33 ranges may each be locally below threshold; global lengths around
63/64/65/66 expose the local32 boundary instead.

No speedup is claimed. Issue #182's allocation/timer/benchmark-settings
acceptance is conditional on such a claim; kernel activation is not a speed
measurement. Any scoped benchmark evidence uses the existing workflow and
explicit settings, not an invented pipeline or an unrequested optimization.
Original thread84 semantic/API mapping and MPI activation remain explicit
acceptance work; the result above must not be promoted to full issue completion.
