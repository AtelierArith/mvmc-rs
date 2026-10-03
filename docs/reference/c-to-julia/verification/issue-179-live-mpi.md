# Issue 179: optional executable MPI verification

This gate is incomplete. Successful CLI launches are **EXECUTION_ONLY**, not
deterministic parity. InterAll is excluded. C defines the numerical/input
contract; independent Julia 1.13.1 observations are secondary evidence where
they agree with C. Computed values require justified #190 componentwise bounds;
RNG and discrete contracts remain exact.

## Reproduce in the Linux container

Run from `/workspaces/mvmc-rs`, with `CARGO_TARGET_DIR` isolated at
`/tmp/mvmc-issue179-target` for the old diagnostic container. In the current
Dev Container set `MPI179_CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-mpich`
to keep builds in its named cache volume. The launcher is selected from the
observed Open MPI/MPICH version, not silently replaced. Julia uses its own
`MPI.mpiexec()`; do not mix MPI ABIs. A real-world-size/root-broadcast preflight
must pass before CLI cells are accepted as MPI execution evidence.

```sh
bash scripts/verify_mpi_issue179.sh
MPI179_JULIA=/path/to/julia-1.13.1 \
  bash scripts/verify_mpi_issue179_states.sh /tmp/mvmc-issue179-evidence.EXAMPLE
PYTHONDONTWRITEBYTECODE=1 uv run --no-project python scripts/test_compare_mpi_issue179.py
bash scripts/verify_mpi_issue179_physcal.sh
bash scripts/verify_mpi_issue179_rust_states.sh /tmp/mvmc-issue179-evidence.EXAMPLE /tmp/mvmc-issue179-states.EXAMPLE/state-build.json
bash scripts/verify_mpi_issue179_workers.sh /tmp/mvmc-issue179-evidence.EXAMPLE
```

The CLI matrix names 81 cells: ranks 2/4, real/cmp/FSZ, split 1/2,
direct/CG, NStore 0/1, identity/standard/OptTrans projections, invalid CG,
valid C/Julia uneven split-3 widths,
missing input, output failure, and a four-rank/three-sample empty-work candidate.
Grouped CG, grouped OptTrans, and grouped FSZ standard projection are explicit
rejection cells. A requested success returning unsupported fails the gate.
The empty-work candidate is not proof of correct sampler partition semantics.

The state script now selects every named expected-success input from the CLI
manifest, including CG/NStore=0, standard QP, OptTrans, uneven group widths and
empty measurement ranks, at each of inner workers 1/2/4 and prefixes 1/2/3.
`MPI179_CELL_REGEX` can restrict an explicitly labelled development run; a
restricted run is not the full matrix. The prior twelve-input sweep had 36 cells,
plus 2/4-rank collective and callback-failure checks. It observes initial/final
next624, seed/group metadata, configured/local sample counts, saved
configurations, local/reduced counters, local/reduced energy/SR accumulators,
initial/final parameters, and root output files. Newly added opt-in Rust hooks
and observation-only Julia overlays capture update selections, proposals
(including early rejections), each actual Metropolis decision/draw, every actual
sampler-consumed word, and sampling-boundary checkpoints of saved
configurations/counters/next624 before MainCal overwrites live scratch. Missing
or malformed event streams/checkpoints fail the comparator. Sampler draw counts
are counted from those real primitive events, not inferred from final RNG state.
Julia observes actual parameter-initialization API draws. Rust's private SFMT
consumed-word counter now records initialization and total words independently
of the sampler observer; cloned next624 peeks leave the actual count unchanged.
All 24 SFMT tests, including C golden vectors and counter/reset/clone/bulk tests,
passed in the current repaired-image snapshot. Independent MPI count agreement
still requires validation; adding a counter does not establish trajectory parity.
Expanded CG/NStore=0 independent execution and zero physical-weight numerical
behavior still require fresh validation; naming those cells does not validate them.

### Current-runtime repair (#196)

The repaired current-config container `73c57e563c61` uses actual image
`sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`.
Its MPICH 4.2.0 `ch4:ucx` uses Hydra/internal PMI1, not distro external PMIx;
`LD_LIBRARY_PATH` and `PKG_CONFIG_PATH` identify `/opt/mpich` explicitly.
The standalone C preflight passed world sizes 2/4, integer sums 3/10 and
nonzero-root broadcast 196 on every rank. This is communicator/runtime evidence,
not full C sampling parity. Rust MPI preflight remains pending.

Long-running checks use copied snapshot
`/home/vscode/.cache/mvmc/snapshots/issue179.YPAEVT`, with named-volume target
`/home/vscode/.cache/mvmc/target/issue179-snapshot`. Its source-manifest SHA-256
is `f63f23f670648d0da535c9f84c116d64ffc3a3684fa276fb77d0855531178687`;
it precedes the latest counter entry-point and private candidate lint fixes.
Julia's separately matched MPICH_jll 5.0.2 ABI/launcher remains independent of
Rust/C MPICH 4.2.0. The new-image world-4/width-3 prefix-1 Julia reference passed
trace structure and exact same-chain stream checks: ranks 0/1/2 consumed 12
initialization and 124 sampling words, rank 3 consumed 12 and 120 respectively.
Prefixes 2/3 also passed structural checks, same-chain stream equality and exact
extension of each rank's shorter-prefix event stream. Group-0 sampling-word
totals are 124/220/316 for prefixes 1/2/3; group-1 totals are 120/216/312.
Each stream contains exactly one/two/three sampling checkpoints. These Julia-only
results do not establish Rust/reference agreement.

The initial current-config instance `232f26a94452` uses Ubuntu MPICH 4.2.0,
ch4:ucx, configured with external PMIx and Hydra. Independent C and Rust
diagnostics observe singleton worlds under a multi-process launch. This matches
the maintainer explanation in [upstream MPICH #7064](https://github.com/pmodels/mpich/issues/7064).
Its Rust MPI matrix results are invalid, not parity passes. Julia's separately
selected MPICH_jll 5.0.2/ch4:ofi launcher does observe genuine worlds; never use
the system launcher for that different ABI.

[Open follow-up #196](https://github.com/AtelierArith/mvmc-rs/issues/196) repairs
the Dockerfile using official MPICH 4.2.0 source, preserving ch4:ucx, with
`CC=gcc CXX=g++ FC=gfortran`, `--prefix=/opt/mpich --with-device=ch4:ucx
--with-ucx=/usr --with-pm=hydra --with-pmi=pmi1 --without-pmix`.
Archive origin: `https://www.mpich.org/static/downloads/4.2.0/mpich-4.2.0.tar.gz`;
SHA-256: `a64a66781b9e5312ad052d32689e23252f745b27ee8818ac2ac0c8209bc0b90e`.
The runtime/preflight is not yet validated. Existing containers/volumes are not
removed. The explicit developer command is:

```sh
MPI179_CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-mpich-hydra \
  MPI179_IMAGE_ID='<docker inspect .Image of this container>' \
  bash scripts/verify_mpi_issue196_runtime.sh
```

It independently compiles `c_toolbox/mpi_issue196_world.c` (original diagnostic,
no vendored extraction), checks genuine C worlds at 2/4 ranks, then launches
Rust tests for group-width endpoints including width 3, global totals/nonzero
root broadcasts, group-only QP overlap with stale outside-range NaNs and empty
QP ranks, coordinated status, and six statistical counter slots. Cargo tests
do not invoke the C probe. Numerical log(2) checks permit four machine epsilons
for well-conditioned elementary-function roundoff; integer contracts stay exact.

The worker interaction script executes all 46 supported identity/standard
input combinations at workers 1/2/4 (138 CLI/state cells). It fixes BLAS threads
to one and uses an explicit inner threshold of one to exercise parallel paths
on small inputs. Exact configurations, counters and next624 are checked across
workers; accumulator/SR and root-output comparisons require justified bounds.
Requested/configured threads, threshold, observed pool capacity and effective
QP worker count are recorded separately per rank. The capacity probe runs after
execution and does not establish that a particular kernel activated the pool.

The separate grouped PhysCal matrix names eight cells: 2/4 ranks, normal
real/cmp Green positives, FSZ multi-QP rejection, and grouped Lanczos rejection.
All eight passed their execution/rejection gates in
`/tmp/mvmc-issue179-physcal.GBpLp2`; the earlier four-positive-only run is
`/tmp/mvmc-issue179-physcal.5VNJIr`. Neither compares independent Green values.
The Rust-only state sweep captures all 36 supported identity combinations from
the CLI matrix as CAPTURED_NOT_COMPARED. Supply its exact build artifact JSON;
it records the executable and input hashes and does not pretend the current
dirty source necessarily matches a previously built executable.

Every worker is bounded by timeout. Reference failure, missing/malformed
comparison data, or missing bounds makes the script exit nonzero. Supply
`MPI179_ATOL`, `MPI179_RTOL`, and a nonempty `MPI179_JUSTIFICATION` for state
numeric gates only after explaining the first numerical divergence. Optional
CLI reference gates also require `MPI179_ROOT_ATOL` and `MPI179_ROOT_RTOL`.
Discrete-only diagnostics do not check computed numbers or claim full parity.

## Provenance and acceptance status

Evidence is created in fresh `/tmp` directories, not committed. Source manifests
hash tracked and untracked draft code rather than recording only HEAD. Source
changes during a run invalidate milestone validation and are reported explicitly.
Freeze shared writes and repeat the final gate after runner repairs/integration;
the initial execution run cannot certify a subsequently edited runner.

An explicit-reference failure audit using `MPI179_JULIA=/bin/false` completed
all 81 CLI cells in `/tmp/mvmc-issue179-evidence.XHGe2X`. All 53 requested
references exited 1 and the overall gate exited 1. Seven comparator regression
tests include executing the actual CLI awk gate against numerical mismatch,
NaN/overflow, malformed shape, extra rows, and empty comparison data.

On the post-integration checkout, `/tmp/mvmc-issue179-evidence.wSR1WK` recorded
53 EXECUTION_ONLY cells and 28 observed rejections, with no launch timeout.
Two split-3 rejections were incorrectly classified by the initial script:
C/Julia allow these inputs with a warning. The current script treats them as
expected-success cells and fails on that Rust restriction. Its
source-changed flag made the overall gate fail. The independent state run at
`/tmp/mvmc-issue179-states.KqWpM4` exposed, for two ranks/split 1/three configured
samples, Rust chain lengths 2/1 versus Julia 3/3 and reduced burn marker 2
versus 1. These are discrete runner-contract differences, not floating-point
error. Group communicator metadata also differs. Coordinate fixes with the
runner owner; do not repair semantics inside the observation wrapper.

Live 2/4-rank synthetic collective tests cover empty buffers, uneven/zero rank
contributions, expected global sums, fresh contributions detecting extra group
multiplicity, and one-rank coordinated failure. A four-rank callback failure
terminated all ranks successfully under a 30-second timeout. These are protocol
checks, not full executable sampling parity or proof of zero physical-weight
handling. Proposal/decision/draw-count observation hooks are an outstanding
implementation-owner dependency for #179 acceptance.

## Mapping repair and current-image blocker

All evidence above came from old container `60e8b0ba99a4`, Open MPI 4.1.6,
with only a workspace bind. It is not the current documented Dev Container.

The user-authorized balanced mapping repair changes `parallel.rs`: NSplitSize
is communicator width, so world2/width2 and world4/width4 each have one chain;
world4/width2 has two chains. MPI broadcast code was not edited. Endpoint
integration tests verify global-root seed broadcast across all groups/local
ranks. C/Julia remainder partitioning is also restored: length10/size4 gives
`0..2, 2..4, 4..7, 7..10`, with extras on the last ranks. Small work assigns
one item to the first active ranks; zero work is empty everywhere. Six targeted
partition tests pass in the new image. Non-divisible group support and genuine
QP ownership/sampler lifecycle remain runner work; no numerical tolerance can
hide these discrete contract differences.

The new instance `232f26a94452` was created with Dev Container CLI 0.87.0 and
unique label `mvmc.issue179=reference-20261003-d05cde3`, from current config.
Its UID-adjusted image ID is
`sha256:ed25d1f45ee94e5c7e96be0ec902d3f683c44257fef564ca6699519ee34b1dda`.
Named Cargo/Linux cache volumes and the read-only Cargo configuration mount were
confirmed. Observed Rust 1.99.0, MPICH 4.2.0/ch4:ucx, OpenBLAS 0.3.26 LP64,
glibc 2.39-0ubuntu8.6, kache 0.28.1; a two-target probe demonstrated one new local
cache hit. The checksum-verified Julia 1.13.1 reference uses bundled OpenBLAS
ILP64 and MPICH_jll 5.0.2/ch4:ofi. Its world4/width4 reference completed with
group 0, seed 1 and three chain samples on every rank. Per-rank provenance hashes
actual reference source, SFMT dependency binaries where present, input
files and the Julia manifest.

**Current image cannot provide final Rust MPI verification:** its Ubuntu MPICH
library is configured with external PMIx while Hydra supplies PMI. A separately
compiled C rank probe and Rust both report singleton rank0/size1 under `mpirun
-n 2`, despite PMI_RANK=0/1 and PMI_SIZE=2. This matches the
[MPICH upstream package/launcher diagnosis](https://github.com/pmodels/mpich/issues/7064).
The new 81-cell attempt (`/tmp/mvmc-issue179-evidence.TKW0rU`) and 138-cell worker
attempt (`/tmp/mvmc-issue179-workers.zggC16`) are invalid as MPI interaction
evidence and fail overall. Missing rank files are failures, not agreements.
No MPI backend was replaced, no container removed, and live reference/test
processes were not cancelled. The container owner must repair the runtime/launcher
combination before final MPI gates can pass; old Open MPI diagnostics do not
substitute for that validation.

The new fail-fast preflight was executed in
`/tmp/mvmc-issue179-evidence.rk6LR9`: it exited 1 before the CLI matrix because
the actual world size was 1, not the requested 2. The independent reference
worker-axis process subsequently terminated with exit 1 under handle `99007`,
with evidence `/tmp/mvmc-issue179-states.gAscKI`; its singleton Rust captures
cannot be treated as MPI parity. These are historical defective-image results,
not the repaired runtime or current trace coverage.

## Repaired-runtime actual captures after main integration

PR #199 repaired the current Dockerfile's internal-PMI/Hydra runtime; its isolated
current-main proof is recorded separately in [issue-196-mpi-runtime.md](issue-196-mpi-runtime.md).
#196 and #179 remain open. The #179 container `73c57e563c61` uses actual image
`sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`,
system MPICH 4.2.0 and named-volume targets. Julia 1.13.1 still uses its own
MPICH_jll launcher and pinned reference, not the newer PR54 writer commit.

Frozen source snapshot `/home/vscode/.cache/mvmc/snapshots/issue179-sr.WbN8B8`
has source-manifest SHA-256
`e492fb961e0df28685a9298e460bc1e8867eedac143f2c36abed01c8386c0e4b`.
Build handle 64378 exited 0. Julia handle 43637 and Rust handle 28663 exited 0:
worlds 2/4, requested width 3, prefixes 1/2/3, Rust workers 1/2/4 (18 Rust
captures, six independent references). Both post-run source checks passed.
Evidence root: `/home/vscode/.cache/mvmc/issue179-sr-independent`.

Actual observations retain initialization/consumed words, proposals, acceptance
decisions, saved configurations, next624, local/reduced accumulators, SR updates,
and original pre-factorization SR matrix/rhs plus actual solved increment on each
rank. Worker sidecars count actual kernel entries, separately from a post-run
pool-capacity probe. Julia diagnostics consume owned solve observations after the
run; they do not rerun or replace the solver. The reusable helper's hash is part
of reference provenance.

Pauli's independent audit reported 108 actual per-rank solve pairs, exact
dimension/status/active-index/settings agreement, maximum condition2 estimate
`2.8515440521e7` and maximum normwise backward error `4.9380359475e-17`.
World4 solve2 matrix/rhs/increment differences were approximately
`1.39e-17`, `2.17e-19`, `8.68e-12`. These measurements are conditioning evidence,
not a certified condition bound or an adopted numerical tolerance.

Raw real-mode imaginary optimization flags differ between the ports. C
`GetInfoOpt` does not write these slots when the declaration is real; `OptFlag`
is malloc-backed, so zero is not a native expectation for unwritten slots.
Keep raw metadata intact, compare C-written slots using the family-specific
written mask (including the orbital-parallel exception), and compare actual
active indices exactly. Until that separate contract comparison is implemented,
the existing whole-raw-flags comparator does not establish full parity.

The first full-matrix manual launches ended incomplete (nine Rust and three
Julia captures), because launchers shared the matrix reader's stdin. They are
not full coverage. Corrected bounded launches isolate reader fd3 and stdin,
require 495 Rust / 165 Julia captures, and retain every exit status under
`/home/vscode/.cache/mvmc/issue179-sr-matrix-v2`. Handles 15476/81717 were live
when this checkpoint was written. No full matrix or numerical PASS is claimed.

## Owned patch paths (uncommitted)

The partition implementation was added to scope by the user after the mapping
report. Scope was later expanded to MPI/reducer/sampling/SFMT draw counters;
runner/CLI and Banach's worker kernels remain separately owned. The list below
is the original verification-path inventory, not a complete production diff:

```text
crates/mvmc-core/src/parallel.rs
crates/mvmc-core/tests/mpi_issue179_state.rs
crates/mvmc-core/tests/mpi_issue179_mapping.rs
scripts/compare_mpi_issue179.py
scripts/test_compare_mpi_issue179.py
scripts/mpi_issue179_environment.sh
scripts/mpi_issue179_reference_provenance.jl
scripts/verify_mpi_issue179.sh
scripts/verify_mpi_issue179_julia_launch.jl
scripts/verify_mpi_issue179_reference.jl
scripts/verify_mpi_issue179_state.jl
scripts/verify_mpi_issue179_states.sh
scripts/verify_mpi_issue179_physcal.sh
scripts/verify_mpi_issue179_rust_states.sh
scripts/verify_mpi_issue179_workers.sh
docs/reference/c-to-julia/verification/issue-179-live-mpi.md
```
