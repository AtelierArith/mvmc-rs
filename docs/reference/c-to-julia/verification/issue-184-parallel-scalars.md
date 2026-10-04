# Bounded A200/A201 scalar communicator contract

Related to #184, #179 and #185; not full issue completion.

Current SOURCE integration base: main `5b0874eb70b72e2a7993662af757d50b25dadc17`.
Main299's P02 step, main300 joins and main301 parameter comments/joins are preserved.
This integration/transport repair is not yet validated or published.
Transport binding controls below tested the main1b0 SOURCE integration, not a
main5b087 model/native execution; the guard and installer bytes are identical.

Historical integration base: main `1ad3f582f0c6bd35ac522393e65b972a739234a1`.
Resolved merge `6fee8422d4a810ab5b00d1a81dbed669d59a9b55` has tested tree
`03f2ce055cb9581d0482440b60e0b69166e52d0c`. Both scalar and parameter
diagnostic modules/parameter exports are retained, as are main294–298 changes.
Session72857: terminal0/prior0/post0/cleanup0; scalar1, parameters9,
sync/SR25 (220 unselected), original-boundaries2 and shared-orbital1 passed.
MPI strict Clippy, fmt and core MPI docs passed (zero doctests). All complete
source/runtime/tools/scripts/selected-binary posts and independent replays0.
Receipt HOST/container: `/tmp/issue292-main298-proof.uP36YW`.
The following documentation update is post-validation only; executable
sources are unchanged. New exact-head native worlds2/4 CI remains pending.

Historical integration base: main `14a30849ae4fa9dd78900f113257389d906008ee`.
Resolved merge `393dbda7e6234f73f68a15bccdd98ad68dd3582e` preserves both
PR293 parameter exports and this scalar module. Its tested tree is
`cc3795db1f66c8535bc285da83ef4337525c3ff5`.
Session97222 completed terminal0/prior0/post0/cleanup0: serial1,
parameters9 and sync/SR24 passed (220 unselected); targeted MPI strict
Clippy, workspace fmt and core MPI doctests passed (zero doctests).
All source/runtime/tools/scripts/selected-binary postchecks passed.
Container receipt: `/tmp/issue292-integration-proof.z5eEhh`; host copy:
`/tmp/issue292-integration-proof.z5eEhh/issue292-integration-proof.z5eEhh`.
This documentation update is post-validation only; executable sources are
unchanged. Its later exact-head native CI passed; this is not new-head proof.

Historical publication base: main `87cd786c7086618d6f13ec70110a75377d92c337`.
Historical validated source: immutable PR291 head
`c6418c1fa7627bb99964605074646a32a3942d3c` plus the five scalar paths.
The base versions of `lib.rs`, `mpi.rs`, and `reducer.rs` are byte-identical
between these revisions. Publication preserves the merged PR291 signed
`sampling_max_info(i32)` and Pauli's separate #290 work. This is not a fresh
main87cd execution or a whole-workspace proof.

## Source authority and declared differences

Original Julia reference is `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
`MVMCOptimizers.jl/src/parallel.jl` (`build_parallel_context`, `_comm`,
`allreduce_sum_scalar`, `allreduce_max_scalar`) and
`MVMCOptimizers.jl/test_unit/test_unit_parallel.jl` serial scalar assertions.
Original source SHA-256 values respectively:
`a7d72725f193a121f79267be6f931e2439fecdf104de01a0cb1d3a5ca8d91fc1`
and `a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`.

World, sampling-group and cross-group selectors use actual comm0/1/2.
Every rank participates in comm2, which joins equal local ranks across
groups, including a short final group. Serial operations return their input.
A bare MPI world has no built sampling/cross-group context and rejects those
selectors instead of fabricating a world reduction or NULL identity.
Signed maximum is bounded to C-compatible `i32`, not every Julia Integer
type. The sampling maximum delegates to PR291's existing operation.
Complex sums reduce real and imaginary components separately, matching the
Rust representation; no MPI complex-datatype bitwise equivalence is claimed.

## Actual bounded validation

Container `happy_jackson`, source `/tmp/issue184-scalars-c641`, receipt
`/tmp/issue184-scalars-sharedowner-proof.AGJoR5` (also copied to that host path).
Owner session33526: terminal0, prior0/post0, cleanup0. Seven command gates0:

- Locked test-fast nextest inventory: exactly one serial and one ignored native identity.
- Serial: 1PASS, UUID `8dbdd2b7-880d-4584-948b-f3c680527b7d`, 0 skipped.
- Native worlds2/4: all six ranks each 1PASS, 0 failures, 0 ignored.
- Targeted strict MPI Clippy, workspace fmt check, core MPI doctests: all0.
  Doctests contain zero tests: compile proof only.

Native controls use widths1/2/3/world, independent arithmetic membership,
dyadic real/complex sums, original serial 4.5/2+3i/7, negative maxima and
`i32::MIN`, with two same-order collective passes. There are no numerical
tolerances, model, sampling, RNG, or C/Julia runtime/oracle calls.
Commands are in `commands.txt`; full list in `inventory.log`; rank captures
in `world{2,4}-rank*.stdout`. Source inventory before/after comparison and
source/tools/scripts/runtime/selected-binary postchecks all0.

Rust1.99.0; MPICH4.2.0 ch4:ucx, Hydra/internalPMI1, `--without-pmix`;
loaded MPI is `/opt/mpich/lib/libmpi.so.12`. All configured BLAS/thread
controls1. Native ELF SHA-256:
`afeb4a99a2de9d93a2730124d3e5423390bac746e1da0c9628b36537d130b73f`.
Static fixture hydration used the pinned Julia gitlink
`c0788c34a6a5753c611633a97cd1ea233203320c`, without running Julia.

## Preserved failed owner attempt

First receipt `/tmp/issue184-scalars-validation.eZR8Qq`, session49921,
remains terminal83/post1/cleanup1: an obsolete owner helper exhausted its
eight scans during disappearing compilation children. Cargo subsequently
finished, but its wait status was lost; its ELF metadata is not validation
PASS. No scalar assertion had run. The successor uses the existing reviewed
shared `owned-nextest` helper `d335bc9e5de2d3a8310943b3c43a684c92ab6a2257258bb479617c4b8ba32166`
and separate stopped-launch abort helper `46d45967eca7ea0442e984f3c53fd1ff402cd2a47a261caef905954d2355a3ab`.
Their prior13 mock/4 OS controls were reused, not rerun or replaced.

A later integration preflight `/tmp/issue292-main296-proof.XAfEeL` remains
terminal1/prior1/post1/cleanup0: a HOST helper path was absent in the container;
no Cargo/test ran. Successor `yCQOXS` pinned the same helpers in its receipt
and passed37 on main-da979 integration. The final main1ad receipt above is
separate; neither historical failure was discarded or labelled PASS.

Old head `dfa7f9b054aac245cc8ec3bf84bdc4ec5e15ab41` completed all six CI
checks and native worlds2/4; parent reviewed its six rank PASS receipts.
Head `ae75149f819c6c682935329b2ea4a1490a5ea1c6` also completed all six checks
in run37188904027. Parent subsequently downloaded artifact11298557198 to
`/tmp/issue292-ae-native-review.D82r12` and fully reviewed six rank PASS,
worlds2/4 native0, provider startup and empty POST logs. This review occurred
after the original publication paragraph; it is historical ae751 evidence,
not evidence for head84c741 or the new main1b0 integration.
Head `84c741eea12f5fd60c377a043a8105c2fe29dbaa`, run37190994771,
retains a genuine failed Linux-all provider startup: artifact11298943538,
`/tmp/issue292-provider-failure.wkzTZI/artifact/scalar-mpi-provider`.
Configure/make/install completed; world2 native15 aborted in MPI_Init_thread
with UCX `ibv_create_srq(): Operation not supported` on both ranks.
Installer terminal was prior1/post0 and all six POST logs were empty.
Rust all-feature tests and scalar native gates did not start; world4 startup
did not run. Authoritative run API after all six jobs completed confirmed the
other five succeeded, including both macOS jobs; this is not all-six success.
The SOURCE repair fixes only the Linux-all job's requested UCX_TLS to
`self,sm,tcp` before startup and retains that exact setting for Rust/native.
MPICH4.2.0, ch4:ucx, PMI1/Hydra, pinned archive and all assertions are unchanged.
The setting receipt binds the request; it does not observe negotiated topology.
OpenUCX documents TCP/shared-memory/self transports and UCX_TLS selection:
https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use
No startup/native success is claimed for this unrun repair.
Cheap settings-binding controls subsequently ran once at
`/tmp/issue292-transport-controls.1j8tNs`: positive1/negative8,
inner/outer prior0/post0, all source/tools/original-artifact POST logs empty.
The eight rejected cases cover missing/empty/wrong env, wrong/missing/duplicate/
conflicting receipt line and symlink receipt. No installer/Cargo/MPI/model ran.
This result paragraph is a doc-only post-control update; tested workflow,
installer, guard and controls bytes remain unchanged. Old SOURCE.sha256 retains
its historical pre-control doc hash; the final packaging manifest is separate.
The integrated publication head requires fresh exact-head CI receipts.
Array/root-only/abort APIs, all Julia Integer
widths, full model/MPI matrices and full #184/#185 acceptance remain unproved
by this milestone.
