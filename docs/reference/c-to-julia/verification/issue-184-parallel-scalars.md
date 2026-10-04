# Bounded A200/A201 scalar communicator contract

Related to #184, #179 and #185; not full issue completion.

Publication base: main `87cd786c7086618d6f13ec70110a75377d92c337`.
Actual validated source: immutable PR291 head
`c6418c1fa7627bb99964605074646a32a3942d3c` plus the five scalar paths.
The base versions of `lib.rs`, `mpi.rs`, and `reducer.rs` are byte-identical
between these revisions. Publication preserves the merged PR291 signed
`sampling_max_info(i32)` and Pauli's PR290 changes. This is not a fresh
main87cd execution or a whole-workspace proof.

## Source authority and declared differences

Original Julia reference is `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
`MVMCOptimizers.jl/src/parallel.jl` (`build_parallel_context`, `_comm`,
`allreduce_sum_scalar`, `allreduce_max_scalar`) and
`MVMCOptimizers.jl/test/test_unit_parallel.jl` serial scalar assertions.
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

Regular exact-head Linux native CI integration is a separate prospective
workflow change until executed. Array/root-only/abort APIs, all Julia Integer
widths, full model/MPI matrices and full #184/#185 acceptance remain unproved
by this milestone.
