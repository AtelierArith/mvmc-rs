# Standalone original-C real-FSZ direct SR audit

Developer-only `real_fsz_direct_sr_audit.c` includes complete, unchanged
`extern/mVMC-1.3.0/src/mVMC/stcopt.c` (SHA256
`43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2`)
and `stcopt_dposv.c` (SHA256
`2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca`).
No numerical-body changes: copyright/GPL notices remain in
the included sources; upstream license is `extern/mVMC-1.3.0/COPYING`.

Harness-only serial communicator/timer stubs and a DPOSV observation wrapper
provide a standalone kernel check, **not full C executable, MPI, sampling or
RNG validation**. The wrapper records the original assembled matrix/rhs,
calls original LAPACK DPOSV exactly once, then records actual INFO/increment.
Original StochasticOpt selects components and applies its original update.
The matched post-sync comparison also executes original SetFlagShift and
SyncModifiedParameter, including its helper bodies extracted verbatim from
`parameter.c`, starting at `void SyncModifiedParameter(` and ending immediately
before the last `#endif`. The original copyright/license comment is retained.
Source SHA256 is
`46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0`.
The external acquisition stores the actual excerpt and its hash. Only supported
Gutzwiller/Jastrow plus mixed AP/P orbital input is accepted by this probe;
no RBM/DH/OptTrans path is claimed. Original C header widths determine the
Slater width (AP + two P blocks), not Julia's reported parameter count.
No Rust-generated expected values, runtime Rust dependency, or algorithm fix.

Input protocol: NPara/StaDel/StepDt/RedCut, Gutz/Jast/Slater declared widths,
pairs of written-mask/defined flag
for all2*NPara components, complex parameter pairs, size*(size+2) real-buffer
pairs, and size HO pairs. Julia's separately allocated HO is explicitly placed
in the C contiguous real OO/HO/O tail before original StochasticOpt copies the
entire real allocation into its complex shadow. This is a memory-layout adapter,
not a changed numerical formula. Unknown C flag cells are tested with different
assigned values; they must not be presented as native allocator zeros.

Build (Linux x86_64, GCC, scalar assembly, installed LAPACK/BLAS):

```sh
julia +1.13.1 --compiled-modules=existing --project=extern/Julia-mVMC c_toolbox/run_real_fsz_direct_sr_audit.jl /absolute/reference-stage /absolute/NEW-external-stage
# The acquisition records its exact gcc command, including its generated-excerpt include directory.
OPENBLAS_NUM_THREADS=1 /absolute/external-stage/real-sr /absolute/external-stage/input.txt 0
OPENBLAS_NUM_THREADS=1 /absolute/external-stage/real-sr /absolute/external-stage/input.txt 1
```

The acquisition records actual source/compiler/input/executable/linked-library
hashes, both unknown-flag runs, before-sync and after-sync parameters, original
active mapping/matrix/rhs/INFO/increment, and BigFloat256 residual diagnostics.
These are retained-reference-buffer kernel checks, not independently regenerated
C sampling: input OO/HO came from the labelled adapted reference generator.
All commands
are optional developer commands; ordinary Cargo builds/tests never read or
execute toolbox files.

## Executed matched-boundary result

Linux x86_64 host, Julia1.13.1, GCC13.3.0, original C DPOSV linked through
system LP64 LAPACK/BLAS/OpenBLAS; Julia residual diagnostics use its ILP64
OpenBLAS. Actual acquisition15087 terminal0, stdout chunk032677, stage
`/tmp/mvmc-pauli-real-fsz-direct-audit-20261003-d`, reference stage
`/tmp/mvmc-threaded182-real-public-20261003-g`. Reproduce with the command
above using these exact paths and a fresh output destination.

Original C selects20 components:
`6,8,10,12,14,16,20,22,24,26,28,30,32,34,36,38,40,42,44,46`.
Both unknown-imaginary-flag assignments0/1 yield the same active mapping,
matrix, rhs, INFO0, increment, pre-sync parameters and post-sync parameters.
Original metadata differs: fixed/cut counts26/2 versus12/16. This is explicitly
not proof that native unwritten flags are zero or all SR diagnostics agree.

For this actual matrix, Float64 condition2 estimate873.7042302836861 is not a
certified condition bound. BigFloat256 residual infinity norm is9.92033e-20;
normwise backward error1.24506595e-17, componentwise7.16839586e-17.
Matched post-sync C/reference final parameters have observed maximum absolute
difference0.0. An earlier raw post-solve versus reference post-sync comparison
was incomparable (difference0.00475143): original C Slater normalization
resolves it without numerical tolerance or altered solver arithmetic. Earlier
stages `...-a`, `...-b`, `...-c` remain intact; `-b` is the unsynchronized probe.

Persisted actual stdout and full compiler/input/source/library identities:
`docs/reference/c-to-julia/verification/evidence/issue-180-real-fsz-direct-15087-audit.stdout.txt`
and corresponding `-provenance.txt`. External stage additionally retains the
entire assembled operator/rhs/increment and both C outputs with pre/post-sync
parameters. No new tolerance, full-C sampling claim or model-milestone
completion follows from this focused check.
