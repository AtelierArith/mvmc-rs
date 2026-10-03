# Issue 179: fixed published-Julia operands, authoritative C MPI CG

This optional developer probe is not full C sampling parity and is never invoked
by Cargo. `mpi_issue179_cg_replay.c` reads independent Julia operands exported by
`scripts/mpi_issue179_export_cg_operands.py`; no Rust-generated expectations are
used. The comparer is `scripts/mpi_issue179_compare_cg_replay.py`; the independent
residual reader is `mpi_issue179_cg_replay_residual.jl`.

## Retained execution

Container `73c57e563c61`, image
`sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`,
MPICH 4.2.0 internal Hydra PMI, GCC 13.3, LP64 OpenBLAS 0.3.26.
Evidence root inside that container:
`/home/vscode/.cache/mvmc/issue179-C-fixed62b.DWAPKq`.
`results.tsv`, `comparison-results.tsv`, and `residual-results.tsv` each contain
five rows with exit 0: real solves 0/1/2 and complex solves 0/1. Every solve uses
four ranks, width 1, three local samples, global weight 12, shift 1e-5, tolerance
1e-10, default maximum iterations n (real n=10, complex n=20). C launches were
bounded by `timeout --kill-after=5s 30s`. Residual handle 54955 terminated 0.
The source/binary manifest was independently rechecked successfully after capture.

Per-solve directories `real-step0` through `real-step2` and `cmp-step0` through
`cmp-step1` retain rank records, launch logs, comparison JSON and residual stdout.
`real-inputs` and `cmp-inputs` retain lossless hexadecimal operands and provenance
hashes for the original independent Julia records. Source is frozen under
`source/`; `source-binary-sha256.txt` records the actual inputs and executables.

The Julia operands came from published PR54 head
`62b0f97f076fb55c71c3ab0caa041a9adff94e04`, optimizer SHA-256
`b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3`.
The original capture root is
`/home/vscode/.cache/mvmc/issue179-62b-six.BQxPjU`.

## Numerical source and observation boundaries

GPL upstream extracts remain unmodified: `ctest_cg_main_upstream.inc` contains
the license and Main/operator from `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c`
(lines 1–21 and 255–426); `ctest_cg_dot_upstream.inc` contains the dot helper
from `stcopt_cg.c` lines 26–34. Their SHA-256 values are respectively
`9f941cc5f9208eed0efc80937f292837c48b7bc7d078f467715ca001d9e9f43f`
and `f36c3f7d0aa0fc22e7e3cbffde3c95b79808112979e1e0ebc8fdd853d995d214`.
The original `safempi.c` supplies collective chunking; SHA-256
`45ae95e699aad61e42fe3c254194bc0336e36325b30ab5d6385fbbcaa1682440`.
See `ctest_cg_refresh.md` for extraction provenance and licensing.

Compilation uses `mpicc -std=c11 -O0 -ffp-contract=off -Wall -Wextra
-Wno-unknown-pragmas`, the upstream include directory, and `-lopenblas -lm`;
the real variant additionally defines `MVMC_SRCG_REAL`. No OpenMP is enabled.
MPI observation wrappers call actual PMPI broadcast/allreduce; the original
barrier and numerical bodies remain intact. Original debug printing exposes
actual delta, threshold, x/r/d without replacing arithmetic or decisions.

Executed driver SHA-256:
`4d4375a4fe711709e244b57a7ddb7535dabeec9d4fc572b15e718dc91c01f55a`.
Real and complex binary SHA-256:
`204f30e7386fb043a759128e1bfc427b70382fcb6a10794382314ca29d197cf7`,
`b73f78345e1836e4a7ca2e576119c5e93ec1f91f8fd412d3a3df453ae7a8dd50`.
Comparer SHA-256:
`bd5163fd7962b6343ade682e57eeaa2ce87ca01786932b51c884a239b349e812`.
Executed residual reader SHA-256:
`fc55a92640d63b1393cab6ec3f2469751b4e7348ba403d9c8b9d1a5393e5bd42`.

## Scope and numerical interpretation

Only the first solve has common prepared operands in C, Rust and Julia. On that
solve, all observed C/Rust operator phases and final x/r/d agree numerically;
this incidental equality is diagnostic, not a required bitwise acceptance gate.
Later Rust prepared operands differ: the comparer explicitly reports
`NONCOMMON_PREPARED_OPERANDS_NO_SOLVER_COMPARISON`. Later C/Julia fixed-operand
diagnostics do not extend the first-common-solve Rust coverage.

The first C/Julia difference is GLOBAL product 1, following equal LOCAL products:
real index 2 differs by 1.0842021724855044e-19; complex index 15 by
5.421010862427522e-20. The two four-rank MPI runtimes differ. A boundary-only
rounding model `2*gamma_3*sum(abs(local products))`, with u=2^-53 and
gamma_3=3u/(1-3u), gives 1.4077007058607085e-18 and
1.84890167280025e-19 respectively, assuming normal finite additions without
overflow/underflow. This is not a CG forward-error or trajectory tolerance.

The residual audit reconstructs the actual system with 256-bit BigFloat and
evaluates actual C solutions independently. First real solve: condition estimate
9.3545e6, backward error 1.5750e-8; first complex solve: 1.0470e7 and 4.8263e-8.
Condition estimates are not certified bounds. Real solve 1 reaches its 10-iteration
limit with backward error 2.7619e-7; this is not a convergence claim. No arbitrary
downstream forward tolerance is adopted. Iteration counts are diagnostics, not an
exact cross-language gate (real solve 2 returns C=6, Julia=7).

First-solve residual stdout hashes are
`a6714a5ffeb35eb5f88c9f8b8c2ead5521cd74ace8ac7633a169940f2c231edb`
and `48dd9bf32baae457b13aa269f434aca8701982f8b3fef78115e411acbc26d6fe`.
