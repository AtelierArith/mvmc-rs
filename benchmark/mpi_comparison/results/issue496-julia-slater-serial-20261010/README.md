# Serial checked Julia Slater update (#496)

Upstream PR #76 was merged as `02afdae0a19732c727e0d09132d9761c57d68fcb`. Its source tree is identical to the measured `043ec52` tree (`git diff 043ec52 02afdae` is empty). Updating the reference pin does not relabel or regenerate these measurements. Linux Julia 1.12/1.13 CI was still running at merge time; the locally installed production suite results below are separate evidence.

Julia source `043ec52b9c9a1542083c233bd1923fe24f36d03c`, upstream [julia-patch PR #76](https://github.com/tmisawa/Julia-mVMC/pull/76), based on merged PR #75 (`7b7f63a`, identical tree to b672). Linux x86_64 Dev Container, Ryzen 9 PRO 8945HS, Julia 1.13.1, MPICH4.2, MPI4 × compute-thread4, BLAS1. Julia LinearAlgebra OpenBLAS0.3.30 ILP64, native PfaPack and C OpenBLAS0.3.26 LP64. Frozen versioned benchmark Manifest and actual package/native-library hashes are recorded in provenance.json.

Hubbard periodic half-filled chains, t=1,U=4,Lsub=4,NSPGaussLeg=8,NMPTrans normalized to1,NQPOptTrans1: eight QP planes total; sampling two per rank, measurement all eight for each rank's local configurations. Opt300 with averaging window300, 300 saved samples/step (75/rank), full warmup and three measurements.

The typed serial kernel validates mappings/cache shapes once and then evaluates the original four complex C-order spin-block expressions. Original builders and duplicate nonzero selection remain in use. Unsupported/partial inputs and Debug logging use the retained reference loop. This is a combined function-barrier, repeated-check/branch and field-access improvement, not solely bounds-check elimination.

## Same-MPI alternating Julia comparison

| Sites | Previous Julia median (s) | Candidate median (s) | Reduction |
|---|---:|---:|---:|
| 32 | 3.588680265 | 3.355968996 | 6.49% |
| 64 | 13.485768732 | 12.586608366 | 6.67% |

Both variants were installed once before warming, with typed Ref selection and no timed eval/method invalidation. Every candidate pair was faster and all 300×6×3 output values per size matched the baseline with observed max absolute difference zero. These are diagnostic Julia-vs-Julia comparisons; the following confirmation runs use unmodified installed production code.

## Fresh native C / installed Julia production comparison

| Sites | C median (s) | Julia median (s) | Reduction |
|---|---:|---:|---:|
| 32 | 3.424090 | 3.385803881 | 1.12% |
| 64 | 13.579140 | 12.802833085 | 5.72% |

C and Julia batches alternate forward/reverse, with one discarded C warmup and Julia full warmup per fresh batch. L64 Julia beats C in all three pairs. L32 Julia is slower in two individual pairs despite the lower median, so this does not yet establish a robust L32 performance advantage. Further profiling/tuning is required; issue #496 remains open. C rank-zero internal All and Julia warmed maximum-rank production API include different timing boundaries. Startup, JIT and warmups are excluded.

Installed production tests: 41,339 unit tests (including33 new assertions), 25 Slater tests and15 base tests pass with unchanged numerical budgets. Independent analytic complex four-block expectations and actual-dispatch failure/partial-cache/Debug/mirror tests are in the upstream commit. `audit-summary.json` records exact native SFMT624/index/count/reseed and saved/tmp/burn-in configurations/counters in all16 size×20/300×rank cases versus reviewed b672. Timing/output agreement alone is not C numerical/RNG parity proof.

Raw suite/component/worker logs and C timers are retained unchanged in `raw-evidence.tar.gz`. Full diagnostic outputs and profiles remain under `/home/vscode/.cache/mvmc/issue496-julia-slater-serial-alternating`; full paired outputs under `/home/vscode/.cache/mvmc/issue496-julia-slater-fresh-c-pair-root`. `fresh-c-pair/reproduce.py` is the exact executed uv utility; its cache/source/project arguments and existing C inputs are recorded in provenance/input hashes. Optional imported helper snapshots are in `fresh-c-pair/support/`. Primary source/project/native binaries and inputs are hashed before and after measurement.

## Updated stage profile

A separate full300 CTimer/Profile run keeps the installed043ec52 source, 4×4 workers and BLAS1; MAINCAL/SLATER/WEIGHTAVG diagnostics are enabled and CALHAM1 diagnostics are disabled to preserve the production Hamiltonian path. These are instrumented observations, not primary timings. Timer stages may nest and include waits. `profile-summary.json` and per-rank CTimer files retain measured stages.

L32 Slater update20 falls from the preceding owned/typed profile0.18927s to0.02354s; current output22 costs0.08733s and parameter sync23 costs0.03096s. L64 current Slater update20 is0.08865s, output22 is0.34822s and sync23 is0.10674s. Repeated parameter packing in output/sync/SR preprocessing is the next measured investigation. Full sampling profiles remain in `/home/vscode/.cache/mvmc/issue496-profile/slater-serial-final-L{32,64}-full300/`.
