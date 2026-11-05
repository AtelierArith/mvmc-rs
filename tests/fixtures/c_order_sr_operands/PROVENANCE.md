# Native C step-1 SR operands (#358)

Step-1 sampled operands from the real C 1.4.0 `vmc.out`, used by
`crates/mvmc-core/tests/c_sr_operands_358.rs`. Rust tests only read these
files; they never run C or the toolbox.

Each `*.txt` has `energy`, `ho`, `o`, and `oo` (CG: `<O>` then `<OO>` diagonal,
2*SROptSize; direct NStore=0: SROptSize^2 matrix) or `o_store` (direct
NStore=1), as big-endian IEEE-754 hex doubles printed by C with `%.17e` and
converted exactly. Model/mode per file name: `<case>-<cg|direct>-store<N>`.
Regenerated from mVMC 1.4.0 for #695; the former 1.3.0 records were replaced.

- Upstream: `extern/mVMC-1.4.0` at e83aa6cb1e37bfc4876848787ed2f61721a7e972.
  `src/mVMC/vmcmain.c` SHA-256
  `da14885da14fd5b98feb27f7c53efd6fa229a4a33d27c30fe44848e6514394f0`;
  `vmccal.c` SHA-256
  `c06d5eef360ee3e442d1843ec218f0beb34fc7813e5c6de46768b4d6999055d6`;
  `stcopt_cg_impl.c` SHA-256
  `84a1bb1867a957f25d8a822f67213ff044bf52448d96263c2efdf085bb815dd5`.
- Instrumentation: `c_toolbox/sr_operand_dump/dump_sr_operands.patch`
  (SHA-256 `07b2c380ff49005426db04bf9483fe41a31f474c37e050013079cf15820cf500`)
  changes only the number formats of upstream's own `_DEBUG_DUMP_SROPTOO` and
  `_DEBUG_DUMP_SROPTO_STORE` dumps to `%.17e`, declares their loop index and
  guards the store dump by `NStoreO != 0`. No numerical code is touched.
- Build: `c_toolbox/sr_operand_dump/build.sh` (CMake 3.28.3 Release, GCC/GNU
  Fortran 13.3.0 Ubuntu 13.3.0-6ubuntu2~24.04.1, MPICH 4.2.0,
  `-DCMAKE_C_FLAGS="-D_DEBUG_DUMP_SROPTOO -D_DEBUG_DUMP_SROPTO_STORE"`, system
  OpenBLAS 0.3.26 `libopenblas.so.0`, bundled BLIS). Executable SHA-256
  `65661c43adaa0e7dcaeed1a03e041833433b49ff17eb4067947d5ee828b0c334`.
- Environment: Linux x86_64 Dev Container image
  `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`
  (`vsc-mvmc-rs-...`), one MPI rank, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`.
- Inputs: `inputs/<case>/`, copied unchanged (namelist paths flattened to the
  basenames) from the sources in `inputs/SOURCES.txt`: `heisenberg_chain_real`,
  `hubbard_chain_real`, `tests/fixtures/dh2/production_real`,
  `dh4/production_dh4_real`, `dh4/production_dh24_real`,
  `hubbard_chain_pairhop_real`, `opttrans/run_opt_real` (run with `-o`).
  Modpara overrides: `RndSeed 1`, `NSROptItrStep=NSROptItrSmp=1`,
  NSRCG/NStore as in the file name and `NSRCGAbortOnFail 0`. The operands are
  printed before the SR solve; `NSRCGAbortOnFail 0` (a 1.4.0 option) only lets
  the run finish where the step-1 CG solve meets a 1.4.0 breakdown guard, so
  that `zvo_out_001.dat` (the energy) is written. The Rust test applies the
  same overrides.
- Command (from the repository root in the Dev Container):
  `c_toolbox/sr_operand_dump/build.sh /tmp/mvmc-sr-dump` then
  `PATH=/opt/mpich/bin:$PATH python3 c_toolbox/sr_operand_dump/capture.py --vmc /tmp/mvmc-sr-dump/build/src/mVMC/vmc.out --out tests/fixtures/c_order_sr_operands`.

Scope: standalone native single-rank operand checks, not C sampling or CG
trajectory parity. Observed Rust-to-C agreement on Linux x86_64 against these
1.4.0 records: at most 8.6e-16 scaled (`|d| / (ABS/REL + |c|)`, ABS = 1e-14,
REL = 1e-13; `docs/NUMERICAL_COMPARISONS.md`, "C-order real kernels and CG
amplification"). The `opt_real` file's last two entries per array (C
real-mode OptTrans slot defect, #370) are not compared. `rbm_real` has no file:
C ignores RBM in real mode (#669).
