# Native C step-1 SR operands (#358)

Step-1 sampled operands from the real C `vmc.out`, used by
`callback_tests::assert_c_step_one_operands`. Rust tests only read these
files; they never run C or the toolbox.

Each `*.txt` has `energy`, `ho`, `o`, and `oo` (CG: `<O>` then `<OO>` diagonal,
2*SROptSize; direct NStore=0: SROptSize^2 matrix) or `o_store` (direct
NStore=1), as big-endian IEEE-754 hex doubles printed by C with `%.17e` and
converted exactly. Model/mode per file name: `<case>-<cg|direct>-store<N>`.

- Upstream: `extern/mVMC-1.3.0` at d73d06bd529d3b2573f38eb5817c4a5f52971006.
  `src/mVMC/vmcmain.c` SHA-256
  `fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`;
  `stcopt_cg_impl.c` SHA-256
  `41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9`.
- Instrumentation: `c_toolbox/sr_operand_dump/dump_sr_operands.patch`
  (SHA-256 `07b2c380ff49005426db04bf9483fe41a31f474c37e050013079cf15820cf500`)
  changes only the number formats of upstream's own `_DEBUG_DUMP_SROPTOO` and
  `_DEBUG_DUMP_SROPTO_STORE` dumps to `%.17e`, declares their loop index and
  guards the store dump by `NStoreO != 0`. No numerical code is touched.
- Build: `c_toolbox/sr_operand_dump/build.sh` (CMake Release, GCC/GNU Fortran
  13.3.0 Ubuntu 13.3.0-6ubuntu2~24.04.1, CMake 3.28.3, MPICH 4.2.0,
  `-DCMAKE_C_FLAGS="-D_DEBUG_DUMP_SROPTOO -D_DEBUG_DUMP_SROPTO_STORE"`, bundled
  OpenBLAS 0.3.26 via the system `libopenblas.so.0`, BLIS downloaded by upstream
  CMake). Executable SHA-256
  `0c78bde585e850e4d8ca04447370ce630cc3aeb8d146b2411fddded0a103ee04`.
- Environment: Linux x86_64 Dev Container image
  `sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`
  (`vsc-mvmc-rs-...-uid`), one MPI rank, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`.
- Inputs: the same namelists as the Rust tests (`heisenberg_chain_real`,
  `hubbard_chain_real`, `tests/fixtures/dh2/production_real`,
  `dh4/production_dh4_real`, `dh4/production_dh24_real`,
  `hubbard_chain_pairhop_real`, `opttrans/run_opt_real` with `-o`), seed 1,
  `NSROptItrStep=NSROptItrSmp=1`, NSRCG/NStore as in the file name.
- Command (from the repository root in the Dev Container):
  `c_toolbox/sr_operand_dump/build.sh /tmp/mvmc-sr-dump` then
  `python3 c_toolbox/sr_operand_dump/capture.py --vmc /tmp/mvmc-sr-dump/build/src/mVMC/vmc.out --out tests/fixtures/c_order_sr_operands`.

Scope: standalone native single-rank operand checks, not C sampling or CG
trajectory parity. Observed Rust-to-C agreement is at most 6e-16 scaled
(`docs/NUMERICAL_COMPARISONS.md`, "C-order real kernels and CG amplification").
The `opt_real` file's last two entries per array (C real-mode OptTrans slot
quirk) are not compared. `rbm_real` has no file (see the same section).
