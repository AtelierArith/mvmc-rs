# Issue #347: C `vmc.out -b` binary parameter layout

Expected files in `c_expected/` were produced by the **unmodified** authoritative C
`vmc.out` (`extern/mVMC-1.3.0`, no patch, no debug flags) and checked in as data. Rust
tests read only these files; they never build or run C.

| Item | Value |
|---|---|
| Platform | Linux x86_64 Dev Container (Ubuntu 24.04), image `vsc-mvmc-rs-*-uid` |
| Compiler | gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), MPICH `/opt/mpich/bin/mpicc`, Release, CMake `-DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF` |
| `vmc.out` SHA-256 | `629ec3b317d31ed3bbab91f26ae39a4bde8e4a7cdda3c170434ab1f33d99a4a3` |
| `vmcmain.c` SHA-256 | `fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63` |
| `initfile.c` SHA-256 | `8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8` |
| Run environment | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, single MPI process |
| Reproduction | `c_toolbox/issue347_varbin/README.md` |

## Cases

All use the 6-site Heisenberg chain from `tests/fixtures/physcal_181/two-samples/heisenberg_chain_real`.

| File | C command | Notes |
|---|---|---|
| `opt_even_zvo_varbin_001.dat` | `vmc.out -b namelist.def` (NVMCCalMode=0, 4 steps) | NPara=14, header `0e000000 04000000`, 456 bytes |
| `opt_odd_zvo_varbin_001.dat` | same with `namelist_odd.def` (Jastrow family removed) | NPara=13 (odd) |
| `opt_initpara_zvo_varbin_001.dat` | `vmc.out -b namelist.def initpara.dat` | positional initial parameters in mode 0 |
| `physcal_zvo_varbin_{007,008}.dat` | `vmc.out -b namelist.def zqp_opt.dat` (NVMCCalMode=1) | header `(NPara, 1)`; the physcal_181 inputs |
| `physcal_noinit_zvo_varbin_{007,008}.dat` | `vmc.out -b namelist.def` (NVMCCalMode=1, no parameter file) | `InitParameter` draws are the parameters |
| `physcal_noinit_zvo_out_007.dat` | same | energy row for the no-parameter-file PhysCal |

`opt_inputs/` derives from the physcal_181 inputs: `NVMCCalMode 0`, `NSROptItrStep 4`,
`NSROptItrSmp 2`, `NVMCSample 20`, `NDataIdxStart 1`, and the Green-function keywords removed
from `namelist.def` (the Rust optimizer rejects TwoBodyGEx). `initpara.dat` is the physcal_181
`zqp_opt.dat`.

## Layout facts established by these runs

* Optimization writes `zvo_varbin_001.dat` (the `NDataIdxStart` index) and **no** `zvo_var_001.dat`;
  PhysCal writes one `zvo_varbin_NNN.dat` per sample and no `zvo_var_NNN.dat`.
  `zvo_out`, `zvo_time`, `zvo_SRinfo`, `zqp_*` files are unchanged by `-b`.
* Header: native-endian `int NPara`, `int NSROptItrStep` (PhysCal: 1). The optimizer then appends
  one block per step.
* **C defect (not reproduced by Rust):** each block is `fwrite(Para, sizeof(double), NPara, ...)`
  with `double complex *Para`, so a block holds only `NPara` doubles: the first `ceil(NPara/2)`
  parameters as interleaved (re, im), the last one cut to its real part when NPara is odd, and
  the other parameters are never written. A 4-step even run is `8 + 4*14*8 = 456` bytes; PhysCal
  is `8 + 14*8 = 120`; the odd run (NPara=13) is 424 bytes (6.5 parameters per block).
* **Rust (user decision):** same header, but the full `2*NPara`-double block per step. The test
  compares C's `NPara` doubles with the leading `NPara` doubles of Rust's block and checks that
  the Rust block contains every text-mode parameter.

## Comparison policy (docs/NUMERICAL_COMPARISONS.md)

Header bytes: exact. PhysCal blocks are parameters read from a file (or `InitParameter` draws),
not computed results, and C's `NPara` doubles equal the leading `NPara` doubles of Rust's block
exactly.
Optimizer step 0 is identical to C; later steps depend on the SR solve (BLAS operation order)
and differ by about 5e-10 absolute (values are O(1)); the test uses `1e-8` absolute with the
step-0 row required exactly and the RNG-driven control path unchanged (see #358 for the SR
operand analysis).
