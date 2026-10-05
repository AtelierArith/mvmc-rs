# Native C PhysCal / Lanczos references (issue #181)

Standalone developer tools. Cargo builds and tests never read, compile or run
anything here; they consume only the checked-in files under
`tests/fixtures/native_c_physcal_181/`.

| File | Purpose |
| --- | --- |
| `build.sh` | Builds `vmc.out` from a copy of `extern/mVMC-1.3.0` (Release, GCC, MPICH). `plain` = unmodified; `dump` applies `state_dump.patch`. |
| `state_dump.patch` | Additive probe: after every PhysCal sample writes `state_dump_NNN.txt` (EleIdx/EleCfg/EleNum/EleProjCnt/EleSpn, `Counter[0..6]`, RNG draw count, next 624 SFMT words). The RNG dump saves/restores state and count. No numerical code changes; `generate.py` asserts every `zvo_*` output of the probe build is byte-identical to the unmodified build. |
| `generate.py` | Runs every row of `scenarios.tsv` (one MPI rank, `OMP_NUM_THREADS=1`) and writes `inputs/`, `zqp_opt.dat`, `expected/`, `native-state/`, `time-rows/`, `provenance.txt`. |
| `lanczos_sensitivity.py` | First-divergence analysis of Lanczos alpha (see the matrix document). |
| `fsz_sensitivity.py` | One-ulp input sensitivity of the FSZ energy (see the matrix document). |

Reproduction, inside the Linux x86_64 Dev Container at the repository root:

```sh
c_toolbox/physcal_native/build.sh /tmp/pn
c_toolbox/physcal_native/build.sh /tmp/pnd dump
python3 c_toolbox/physcal_native/generate.py \
  --vmc /tmp/pn/build/src/mVMC/vmc.out --vmc-dump /tmp/pnd/build/src/mVMC/vmc.out \
  --mvmc-commit "$(git rev-parse HEAD:extern/mVMC-1.3.0)" \
  --out tests/fixtures/native_c_physcal_181
```

`scenarios.tsv` columns: name, source directory (`inputs/` + `zqp_opt.dat`),
Rust `--mode`, C `-o`/Rust `--opt-trans` flag, `;`-separated stages of modpara
`Key=Value` overrides (several stages rerun in one working directory),
deterministic fixed-parameter perturbation amplitude (used where the original
fixed state is an exact eigenstate and Lanczos alpha is singular), class
(`compare`, `c_rejected`, `c_singular`, `c_defect_not_reproduced`) and an
optional variant (`drop=KEYWORD,...` removes namelist keywords; `zqp=c_opt`
takes the fixed parameters from one native-C optimization step; `zero_in=FILE,...` zeroes
overlay values; `ranks=N` runs the measured stage under `mpiexec -n N`). `make_fsz_sources.py`
assembles the FSZ + DH/RBM/OptTrans inputs under `tests/fixtures/native_c_physcal_181/_sources/`.
