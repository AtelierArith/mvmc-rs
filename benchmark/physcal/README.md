# PhysCal benchmark inputs (Hubbard chain)

Fixed-parameter VMCPhysCal (`NVMCCalMode=1`) inputs at the Hubbard-chain
sizes/settings of the C-vs-Julia performance reports. They reuse the
`bench-hubbard` lattice, interaction and correlation declarations and add a
committed fixed-parameter file, so `mvmc-cli --physcal` and Julia's
`run_phys_cal_from_namelist` measure the measurement path rather than
optimization.

## Layout

```
inputs/hubbard_chain_L16/
  StdFace.def      # generator input (provenance of the copied .def files)
  namelist.def     # Expert-mode entry point
  modpara.def      # NVMCCalMode=1, NDataQtySmp=100
  zqp_opt.dat      # fixed parameters (single C-format line)
  *.def            # modpara / trans / coulombintra / green* / correlation factors
```

Models: `hubbard_chain_L16`, `hubbard_chain_L24`, `hubbard_chain_L32`.
Hubbard chain, `Lsub=4`, `U=4.0`, `t=1.0`; half filling (`Ncond=L`, `2Sz=0`);
`NSPGaussLeg=8`, `NSPStot=0`; `NSplitSize=1`, `NStore=1`, `RndSeed=1`;
`NDataQtySmp=100` (the PhysCal sample count; PhysCal ignores
`NSROptItrStep`/`NVMCSample`).

## Provenance

The `.def` files are copied from
`benchmark/hubbard_chain/inputs/hubbard_chain_L*`, whose provenance is the C
StdFace generator (see `benchmark/hubbard_chain/README.md`). Only
`modpara.def` is edited for PhysCal:

- `NVMCCalMode 0 -> 1`
- `NDataQtySmp 1 -> 100`

`StdFace.def` is kept from the optimization input; regenerating it would
recreate the `NVMCCalMode=0` parameters, so the two PhysCal edits above are
required afterwards.

`zqp_opt.dat` is a C-format optimized-parameter line used only as the fixed
input for timing; it is not treated as a numerical expectation. It was
produced by running the corresponding optimization input (whose
`NVMCCalMode=0`) and copying the result:

```sh
target/release/mvmc benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def \
  --nsteps 300 --nsmp 30 --seed 1 --out-dir /tmp/physcal-opt-L16
cp /tmp/physcal-opt-L16/zqp_opt.dat \
  benchmark/physcal/inputs/hubbard_chain_L16/zqp_opt.dat
```

Repeat for `L24`/`L32`.

## Running

```sh
cargo run -p xtask -- bench-physcal-hubbard --reps 3 --warmups 1 --threads 1
```

Single model:

```sh
cargo run -p xtask -- bench-physcal-hubbard --model hubbard_chain_L16 \
  --reps 3 --warmups 1 --threads 1
```

The task writes `target/bench/physcal_hubbard.csv` and
`target/bench/physcal_hubbard_report.md`; new baselines are archived under
`results/`. With `MVMC_C_TIMER=1` and `--keep-output`, each side's
`zvo_CalcTimer.dat` is kept for the section breakdown.

`--julia-root` selects the reference checkout. The Julia section timer needs
the `julia-patch` branch (PR tmisawa/Julia-mVMC#54); the committed reference
`extern/Julia-mVMC` still runs the benchmark without timers.

## Results

Baselines in `results/` were captured on Darwin arm64, rustc 1.99.0, Julia
1.13.1, one thread.
