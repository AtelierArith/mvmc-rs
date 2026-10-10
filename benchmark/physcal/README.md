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
`NDataQtySmp=100` (the number of measurement groups). Each group calls the
sampler, which uses `NVMCSample` to determine the number of saved configurations;
`NSROptItrStep` does not control PhysCal. See C `vmcmain.c`'s `VMCPhysCal` loop
and `vmcmake.c`'s `nOutStep`/saved-sample bounds.

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

## Observable comparison

`bench-physcal` and `bench-physcal-hubbard` also compare the measured PhysCal
observables, not only the energy. For the last measured repetition they walk the
Rust and Julia output directories, match files by name and compare them token by
token:

- families: `zvo_cisajs`, `zvo_cisajscktalt`, `zvo_cisajscktaltex`, and, when
  `NLanczosMode > 0`, `zvo_ls_out`, `zvo_ls_qqqq`, `zvo_ls_cisajs`,
  `zvo_ls_cisajscktalt`, `zvo_ls_cisajscktaltex`;
- integer/discrete columns must match exactly; floating-point values are checked
  componentwise against the documented PhysCal bounds (absolute `1e-10`,
  relative `1e-9`);
- a row/column count or discrete-token difference aborts the benchmark.

`bench-physcal` prints the per-family `files`, `values`, `max|Δ|`, `max_rel` and
`status`; `bench-physcal-hubbard` adds the same table to its Markdown report.

`--julia-root` selects the reference checkout. The committed reference
`extern/Julia-mVMC` pins upstream `main` including the section timer changes
merged in [Julia PR #54](https://github.com/tmisawa/Julia-mVMC/pull/54).
A separate `julia-patch` checkout is no longer required for section timings.

## Results

Baselines in `results/` were captured on Darwin arm64, rustc 1.99.0, Julia
1.13.1, one thread.

`physcal_hubbard_2026-10-06.md` (Linux x86_64, system OpenBLAS) is the
Rust-slower baseline and `physcal_hubbard_2026-10-06_green_gap.md` is the same
benchmark after the issue #442 fix. After the fix Rust is faster at L16 (1.12x)
and L24 (1.06x) and within ~2% at L32; `|ΔE|=0` and every observable family
matched.

`physcal_hubbard_2026-10-06_c_order.md` records the issue #449 change (the two-hop kernel, the
complex Pfaffian refresh and the Gram follow the C operation order instead of Julia's
FMA/lane-split order): no slowdown, observables now differ from Julia at roundoff level
(max|diff| <= 2.5e-13) because the Julia reduction order is no longer reproduced.

Reproduce both tasks (four `physcal_ref` fixtures and the Hubbard L16/L24/L32
chain) with:

```sh
scripts/run_physcal_benchmark.sh [reps] [warmups] [threads]
```
