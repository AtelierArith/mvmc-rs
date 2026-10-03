# Hubbard-chain Rust vs Julia benchmark inputs

Expert-mode inputs for a repeatable Rust-vs-Julia speed comparison on the
parameters used by the C-vs-Julia performance reports. They mirror the
`Hubbard chain` setup in
`docs/reference/c-to-julia/performance/2026-06-17-julia-mvmc-hubbard-locenergy-slater-calham1-optimization-record.md`
and the roadmap `docs/reference/c-to-julia/roadmaps/2026-06-14-julia-mvmc-v0.4-summary-and-v0.5-roadmap.md`:

- Hubbard chain, `L=16`, `L=24`, `L=32`
- `Lsub=4`, `U=4.0`, `t=1.0`
- half filling: `Ncond=L`, `2Sz=0`
- `NSPGaussLeg=8`, `NSPStot=0`
- `NSplitSize=1`, `NStore=1`, `NSRCG=0`, `RndSeed=1`
- `NSROptItrStep=300`, `NVMCSample=300`

The reports run `R=4` MPI ranks, which partitions sampling and performs
collective reductions, so their timing is not reproducible by a single-rank run.
This benchmark is a **separate serial (`R=1`), one-thread** measurement: it does
not reproduce the report's `R=4` timing and is not its per-rank condition. The
Rust `mvmc-cli` has an `mpi` feature, but this task does not build or launch an
MPI reducer; expanding to `R=4` is future work. `NVMCSample` is not stated in
the report, so it is set equal to the 300 SR steps.

## Layout

```
inputs/hubbard_chain_L16/
  StdFace.def      # generator input (provenance)
  namelist.def     # Expert-mode entry point
  *.def            # modpara / trans / coulombintra / green* / correlation factors
```

## Provenance

Generated with the C StdFace input generator shipped in
`extern/mVMC-1.3.0/src/StdFace` (`mVMC-1.3.0`). The generator was built
standalone; only `-lm` is required:

```sh
clang -O2 -D_mVMC -DMEXP=19937 \
  -I. -I../../common -I../../sfmt \
  StdFace_main.c StdFace_ModelUtil.c ChainLattice.c SquareLattice.c \
  TriangularLattice.c HoneycombLattice.c Ladder.c Kagome.c Orthorhombic.c \
  Pyrochlore.c Wannier90.c FCOrtho.c setmemory.c export_wannier90.c dry.c \
  -lm -o mvmc_dry.out
```

Then, from inside each `inputs/hubbard_chain_L*/` directory:

```sh
mvmc_dry.out StdFace.def
```

This produces the committed `*.def` files (only `namelist.def`, `modpara.def`,
`locspn.def`, `trans.def`, `coulombintra.def`, `greenone.def`, `greentwo.def`,
`gutzwilleridx.def`, `jastrowidx.def`, `orbitalidx.def`, `qptransidx.def` are
kept; `geometry.dat` and `lattice.gp` are dropped). Regeneration is an optional
developer step: the committed files are self-contained and the benchmark runs
without rebuilding StdFace or C.

## Running

```sh
cargo run -p xtask -- bench-hubbard --steps 300 --reps 3 --warmups 1 --threads 1
```

## Results

Warmup-excluded medians over 3 repetitions, internal
`run_para_opt_from_namelist` wall clock, one thread (`Darwin arm64`, Julia
1.13.1, Rust OpenBLAS, Julia `lbt` -> `libopenblas64_`). `speedup = julia / rust`;
this is the serial `R=1`, one-thread condition, not the report's `R=4`.

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | \|ΔE\| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 4.183 | 3.069 | 0.73x | 0.00e0 |
| hubbard_chain_L24 | 8.881 | 6.488 | 0.73x | 0.00e0 |
| hubbard_chain_L32 | 16.254 | 12.228 | 0.75x | 0.00e0 |

Initial baselines are archived under `results/`; the task writes new CSV and
Markdown reports to `target/bench/` for comparison. The section-timer
breakdown that localizes the gap to `VMCMainCal`
(`CalHamiltonian1` / `ReturnSlaterElmDiff`) is in
`results/hubbard_chain_2026-10-03_sections.md` (tracked as issue #207).

