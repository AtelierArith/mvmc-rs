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

The reports run `R=4` MPI ranks; the Rust port has no MPI support, so this
benchmark uses `R=1` (serial) and one thread, which corresponds to the report's
per-rank, one-thread condition. `NVMCSample` is not stated in the report, so it
is set equal to the 300 SR steps.

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
1.13.1). `speedup = julia / rust`; the Rust port does not yet have an MPI path,
so this compares the report's single-rank, one-thread condition.

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | \|ΔE\| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 4.210 | 3.102 | 0.74x | 0.00e0 |
| hubbard_chain_L24 | 9.040 | 6.539 | 0.72x | 0.00e0 |
| hubbard_chain_L32 | 16.190 | 11.966 | 0.74x | 0.00e0 |

Initial baselines are archived under `results/`; the task writes new CSV and
Markdown reports to `target/bench/` for comparison. The section-timer
breakdown that localizes the gap to `VMCMainCal`
(`CalHamiltonian1` / `ReturnSlaterElmDiff`) is in
`results/hubbard_chain_2026-10-03_sections.md` (tracked as issue #207).

