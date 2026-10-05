# Rank-wise native-C MPI optimization matrix (issue #179)

Output of the instrumented native `vmc.out` (stock `extern/mVMC-1.3.0` plus the two
read-only patches listed in `PROVENANCE.txt`) for the cells in `cells.txt`. One
optimization step per cell (`NSROptItrStep = NSROptItrSmp = 1`), seed from the input,
no initial-parameter file. Regenerate (explicit developer command, Dev Container):

```sh
c_toolbox/mpi_matrix_179/build.sh /tmp/mvmc-179-c
SOURCE_COMMIT=$(git -C extern/mVMC-1.3.0 rev-parse HEAD) \
  python3 c_toolbox/mpi_matrix_179/generate.py --vmc /tmp/mvmc-179-c/build/src/mVMC/vmc.out \
  --work /tmp/mvmc-179-runs --out tests/fixtures/mpi_matrix_179
```

Per cell `<model>-<solver>-r<world>w<NSplitSize>-s<samples>/`:

* `rank<r>.txt`: this rank's `Counter[0..6]` after `ReduceCounter`, every saved
  `EleIdx`, and the full SFMT state (`psfmt32[624]`, cursor) after the step.
* `operands.txt`: rank 0's reduced `SROptHO`, `SROptO` (rank-local last-sample
  derivative, not reduced by C), `SROptOO` in `%.17e` (real and imaginary parts) and
  the step energy.

Models: `real`, `cmp` (Heisenberg chain), `fsz1` / `fsz2` (FSZ, `NQPFull` 1 / 2),
`ot` (Hubbard chain with DH and OptTrans, `NQPOptTrans = 3`, `-o`). Solvers: `d0`
(direct, `NStore 0`), `d1` (direct, `NStore 1`), `cg` (`NSRCG 1`). Worlds 2 and 4,
widths 1, 2 and the world size; `s7` cells use 7 saved samples (uneven partitions),
`s3` cells use 3 samples with 4 ranks (empty measurement work) and `fsz1` with
width 4 (three ranks own an empty QP range). `c_sr_error = 1` marks the two cells
where C itself fails the SR solve (3 samples, one group).

## Reproducibility and the C defect

Two independent generations are byte-identical except `<OO>` of the grouped
SR-CG cells, which C builds from unwritten `malloc` memory (`vmccal.c:314-318`,
tmisawa/Julia-mVMC#61): the first entries held denormal garbage (`8.96e-311` versus
`9.48e-311`). Those `<OO>` values are therefore not stored. Grouped stored-O
(`d1`, width > 1) `<OO>` is deterministic here but loses the samples of the ranks
after the first (calloc'd zeros on this build); `mpi_matrix_179` asserts the Rust
value against Rust grouped direct instead.

## Tolerances and what is exact

Exact: counters, every saved `EleIdx`, the generator state (words and cursor), per
rank. Computed values (`<HO>`, `<OO>`, `<O>` on rank 0, energy): `|a-b| <= 1e-13 +
1e-12 |b|`. Observed serial worst case is about 2 ulp (last-bit summation order,
`docs/NUMERICAL_COMPARISONS.md`); the absolute term covers exact-zero entries that
hold roundoff.
