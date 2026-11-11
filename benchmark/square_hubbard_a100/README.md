# Square-lattice Hubbard inputs for the A100 benchmark

This directory holds 14 Expert-mode inputs of the half-filled square-lattice
Hubbard model with fixed optimized parameters. Each input has a C-mVMC
reference measurement. Use them to check the resident GPU PhysCal (and the CPU
runner) against C, and to time both on the same workload. The benchmark round is
tracked in the "A100 benchmark round 1" issue.

## Cases

All cases use `t = 1` and half filling (`Ncond = Nsite`, `2Sz = 0`). The
wave function has real antiparallel orbitals on a 2×2 sublattice, with
Gutzwiller and Jastrow factors. Every case uses `NExUpdatePath = 1`.

- **NP**: no projection, `NSPGaussLeg = 1`, `NMPTrans = 1` (P/P) or `-1` (AP/P).
  One QP plane.
- **SP**: K = (0,0) translation projection over the four 2×2 coset translations,
  plus S = 0 spin projection. `NSPGaussLeg = 8`, `NSPStot = 0`,
  `NMPTrans = 4` (P/P) or `-4` (AP/P). 32 QP planes.
- **AP/P**: antiperiodic in x and periodic in y (varbench "PA").

| Case | E/N (C reference) | Other reference |
|---|---|---|
| `pp_L4_U4_NP` | −0.833625(70) | 4×4 ED −0.851366 |
| `pp_L4_U4_SP` | −0.849956(39) | 4×4 ED −0.851366 |
| `pp_L4_U8_NP` | −0.49935(18) | 4×4 ED −0.529305 |
| `pp_L4_U8_SP` | −0.52582(10) | 4×4 ED −0.529305 |
| `pp_L8_U4_NP` | −0.84911(11) | AFQMC −0.860359(63) |
| `pp_L8_U4_SP` | −0.854490(50) | AFQMC −0.860359(63) |
| `pp_L8_U8_NP` | −0.50580(11) | AFQMC −0.525656(78) |
| `pp_L8_U8_SP` | −0.51335(15) | AFQMC −0.525656(78) |
| `pp_L16_U4_NP` | −0.850079(50) | — |
| `pp_L16_U4_SP` | −0.85115(15), one high-variance bin | — |
| `ap_p_L8_U4_NP` | −0.851515(52) | — |
| `ap_p_L8_U4_SP` | −0.857538(44) | — |
| `ap_p_L16_U8_NP` | −0.50499(18), bin 1 not relaxed | AFQMC −0.524414(117) |
| `ap_p_L16_U8_SP` | −0.507444(107) | AFQMC −0.524414(117) |

The E/N values are 10-bin means with the standard error between bins. AFQMC is
the varbench total energy (Qin, Shi and Zhang, PRB 94, 085103 (2016)) divided by
`Nsite`. VMC energies are variational, so they lie above ED and AFQMC.
`cases.json` holds the full numbers:

- double occupancy D/N and S(π,π)/N, with standard errors;
- notes on the bins;
- the exact sampling layout of each reference.

## Reference measurements

Each reference is a fixed-parameter C-mVMC measurement (`NVMCCalMode = 1`) with
the `zqp_opt.dat` in this directory. The solver is a C-mVMC development build
(commit `d4430b9d`).

- Fugaku (A64FX), 1 node, 48 MPI ranks × 1 OpenMP thread, `NSplitSize = 1`.
  This gives 48 independent chains and `48 × NVMCSample = 48,000` samples per
  bin.
- Exception: `pp_L8_U4_NP` and `pp_L8_U4_SP` ran on an Intel x86_64 Linux
  cluster with 24 ranks, so 24,000 samples per bin.
- All cases: 10 bins (`NDataQtySmp = 10`), `NVMCWarmUp = 100`,
  `NVMCInterval = 1`.
- `ap_p_L16_U8_SP` combines 8 + 2 bins from two jobs with the same parameters.

Conventions:

- D/N is `(1/N) Σ_i <n_i↑ n_i↓>`.
- S(q) is `(1/N) Σ_ij e^{iq·(r_i−r_j)} <S_i·S_j>`, with the full spin and not
  connected. `cases.json` reports S(π,π)/N.
- Sites are numbered x-fastest: `site = x + L*y`.

Every committed file is the executed input, byte for byte. Its SHA-256 matches
the reference run's result manifest. The parameter file is stored as
`zqp_opt.dat`.

## Generated files

`greenone.def` and `greentwo.def` are not committed: for `L = 8` with all site
pairs they exceed 1 MB. `generate_green_defs.py` writes them, and `cases.json`
records the SHA-256 that the reference run used.

- `measurement_origins = "all"`: every ordered site pair (`6N² − 2N`
  two-body rows).
- `measurement_origins = "sublattice"`: full rows only for the 2×2 unit-cell
  origins `{0, 1, L, L+1}`.

The requests are:

- one-body `c†_{is} c_{js}`;
- density `n_{is} n_{jt}`;
- transverse spin `c†_{i↑} c_{j↑} c†_{j↓} c_{i↓}` and its spin flip, for `i ≠ j`.

## Preparing a run

```bash
uv run --no-project python benchmark/square_hubbard_a100/prepare_run.py --list
uv run --no-project python benchmark/square_hubbard_a100/prepare_run.py \
    pp_L8_U4_SP /path/to/new-run --walkers 4800 --windows 10 --seed 79
```

`prepare_run.py` does the following:

1. Copies the case into a new directory and generates the two Green-function
   definitions.
2. Verifies every file against `cases.json`.
3. Optionally replaces the sampling keys of `modpara.def` (`NVMCSample`,
   `NDataQtySmp`, `NVMCWarmUp`, `NVMCInterval`, `RndSeed`).
4. Writes `run_manifest.json` with the changes and the final hashes.

It never changes physical parameters. Then run:

```bash
mvmc-cuda resident-physcal namelist.def zqp_opt.dat <new-output-dir>   # resident GPU
mvmc namelist.def zqp_opt.dat                                          # CPU runner
```

The resident GPU PhysCal takes its walker count from `NVMCSample` and its window
count from `NDataQtySmp`. One window advances every walker by the interval and
measures one configuration per walker. Matching one C bin therefore needs
`walkers × windows = 48,000` samples (24,000 for the two `pp_L8_U4` cases).

The resident GPU currently returns only the mean over all windows. Until
per-window output exists, treat independent runs with different seeds as bins.

## Comparison

The GPU and C use different random number generators (cuRAND XORWOW versus
SFMT), so their results agree only statistically. The benchmark criterion is
`|ΔE/N| ≤ 3σ` with `σ = sqrt(SE_GPU² + SE_C²)`, and the same for D/N. A
disagreement must reproduce with a second seed. This check does not replace the
deterministic Julia GPU fixtures of `gpu/mvmc-gpu-cuda`.

Resident GPU coverage, as of `fae8eb14`:

- Real normal orbitals, Gutzwiller/Jastrow, QP projection and
  `NExUpdatePath = 1` are supported.
- `NMPTrans < 0` is normalized to its absolute value. The antiperiodic signs
  come from `qptransidx.def`.
- No resident fixture has the AP/P square geometry or `NMPTrans = 4` with
  `NSPGaussLeg = 8`. These cases are new coverage.
