# Issue #181: serial PhysCal and Lanczos scenario matrix against native C

Status vocabulary: **Pass** = compared with the native C `vmc.out` (exact for
discrete data, toleranced for computed values), **Rejected** = C (and Rust)
reject the input, **NotRun** = no native-C evidence in this slice, **Excluded**
= owned by another issue (InterAll). Julia fixtures under
`tests/fixtures/physcal_181/` remain independent Julia 1.13.1 references; this
document adds the previously missing native-C layer.

## Baseline and provenance

* Rust: branch `feat/issue181-physcal-lanczos-matrix` on main `376afa2e` (#180 ctest gate).
* C: vendored mVMC 1.3.0, `extern/mVMC-1.3.0` submodule commit
  `d73d06bd529d3b2573f38eb5817c4a5f52971006`; GCC 13.3.0, Release, MPICH, Linux
  x86_64 Dev Container, 1 rank, `OMP_NUM_THREADS=1`. `vmc.out` SHA-256
  `48cf78a6ce464bc97ea771052892bd1ef02002870181d5ce91e25fae7bfe034b`
  (unmodified). Probe build (`state_dump.patch`, additive printing only, outputs
  verified byte-identical for every scenario) SHA-256
  `0e6d1f9c9df1a39ddb587c95f2dda859600049201c6187afb891151a624e2363`.
* Fixed parameters are the checked-in `zqp_opt.dat` of each scenario passed as the
  second `vmc.out` argument; `In*` overlays are namelist keywords. Seed 1.
* Per-scenario provenance (sources hashes, stages, exit codes) is in
  `tests/fixtures/native_c_physcal_181/<scenario>/provenance.txt`.
* Rust tests never run C: `crates/mvmc-cli/tests/issue181_native_c_physcal.rs` and
  `crates/mvmc-core/tests/issue181_lanczos_native_c.rs` read checked-in files only.

Exact (per scenario, every sample of the final stage): file inventory including
C's `_time_` file and stale files of reruns, `zvo_time` rows (acceptance
`Counter[0..6]` columns), `EleIdx/EleCfg/EleNum/EleProjCnt/EleSpn`, `Counter[0..6]`,
cumulative primitive RNG draw count, and the next 624 SFMT words. Toleranced: all
numerical `zvo_*` text files (rows, columns, indices and headers exact).

## Matrix 1: six Julia PhysCal models

| Model | Fixed parameters / inventory | RNG, configs, counters (exact) | Per-sample energy (`zvo_out`, `zvo_var`) | OneBodyG / TwoBodyG / TwoBodyGEx | Max abs diff vs C |
| --- | --- | --- | --- | --- | --- |
| Heisenberg real | Pass | Pass | Pass | Pass | 1.1e-16 |
| Heisenberg complex | Pass | Pass | Pass | Pass | 4.4e-16 |
| Heisenberg FSZ | Pass | Pass | Pass (tol 1e-11) | Pass | 3.1e-12 |
| Hubbard real (+ Lanczos 2 in file) | Pass | Pass | Pass | Pass | 1.4e-13 (moments) |
| Hubbard DH2+DH4 real | Pass | Pass | Pass | Pass | 5.3e-15 |
| Kondo real | Pass | Pass | Pass | Pass | 1.8e-15 |

Tests: `heisenberg_real_matches_native_c`, `heisenberg_complex_matches_native_c`,
`heisenberg_fsz_matches_native_c`, `hubbard_real_lanczos2_matches_native_c`,
`hubbard_dh_real_matches_native_c`, `kondo_real_matches_native_c`.

## Matrix 2: non-InterAll Lanczos modes 1 and 2

Models whose fixed state is an exact eigenstate (Heisenberg) give a singular alpha;
the Lanczos rows use a deterministic 0.05 perturbation of the fixed parameters.

| Model | Mode 1 | Mode 2 | Notes |
| --- | --- | --- | --- |
| Heisenberg real | Pass | Pass | perturbed fixed state |
| Heisenberg complex | Pass | Pass | perturbed fixed state |
| Heisenberg FSZ | Rejected | Rejected | C readdef.c: "Lanczos mode is not supported when orbital is general"; Rust now reports the same message |
| Hubbard real | Pass | Pass | `hubbard_lanczos1`, `hubbard_chain_real` |
| Hubbard real, complex path (`--mode cmp`) | NotRun | Pass | real fixed values through the complex arithmetic path (C runs real arithmetic) |
| Hubbard DH real | Pass | Pass | |
| Kondo real | Pass | Pass | perturbed fixed state |
| All six non-InterAll terms (hopping, intra, inter, Hund, exchange, pair hop), real and `--mode cmp` | Pass | Pass | `all_terms_lanczos{1,2}_{real,cmp}` |
| `hubbard_chain_lanczos`, `spin_chain_lanczos` (historical C reference inputs, regenerated natively) | Pass | Pass | |
| InterAll | Excluded | Excluded | owned separately |

Lanczos formula (`CalculateEne`, `CalculatePhysVal`) fed with the native C moments
reproduces native C `zvo_ls_out` to 1e-13 relative on Linux (8 models,
`issue181_lanczos_native_c`), so every end-to-end difference below comes from the
moments.

## Matrix 3: parameter families, overlays, flags

| Combination | Result | Scenario / test |
| --- | --- | --- |
| `InDH2` + `InDH4` overlays | Pass (max 5.3e-15) | `hubbard_chain_dh_overlays` |
| DH2 only with overlay, fixed values from a C optimization step | Pass (bit-exact) | `hubbard_dh2_only_overlay` |
| DH4 only with overlay | Pass (bit-exact) | `hubbard_dh4_only_overlay` |
| `InGutzwiller` + `InJastrow` + `InOrbital` overlays | Pass (3.6e-15) | `heisenberg_overlays_gjo` |
| OptTrans (3 sectors, `-o`, `InOptTrans`) with DH overlays, real mode | Pass (1.4e-16) | `hubbard_chain_dh_opttrans` |
| All nine RBM sections (`In*RBM*`) + OptTrans + DH overlays, complex, `-o` | Pass (7.1e-15) | `hubbard_chain_dh_rbm_opttrans` |
| Real-mode RBM | NotRun | C ignores RBM in real mode (tmisawa/Julia-mVMC#59); not reproduced |
| Real-mode OptTrans slot shift (#55) | Pass for the above fixture | the fixed file follows the C layout; defect handling stays as pinned in #370 |
| FSZ with DH2/DH4/RBM/OptTrans | NotRun | no Julia PhysCal model; C FSZ + DH is covered by the optimization fixtures |

## Matrix 4: sample counts, indices, rerun behavior

| Behavior | Result | Scenario |
| --- | --- | --- |
| Two samples (`NDataIdxStart 7`, `NDataQtySmp 2`) | Pass | all six models |
| `NDataIdxStart 3`, `NDataQtySmp 3` | Pass | `idx_start3_qty3` |
| `NDataIdxStart 1`, `NDataQtySmp 3` with Lanczos 2 | Pass | `idx_start1_qty3_lanczos2` |
| Negative start index (`zvo_*_-01.dat`, `_000`, `_001`) | Pass | `idx_negative_start` |
| Rerun with fewer samples into the same directory: stale file kept, overlapped files truncated, `_time_` per run | Pass | `rerun_truncate` |
| Rerun with Lanczos switched off: stale `zvo_ls_*` kept untouched | Pass | `rerun_lanczos_off` |
| Append mode | n/a | C `InitFilePhysCal` opens with `"w"`; there is no append path |
| Binary output (`-b`), MultiDef (`-m`), `-F` flush | NotRun | outside this slice (#346 covers `-F` formatting) |
| One-configuration FSZ diagnostics (warm-up 0, 1/2/10 samples; 50 samples) | Pass (tol 3e-10 / 3e-10 / 3e-10 / 1e-11) | `fsz_*` |

## First-divergence analysis and tolerances

1. **Non-FSZ real/complex/DH/OptTrans/RBM.** All discrete data exact. Largest
   numerical difference 7.1e-15 absolute (energies), 1.4e-13 absolute on Lanczos
   moments of size 1e1..1e4 (relative 1e-15). Bound used: absolute 1e-13 plus
   relative 1e-10 (moments: 1e-13 plus 1e-12 relative).
2. **FSZ.** Configurations and RNG are exact, yet the energy of the very first
   configuration (warm-up 0, one sample) differs by 1.1e-10. Perturbing every
   fixed parameter by one ulp (relative 1e-16, `fsz_sensitivity.py`) moves the
   Rust energy by up to 1.1e-11, while the same probe on the complex non-FSZ model
   moves it by < 3e-15. The FSZ Pfaffian/inverse chain therefore amplifies
   roundoff by about 1e5, which accounts for the observed difference; the
   differences shrink with the sample average (3.1e-12 at 200 samples, 2.6e-12 at
   50). Bounds: 1e-11 absolute (200 and 50 samples), 3e-10 for the
   one-to-ten-configuration diagnostics, relative 1e-9.
3. **Lanczos alpha-dependent outputs** (`zvo_ls_out`, `zvo_ls_cisajs`,
   `zvo_ls_cisajscktalt*`). `lanczos_sensitivity.py` transcribes `CalculateEne`
   and reproduces both implementations' own output exactly from their own
   moments; moments differ by 1.2e-16..1.1e-15 relative and the output differs by up
   to 3.5e-7 relative (5e-9 absolute, 3.8e-8 for the corrected Kondo Green
   functions): amplification 1e3 .. 8e8 of the moment roundoff, set by the
   near-degenerate stationary-point equation. Bound: 1e-7 absolute plus 1e-6
   relative, only for these files. Scenarios with a better-conditioned alpha
   agree to 1e-13 (`all_terms`, DH, `hubbard_chain_lanczos`).

## Rust defects found and fixed (C is authoritative)

* **Exchange counters.** Normal (non-FSZ) real and complex samplers never
  counted exchange attempts/accepts (`Counter[2]`, `Counter[3]`), so `zvo_time`
  reported `acc_ex = 0`, `n_ex = 0` for every exchange-path model (Heisenberg,
  Kondo). Fixed; the `zvo_time` rows and counters now equal C exactly. (Julia has
  the same omission; it is not a C behavior.)
* **`pow(H1, 3)`.** `CalculateEne` uses libm `pow`; Rust's `powi(3)` rounds
  twice, shifting the discriminant by one ulp and, through the ill-conditioned
  alpha, `zvo_ls_out` by up to 6.5e-8 relative. Fixed with `powf(3.0)`.
* **Lanczos failure writer.** When `CalculateEne` fails (negative discriminant or
  `|dnorm/H1| < 1e-12`) C returns -1 before any `fprintf`: every `zvo_ls_*` file
  is zero bytes and the run continues. Rust wrote a NaN triple and moments. Now
  identical for those two failures; non-finite arithmetic (all-zero moments)
  still writes NaN like C's IEEE propagation.
* Rust's FSZ + Lanczos rejection now carries C's message.

## C defects and limits (math kept correct, not reproduced; for the maintainer to report)

* `readdef.c:610-617`: `if (NSPGaussLeg > 1) {...} else if (Lanczos != 0) error`.
  With a general orbital and `NSPGaussLeg > 1`, C accepts Lanczos and writes
  `-nan -nan -nan` moments (`heisenberg_fsz_lanczos1_gauss8`). Rust rejects.
* Singular alpha (exact eigenstate, zero energy variance): native C sample 007
  of `heisenberg_real_lanczos1_exact_state` writes uninitialised-memory denormals
  (`0 4.9e-324 1.7e-312`) and the complex model returns -1 depending on 1-ulp
  noise. These cells are `c_singular`: only non-Lanczos outputs, state and the
  zero-byte contract are compared.
* Real-mode RBM ignored by C (#59) and the real-mode OptTrans slot shift (#55) are
  unchanged.

## Not run / remaining

* InterAll and its Lanczos (separate owner); FSZ + DH/RBM/OptTrans PhysCal;
  real-mode RBM; `-b`, `-m`; MPI and threaded PhysCal (#182, #178/#179);
  native macOS (Linux numerical reference only; the libm-dependent
  Lanczos-formula test uses a 1e-6 bound off Linux/glibc, unverified on macOS).
* Julia-side: the historical Julia 1.13.1 references in `tests/fixtures/physcal_181`
  are unchanged. The Julia runner cannot preserve fixed RBM/OptTrans values (its
  restoration guard rejects them); native C accepts them, so those two models now
  have their first native-C PhysCal comparison.

## Commands

```sh
# regenerate (Dev Container, repository root); see c_toolbox/physcal_native/README.md
c_toolbox/physcal_native/build.sh /tmp/pn && c_toolbox/physcal_native/build.sh /tmp/pnd dump
python3 c_toolbox/physcal_native/generate.py --vmc /tmp/pn/build/src/mVMC/vmc.out \
  --vmc-dump /tmp/pnd/build/src/mVMC/vmc.out --mvmc-commit <submodule commit> \
  --out tests/fixtures/native_c_physcal_181
# analyses (optional)
python3 c_toolbox/physcal_native/lanczos_sensitivity.py <scenario dir> <Rust output dir>
python3 c_toolbox/physcal_native/fsz_sensitivity.py fsz_warm0_sample1 fsz 1e-16
# Rust tests (no C)
cargo nextest run -p mvmc-cli --cargo-profile test-fast --test issue181_native_c_physcal
cargo nextest run -p mvmc-core --cargo-profile test-fast --test issue181_lanczos_native_c
```
