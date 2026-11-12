# Reported AP/P square input (#730)

`ap_p_L8_U4_SP` is the actual 8×8 Hubbard input used in the #730 report:
antiperiodic x, periodic y, U=4, half filling, real normal orbitals,
Gutzwiller/Jastrow, `NMPTrans=-4`, `NSPGaussLeg=8` (32 QP planes), and
16 negative QP translation signs. Site numbering is x-fastest.

Source: `AtelierArith/mvmc-rs`, branch `gpu_test_misawa`, commit
`7b93c39227faf8e48b34474677e9bafac6823b32`, directory
`benchmark/square_hubbard_a100/inputs/ap_p_L8_U4_SP`.
`provenance.json` records the source and extracted SHA-256 hashes. Every source
file was checked against that revision's `cases.json` before copying.

The only extraction change removes the `OneBodyG` and `TwoBodyG` entries from
`namelist.def`. Those generated request files are unnecessary for projected
overlap and Hamiltonian checks; the Hamiltonian, projection tables, sampling
input and fixed parameters are otherwise byte-identical. The Opt test copies
this fixture to a temporary directory and changes only `NVMCCalMode` to zero.
Explicit test controls select one walker, no proposals/warmup, one observation,
and seed 79; no sampling-length change is hidden in the stored input.

The source benchmark used this parameter record for fixed-parameter measurements
with C-mVMC **development build `d4430b9d`** on Fugaku. It does not separately
record the parameter-generation compiler or environment.
These files are physical **inputs**, not C 1.4.0 numerical expected values.
The statistical reference numbers from that development build are not copied
or asserted here. The numerical expectations are evaluated independently on
the same walker by the Rust CPU C-order kernels ported from authoritative
C 1.4.0 (`e83aa6cb`): Slater assembly, Pfaffians, projected overlap and local
energy. The CPU reference explicitly uses the input's negative translation
flag and does not reuse resident sign tables. Ordinary Rust tests require no
C/Julia runtime, toolbox or Git access.

Host tests check every resident Slater sign against `translated_site` on the
unmodified parsed input, for Opt and PhysCal. CUDA tests compare projected
overlap and energy for both entry paths. Bounds use the existing #656
`16*n*epsilon*kappa` factorization/ratio budget, scaled by the absolute weighted
Pfaffian sum or Hamiltonian term sum. The projected-overlap bound must stay
below `1e-5*|IP|` so the check cannot become vacuous near a node.

Validation used Linux x86_64, Rust 1.99.1, RTX 3060, NVRTC 12.9 and device
cuBLAS 12.9.2. The CPU test binary links `/opt/openblas/lib/libopenblas.so.0`
(OpenBLAS 0.3.26, DYNAMIC_ARCH), with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`; the standalone test dependency graph enables PfaPack's
`blas-backend`. On this environment both routes gave these pointwise comparisons:
the overlap difference was `1.135e-4` at magnitude `1.144e9` (bound `9.988e1`,
Frobenius condition number `3.297e5`), and local-energy difference `1.421e-14`
(bound `1.425e-5`). These observations are not checked-in float goldens.
The stage diagnostic observed no Slater-entry difference; the first floating
difference was in Pfaffian plane zero, with maximum per-plane difference
`2.768e-4`. This locates the overlap rounding at factorization rather than
translation/sign construction, motivating the existing condition-scaled bound.
The one-walker PhysCal second moment also equals the squared local energy
within its explicit rounding bound.
Dropping the AP signs changes the projected overlap by `2.354e8` on this
configuration, far beyond the rounding bound. The test requires a change
greater than ten times the overlap or energy bound as a regression control.

Reproduce the focused checks on a CUDA host:

```sh
cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked \
  --lib -E 'test(square_ap_p)' --run-ignored all --no-fail-fast --retries 0 \
  --success-output immediate
```

The A100 L8/L16 rerun against the reported C bin statistics remains a separate,
unverified acceptance item of #730; these per-configuration checks do not
replace it. #727 remains the open A100 benchmark tracker.
