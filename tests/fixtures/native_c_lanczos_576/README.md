# C Lanczos InterAll references

`interall_real1` and `interall_real2` are independent unmodified C mVMC
1.4.0 PhysCal executions using `NLanczosMode=1` and `2`. The base inputs and
fixed parameters are copied from `native_c_physcal_181/all_terms_lanczos1_real`
and `all_terms_lanczos2_real`, respectively; `InterAll` is added using the
12-row input from `complex_uhf/uhf_interall_exchange/input/interall.def`,
plus six Hermitian-paired rows with distinct-site, same-spin and
number-times-hopping operators, including complex coefficients.
Each row conserves spin within both bilinears. Each fixture contains the full
actual input, C output and per-run provenance. An additive probe records the
saved configurations, acceptance counters, primitive RNG draw count and next
624 SFMT words; the generator verifies that the probe and unmodified executable
produce identical `zvo_*` outputs. The CLI regression compares those integer
records exactly.

`interall_complex1` and `interall_complex2` use the same inputs with
`ComplexType=1` in the orbital header and a deterministic 0.005 perturbation
of the fixed complex parameters by the generator. The stored parameter files
are the resulting values (the 1.4.0 regeneration reran C on these stored files
with no further perturbation, so `provenance.txt` records `zqp_perturb 0`).

`interall_nodal1` and `interall_nodal2` add the same InterAll file to the
real `nodal_neel` scenarios described in `c_toolbox/nodal_481/README.md`.
The selected input exercises C's zero-overlap expansions; it does not assert
that every added InterAll row reaches a nodal branch.

The reference was regenerated from `extern/mVMC-1.4.0` (`e83aa6cb`) for #695 in the
Linux x86_64 Dev Container image `503e36a6615e`
(`vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32`),
with GCC 13.3.0, MPICH 4.2.0, OpenBLAS 0.3.26 and CMake 3.28.3 Release; executable
SHA-256 and source commit are in `provenance.txt`. C ran with one MPI rank,
one OpenMP thread and one OpenBLAS thread. These are end-to-end sampled runs,
not standalone kernel results.

On the Linux x86_64 host, with the generated inputs and exact C/Rust RNG and
saved-state agreement, the largest observed absolute difference against these
1.4.0 references is `9.1e-13` in the 16 Lanczos moments (`zvo_ls_qqqq`) and
`7.8e-11` (`1.1e-13` relative, `interall_real1`/`2`, `zvo_ls_out_001.dat`,
value `7.17e2`) in the alpha-dependent energy output. Both C 1.4.0 and Rust use
`rsbOld = rbOld + t*Nsite` in the real two-electron update; the exchange
updates of the real InterAll runs accumulate its roundoff into the Lanczos
energy. Differences are measured at the formatted output precision. Lanczos
outputs use `max(1e-10, 1e-12 |value|)`; exact integer state checks are
independent of this bound. See the issue #181 Lanczos sensitivity audit for the
conditioning of the alpha solve. The macOS CI result must be recorded
separately; these Linux differences are not a macOS measurement.

SHA-256 of relevant vendored 1.4.0 sources (paths under `src/mVMC/`):

| File | SHA-256 |
| --- | --- |
| `lslocgrn.c` | `cb255a8929a03b37e0d98575828c348a893fccd30f182a07770d79c2ef9efeaf` |
| `lslocgrn_real.c` | `175210aa915e25e50b3ca6362b0e7268133e0f70f5cb4cf6691fac6163526445` |
| `readdef.c` | `9e1f2c635bffb2acdf3210613ab7330c5b28967fae928b303f3b151dfa1b7ef3` |
| `calham.c` | `94e3053a1c05ccee6e1c4359c94f24ce1d86cc07cbe4036231a37cd9622819b4` |
| `calham_real.c` | `aea8517ee6e0c637c1dd24ec08b516b0d2f87edab5763928d2845b4ae4eee705` |

For regeneration, build the unmodified executable following
`c_toolbox/physcal_native/README.md`; copy each fixture's `inputs/` and
`zqp_opt.dat` to a separate source directory, provide those two directories as
rows of the existing `scenarios.tsv` format, and run
`uv run --no-project python c_toolbox/physcal_native/generate.py --vmc <vmc.out> --vmc-dump <probe-vmc.out> --mvmc-commit <C-commit> --out <output>`.
The generator records executable and selected source hashes. Rust tests consume
only the checked-in fixture and never execute C.
