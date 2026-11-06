# Issue #180: upstream ctest rule gate for all thirteen models

Related to #180 and #185. Not a deterministic-parity claim.

## Contract implemented

C `extern/mVMC-1.4.0/test/python/runtest.py` runs `vmc.out -s StdFace.def initial.def`
unchanged (upstream `NSROptItrStep`/`NSROptItrSmp`, input seed), reads the first two values of
the result, and exits non-zero when, for either value,
`|calculated - ref_mean| >= 3 * ref_std` **and** `|calculated - ref_mean| >= 1e-8`.
Julia's `test/integration/ctest_equivalent.jl` mirrors this rule exactly. The Rust gate
`rust_ctest_upstream_rule_selected_models` (`crates/mvmc-core/tests/ctest_equivalent.rs`,
ignored; `MVMC_RS_CTEST_UPSTREAM_MODELS=<names|all>`) applies the same rule with the same
`passes()` predicate (threshold unit test `ctest_failure_requires_both_thresholds`),
at each input's own run length (not 20 or 50 steps). References are the C-shipped
`ref_mean.dat`/`ref_std.dat`, copied with provenance and hashes to
`tests/fixtures/ctest_upstream_reference/` (see its `PROVENANCE.md`); inputs are
hash-verified against the Julia reference bundle. Nothing was generated from Rust, no
tolerance or reference was changed.

This is the statistical model-level contract. Per #358, SR-CG parameter trajectories are
ill-conditioned (about 1e-16 operand differences amplify to 1e-4..1e-2 in parameters), so
long-run trajectories are deliberately not compared tightly; prefix/RNG gates remain the
deterministic evidence. A pass at one seed means "within the upstream acceptance band",
not that the trajectory equals C.

## Why the nine "unsupported" models are not blocked on current main

The premise "4/13 supported" no longer matches the code: `MODELS` in `ctest_equivalent.rs`
lists all thirteen, each then with canonical 1/2/3/50 prefix fixtures
(`tests/fixtures/ctest_model_prefixes/`, removed in #702), and every model's input is accepted by the Rust
runner (parser, validation, sampler, Hamiltonian incl. Kondo exchange/Hund/CoulombInter,
FSZ, RBM, tetragonal/momentum projection). The earlier "Unsupported" labels were harness
declarations that have since been replaced; no missing feature or C-justified rejection was
found. The remaining gap was the absence of the upstream-rule gate, now added. The
supplementary 20-step independent-reference gate stayed MissingFixture (no independent
step-20 references existed; none were fabricated from Rust) and was removed in #702.

## Rerun on main after the C 1.4.0 move (#689)

Rerun on 2026-11-06 at main `f682a3b0` (after #682, #695, #714): the
`MVMC_RS_CTEST_UPSTREAM_MODELS=all` gate passed for all thirteen models, including
the three FSZ models (nextest run `0a718a56-1911-452c-ac19-c390b569a3ba`, 240 s
total). Intel Xeon E5-2699 v3, rustc 1.99.0 (b940084d7), `test-fast` profile,
default features, system OpenBLAS (`libopenblas.so.0`, openblas-pthread) with the
default thread settings (no `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` set);
references are the C-shipped files of mVMC v1.4.0 (`e83aa6cb`); inputs from
Julia-mVMC `73140156`.

```sh
MVMC_RS_CTEST_UPSTREAM_MODELS=all cargo nextest run -p mvmc-core --cargo-profile test-fast \
  --test ctest_equivalent -E 'test(rust_ctest_upstream_rule_selected_models)' \
  --run-ignored only --no-fail-fast --retries 0 --success-output immediate
```

| Model | steps/window | col0 calc | col0 ref | col0 sigma | col0 diff | diff/sigma | col1 diff | col1 diff/sigma | time | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heisenberg_chain_real | 1000/100 | -2.802775638 | -2.802775637 | 1.27e-09 | 8.94e-10 | 0.70 | 0 | n/a | 6s | pass |
| hubbard_chain_real | 500/50 | -3.662029893 | -3.665762089 | 1.78e-03 | 3.73e-03 | 2.10 | 0 | n/a | 2s | pass |
| heisenberg_chain_cmp | 1000/100 | -2.802775638 | -2.802775637 | 9.18e-10 | 7.26e-10 | 0.79 | 1.50e-10 | 0.66 | 10s | pass |
| heisenberg_chain_fsz | 5000/500 | -2.898795605 | -2.898726313 | 7.65e-05 | 6.93e-05 | 0.91 | 8.59e-06 | 0.20 | 27s | pass |
| hubbard_chain_cmp | 500/50 | -3.665850883 | -3.666230175 | 1.61e-03 | 3.79e-04 | 0.23 | 8.88e-05 | 0.92 | 3s | pass |
| hubbard_chain_fsz | 200/20 | -4.002037731 | -4.001453451 | 2.49e-03 | 5.84e-04 | 0.23 | 6.32e-05 | 0.21 | 22s | pass |
| kondo_chain_real | 200/20 | -12.67670473 | -12.67731698 | 6.55e-04 | 6.12e-04 | 0.93 | 0 | n/a | 6s | pass |
| kondo_chain_cmp | 200/20 | -12.67717063 | -12.67704858 | 3.12e-04 | 1.22e-04 | 0.39 | 2.50e-04 | 0.99 | 27s | pass |
| kondo_chain_stot1_cmp | 200/20 | -9.835387491 | -9.834352918 | 4.33e-04 | 1.03e-03 | 2.39 | 1.16e-04 | 0.62 | 14s | pass |
| general_rbm_cmp | 1500/100 | -3.264704373 | -3.267960515 | 2.28e-02 | 3.26e-03 | 0.14 | 5.67e-04 | 0.22 | 65s | pass |
| hubbard_tetragonal_real | 200/20 | -10.22333659 | -10.22047683 | 6.82e-03 | 2.86e-03 | 0.42 | 0 | n/a | 5s | pass |
| hubbard_tetragonal_momentum_projection_real | 200/20 | -10.24933511 | -10.24960604 | 1.72e-03 | 2.71e-04 | 0.16 | 0 | n/a | 31s | pass |
| kondo_chain_fsz | 200/20 | -14.6368024 | -14.63830588 | 9.12e-04 | 1.50e-03 | 1.65 | 5.27e-04 | 1.54 | 23s | pass |

No model failed; the largest deviation is kondo_chain_stot1_cmp col0 at 2.39 sigma
(single seed, no reseeding or averaging). Compared with the 1.3.0-era run below, ten
models reproduce the printed values exactly; kondo_chain_real, kondo_chain_cmp and
kondo_chain_stot1_cmp changed (Rust's local-spin trajectories changed between the two
heads; the cause was not bisected here). This is a statistical gate: deterministic
C 1.4.0 parity is checked by the fixture tests, not by this rule. The prefix and 20-step gates that the driver once reported as
supplementary were removed in #702, so the driver now runs only this gate.

## Historical results (mVMC 1.3.0 era, all 13)

Driver `scripts/run_ctest_long_183.py all <dir>` => status **Pass**. Intel Xeon E5-2699 v3,
rustc 1.99.0 (b940084d7), OpenBLAS 0.3.26 Haswell, 1 thread (OPENBLAS/OMP/MKL/BLIS=1);
mVMC v1.3.0 (`d73d06bd`), Julia-mVMC checkout `c0788c34`. Re-run on main including #374 (C ModPara reader) and #371 (C-order real kernels), Rust head `9e325e79`; every calculated value, diff and diff/sigma is unchanged to the printed precision from the pre-#371 run on `d2e01e23`.
Column 0 is the real energy window mean, column 1 the imaginary part. Zero-sigma columns
(real models) are exactly 0 on both sides (diff 0).

| Model | steps/window | col0 calc | col0 ref | col0 sigma | col0 diff | diff/sigma | col1 diff | col1 diff/sigma | time | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heisenberg_chain_real | 1000/100 | -2.802775638 | -2.802775637 | 1.27e-09 | 8.94e-10 | 0.70 | 0 | n/a | 9s | pass |
| hubbard_chain_real | 500/50 | -3.662029893 | -3.665762089 | 1.78e-03 | 3.73e-03 | 2.10 | 0 | n/a | 3s | pass |
| heisenberg_chain_cmp | 1000/100 | -2.802775638 | -2.802775637 | 9.18e-10 | 7.26e-10 | 0.79 | 1.50e-10 | 0.66 | 13s | pass |
| heisenberg_chain_fsz | 5000/500 | -2.898795605 | -2.898726313 | 7.65e-05 | 6.93e-05 | 0.91 | 8.59e-06 | 0.20 | 42s | pass |
| hubbard_chain_cmp | 500/50 | -3.665850883 | -3.666230175 | 1.61e-03 | 3.79e-04 | 0.23 | 8.88e-05 | 0.92 | 3s | pass |
| hubbard_chain_fsz | 200/20 | -4.002037731 | -4.001453451 | 2.49e-03 | 5.84e-04 | 0.23 | 6.32e-05 | 0.21 | 178s | pass |
| kondo_chain_real | 200/20 | -12.67767184 | -12.67731698 | 6.55e-04 | 3.55e-04 | 0.54 | 0 | n/a | 12s | pass |
| kondo_chain_cmp | 200/20 | -12.67644705 | -12.67704858 | 3.12e-04 | 6.02e-04 | 1.93 | 2.13e-05 | 0.08 | 56s | pass |
| kondo_chain_stot1_cmp | 200/20 | -9.835010778 | -9.834352918 | 4.33e-04 | 6.58e-04 | 1.52 | 4.39e-04 | 2.35 | 27s | pass |
| general_rbm_cmp | 1500/100 | -3.264704373 | -3.267960515 | 2.28e-02 | 3.26e-03 | 0.14 | 5.67e-04 | 0.22 | 112s | pass |
| hubbard_tetragonal_real | 200/20 | -10.22333659 | -10.22047683 | 6.82e-03 | 2.86e-03 | 0.42 | 0 | n/a | 7s | pass |
| hubbard_tetragonal_momentum_projection_real | 200/20 | -10.24933511 | -10.24960604 | 1.72e-03 | 2.71e-04 | 0.16 | 0 | n/a | 50s | pass |
| kondo_chain_fsz | 200/20 | -14.6368024 | -14.63830588 | 9.12e-04 | 1.50e-03 | 1.65 | 5.27e-04 | 1.54 | 122s | pass |

No model failed. Largest deviations: kondo_chain_stot1_cmp col1 2.35 sigma, hubbard_chain_real
col0 2.10 sigma, kondo_chain_cmp col0 1.93 sigma. These are inside the 3 sigma band but are
single-seed samples; no reseeding or averaging was used to select them.

Supplementary gates in the same driver run (not part of the verdict, never Pass; both
removed in #702): `rust_ctest_equivalent_selected_models` MissingFixture,
`canonical_models_match_independent_prefix_oracles` Unverified (no independent step-20 refs).

## Not claimed

Deterministic 20-step parity, C sampler/SR parity of trajectories, macOS results, MPI,
and Lanczos/InterAll (owned separately) are not covered. #180 acceptance criteria that
need independent 20-step references remain open.
