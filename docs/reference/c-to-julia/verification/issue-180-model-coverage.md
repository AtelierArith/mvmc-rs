# Issue #180 model coverage

## Current acceptance scope: long baseline20

The user's latest instruction changes the long runner baseline from50 to20
optimization steps. Short prefixes1/2/3 remain; deliberate failure-boundary
prefixes such as27/28/29 remain separate regressions. The CG kernel's
limits1–41, including residual refresh20/40, are not long-run baselines and
remain unchanged. This applies to serial, CLI, native MPI and worker/thread
acceptance runs, with each execution configuration explicitly recorded.

Fresh20 references must execute20 steps from the same reviewed Julia PR54
commit `62b0f97f076fb55c71c3ab0caa041a9adff94e04`, with the parameter window
and effective NSROptItrSmp recorded consistently. A50 endpoint cannot be
truncated or relabelled as20 final parameters/window output. Completed50
artifacts and historical results below remain archived evidence, not fresh20
acceptance. No numerical comparison bounds are changed by this workload
change. Reference regeneration and the active prefix-test selections are
coordinated with their owners; this documentation does not claim fresh20
generation or validation has completed.

Fixed-input/fixed-seed repeatability uses the same implementation, backend,
MPI/worker/thread configuration. Primitive RNG initialization, conversions,
output/state remain exact for matching draw order/count. Numerically
dependent branch/acceptance divergence must identify the first arithmetic
cause and decision threshold; math-function error is not blanket Monte Carlo
noise or permission to hide algorithm/input defects.

## Historical50 checkpoint (not current20 acceptance)

All thirteen canonical native Linux short/50-step gates passed, including
GeneralRBM's additional CG variant. Final integrated milestone/merge
validation is **incomplete**: the all-thirteen long run preceded integration,
and final workspace/native macOS checks are owned/coordinated separately.
An input inventory, accepted orbital reader
section, or successful one-step execution does not establish deterministic
parity or a long ctest summary. Missing evidence remains unverified.
Lanczos/InterAll, runner changes, MPI, and the portable numerical policy (#190)
are owned separately. No commits were made for this work.

Audit date: 2026-10-03. Shared Rust checkout initially
`30d8d69ffc6d2700c56fda81841827a49e72e57c` with pre-existing changes; Julia
reference checkout `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`. Native Linux
x86_64, Rust 1.99.0 (`b940084d7`, LLVM 23.1.1), system OpenBLAS package 0.3.26.
Package metadata alone does not establish the backend linked by every test.
Host `julia +1.13.1 --version` confirms Julia 1.13.1. The earlier unavailable
observation is superseded. New oracle runs use the pinned Manifest-v1.13.toml.
The GeneralRBM test binary links `/lib/x86_64-linux-gnu/libopenblas.so.0`.
Canonical `all` prefix selection now includes all thirteen models, plus a
second GeneralRBM CG/NStore=0 solver variant of the same model.

The following counts and seeds come from each pinned ctest input, not from
example inputs. AP/P acceptance below is an actual standalone native C reader
result; it does not establish acceptance of all other input sections. See
[probe provenance](../../../../c_toolbox/ctest_orbital_contracts.md).

The table records the initial evidence gap, not current pass status; the
execution record below supersedes it.

| Model | Mode | Steps/window | Seed | Native C orbital sections | Independent prefix evidence before this work | Initial missing coverage |
| --- | --- | --- | --- | --- | --- | --- |
| heisenberg_chain_real | real | 1000/100 | 1 | AP accepted | Existing 1/2/3/50 SR fixtures | Native long summary in this session |
| hubbard_chain_real | real | 500/50 | 1 | AP accepted | Existing 1–10/50 SR fixtures | Native long summary in this session |
| heisenberg_chain_cmp | cmp | 1000/100 | 1 | AP accepted | Existing 1/2/3/50 SR fixtures | Native long summary in this session |
| heisenberg_chain_fsz | fsz | 5000/500 | 1 | AP/P accepted | Existing 1/2/3/50 SR fixtures; label historical/mixed-C provenance | Native long summary and current numerical policy |
| hubbard_chain_cmp | cmp | 500/50 | 1 | AP accepted | No canonical model prefix fixture located | Independent configs/RNG/SR/output prefix oracle |
| hubbard_chain_fsz | fsz | 200/20 | 1 | AP/P accepted | No canonical model prefix fixture located | Independent FSZ configs/spins/RNG/SR/output oracle |
| kondo_chain_real | real | 200/20 | 1 | AP accepted | PhysCal evidence belongs to #181, not this optimization gate | Optimization local-spin trajectory and SR oracle |
| kondo_chain_cmp | cmp | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Complex local-spin trajectory and SR oracle |
| kondo_chain_stot1_cmp | cmp | 200/20 | 123456789 | AP accepted | No canonical model prefix fixture located | Nonzero total-spin projection trajectory/SR oracle |
| general_rbm_cmp | cmp | 1500/100 | 12395 | AP accepted | Canonical mixed C-counter/Julia SR 1/2/3/50 fixtures already exist | Current native discrete gate and long summary |
| hubbard_tetragonal_real | real | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Lattice-specific trajectory/SR oracle |
| hubbard_tetragonal_momentum_projection_real | real | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Momentum-sector weights/signs and trajectory/SR oracle |
| kondo_chain_fsz | fsz | 200/20 | 1 | AP/P accepted | No canonical model prefix fixture located | FSZ local-spin trajectory/SR oracle |

“Missing oracle” does not mean an algorithm is absent. In particular, the
samplers already consume local-spin mappings, and Hamiltonian evaluation
already supports Exchange/Hund/CoulombInter. The old blanket claim that Kondo
Hamiltonian parity is unimplemented requires a first-divergence investigation;
it is not a diagnosis. Never repair these coverage gaps by enabling flag pairs,
filling incomplete C inputs implicitly, or recycling another model's goldens.
The historical malformed inputs and explicit replacements documented in
`tests/fixtures/c_orbital_inputs/README.md` remain untouched.

## Gates and interpretation

The ctest long gate is ignored by default and uses `support::require_gate`.
It preserves the input's seed and optimization/window counts. Its final two
summary columns use the upstream failure rule: a difference fails when it is
both at least three reference standard deviations and at least `1e-8`.
Nonfinite summaries/references and negative standard deviations fail. This
statistical gate cannot substitute for exact trajectory comparisons.

The new `ctest_model_prefixes` gate independently restarts the canonical
GeneralRBM input at seed 12395 for prefixes 1, 2, 3, and 50, for direct stored
SR and CG. It checks every saved configuration, burn-in state, counters, and
the next full 624-word SFMT block exactly, before any numerical comparison.
It reads only existing checked-in fixtures. Its expanded 8.29-second rerun
(nextest 61021156-8160-4d95-9a91-7a1ed4206f17) also passed independent energy,
mapped parameters and `zvo_out` at all eight solver/prefix combinations,
plus direct prefix-1 sampled SR OO/HO. Full declared-slot SR/output coverage
is also checked against the new independent observer below. The existing
`canonical_general_rbm_complex_reference_uses_native_c_counter_order` runner
gate uses #190's numerical policy. These fixtures are
mixed references, not full C executable validation.

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  --test ctest_equivalent --run-ignored only \
  -E 'test(inventory_all_thirteen_ctest_input_contracts)' --success-output immediate
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  --test ctest_equivalent --run-ignored only \
  -E 'test(audit_thirteen_ctest_one_step_execution_paths)' --success-output immediate
MVMC_RS_CTEST_PREFIXES=1 cargo nextest run -p mvmc-core \
  --cargo-profile test-fast --test ctest_model_prefixes --run-ignored only \
  -E 'test(general_rbm_ctest_short_prefix_and_twenty_step_discrete_trajectory)' \
  --no-fail-fast --retries 0 --success-output immediate
MVMC_RS_CTEST_PREFIX_MODELS=all cargo nextest run -p mvmc-core --locked \
  --cargo-profile test-fast --test ctest_model_prefixes --run-ignored only \
  -E 'test(canonical_models_match_independent_prefix_oracles)' \
  --no-fail-fast --retries 0 --success-output immediate
MVMC_RS_CTEST_MODELS=heisenberg_chain_real cargo nextest run -p mvmc-core \
  --cargo-profile test-fast --test ctest_equivalent --run-ignored only \
  -E 'test(rust_ctest_equivalent_selected_models)' --success-output immediate
```

The inventory and execution-audit commands are diagnostics, not #180 parity
execution. Explicit invocation of the long or prefix gate with absent, empty,
or `skip` selection fails; normal runs show the gate as ignored.

## Actual execution record

- Executed: standalone native C AP/P section audit, all 13 AP and three P
  sections accepted after correcting probe storage.
- Executed: `uv run --no-project python scripts/check_orbital_contracts_c_parity.py`,
  all 92 existing C contract cases passed and the excerpt matched upstream.
- Executed: Rust inventory and all-thirteen one-step execution audit passed;
  these are diagnostics, not deterministic or long parity coverage.
- Executed: GeneralRBM exact 1/2/3/50 configuration/counter/RNG checks passed
  for direct stored SR and CG. The corrected temporary-output rerun passed in
  8.15 seconds (nextest run 480fdc6d-0cf8-4756-9db9-4d17825a1809).
- Executed: all-thirteen native long statistical gate passed in 792.54 seconds,
  nextest run 3b26e52f-bdfc-4871-a454-a8dabc05186f (handle 90495). Every model
  reported a passing statistical summary. This run started before integration;
  it is not final integrated-runner milestone validation.
- Executed but failed: fresh canonical prefix gates for twelve no-RBM models,
  nextest runs 4a527ad2-6723-4179-97ff-b7ac513668e0 and
  4db86812-fa55-41b0-85db-40b07479d8ef. Each reached prefix 1 output comparison
  after matching exact configurations/counters/RNG and numerical parameters,
  energy and normalized SR buffers. Julia's mapped orbital output duplicates
  or reorders declared slots; C `outputData()` writes contiguous `Para[i]`.
  The first missing contract is complete C-slot output verification, not a
  demonstrated sampler/SR algorithm defect. Prefixes 2/3/50 were not reached.
- Unrun: final integrated validation of the newly checked-in twelve-model
  fixtures, full GeneralRBM quantity coverage, C optimization-window output
  parity and native macOS.

Integration: #190/#193 numerical policy is now merged in the shared checkout.
The preceding runs started before integration; final milestone verification
requires the integrated runner review and rerunning affected gates.

The current integration commit is `d05cde3e5fa8bed93ea4100dca969c6455d3c295`.
Fresh Julia 1.13.1 canonical observations for Hubbard complex, normal Kondo
real/complex/Stot=1, and both tetragonal models completed at 1/2/3/50 steps in
`/tmp/mvmc-ctest-oracles.0XGTlt`. Their provenance includes the pinned manifest,
source/input SHA-256 hashes, one Julia BLAS thread and ILP64 OpenBLAS.
Generation alone is not parity execution. The remaining six no-RBM models
also completed independent observation in
`/tmp/mvmc-ctest-oracles.nSLHLv`, with the independently validated native C
FSZ energy bridge for FSZ. All 48 input SHA-256 manifests were independently
checked against the canonical inputs. No expectation comes from Rust output.
The failed Rust gates above used these stages. A corrected output observer
retains original Julia output and emits a separate C-declared-slot stream
from the same pre-SR Julia data, with the C writer source hash recorded.
Its twelve-model regeneration completed successfully as handle 66478. All
48 prefix observations are now checked in under
`tests/fixtures/ctest_model_prefixes`, including input hashes, seed/solver
settings and initial-overlay provenance. Fixture presence does not establish
Rust parity. Handle 13138 finished in 125.56 seconds, nextest run
4ef901ce-cbcc-49e1-8e79-0a2aede97f16: all twelve models reached all four
prefixes, matching exact configurations/counters/RNG and numerical energy,
parameters, normalized SR buffers and both `zvo` output streams. The overall
gate failed explicitly for all twelve models' missing C optimization-window
output fixtures. This is executed partial quantity coverage, not a full pass.
All 481 copied observation files were byte-compared with the external
generation successfully. The latest ordinary nextest run passed four support
tests and ignored both gates; those support tests are not #180 execution.
The ignored gate defaults to checked-in fixtures and requires explicit
`MVMC_RS_CTEST_PREFIX_MODELS`; an optional external root is only for developer
regeneration checks. It never invokes a reference runtime.

Same-input qualification: the archived Hubbard real and Heisenberg FSZ SR
prefix generators omit the canonical `initial.def` overlay. They are valid
historical workloads but cannot supply canonical ctest prefix evidence.
The fresh observer preserves this overlay; the long gate also preserves it.

### First identified production output gap (requires runner/I/O owner)

Rust `io.rs::output_opt_data` emits final Gutzwiller/Jastrow/RBM/Slater values
only. This is not authoritative C `avevar.c::OutputOptData`: C stores energy,
energy squared and the post-SR declared parameters through `StoreOptData`,
then emits that optimization window's means and deviations (or its special
single-window real-only branch). The independent Julia final-parameter writer
also differs from C and must not be treated as a C output oracle. Source
SHA-256: `509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a`;
boundaries `StoreOptData` lines 82-92, `OutputOptData` starting line 94, and
`CalcAveVar` lines 34-54. The call occurs after SR/synchronization in
`vmcmain.c:511-512`. Fixing this production writer is outside this agent's
owned paths. Full output parity remains blocked even if parameter checks pass.

The missing C-window oracle has now been independently generated for all
thirteen models at 1/2/3/50, plus GeneralRBM CG: 56 actual standalone C
aggregations. The probe executes verbatim `StoreOptData`, `CalcAveVar` and
`OutputOptData` bodies on independently observed Julia/C-contract histories;
it neither invokes Rust nor uses Rust results. Checked-in expected streams,
standalone chronological inputs and per-case hashes/compiler provenance are
in `tests/fixtures/ctest_model_prefixes`. See
[C-window extraction/reproduction](../../../../c_toolbox/ctest_opt_window.md).
The first history metadata used mapped RBM row counts; the aggregation
adapter rebuilds declared widths from canonical inputs, preserving every
numerical history line. No flag pairs or invalid C model input are enabled.
Rust comparison handle 86844 completed (nextest
1f6953b3-a220-4b3a-bdd9-10c9de4d0938, 9.06 seconds): Heisenberg real and
GeneralRBM direct/CG reached all prefixes and matched configurations/RNG,
declared parameters, energy, SR OO/HO and `zvo` output before hard failures
at C-window field 0. Heisenberg real prefix 1: Rust `0`, C
`-1.17506640710817334`. GeneralRBM direct prefix 1: Rust parameter
`-0.111266893280731596`, C energy `11.5114979860913582` (CG has the same
expected energy). This is a field/order algorithm mismatch, not a floating
error or sampling drift. The production writer repair belongs to Goodall.
All-model comparison handle 58267 completed in 124.74 seconds, nextest
a53bb958-0479-4577-bf71-4c8ed6479213. All thirteen canonical models plus
GeneralRBM CG reached all four prefixes: 56 executed exact configurations/RNG
and numerical parameters/energy/SR OO/HO/`zvo` checkpoints matched. The full
gate failed at C-window field 0 for every model/variant. Fields beyond the
first mismatch were not compared; full C-window parity remains unverified.
The expanded historical GeneralRBM gate including direct prefix-1 stored
Gram data passed in 7.88 seconds, nextest
f1843e13-6ba4-4cda-93d0-c188f754f5ce. CG diagnostics use #190's existing
formatted-output bound and exact integer columns; the latest CG run
5428fc71-929d-47fe-9f49-130da07cad61 passed these and all four quantity prefixes
before failing C-window field 0, in 3.48 seconds. It is separate from the
preceding all-model binary. Final writer repair and its integrated rerun
remain outstanding. No commits or production runner/I/O/MPI edits were made
by this agent; uncertain root-owned generated files remain untouched.
Ordinary harness verification passed nine support/statistical-rule tests and
ignored five gates (nextest 1f42e019-95f8-4d6a-a693-c5a7370d4918); these are
not model execution coverage. Owned Rust files pass rustfmt and diff checks.

Effective-window/source audit: short prefixes override both step and window
to 1/2/3/50 explicitly in Rust and Julia; the native C history header uses
that window. Original canonical counts are also recorded. No min-with-original
clamp, seed change or implicit input repair is used. All 56 standalone input
hashes verified against C aggregation provenance; the extracted C bodies
match the authoritative source. Native C energy/aggregation link libm/libc
only, while Julia uses one-thread ILP64 OpenBLAS 0.3.30. Source and actual
backend hashes are recorded in fixture metadata.

The new ordinary `ctest_window_fixtures` consumer is an offline aggregation
and policy check, not model execution. Its first build was unrun due to the
in-progress writer/state API transition (`OptDataPoint.energy_squared` and
new output writer state argument). Rerun it and all 56 model prefixes after
Goodall/Ramanujan finish that transition; no successful result is inferred
from test registration. C-window comparison budgets and propagation are
documented in the standalone probe notes; the one-sample branch has no
deviation field and does not call `CalcAveVar`.
After the API transition, the offline consumer passed 56 finite-domain cases
in 0.023 seconds, nextest 4650fe69-ab92-4da1-849e-846dfa329460. Its measured
maximum absolute arithmetic difference from actual C outputs was zero;
assertions still use the portable 1e-11 absolute/relative policy, not bitwise
comparison. This is not 56 model runs. The companion full model-prefix gate
passed all 56 prefixes in 156.55 seconds in the same nextest run (handle
58378): exact configurations/RNG, energy, parameters, SR, CG diagnostics
and full C-window output means/deviations/layout. The earlier first-field
writer mismatch is resolved by Goodall/Ramanujan's production edits, not by
changing independent expectations. Fields beyond the previous mismatch were
now compared successfully. No oracle regeneration from Rust was performed.
The strengthened ordinary fixture consumer also calls the production writer
for all 56 independent histories, with mixed integer optimization flags and
no Slater mappings to ensure all reserved declared slots remain in the stream.
It passed in 0.108 seconds, nextest 3e6814a3-2794-47e5-82b1-11abad4703c3.
It enforces exact one-window literal-zero formatting and finite nonnegative
multi-window deviations in addition to portable numerical/layout comparisons.
These are fixed-history writer tests, not additional model executions.

Update this execution record with observed outcomes; do not infer success from
test registration, fixture presence, or compilation.

### Julia retained-slot repair (separate reference checkpoint)

[Julia PR54](https://github.com/tmisawa/Julia-mVMC/pull/54) now includes
`973184d49a16ea26b54a0ba609d10f86fa7c1857`, parent
`87dd33fe26c50cac70e7bcb38e8cb41f35681f39` (the accepted full-storage writer).
This is Julia production lifecycle repair, not an observer-only correction:
declared unmapped slots survive initialization, fixed/In overlays, direct/CG
SR, pack/unpack and full-width normalization. Four focused files pass
171 assertions (54 retained + 35 sync + 19 writer + 63 stochastic); own
session66316 terminal0 and parent-reported34945 terminal0. Environment:
Julia1.13.1, Linux x86_64, actual OpenBLAS0.3.30 ILP64/Haswell, one thread.

Independent unchanged C InitParameter probe: four draws, parameter maximum
absolute difference0.0 on both same-width reinitializations, exact next624.
The numerical bound was tightened BEFORE session66316 from abs1e-16/rel1e-14
to abs `eps(Float64)*0.01` / rel `2*eps(Float64)` for the small scaled draw.
Own before/after tracked-diff SHA256 is identical:
`dc9c797558c1778c0c14f1e552bf798164c784fd03a196e4568070f0cb43e809`;
retained helper SHA256
`c4d3d7ce66f0c203b21021b9f66c10b16e153ab10670fbb4bbf5a036c4be5392`;
retained regression SHA256
`3415a093b102715ed7e8846133e136d182d24240a697dafe208870d679a170e4`.
Parent34945's separate before/after hash transcript was not supplied to this
agent; do not reconstruct that record from the own-session hashes.

The public commit contains fourteen reviewed paths and no Manifest/cache.
Shared `extern/Julia-mVMC` remains clean at pinned
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`; existing Rust fixtures/goldens
were not regenerated or relabeled. This focused reference checkpoint does not
supersede historical prefix evidence, demonstrate real MPI/full trajectory,
or by itself close the reserved-FSZ-CG col48/prefix aggregation audit.

### Reproducibility checkpoint before parent freeze

The latest observed shared HEAD is `2edb6794baf76af6cd83a95ff0b54fc4aa10398e`.
This is a post-run source observation, **not** a reconstructed hash of the
156.55-second execution snapshot. Runner/MPI APIs are still changing; the
fresh thirteen-model long gate waits for the parent's freeze. No native macOS
result is claimed. Last observed SHA-256 values:

```text
44113e2bc794c21f297be88c95f0e72b7b33a4eea5e99f953b6699fd06aa4cc3  crates/mvmc-core/tests/ctest_equivalent.rs
861dea139e5e8d35d67fc504d7f9a8d7505811dfdb1620e8c7e81f340cc6285c  crates/mvmc-core/tests/ctest_model_prefixes.rs
817c49c26a180947a646ef4977f24b4a76637b6cf80eccdc3c62edbc82ffb6c6  crates/mvmc-core/tests/ctest_window_fixtures.rs
44dd05ccf0fc7d58c354f41c67994ca632d1a37f99947668e41646db041ab549  crates/mvmc-core/src/io.rs
d1d712958feacba99e5d261e123f91a7f55facf05acd061fede4533be6d01958  crates/mvmc-core/src/run.rs
a912eab6001dc2ed336a2a15c87c0a869153a049e4702c323cb06d0fdb3b3e74  crates/mvmc-core/src/state.rs
```

Archive audit: 848 files; 56 each of model-settings, canonical input manifests,
C-window provenance records and expected C-window streams. All 56 canonical
input manifests verified from their respective reference `inputs/` directories
(CG uses the canonical GeneralRBM directory). All 56 declared C-history hashes
verified against their provenance, and `post-generation-source-audit.sha256`
verified against the current pinned reference sources. Settings record the
original seed, solver/storage choices, initial overlays and original/effective
step/window counts; short runs override both counts, without a clamp.
The normal fixture consumer and optional model gate contain no subprocess
invocation or toolbox reads. They use standalone checked-in expectations;
the optional gate additionally parses canonical reference input data, not a
Julia runtime. Reproduction commands remain optional toolbox developer work.

Numerical evidence boundary: the C-window scalar aggregation bound is justified
in `c_toolbox/ctest_opt_window.md`; it introduces no solver tolerance. Existing
`tests/support/numerical_comparison.rs` provides the componentwise comparison
mechanism, not a condition estimate or residual certificate. Existing
`cg_fixed_input_matches_julia_through_residual_refresh` separately checks an
explicit Gram residual on fixed CG problems; those are not per-model metrics
for these new fixtures. The archived prefix fixtures do not include per-prefix
direct-SR condition estimates or backward-residual metrics. Consequently the
successful 1e-11 energy/parameter and 1e-12 SR-buffer gates must not be described
as a newly established condition/residual bound for every model. That evidence
gap remains explicit for the numerical-policy owner; no #190 helper, numerical
tolerance, solver or oracle expectation was changed during this audit.

After the gate-only integration, observed HEAD is
`6cd03743ea2c3ca7ef189415ff5ade4e0ca3598c`. Shared production drafts remain
unstaged; this integration is not full numerical validation. The observed
`ctest_equivalent.rs` hash is now
`61ebd3be15009239036f8226f4eab1ae426887b444c586de0a98218072a88719`,
and `io.rs` is
`fd7ab1952a2e5cf579752403a5d80911a41011f547142e7fe8a5b78f4886217c`.
The preceding observed `run.rs`, `state.rs` and `ctest_model_prefixes.rs` hashes
are unchanged. These observations do not retroactively pin the successful
prefix binary. Fresh long execution remains deferred until parent freeze.
The ordinary standalone fixture/writer consumer was rebuilt after integration
and passed all 56 fixed histories in 0.082 seconds, nextest
`77a03a8e-b2eb-492d-bc98-fb67d494ca11`. This is not a fresh model-prefix or
long execution. No reference expectations were regenerated.

### Subsequent direct-SR and DH evidence

The independent serial pre-factorization/post-original-substitution observer
completed all 52 canonical direct endpoints (13 models times 1/2/3/50), with
the same verified native C-FSZ energy bridge and original input seeds. Actual
unfactored matrices, gradients, original increments, active indices/flags,
settings, source/environment provenance and exact configs/next624 words are
archived separately in `tests/fixtures/ctest_direct_sr_metrics`. All 52 discrete
records match the existing independent prefix archive. The optional high-
precision archive audit passed all 52 existing `128*n*epsilon` backward checks:
maximum normwise backward error `4.677538293263316e-17`, componentwise error
`3.509860472961454e-16`, condition-2 estimate `3.34976100140845e7`.
These are actual original increments, not reconstructed parameter differences.
Endpoint conditioning supports backward stability, not a blanket portable
forward allowance or MPI budget. The generation-revision observer sources are
also archived with hashes; later reusable-hook refactoring is separately
labelled. No #190 policy/helper/tolerance change was made.

Nine DH2/DH4/combined three-step independent histories were captured and
aggregated by actual C avevar bodies, including actual AP/P auxiliary splits
and every DH declared slot. The newly granted `dh2_runtime.rs`/`dh4_runtime.rs`
tests compare complete pre-SR declared `var` streams first and retain the
historical Julia DH-omitting subsequence only as a separately labelled check.
Their full C-window expectations remain post-SR/sync. `runner_config.rs`
retains rejection of oversized windows: authoritative allocation uses malloc
without any clearing, collection leaves leading rows unwritten, and aggregation
reads every requested row. The safe bounds-only probe never reads indeterminate
numerical storage. See `c_toolbox/ctest_window_bounds.md` for hashes and executed
index cases; no undefined C output becomes an expectation.
After the full-var subsequence offset repair, nextest
`026ef308-763e-4adf-ae65-1765f0d8174f` passed all 16 tests across the three repaired
targets and the four observer invariance tests, in 12.052 seconds. The parent
independently passed the three targets (12 tests, 8.759 seconds), nextest
`209651a8`. These DH cases are not additional execution of the 13-model matrix.

A fresh explicitly matched step/window 56-prefix gate passed in 149.754 seconds,
nextest `a6e1a0cc-d43f-4861-9e74-300f26a9a3ea` (handle 91813), after restoration
of the public oversized-window guard. It verifies complete C-declared pre-SR
`var` as well as post-SR C-window output, exact discrete/RNG and energy/SR/params.
It preceded subsequent opt-in SR-observer and full-var writer drafts, so final
frozen-source verification remains required; it is not retroactively relabelled.

Explicit expanded ownership now includes diagnostic-only `src/sr.rs` hooks,
new `src/sr_observer.rs` and its integration test, beyond the original ctest
scope. See `issue-180-sr-observer.md`. The original matrix assembly, POTRF/POTRS
calls, error interpretation and parameter updates are unchanged; observations
are owned copies only while a thread-bound RAII token is enabled. Five observer
integration tests passed in 0.095 seconds, nextest
`22fd4709-32fa-41fb-8017-38c1e2a69598`, including the original positive-factor-INFO
followed by nonfinite substitution path. The disabled-payload unit test passed,
nextest `4387894d-eac6-45a7-9c8e-89dd3082010a`. Focused strict clippy passed on
the library and these owned test targets after public-doc/array-chunk repairs;
no lint allowance was added. MPI actual-system capture belongs to Wegener's
harness; serial fixtures are not an MPI acceptance budget. Fresh thirteen-model
long/workspace/native macOS acceptance remains pending parent freeze/coordination.
