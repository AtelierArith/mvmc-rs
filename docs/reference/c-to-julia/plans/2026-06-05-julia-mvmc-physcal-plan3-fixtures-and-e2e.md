# Plan 3 — PhysCal factored Green: runner, comparison helpers, C reference fixtures, e2e gate

This historical Julia plan records the original comparison requirements.
For current Rust and optional reference checks after #186, use the numerical
bounds and exact deterministic controls in
[NUMERICAL_COMPARISONS.md](../../../NUMERICAL_COMPARISONS.md).

Spec: [`../specs/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md`](../specs/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md)
Predecessors: Plan 1 (parser, merged PR #8), Plan 2 (factored compute, PR #9 open against `develop`).

> Status: DRAFT, design-reviewed 2026-06-05 (5-dimension adversarial pass, 31
> findings, verified against source). Corrections folded in below. No code
> changes yet; implementation starts after sign-off and after PR #9 merges to
> `develop`.
>
> **Design-review corrections applied (most-impactful first):**
> - **CRITICAL — runner double-init (§5.2 rewritten).** `vmc_phys_cal!` already
>   does its own `init_parameter!` (RNG match) + `init_qp_weight!` internally, so
>   the runner must NOT call them; the original "mirror run_para_opt" order would
>   double-consume the RNG and desync every fixture from C. Corrected order: the
>   runner does only load → overlays → sync, then `vmc_phys_cal!`.
> - **HIGH — comparison tolerance (§5.3).** Factored + direct-DC files are
>   product/squared accumulators (like `<H^2>` col-3); use `rtol=1e-9` for those,
>   `1e-10` only for the linear one-body file. Reuse `runtests.jl` TOL constants.
> - **HIGH — no run-gate env toggle exists (§6.3/§7/§8).** Integration is run by
>   explicit file invocation; only a model-subset filter env var exists. Invocation
>   model corrected; `MVMC_RUN_INTEGRATION` placeholder removed.
> - **MEDIUM** — exact-string compare must be raw-bytes, no strip (§5.3);
>   fixture isolation from the shared optimization namelist (§6.1); RNG-alignment
>   validated empirically on one system first (§5.2).
> - **LOW** — loader caveat list completed (SpinJastrow/DH omission, NSROptItrSmp
>   enforcement) and `read_opt_para_file!` explicitly scoped + guarded (§4).
>
> **User-review additions folded in (2026-06-05):** corroborated the runner
> double-init (High) and env-toggle / loader-scope findings, plus three new ones —
> drop the ambiguous `nsmp` runner kwarg (modpara is source of truth, §5.2);
> make `NVMCCalMode` file management explicit via a separate `physcal_inputs/`
> (§6.1/§6.2); raise non-float tokens as a stable `error()` with a test (§5.1).
> Decisions §9 resolved: 3a/3b split, branch after PR #9, macOS-local C reference
> with rtol/atol fallback, fixture isolation, no `nsmp` override.

## 1. What Plan 1/2 already delivered

The spec's "Proposed Changes" §1–§7 are done:

- §1/§2/§6 parser + `GreenTwoExTerm` + strict `parse_green_two_ex_def` (Plan 1).
- §3 canonical one-body list + 1-based index resolution
  (`build_canonical_cis_ajs_idx`, `resolve_cis_ajs_ckt_alt_idx`).
- §4 accumulation `w·local[idx0]·conj(local[idx1])` (`accumulate_factored_green!`).
- §5 output numbering `ismp + NDataIdxStart` + `output_dir`.
- §7 FSZ runtime guard (`validate_factored_green_supported`).
- Plus review fixes: fmt-1 (out/var truncate), NLanczosMode>0 reject, docstrings.

Unit/contract tests 1–6 from the spec's Testing section are effectively covered
by `test_unit_physcal_factored_green.jl` (incl. test 2's "constituent not in
greenone" case and test 3's 1-based resolution).

## 2. What Plan 3 must still deliver (spec §8 + Testing §7 + Fixtures + Docs)

1. A **PhysCal runner** `run_phys_cal_from_namelist` (spec Proposed §8 / Finding 7).
2. A **fixed-parameter loader** `read_opt_para_file!` (strict, non-perturbing,
   asserts consumption).
3. **Green-file comparison helpers** (one-body 6-col, direct 10-col, factored
   value-only; exact-string first, then `rtol=1e-10`/`atol=1e-12` fallback,
   recorded per quantity) (Finding 8).
4. **C reference fixtures** for 4 systems (`heisenberg_chain_real`,
   `heisenberg_chain_cmp`, `hubbard_chain_real`, `kondo_chain_real`):
   hand-authored `greentwoex.def` + committed C outputs + fixed `zqp_opt.dat`.
5. **e2e integration gate** (spec Testing §7).
6. **Docs**: manual `04_physics_calc.md`, `MVMCOptimizers.jl/README.md`,
   `test_unit/INDEX.md`, `test/integration/reference/README.md`.

## 3. Split: Plan 3a (pure Julia, no C) vs Plan 3b (needs C runs)

Plan 3a is fully testable without any C binary and unblocks review fast.
Plan 3b carries the external dependency (build + run C-mVMC offline).

| | Plan 3a | Plan 3b |
|---|---|---|
| `read_opt_para_file!` + tests | ✅ | |
| `run_phys_cal_from_namelist` + tests | ✅ | |
| comparison helpers + tests | ✅ | |
| hand-author `greentwoex.def` ×4 | | ✅ |
| generate + commit C references ×4 | | ✅ |
| e2e gate (`run_phys_cal_from_namelist` vs C ref) | | ✅ |
| docs | partial (API) | reference/README + manual flip |

## 4. Key findings that shape the design (verified against source)

- **`read_initial_def!` reads C's `zqp_opt.dat` layout for the 4 target systems.**
  C `OutputOptData` (`avevar.c:94-163`) with `NSROptItrSmp > 1` writes
  `6 floats (E re/im/var, E2 re/im/var) + 3 floats/param (re, im, var)` in the
  order Gutzwiller → Jastrow → SpinJastrow → DH2 → DH4 → RBM blocks → Slater →
  OptTrans (`Child_OutputOptData` `avevar.c:74-77`; `NProj` def `readdef.c:883-885`).
  `read_initial_def!` (`initial_params.jl:34`) expects `6 + 3·(n_proj + n_slater)`
  floats, reading (re, im) per triple. The 4 fixtures
  (heisenberg_real/cmp, hubbard_real, kondo_real) have NO SpinJastrow / DH / RBM /
  OrbitalGeneral, so C's `NProj` collapses to Gutzwiller+Jastrow = Julia's
  `n_proj`. Verified by token count of the two already-committed fixtures:
  hubbard `zqp_opt.dat` = 63 = 6 + 3·(2+5+12); kondo = 234 = 6 + 3·(5+7+64).
  So reuse is sound **for these systems**.
  - Caveat 1 (`NSROptItrSmp == 1`): the `avevar.c:106-109` branch writes
    `n=(2+NPara)` `(real, 0.0)` *pairs* — a different layout. Fixtures must use
    `NSROptItrSmp > 1`. This is currently enforced only implicitly (a pairs-layout
    file is always shorter than `6 + 3·NPara`, so the too-short check rejects it);
    add an explicit note in the fixture procedure rather than relying on that
    (design-review LOADER-2).
  - Caveat 2 (incomplete category coverage — design-review LOADER-1): C's `NProj`
    also includes **SpinJastrow + 6·NDH2 + 10·NDH4** parameters, written *before*
    Slater; `read_initial_def!` counts only Gutzwiller+Jastrow (and SpinJastrow
    isn't even parsed today). For a DH/SpinJastrow model the loader would
    mis-attribute triples and reject with a misleading message (fails loud, not
    silently wrong). Harmless for the 4 fixtures, but `read_opt_para_file!` must
    be **explicitly scoped** to Gutzwiller+Jastrow+Slater and **guard** non-empty
    `doublon_holon_2site_terms` / `doublon_holon_4site_terms` (and RBM), like the
    existing RBM guard. Do not frame the loader as a general zqp_opt.dat reader.
  - Caveat 3: `read_initial_def!` *warns and returns false* on RBM / OptTrans /
    short files; the PhysCal gate must instead **fail hard** and **assert
    consumption**. So `read_opt_para_file!` is a strict wrapper, not a rename.
  - Caveat 4: `read_initial_def!` does **not** perturb (perturbation is the C
    test driver's external `random.uniform`), so reusing it gives the required
    non-perturbing load.
- **Existing integration infra to build on:** `Julia-mVMC/test/integration/`
  has `ctest_equivalent.jl`, `ctest_models.jl`, `runtests.jl`, `tools/`, and
  `reference/<system>/{inputs,ctest_ref}` for all 4 target systems already.
  Plan 3 adds a PhysCal-only namelist + `greentwoex.def` and a new PhysCal
  reference subtree (the existing `ctest_ref/` is optimization mean/std, not
  Green files). See §6.1 on isolating the PhysCal inputs from the shared
  optimization namelist.
- **`run_para_opt_from_namelist`** (`run_para_opt_from_namelist.jl:59`) is a
  PARTIAL template (path parsing, seed convention, timers) — but its
  parameter-init steps must NOT be copied: `vmc_phys_cal!` already does
  `init_parameter!` (RNG match) and `init_qp_weight!` internally. See §5.2's
  CRITICAL correction; the runner does only load + overlays + sync before
  `vmc_phys_cal!`.

## 5. Plan 3a — detailed steps (TDD, pure Julia)

### 5.1 `read_opt_para_file!(data, path) -> Int`  (`MVMCOptimizers.jl/src/`)

- Strict, non-perturbing loader for a committed C `zqp_opt.dat` (NSROptItrSmp>1
  layout), scoped to the **Plan 3 fixture layout only** (Gutzwiller + Jastrow +
  Slater; no SpinJastrow / DoublonHolon / RBM / OptTrans — design-review
  LOADER-1 / user-review Low). Reuse the validated parsing of `read_initial_def!`
  but:
  - `error(...)` (not `@warn`+`false`) on: missing file, RBM-bearing model,
    **non-empty DoublonHolon (DH2/DH4) terms** (scope guard), too-short data,
    trailing non-triple floats.
  - **Non-float token → stable `error(...)` (user-review Low).** `read_initial_def!`
    uses `parse.(Float64, ...)`, which throws a bare `ArgumentError` on a garbled
    token. A public-ish strict loader should instead use `tryparse` and raise a
    clear, testable `error("read_opt_para_file!: non-numeric token '...' in <path>")`.
  - Return the number of parameters consumed (`n_proj + n_slater`) so the gate
    can assert it is `> 0` and matches expectations.
  - Same commit-after-validate discipline (no partial mutation on failure).
- Decision to confirm in review: implement as a thin strict wrapper that calls a
  shared internal `_load_para_triples!(data, values)` extracted from
  `read_initial_def!` (refactor read_initial_def! to delegate), so the two
  loaders cannot drift. (Avoids duplicating the triple layout in two places.)
- Tests (`test_unit/test_unit_read_opt_para.jl`):
  - golden: a small in-memory `zqp_opt.dat` string (6 + triples) loads into
    Gutzwiller/Jastrow/Orbital values exactly; returns the right count.
  - strict failures throw (missing file; empty; short; trailing 1–2 stray floats;
    **non-float token**; RBM model; **non-empty DoublonHolon**) — assert on
    message substrings, not types (codebase convention).
  - non-perturbation: loading twice yields identical values (no RNG touch).

### 5.2 `run_phys_cal_from_namelist(namelist_path; opt_para, mode, seed, output_dir)`

> **Signature correction (user-review Medium — `nsmp` is ambiguous for PhysCal).**
> The original draft carried `nsmp` over from `run_para_opt_from_namelist`, but in
> PhysCal three different counts are easy to confuse: `NDataQtySmp` (number of
> per-sampling output sets / files), `NVMCSample` (Monte-Carlo samples per set),
> and `NSROptItrSmp` (only relevant to the *opt-side* `zqp_opt.dat` layout). A
> single `nsmp` override is a footgun. **Decision: drop the override; `modpara`
> (from the namelist) is the source of truth** for `NDataQtySmp` / `NVMCSample`.
> If an override is ever needed, expose it as an explicitly named kwarg
> (`n_data_qty_smp` / `nvmc_sample`), never a bare `nsmp`. The deterministic gate
> wants the fixture's modpara verbatim anyway.

> **CRITICAL design correction (design-review F1/F2, 3× confirmed).** The runner
> must NOT mirror `run_para_opt_from_namelist`'s parameter-init steps.
> `vmc_phys_cal!` is **not** structured like `vmc_para_opt!`: it ALREADY does its
> own `init_parameter!` internally — a save→`init_parameter!(rng)`→restore dance
> (`vmc_phys_cal.jl:69-86`) whose sole purpose is to consume the SFMT RNG exactly
> once to match C's single `InitParameter()` — AND it ALREADY calls
> `init_qp_weight!` (`vmc_phys_cal.jl:160`) and `update_slater_elm_*`
> (`:162-168`). If the runner also calls `init_parameter!` / `init_qp_weight!`,
> the RNG is consumed **twice** (~`n_slater` extra draws) before sampling, so the
> Monte-Carlo stream desyncs from C and **all 4 e2e fixtures fail** by
> sampling-noise magnitude (≫ rtol), not BLAS rounding. This would be
> misdiagnosed as a platform/BLAS reproducibility problem (§6.2).

- Reuse `run_para_opt_from_namelist`'s scaffolding only for: relative-path
  parsing, the RNG seed convention (`rnd_seed>0 ? rnd_seed : 11272`), and
  optional timers. The parameter source is the committed fixed file and the
  terminal call is `vmc_phys_cal!`, which owns init_parameter!/init_qp_weight!.
- **Corrected init order** (the runner does load + overlays + sync only; the RNG
  consumption and qp-weight/Slater init happen *inside* `vmc_phys_cal!`):
  1. `parse_expert_mode_files(namelist)`
  2. seed RNG (SFMT19937, C convention)
  3. `read_opt_para_file!(data, opt_para)` — overwrite term `.value`s with the
     fixed params; assert consumed `> 0`. **No `init_parameter!` precondition**
     (design-review F2): the loader only sets `.value` on parser-created
     `gutzwiller/jastrow/orbital_terms`, and `n_slater` comes from
     `modpara.n_orbital_idx` (set at parse). So it runs right after parse.
  4. `read_input_parameters!(data)` — In*.def overlays (C `ReadInputParameters`,
     after the fixed load so overlays win, matching C order).
  5. `sync_modified_parameter!(data)` — Slater rescale + GJ shift (C
     `SyncModifiedParameter`).
  6. `vmc_phys_cal!(data; rng, output_dir)` — its internal save/restore
     preserves the loaded+synced values across its single internal
     `init_parameter!` (RNG match), then `init_qp_weight!` + `update_slater_elm`
     use those values. **No extra init in the runner.**
- Verify during implementation (against `vmc_phys_cal.jl:69-168`): that the
  save/restore at :70-86 covers exactly the params `sync_modified_parameter!`
  mutates (gutzwiller/jastrow/orbital — it does), so the synced values survive
  the internal init_parameter!. Confirm `vmc_phys_cal!` does NOT itself call
  `sync_modified_parameter!` (it does not, as of the merged Plan 2), so the
  runner must.
- `opt_para` resolution: explicit path required for the gate (no `:auto`
  guessing — the gate must be deterministic about which file it consumed).
- The validate_supported_modpara / validate_factored_green_supported guards
  already run inside `vmc_phys_cal!`; the runner does not duplicate them.
- **RNG-alignment validation (design-review F5):** before committing all 4
  fixtures, validate empirically on ONE system that the SFMT consumption before
  the first `VMCMakeSample` equals C's single-`InitParameter` count, so a later
  tolerance miss is correctly attributed (RNG desync vs BLAS noise). A unit/debug
  assertion on the pre-sampling RNG draw count is cheap insurance.
- Tests (`test_unit/test_unit_run_phys_cal_runner.jl`):
  - argument validation (mode ∈ {:real,:cmp,:fsz}; positive nsmp; missing
    opt_para errors before sampling).
  - the runner routes through the guards: FSZ+factored rejects through the
    runner; a missing/malformed opt_para errors before sampling.
  - (Real numeric check lives in the e2e gate, 3b.)

### 5.3 Comparison helpers  (`test/integration/tools/` — test-only)

- `compare_green_one(julia_path, c_path; rtol, atol)` — 6 columns
  `ri si rj sj real imag`; integer columns must match exactly.
- `compare_green_two_dc(...)` — 10 columns `ri si rj sj rk sk rl sl real imag`.
- `compare_green_factored(...)` — value-only, 2 floats per factored term, all on
  one line; parse and compare pairwise.
- **Per-quantity tolerance (design-review F1, confirmed high).** Do NOT use a
  single uniform `1e-10`. The factored value
  `Σ w·local[idx0]·conj(local[idx1])` and the direct-DC value (4-operator
  per-sample product) are product/squared accumulators — the same structural
  form as `zvo_out` col-3 `<H^2> = Σ w·conj(e)·e`, which the existing harness
  (`runtests.jl:21-29`) already grants `TOL_LOOSE = 1e-9` (vs `TOL_DEFAULT =
  1e-10` for linear accumulators). The one-body file (`phys_cis_ajs += w·local`)
  is linear → `1e-10`. So:
  - one-body (`zvo_cisajs`): `rtol = 1e-10`, `atol = 1e-12`.
  - factored (`zvo_cisajscktaltex`) **and** direct-DC (`zvo_cisajscktalt`):
    `rtol = 1e-9`, `atol = 1e-12`.
  Reuse `runtests.jl`'s `TOL_DEFAULT`/`TOL_LOOSE` constants for consistency.
- **"Exact-string first" = raw bytes (design-review F2, confirmed medium).** The
  byte streams DO match (C `fprintf` and Julia `@printf` agree on `% .18e`,
  including `-0.0`, 3-digit exponents, rounding, and each format's trailing
  whitespace / blank line — verified). But the exact-string check must read with
  `read(path, String)` (raw, NO `strip`/`split`): the two existing harnesses
  (`runtests.jl:78-79`, `ctest_equivalent.jl:32-36`) `strip()`+`split()`, which
  would silently discard the trailing spaces and trailing blank line that the
  bit-for-bit Completion Condition relies on. Only the numeric-fallback path
  tokenizes. Add a self-test that one Julia-written file equals a captured C
  reference byte-for-byte, so "bit-for-bit" is actually exercised.
- Each returns a struct: `{exact::Bool, max_abs_err, max_rel_err, n_values,
  fallback_used::Bool, tol_used}` and the gate logs per-file/quantity whether it
  matched bit-for-bit or via numeric fallback, and at which tolerance (Finding 8:
  "record any fallback per file and quantity").
- Tests (`test/integration/.../test helpers`): identical files (exact=true);
  within-tol diffs (exact=false, fallback ok at the right per-quantity bound);
  beyond-tol (fail); column-count/index mismatch (hard fail regardless of tol);
  raw-byte self-test (trailing space / blank line preserved).

## 6. Plan 3b — detailed steps (needs C-mVMC offline, OMP_NUM_THREADS=1)

### 6.1 Author `greentwoex.def` for the 4 systems

- Small, hand-written factored pair set per system.
- **Fixture isolation (design-review F4 + user-review Medium — resolved).** Adding
  `TwoBodyGEx greentwoex.def` to a system's EXISTING shared `namelist.def` would
  couple that system's *already-green optimization ctest* to `greentwoex.def`
  parsing (Plan 1 made it fatal-if-present), and committing `NVMCCalMode=1` into
  the shared `modpara.def` would disturb the opt integration. **Decision: use a
  separate `physcal_inputs/` set per system** (its own namelist + `greentwoex.def`
  + `modpara` with `NVMCCalMode=1`), referenced only by the gate. The committed
  optimization `inputs/` (NVMCCalMode=0) stays untouched, so a fixture bug cannot
  regress the optimization suite. Add a per-fixture parser unit test for the new
  `greentwoex.def`.
- Committed fixtures may keep all constituents within `greenone.def`
  (simplicity); the "constituent absent from greenone" C-only behavior is
  already locked by unit/contract test 2 (no C run needed) — per spec §9.
- Use real-valued definitions for `_real` systems; the `_cmp` system exercises
  the complex path (conj matters).

### 6.2 Generate + commit C references

> **NVMCCalMode file management (user-review Medium).** Be explicit about which
> files are committed vs temporarily rewritten, so the PhysCal fixture cannot
> regress the optimization integration:
> - The system's `inputs/modpara.def` that the optimization ctest uses stays at
>   `NVMCCalMode = 0` (committed, unchanged).
> - The PhysCal run uses a **separate committed input set** — a
>   `physcal_inputs/` dir (or a distinct `modpara_physcal.def` + namelist) with
>   `NVMCCalMode = 1` — NOT an in-place edit of the shared `modpara.def`.
> - During offline generation, the optimize step (NVMCCalMode=0) and the PhysCal
>   step (NVMCCalMode=1) run in temp/staging copies; only the resulting committed
>   artifacts (fixed `zqp_opt.dat` + the three Green files + the PhysCal input
>   set) land under version control.

Per system, with `OMP_NUM_THREADS=1` and a recorded mVMC commit/build:
1. Build C-mVMC (the project `mVMC/`); set `NSROptItrSmp > 1` so `zqp_opt.dat` is
   the 6+triples layout (§4 caveat 1) — note this in the committed metadata.
2. Optimize (`NVMCCalMode=0`, opt input set) → `zqp_opt.dat` (the fixed params).
3. PhysCal (`NVMCCalMode=1`, the PhysCal input set incl. `greentwoex.def`, fed the
   `zqp_opt.dat` from step 2) → `zvo_cisajs_001.dat`, `zvo_cisajscktalt_001.dat`
   (DC), `zvo_cisajscktaltex_001.dat` (factored).
4. Commit under `test/integration/reference/<system>/physcal_ref/`:
   the PhysCal input set + `zqp_opt.dat` (committed gate input) + the three Green
   outputs (committed expected outputs). Keep the existing `inputs/` (opt,
   NVMCCalMode=0) and `ctest_ref/` untouched.
5. Record provenance in `reference/README.md` (mVMC commit, build flags,
   **generation platform/OS**, `OMP_NUM_THREADS=1`, NSROptItrSmp, seed,
   NDataQtySmp/NVMCSample) per CLAUDE.md policy.

C-reference platform (resolved, user-review): generate on **macOS local +
`OMP_NUM_THREADS=1`** to match existing-reference provenance. Do **not** expect
byte-level cross-platform identity; the per-quantity `rtol/atol` fallback (§5.3)
is the contract, and the generation environment is recorded in metadata. (If the
gate runs in CI on Linux too, the fallback bounds — not exact-string — are what
must hold there.)

### 6.3 e2e gate (spec Testing §7)

> **Invocation correction (design-review fixture-F1 / scope-F2 / runner-F6,
> confirmed high).** There is **no run-gating env toggle** in the existing
> integration suite. `test/integration/runtests.jl` runs unconditionally when
> invoked by file path; `ctest_equivalent.jl`'s only env var is
> `JULIA_MVMC_CTEST_MODELS`, a *model subset filter*, not an on/off gate. So the
> earlier "reuse the existing env toggle" framing was wrong.

- New integration test file under `test/integration/`, run by **explicit file
  invocation** (not `Pkg.test()`), like `runtests.jl` / `ctest_equivalent.jl`.
  Decide invocation explicitly:
  - (a) add it as a third explicit CI step. Then it MUST be cross-platform green,
    so the per-quantity `rtol/atol` contract (§5.3) and fixture provenance must be
    committed up front — the cross-platform reproducibility question (§6.2) is a
    **decision to settle before merge, not an open question**; or
  - (b) introduce a genuine NEW run-gate env var (e.g.
    `JULIA_MVMC_RUN_PHYSCAL_GATE`) and document that the existing suite has none.
  Recommended: (a) with a `JULIA_MVMC_CTEST_MODELS`-style subset filter for
  selective local runs.
- For each of the 4 systems: copy `physcal_inputs/` (incl. `greentwoex.def`,
  NVMCCalMode=1) + the fixed `zqp_opt.dat` into a fresh `mktempdir()`, run
  `run_phys_cal_from_namelist(namelist; opt_para=..., mode=..., output_dir=tmp)`,
  then compare the three produced Green files to the committed reference with the
  §5.3 helpers (per-quantity tolerance).
- Assert the fixed parameters were consumed (loader return count > 0 and equals
  the system's parameter count); fail on missing/malformed `zqp_opt.dat`.
- References are read-only; the run writes only into the temp dir.

### 6.4 Docs

- `docs/manual/04_physics_calc.md` + `MVMCOptimizers.jl/README.md`: move
  factored/product two-body Green (`cisajscktaltex` / `TwoBodyGEx`) from
  "not yet ported" to "supported (non-FSZ)", FSZ caveat noted.
- `test/integration/reference/README.md`: PhysCal fixture generation procedure
  (the §6.2 steps + provenance fields).
- `MVMCOptimizers.jl/test_unit/INDEX.md`: register §5.1/§5.2 unit tests.

## 7. Testing matrix

- Subpackage unit (all CI matrix, no C): read_opt_para (§5.1), runner guards
  (§5.2), comparison helpers (§5.3). Plus the existing Plan-1/2 contract tests.
- Integration (committed C ref, no C binary at test time): the 4-system e2e gate
  (§6.3), run by explicit file invocation (there is NO run-gate env toggle — see
  §6.3); optional `JULIA_MVMC_CTEST_MODELS`-style subset filter.
- Acceptance (spec §"Completion Conditions"): one-body `zvo_cisajs` matches C
  bit-for-bit or within `rtol=1e-10`/`atol=1e-12`; factored `zvo_cisajscktaltex`
  and direct `zvo_cisajscktalt` within `rtol=1e-9`/`atol=1e-12` (per-quantity,
  §5.3); `*_001.dat` names; params consumed; FSZ rejected; manual updated.

## 8. Verification commands

```
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia
# unit (3a) — in MVMCOptimizers.jl:
OMP_NUM_THREADS=1 "$JL" --project=@. -e 'using Pkg; Pkg.test()'
# integration (3b) — explicit file invocation from the Julia-mVMC root, NO env gate
# (mirrors how runtests.jl / ctest_equivalent.jl are run):
OMP_NUM_THREADS=1 "$JL" --project=@. test/integration/<physcal_gate_file>.jl
# optional subset filter, like ctest_equivalent.jl:
OMP_NUM_THREADS=1 JULIA_MVMC_CTEST_MODELS=hubbard_chain_real "$JL" --project=@. test/integration/<physcal_gate_file>.jl
```

## 9. Decisions

Resolved (user-review 2026-06-05):

1. ✅ **3a/3b split: yes** — but 3a lands only after the runner's init
   responsibility is corrected (§5.2), else 3b's C-reference comparison flakes.
2. ✅ **Branch point: from merged `develop` after PR #9** — not on the Plan 2
   branch (keeps the review-fix and the Plan 3 loader/runner design separate).
3. ✅ **C-reference platform: macOS local + `OMP_NUM_THREADS=1`** to match
   existing provenance; do not expect bit-level cross-platform exact — rely on
   the per-quantity `rtol/atol` fallback (§5.3) and record the generation
   environment in metadata (§6.2).
4. ✅ **Fixture isolation: separate `physcal_inputs/`** with `NVMCCalMode=1`;
   committed opt `inputs/` (NVMCCalMode=0) untouched (§6.1/§6.2).
5. ✅ **Runner sample-count override: dropped** — `modpara` is the source of
   truth for `NDataQtySmp`/`NVMCSample`; no ambiguous `nsmp` kwarg (§5.2).

Still to confirm at implementation start:

6. **`read_opt_para_file!` via a shared `_load_para_triples!` refactor of
   `read_initial_def!`** (single source of the triple layout), scoped to
   Gutzwiller+Jastrow+Slater with explicit DH/RBM guards (§4/§5.1). OK to
   refactor `read_initial_def!` to delegate?
7. **e2e gate invocation:** (a) explicit CI step that must be cross-platform
   green at the `rtol/atol` bounds, vs (b) a new opt-in run-gate env var so the
   gate is local/manual until references are trusted. Recommended: (a) once one
   system's RNG alignment + tolerances are validated (§5.2/§6.2); (b) is the safe
   interim if cross-platform behavior is still uncertain.

## 10. Key C reference points (for implementers)

- `avevar.c:94-163` `OutputOptData` / `:65-80` `Child_OutputOptData` —
  `zqp_opt.dat` layout (6 + triples for NSROptItrSmp>1).
- `vmcmain.c` PhysCal init order; `initfile.c:99/104` — `_cisajscktaltex_%03d`
  / `_cisajscktalt_%03d` names.
- `vmcmain.c:665-689` — Green output formats (6-col one-body, value-only
  factored, 10-col DC).
- `calgrn.c:72-79`, `:113-118` — one-body precompute + factored accumulation.
