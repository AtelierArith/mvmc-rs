---
date: 2026-06-06
datetime: 2026-06-06 09:58 JST
model: GPT-5 (Codex)
status: plan
topic: Julia-mVMC v0.3 DH2/DH4 input compatibility design-review plan
updated: 2026-06-06 11:55 JST
review:
  document: docs/reviews/2026-06-06-julia-mvmc-dh2-dh4-compat-plan-review.md
  status: D1-D11 corrections folded into this plan; open questions Q1 (legacy types) and Q2 (DH fixtures) resolved
target:
  repository: Julia-mVMC
  base: develop
  scope: DH2/DH4, InDH2/InDH4, projection layout, projection counts
---

# Julia-mVMC v0.3 DH2/DH4 compatibility plan

## Review status

Independent review:
[`docs/reviews/2026-06-06-julia-mvmc-dh2-dh4-compat-plan-review.md`](../../reviews/2026-06-06-julia-mvmc-dh2-dh4-compat-plan-review.md).

The review found that the core C contract in this plan is correct, but D1-D11
needed to be made explicit before implementation. This revision folds those
corrections into the body of the plan. The most important changes are:

- DH shift requires all Gutzwiller real `OptFlag` entries to be optimized, not
  merely the DH slice.
- All projection-count buffers must be sized with `projection_layout(data).n_proj`.
- DH definition-file opt tables use file order and ignore the first column,
  while `InDH2` / `InDH4` overlays use the first column as the target local index.
- DH-1 must not leave DH inputs in a half-supported runtime state.

Both prior open questions are now resolved (section 8): legacy `DoublonHolon*Term`
types are removed in DH-1, and DH fixtures use two tiers (hand-authored tiny +
StdFace-base-plus-hand-authored-DH overlay), since StdFace cannot emit DH.

## 1. Goal

`DH2` / `DH4` and `InDH2` / `InDH4` を warn-only stub から外し、C-mVMC と同じ意味で計算へ反映する。

Definition of Done:

- `namelist.def` の C keyword `DH2` / `DH4` を読み、C と同じ index table と parameter slice を構築する。
- `InDH2` / `InDH4` が C と同じ局所 index order で projection parameter を上書きする。
- `NProj`、`OptFlag` offset、`init_parameter!`、`initial.def` / `zqp_opt.dat` loader、`LogProjVal` / `LogProjRatio`、`MakeProjCnt` / `UpdateProjCnt`、SR derivative が DH slice を含む。
- DH なしの既存 fixtures と tests は数値・出力ともに変わらない。
- DH ありの committed C reference fixture が pass する。

Non-goals for this plan:

- `InOrbitalParallel`、`InOptTrans`、`OptTrans` block は別 plan で扱う。
- BackFlow は pending のまま。DH を BackFlow path へ流さない。
- SpinJastrow の新規実装はしない。ただし C layout では DH offset の前に `NSpinJastrowIdx` が入るため、projection layout helper は SpinJastrow slot を明示的に持つ。SpinJastrow input が present の場合は、silent zero-slot ではなく hard fail にする。

## 2. C compatibility contract

### 2.1 Projection layout

C の `Proj` layout は以下の順番。

```text
Gutzwiller | Jastrow | SpinJastrow | DH2 | DH4 | RBM blocks | Slater | OptTrans
```

この plan が扱う `NProj` は RBM / Slater より前の projection slice。

```text
dh2_offset = NGutzwillerIdx + NJastrowIdx + NSpinJastrowIdx
dh4_offset = dh2_offset + 6 * NDoublonHolon2siteIdx
n_proj     = dh4_offset + 10 * NDoublonHolon4siteIdx
```

DH2 の parameter index:

```text
idx = dh2_offset + xn + (xi + 2*xm) * NDoublonHolon2siteIdx
```

DH4 の parameter index:

```text
idx = dh4_offset + xn + (xi + 2*xm) * NDoublonHolon4siteIdx
```

ここで `xn` は DH index、`xi=0` は中心 site が holon、`xi=1` は中心 site が doublon。DH2 の `xm` は周囲 2 site の反対状態数 `0..2`、DH4 の `xm` は周囲 4 site の反対状態数 `0..4`。

### 2.2 DH definition files

`dh2.def`:

- Header 5 lines are fixed-format compatible.
- Main table has `Nsite * NDoublonHolon2siteIdx` rows:

```text
i  x0  x1  n
```

- `ArrayIdx[n][2*i] = x0`, `ArrayIdx[n][2*i+1] = x1`。
- Optimization table has `6 * NDoublonHolon2siteIdx` rows:

```text
local_param_index  opt_flag
```

`dh4.def`:

- Main table has `Nsite * NDoublonHolon4siteIdx` rows:

```text
i  x0  x1  x2  x3  n
```

- `ArrayIdx[n][4*i + k] = xk`。
- Optimization table has `10 * NDoublonHolon4siteIdx` rows.

Important distinction:

- In `dh2.def` / `dh4.def`, C reads the first column of the optimization table but discards it. Optimization flags are stored by file row order.
- In `InDH2` / `InDH4`, C uses the first column as the target local DH parameter index.

Julia parser must validate counts exactly. C does not protect every malformed `n` path strongly, but Julia should fail loud on out-of-range index, duplicate or missing rows, and invalid site ids.

### 2.3 Projection counts

`MakeProjCnt` starts from zero and fills all projection counts.

For DH2/DH4:

- singly occupied center site is skipped.
- holon center counts neighboring doublons.
- doublon center counts neighboring holons.
- only the corresponding DH parameter bin is incremented by `1`.

`UpdateProjCnt` first updates Gutzwiller / Jastrow / SpinJastrow incrementally. If any DH exists, C then zeroes the whole DH tail and recomputes DH2/DH4 from the updated `eleNum`. Julia should copy this behavior first. It is simpler and safer than a custom incremental DH update.

FSZ path has the same DH recompute requirement for `ri != rj`. For the on-site spin-flip case `ri == rj`, C returns after SpinJastrow update and does not recompute DH because total occupancy is unchanged. Julia should preserve this early-return structure and test it.

### 2.4 Log projection value and ratio

C uses only `creal(Proj[idx])`:

```text
LogProjVal   = sum(real(Proj[idx]) * projCnt[idx])
LogProjRatio = sum(real(Proj[idx]) * (projCntNew[idx] - projCntOld[idx]))
```

Complex DH values may be parsed and stored, but projection count weighting uses the real part.

### 2.5 DH shift during parameter sync

C `SyncModifiedParameter` does this order:

1. If `FlagShiftDH2 == 1`, run `shiftDH2()` and add its returned value to `gShift`.
2. If `FlagShiftDH4 == 1`, run `shiftDH4()` and add its returned value to `gShift`.
3. Add `gShift` to every Gutzwiller parameter.
4. If `FlagShiftGJ == 1`, run `shiftGJ()`.
5. Rescale Slater.

`shiftDH2()` groups the 3 `xm` bins for each `(d/h, xn)` and subtracts their average. `shiftDH4()` does the same over 5 bins. The subtracted averages are accumulated and shifted back into Gutzwiller.

The shift flags are enabled only when all of these hold:

- `NGutzwillerIdx > 0`
- every Gutzwiller real-part `OptFlag[2*i]` is optimized
- every real-part `OptFlag` entry in the corresponding DH slice is optimized

If any Gutzwiller parameter is fixed, C returns early from `SetFlagShift` and leaves `FlagShiftDH2` / `FlagShiftDH4` disabled, even if all DH parameters are optimized.

## 3. Current Julia gaps to close

- `parse_file_by_type!` has branches for `DoublonHolon2Site` / `DoublonHolon4Site`, but C namelist keywords are `DH2` / `DH4`.
- Current `DoublonHolon2SiteTerm` / `DoublonHolon4SiteTerm` are value-bearing local terms. They cannot represent C's `ArrayIdx[idx][2*Nsite]` / `ArrayIdx[idx][4*Nsite]` tables.
- There is no canonical projection layout helper. `n_proj = length(gutzwiller_terms) + length(jastrow_terms)` is duplicated across parser utilities, optimizer setup, sampling, parameter init, and opt-flag utilities.
- `n_proj_bf` currently counts DH terms and routes nonzero DH models into BackFlow sampling paths. DH must stay in the normal projection-correlator path.
- `make_proj_cnt!`, `update_proj_cnt!`, FSZ update, `log_proj_val`, and `log_proj_ratio` ignore DH.
- `init_parameter!` zeroes only Gutzwiller/Jastrow and uses a DH-free `NProj` when computing Slater `OptFlag` offsets.
- `read_initial_def!` / `read_opt_para_file!` explicitly reject DH models because their `NProj` slice is incomplete.
- `read_input_parameters!` recognizes `InDH2` / `InDH4` but warns and skips.
- `sync_modified_parameter!` implements GJ shift and Slater rescale, but not DH2/DH4 shift.
- Manual docs still describe `dh2.def` / `dh4.def` as stubs.

## 4. Proposed design

### 4.1 Add a canonical projection layout API

Add this in `MVMCExpertModeParsers.jl` so both parser utilities and `MVMCOptimizers.jl` can use it:

```julia
struct ProjectionLayout
    n_gutzwiller::Int
    n_jastrow::Int
    n_spinjastrow::Int
    n_dh2::Int
    n_dh4::Int
    gutzwiller_offset::Int
    jastrow_offset::Int
    spinjastrow_offset::Int
    dh2_offset::Int
    dh4_offset::Int
    n_proj::Int
end
```

Required helper functions:

- `projection_layout(data)::ProjectionLayout`
- `n_projection_parameters(data)::Int`
- `projection_parameters(data, layout)::Vector{ComplexF64}` in exact C order
- `ensure_projection_parameter_storage!(data, layout)` if values are stored outside the old term vectors
- `projection_opt_flag_index(layout, local_proj_idx0)::Int`
- `slater_opt_flag_offset(data, layout)::Int`

Review requirement: every existing `length(data.gutzwiller_terms) + length(data.jastrow_terms)` that affects parameter layout or state size must either be replaced by this helper or explicitly documented as no-DH-only test code.

State construction requirement: every projection-count buffer in `VMCOptimizationState`, `ElectronConfiguration`, sampling workspaces, burn-in storage, and temporary count arrays must be allocated with `projection_layout(data).n_proj`. Add constructor-level assertions where practical so a DH input cannot create a `proj_cnt` shorter than the DH offsets.

### 4.2 Replace the DH data model with C index-table storage

Preferred storage:

```julia
struct DoublonHolon2SiteIndex
    neighbors::Matrix{Int}  # Nsite x 2, zero-based site ids
end

struct DoublonHolon4SiteIndex
    neighbors::Matrix{Int}  # Nsite x 4, zero-based site ids
end
```

Add fields to `ExpertModeData`:

- `doublon_holon_2site_indices::Vector{DoublonHolon2SiteIndex}`
- `doublon_holon_4site_indices::Vector{DoublonHolon4SiteIndex}`
- `doublon_holon_2site_params::Vector{ComplexF64}` length `6 * n_dh2`
- `doublon_holon_4site_params::Vector{ComplexF64}` length `10 * n_dh4`
- `doublon_holon_2site_opt_flags::Vector{Bool}` length `6 * n_dh2`
- `doublon_holon_4site_opt_flags::Vector{Bool}` length `10 * n_dh4`
- `doublon_holon_2site_complex::Bool`
- `doublon_holon_4site_complex::Bool`

The old `DoublonHolon2SiteTerm` / `DoublonHolon4SiteTerm` names should not remain the parser's runtime representation.

Resolution (Q1): remove the legacy value-bearing types and the `doublon_holon_*site_terms` fields on `ExpertModeData` immediately in PR DH-1. These types are internal only (not exported, absent from `examples/` and the manual), and their readers are bounded: `doublon_holon_parser.jl`, `expert_types.jl`, `validation.jl`, `read_input_parameters.jl`, `MVMCExpertModeParsers.jl`, plus `initial_params.jl`, `vmc_para_opt.jl`, `vmc_phys_cal.jl` (which must change for the DH-rejection and `n_proj_bf` work anyway), and one test `test_unit/test_unit_read_opt_para.jl`. Migrate all of them to the index-table model in the same PR rather than keeping a parallel field, which would re-introduce the layout-drift risk this plan removes. A deprecated constructor that builds an index table from legacy `(site1, site2, value)` arguments may be retained for test convenience only; the old runtime field on `ExpertModeData` must not survive.

Matrix orientation is part of the contract: for DH index `n`, `indices[n+1].neighbors[i+1, :]` must equal C's `ArrayIdx[n][2*i : 2*i+1]` for DH2 and `ArrayIdx[n][4*i : 4*i+3]` for DH4. All stored site ids remain zero-based.

### 4.3 Parser and opt flags

Parser changes:

- Add `DH2` and `DH4` entries to the keyword map and matching branches in `parse_file_by_type!`. Legacy long names may exist only as explicitly documented compatibility aliases.
- Parse fixed headers and exact row counts.
- Fill `neighbors` by DH index `n`, not by file order.
- Parse the DH optimization table into local DH opt flags by file row order. Do not trust the first column in `dh2.def` / `dh4.def`; C discards it.
- Initialize `doublon_holon_*_params` to zeros in C slice order.
- Fold DH complex flags into `get_all_complex_flag(data)`.
- Mirror C's imaginary opt-flag convention: if DH `ComplexType > 0`, the imaginary flag mirrors the real flag; otherwise it is false.

OptFlag changes:

- Add `set_dh_opt_flags!(data, layout)`.
- Use `layout.n_proj` for RBM and Slater offsets.
- Update `is_slater_optimized`, `get_slater_opt_flag_index`, `set_orbital_opt_flags!`, and RBM opt-flag offset code.

### 4.4 Parameter initialization and fixed-parameter loading

`init_parameter!`:

- Zero all projection parameters: Gutzwiller, Jastrow, SpinJastrow if present, DH2, DH4.
- Use `layout.n_proj` when deciding Slater `OptFlag` offsets and RNG consumption.
- Include DH complex flags in `AllComplexFlag`.

`read_initial_def!` / `read_opt_para_file!`:

- Remove the DH scope guard only after DH projection storage exists.
- Expected float count becomes `6 + 3 * (layout.n_proj + n_rbm + n_slater + n_opttrans_if_supported)`.
- For this DH plan, RBM / OptTrans may stay guarded as before, but DH must be consumed.
- Load DH2/DH4 triples between Jastrow/SpinJastrow and Slater in exact C order.
- Existing no-DH tests must still pass without changing fixture bytes.

`read_input_parameters!`:

- `InDH2` validates file count equals `n_dh2`.
- `InDH2` local indices `0..6*n_dh2-1` overwrite `doublon_holon_2site_params[local+1]`. Unlike `dh2.def` opt tables, the index column is meaningful here.
- `InDH4` local indices `0..10*n_dh4-1` overwrite `doublon_holon_4site_params[local+1]`. Unlike `dh4.def` opt tables, the index column is meaningful here.
- Present `InDH2` / `InDH4` files validate local-index range, duplicate entries, missing entries, and short files. Violations are hard errors in strict runners.
- Missing optional files may keep current skip behavior. Malformed present files should fail or produce a stable hard error in strict runners.

### 4.5 Projection count and sampling path

Add helpers in `vmc_sampling.jl`:

- `_count_dh2!(proj_cnt, ele_num, data, layout)`
- `_count_dh4!(proj_cnt, ele_num, data, layout)`
- `_recompute_dh_counts!(proj_cnt, ele_num, data, layout)`

Use these in:

- `make_proj_cnt!`
- `update_proj_cnt!`
- FSZ `make/update` equivalents

Implementation rule:

- Construct all state/count buffers with `layout.n_proj`, not a local Gutzwiller+Jastrow count. Assert this in constructors or before count updates.
- Match C's recompute-on-update strategy first.
- Preserve the FSZ `ri == rj` early return: no DH recompute is needed for on-site spin flips because occupancy is unchanged.
- Do not route DH through `vmc_bf_make_sample*` or `vmc_bf_main_cal*`.
- `n_proj_bf` should mean BackFlow only, or be removed/renamed if it is just a branch guard.

`log_proj_val` and `log_proj_ratio` should use `projection_parameters(data, layout)` and must assert that count vector length equals `layout.n_proj`.

### 4.6 Parameter sync

Extend `sync_modified_parameter!`:

- Compute `flag_shift_dh2` and `flag_shift_dh4` with C's full rule: Gutzwiller exists, all Gutzwiller real-part opt flags are optimized, and all real-part opt flags in the corresponding DH slice are optimized.
- Implement `shift_dh2!` and `shift_dh4!` over the DH parameter vectors.
- Add returned DH shift to every Gutzwiller value before the existing GJ shift.
- Keep Slater rescale behavior unchanged.

### 4.7 Docs

Update:

- `Julia-mVMC/docs/manual/02_input_files.md`: `DH2` / `DH4` and `InDH2` / `InDH4` move from stub/warn-only to supported, with remaining caveats.
- `Julia-mVMC/docs/manual/05_compatibility.md`: warn-only table updated.
- `MVMCOptimizers.jl/test_unit/INDEX.md`: add DH parser/count/loader contract tests.
- `docs/TODO.md` or roadmap notes only if the remaining warn-only items need tracking after DH is complete.

## 5. Test plan

Unit / contract tests:

- Parser: minimal DH2 and DH4 files with exact C row counts parse into index tables and opt flags.
- Parser failure: missing rows, extra rows, out-of-range site ids, invalid DH index, invalid local parameter index.
- Parser contract: `dh2.def` / `dh4.def` opt-table first column is ignored and file row order controls opt flags.
- Layout: `n_proj` and offsets for no-DH, DH2-only, DH4-only, DH2+DH4, and SpinJastrow-count placeholder cases.
- State sizing: `VMCOptimizationState` and all projection-count buffers have length `layout.n_proj` for DH inputs.
- Unsupported SpinJastrow: present SpinJastrow input hard-fails until SpinJastrow is implemented.
- Projection counts: hand-computed small occupancies for DH2 and DH4, including singly occupied center skip, holon counting neighboring doublons, and doublon counting neighboring holons.
- Update invariant: after `update_ele_config!` plus `update_proj_cnt!`, DH counts equal a fresh `make_proj_cnt!`.
- FSZ invariant: same recompute invariant for `UpdateProjCnt_fsz` equivalent, including `ri == rj` on-site spin flip where DH counts remain unchanged.
- Log value/ratio: full `projection_params` vector, real-part-only weighting, count length checks.
- `init_parameter!`: DH params zeroed; Slater `OptFlag` offset uses `layout.n_proj`; no-DH RNG sequence unchanged.
- Loader: `zqp_opt.dat` / `initial.def` with G/J/DH/Slater triples loads DH slice in C order and leaves Slater alignment correct.
- `InDH2` / `InDH4`: local overlay indices overwrite the correct DH slice and win after fixed-parameter load; out-of-range, duplicate, missing, and short-file cases hard-fail.
- DH shift: `shift_dh2!` and `shift_dh4!` match C formulas and add the DH shift to Gutzwiller before GJ shift.
- DH shift guard: if any Gutzwiller real opt flag is fixed, DH shift is disabled even when all DH flags are optimized.

Integration tests:

- Existing no-DH ctest-equivalent and PhysCal fixtures must remain pass.
- Add at least one DH2 fixture and one DH4 fixture generated by local C-mVMC.

Resolution (Q2): use two fixture tiers, because StdFace does not emit DH (there is no DH generation code in `mVMC/src/StdFace` and no `dh2.def` / `dh4.def` anywhere in the mVMC tree; DH is expert-mode input only).

- Tier 1 (primary, DH-2 contract): a hand-authored tiny system (e.g. 2–4 site Hubbard with one DH2 index and one DH4 index) where every count bin and the shift value is hand-computable. This is the gold standard for correctness and reviewability.
- Tier 2 (DH-3 integration gate): take a StdFace-generated base (Gutzwiller + Jastrow + Slater) and add hand-authored `dh2.def` / `dh4.def` plus `namelist.def` `DH2` / `DH4` entries on top, then run it through local C-mVMC to produce the reference `zvo_*` outputs committed for byte-for-byte parity. This exercises the realistic G + J + DH layout. A "pure StdFace" DH fixture is not achievable.
- Record C provenance in `metadata.txt` and third-party/provenance docs, matching the Plan 3b fixture convention.

CI:

- Subpackage `Pkg.test()` covers parser/layout/count contract tests.
- Root integration covers committed C reference fixtures.
- Keep C binary out of CI; fixtures are committed artifacts.

## 6. Suggested PR split

### PR DH-1: parser, data model, layout, docs as stub-to-structured

Scope:

- Add DH index-table types and `ProjectionLayout`.
- Remove the legacy `DoublonHolon2SiteTerm` / `DoublonHolon4SiteTerm` types and their `ExpertModeData` fields, migrating all internal readers and `test_unit/test_unit_read_opt_para.jl` to the index-table model in this PR (Q1 resolution).
- Parse `DH2` / `DH4`.
- Apply DH opt flags into global `optimization_flags`.
- Keep runtime paths guarded if DH is parsed but not yet executable. DH inputs must be rejected before optimization/sampling/PhysCal in DH-1, not half-executed.
- Ensure `n_proj_bf` is BackFlow-only and no longer counts DH, so parsed DH does not route into BackFlow stubs.
- Unit tests for parser/layout/opt-flag offsets.

Exit criteria:

- No-DH tests pass.
- DH inputs no longer look like value-bearing fake terms.
- No-DH offset, RNG, output, and integration behavior are unchanged.
- DH runtime inputs fail explicitly until DH-2 connects counts/logs/loaders.
- Review confirms C layout and index order.

### PR DH-2: projection counts, logs, sync, loaders, overlays

Scope:

- Implement DH count/recompute in normal and FSZ paths.
- Extend `log_proj_val` / `log_proj_ratio`.
- Fix `init_parameter!`, `read_initial_def!`, `read_opt_para_file!`, `read_input_parameters!`.
- Implement DH shifts in `sync_modified_parameter!`.
- Remove DH from BackFlow branch guard.

Exit criteria:

- Unit/contract tests pass, including update-vs-recompute invariants.
- Existing no-DH integration fixtures remain pass.
- DH no longer warns or skips in normal supported path.

### PR DH-3: C reference fixtures and manual flip

Scope:

- Generate C DH2/DH4 fixtures using the two tiers from section 5 (hand-authored tiny contract fixture; StdFace base + hand-authored DH overlay for the integration gate — not pure StdFace, which cannot emit DH).
- Add integration gate.
- Update manual compatibility tables and fixture provenance.

Exit criteria:

- DH fixtures pass against committed C references.
- Manual says supported only for the implemented DH scope.
- Remaining warn-only items are limited to `InOrbitalParallel`, `InOptTrans`, and `OptTrans`.

## 7. Design-review checklist

- `DH2` / `DH4` C keywords are parsed, not only legacy `DoublonHolon2Site` names.
- `NProj` includes `6*n_dh2 + 10*n_dh4` and uses the SpinJastrow offset slot even when SpinJastrow is zero.
- State and workspace count buffers are allocated with `layout.n_proj`.
- DH parameter values are stored separately from DH index tables.
- `dh2.def` / `dh4.def` opt-table first column is ignored; file row order controls opt flags.
- `InDH2` / `InDH4` use local DH slice indices, not global `Proj` indices.
- `InDH2` / `InDH4` validate range, duplicate, missing, and short-file errors.
- Slater and RBM `OptFlag` offsets use the same layout helper as projection counts.
- DH does not trigger BackFlow branches.
- `UpdateProjCnt` recomputes DH tail from current `eleNum` after the move.
- FSZ `ri == rj` keeps DH counts unchanged and returns early.
- `LogProjVal` / `LogProjRatio` use `real(param)` only.
- DH shift runs before GJ shift, requires all Gutzwiller real opt flags optimized, and adds its gauge shift into Gutzwiller.
- No-DH RNG, counts, output, and fixtures are unchanged.

## 8. Open questions for review

No open questions remain. Both prior open questions are resolved below.

Resolved by review:

- Legacy `DoublonHolon2SiteTerm` / `DoublonHolon4SiteTerm` (Q1): removed immediately in PR DH-1; their `ExpertModeData` fields do not survive. The types are internal only (not exported, absent from `examples/` and the manual) and have a bounded set of readers that change for the DH work anyway, so a parallel field would only re-introduce layout-drift risk. A deprecated test-only constructor that builds an index table from legacy arguments may be kept. See section 4.2 and PR DH-1.
- DH fixtures (Q2): two tiers — a hand-authored tiny contract fixture (Tier 1) and a StdFace-base-plus-hand-authored-DH-overlay integration fixture run through local C-mVMC (Tier 2). StdFace cannot emit DH (no DH code in `mVMC/src/StdFace`, no `dh2.def` / `dh4.def` in the mVMC tree), so a pure-StdFace DH fixture is not achievable. See section 5.
- Complex DH imaginary opt flags follow C: mirror real flags when `ComplexType > 0`, otherwise false.
- SpinJastrow input is hard unsupported for this plan; the layout helper reserves the offset slot but must not silently accept present SpinJastrow files.
- Malformed present `InDH2` / `InDH4` files hard-fail in strict runners.
