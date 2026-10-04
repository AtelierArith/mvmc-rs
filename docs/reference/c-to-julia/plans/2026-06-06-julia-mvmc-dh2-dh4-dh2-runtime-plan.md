---
date: 2026-06-06
datetime: 2026-06-06 14:20 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC DH2/DH4 DH-2 runtime implementation plan
target:
  repository: Julia-mVMC
  base: develop after PR #12 merge
  expected_branch: feature/v0.3-dh2-dh4-runtime
  prerequisite_pr: "#12 DH-1 parser/layout/runtime guard"
updated: 2026-06-06 14:41 JST
review:
  status: G1-G3 + minor notes folded in (orchestrator review against current code)
  added:
    - "G1: update_parameter_value DH write-back (the one remaining DH-unaware para mapping)"
    - "G2: resolve duplicate sync_modified_parameter!"
    - "G3: SR derivative DH inclusion in DoD"
    - "Scope note: split DH-2 loader order from future RBM-inclusive C order"
related:
  parent_plan: docs/superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md
  dh1_review: docs/reviews/2026-06-06-julia-mvmc-dh2-dh4-dh1-impl-review.md
  dh1_worklog: docs/worklog/2026-06-06_julia-mvmc-dh2-dh4-dh1-implementation.md
---

# Julia-mVMC DH2/DH4 DH-2 runtime implementation plan

## 0. 前提

DH-1（PR #12, commit `7ab379e`）で以下は完了済み。

- `DH2` / `DH4` parser、index-table model、`ProjectionLayout`。
- DH opt flags の global `optimization_flags` 反映。
- `Orbital*` opt flags の final-layout post-loop 適用。
- `SpinJastrow` hard-fail。
- active DH input の runtime guard。

DH-2 はこの guard を解除できるだけの runtime 実装を入れる。
作業開始は PR #12 merge 後、最新 `develop` から新 branch を切る。

## 1. Goal

DH2/DH4 を C と同じ意味で VMC runtime に反映する。

Definition of Done:

- `make_proj_cnt!` / `update_proj_cnt!` / `update_proj_cnt_fsz!` が DH2/DH4 counts を含む。
- `log_proj_val` / `log_proj_ratio` が DH parameter real part を含む。
- `init_parameter!` / `read_initial_def!` / `read_opt_para_file!` / `read_input_parameters!` が DH slice を C order で扱う。
- `sync_modified_parameter!` が DH2/DH4 shift を C order で実装する。
- **SR optimization が DH param を最適化対象に含む（G3）**: SR derivative `O_k = projCnt[k]` が DH slice を含み（buffer sizing で自動）、`update_parameter_value` が DH param を書き戻す（G1）。DH param が実際に更新されることを test で確認する。
- DH runtime guard を解除しても no-DH tests / fixtures が変わらない。
- DH の小さい hand-authored contract tests が pass する。

Non-goals:

- DH C reference fixture gate は DH-3。
- RBM fixed-parameter loader / `zqp_opt.dat` RBM block support は将来 scope。DH-2 では `NRBM > 0` loader は fail-loud のまま維持する。
- `InOrbitalParallel` / `InOptTrans` / OptTrans runtime は別 scope。
- SpinJastrow 実装はしない。present input は引き続き hard-fail。

## 2. 実装順

### DH-2.1: count/log core を先に作る

対象:

- `MVMCOptimizers.jl/src/vmc_sampling.jl`
  - `make_proj_cnt!`
  - `update_proj_cnt!`
  - `update_proj_cnt_fsz!`
  - `log_proj_ratio`
  - `log_proj_val`

実装:

- `_count_dh2!(proj_cnt, ele_num, data, layout)`
- `_count_dh4!(proj_cnt, ele_num, data, layout)`
- `_recompute_dh_counts!(proj_cnt, ele_num, data, layout)`

Counting contract:

- singly occupied center site は skip。
- holon center は neighbor doublon 数を count。
- doublon center は neighbor holon 数を count。
- DH2 index:
  `layout.dh2_offset + xn + (xi + 2*xm) * layout.n_dh2`
- DH4 index:
  `layout.dh4_offset + xn + (xi + 2*xm) * layout.n_dh4`
- `xi = 0`: center holon, `xi = 1`: center doublon。
- `xm = 0..2` for DH2, `xm = 0..4` for DH4。

Implementation note:

- The DH2/DH4 index formulas above are **0-based C offsets**; `proj_cnt` is 1-based in Julia, so add `+1` when indexing (same convention as DH-1 `set_dh_opt_flags!`: `2*(dh2_offset+local_i-1)+1`).
- First implementation should recompute the whole DH tail after ordinary G/J count updates. **Zero the DH tail first, then recompute** (C: `for(idx=offset; idx<nProj; idx++) projCntNew[idx]=0;` before the DH loops).
- The DH recompute reads the **post-move `ele_num`** (C's `UpdateProjCnt` recomputes DH after the move is applied to `eleNum`).
- FSZ `ri == rj` on-site spin flip keeps occupancy unchanged; preserve early return / no DH recompute.
- `log_proj_val` / `log_proj_ratio` must use `real(param)` only, matching C. They read live params via `projection_parameters(data, layout)`, so they reflect DH updates only once `update_parameter_value` writes DH back (G1).

Tests first:

- DH2 hand-computed occupancy cases.
- DH4 hand-computed occupancy cases.
- update-vs-fresh-recompute invariant for normal path.
- update-vs-fresh-recompute invariant for FSZ path.
- log value / ratio real-part-only tests.

### DH-2.2: state sizing and guard removal

対象:

- `MVMCOptimizers.jl/src/vmc_para_opt.jl`
- `MVMCOptimizers.jl/src/vmc_phys_cal.jl`
- `MVMCOptimizers.jl/src/vmc_sampling.jl`
- `MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `MVMCOptimizers.jl/src/stochastic_opt.jl`

実装:

- projection count buffers are always sized with `projection_layout(data).n_proj`。
- Add asserts around `proj_cnt` length where practical.
- **(G1, 必須) SR パラメータ書き戻し `update_parameter_value`（`stochastic_opt.jl:40`）を DH 対応にする。** これは唯一残った DH 非対応の para マッピング（現状 `n_proj = n_gutzwiller + n_jastrow`（:48, DH 除外）かつ DH branch 無し）。
  - `n_proj` を `projection_layout(data).n_proj`（DH 込み）に変更。
  - para_idx が DH2/DH4 range のとき `doublon_holon_2site_params` / `doublon_holon_4site_params` の該当 local index に `delta` を加算する DH branch を追加（DH param は term ではなく専用 vector に入るため、Gutz/Jastrow と同じ writeback では届かない）。
  - 背景: SR の列挙側（`smat_to_para_idx`、`stochastic_opt.jl:208` 以降）は既に DH 込み `n_para` を列挙し DH を最適化対象に選ぶため、書き戻しが DH 非対応だと **DH para_idx が RBM/Slater branch に誤ルートして Slater/RBM を破壊し、DH param は永久に未更新**（silent wrong physics）。DH-1 では DH が runtime-reject されこの経路が未到達のため残っていた。
- Remove active-DH early return guard only after DH-2.1 tests pass.
- Keep `n_proj_bf = 0` as BackFlow-only.

Tests:

- no-DH existing unit/integration pass unchanged.
- DH input no longer returns `1` in entry point solely because DH exists.
- DH input still does not enter BackFlow path.
- **(G1) DH ありで 1 SR step 後に DH param が更新され、Slater / RBM param が破壊されないことを確認する。** あわせて DH param が最適化対象として `smat_to_para_idx` に列挙され、その `O_k` が DH `projCnt` であること（G3）も検証する。

### DH-2.3: fixed-parameter loaders and overlays

対象:

- `MVMCOptimizers.jl/src/initial_params.jl`
  - `_load_para_triples!`
  - `read_initial_def!`
  - `read_opt_para_file!`
- `MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl`
  - `InDH2`
  - `InDH4`

実装:

- Remove DH loader guard.
- Keep RBM-bearing parameter files unsupported in DH-2: if `NRBM > 0`, `_load_para_triples!` continues to fail loud before parsing Slater.
- Projection block order:
  `Gutzwiller | Jastrow | SpinJastrow(0) | DH2 | DH4`
- DH-2 supported fixed-parameter record order (`NRBM == 0`, `NOptTrans == 0`):
  `6 diagnostics | Projection block | Slater`
- Future full C parameter record order:
  `6 diagnostics | Projection block | RBM | Slater | OptTrans`
- Expected triple count for DH-2 uses `projection_layout(data).n_proj` for the projection block and remains `6 + 3*(NProj + NSlater)` only after confirming `NRBM == 0`.
- Keep loader structure/comments clear that future RBM support inserts the RBM block between `Projection block` and `Slater`, not inside the projection block.
- `InDH2` uses first column as local DH2 index, not file order.
- `InDH4` uses first column as local DH4 index, not file order.
- Present overlays validate range, duplicate, missing, and short-file errors.
- `InDH2` / `InDH4` overlays apply **after** the fixed-parameter load (`read_initial_def!` / `read_opt_para_file!`) so the overlay wins, matching C order (`InitParameter → ReadInitParameter → ReadInputParameters`). Existing pipeline order already satisfies this; keep it.

Tests:

- `NRBM == 0` `zqp_opt.dat` with G/J/DH/Slater triples loads DH slice and keeps Slater aligned.
- RBM-bearing `zqp_opt.dat` still fails loud in DH-2, so Slater is not silently read from the future RBM block position.
- `InDH2` / `InDH4` overwrite the correct local DH param.
- malformed overlay hard-fails.
- no-DH loader tests remain byte-for-byte behavior compatible.

### DH-2.4: sync shift

対象:

- `MVMCOptimizers.jl/src/parameter_sync.jl`
- **(G2 解決) 二重定義の整理:** `sync_modified_parameter!` は 2 つある。
  - `parameter_sync.jl:29` — 最適化ループ用（GJ shift + Slater rescale）。`vmc_para_opt.jl:331` / `run_para_opt_from_namelist.jl:181` が呼ぶ。**DH2/DH4 shift はここにのみ入れる。**
  - `parameter_init.jl:247` — init 直後用で **Slater rescale のみ**（GJ/DH shift 無し）。C の `shiftDH2/4` は最適化ループの `SyncModifiedParameter` 側なので、**この parser-side 関数は DH shift 不要・変更しない**。混乱回避のため、必要なら docstring に「init-time Slater rescale 専用」と明記。

実装:

- Add `flag_shift_dh2(data, layout)` and `flag_shift_dh4(data, layout)`.
- Guard rule:
  - `NGutzwillerIdx > 0`
  - every Gutzwiller real OptFlag is optimized
  - every real DH slice OptFlag is optimized
- `shift_dh2!`: for each `(xi, xn)`, average the 3 `xm` bins, subtract from DH params, accumulate into `gShift`.
- `shift_dh4!`: same over 5 `xm` bins.
- Add total DH `gShift` to every Gutzwiller parameter before existing GJ shift.
- Preserve Slater rescale behavior.

Tests:

- DH2 shift hand-computed case.
- DH4 shift hand-computed case.
- if any Gutzwiller real OptFlag is fixed, DH shift disabled.
- DH shift occurs before GJ shift.

## 3. PR split within DH-2

Preferred: keep DH-2 as one PR only if review size stays manageable.

If it grows too large, split:

- DH-2a: counts/logs + runtime guard still present.
- DH-2b: loaders/overlays + guard removal.
- DH-2c: sync shift.

The guard should not be removed until count/log and loader coverage both exist.

## 4. Test commands

Minimum local gate:

```bash
cd /Users/misawatakahiro/Dropbox/Projects/Shin-mVMC/Julia-mVMC

~/.juliaup/bin/julialauncher +1.12 --project=MVMCExpertModeParsers.jl -e 'using Pkg; Pkg.test()'
~/.juliaup/bin/julialauncher +1.12 --project=MVMCOptimizers.jl -e 'using Pkg; Pkg.test()'
git diff --check
```

After guard removal, also run root integration smoke if runtime behavior changes broadly:

```bash
~/.juliaup/bin/julialauncher +1.12 --project=. test/integration/ctest_equivalent.jl
~/.juliaup/bin/julialauncher +1.12 --project=. test/integration/phys_cal_equivalent.jl
```

## 5. Next-thread startup checklist

1. Confirm PR #12 is merged.
2. Fetch and update local `develop`.
3. Create `feature/v0.3-dh2-dh4-runtime` from updated `develop`.
4. Start with DH-2.1 tests for count/log core before editing runtime code.
5. Keep DH guard until DH-2.1 and DH-2.3 are both green.

## 6. Known deferred items

- B3 from DH-1 review: optional exports for helper APIs.
- B4 from DH-1 review: whether to reject `NDoublonHolon*Idx == 0` to match C strictness.
- B5 from DH-1 review: explicit ModPara-before-DH ordering enforcement beyond current loud fail.
- DH-3: C reference fixtures and integration gate.
