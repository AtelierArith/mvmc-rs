---
date: 2026-06-23
datetime: 2026-06-23 18:28 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC Full Lanczos implementation plan
updated:
  - datetime: 2026-06-29 16:48 JST
    model: GPT-5 Codex
    note: Fixed the v0.4.3 P0 implementation scope to R0/R1 only.
  - datetime: 2026-06-29 17:10 JST
    model: GPT-5 Codex
    note: Added the R0/R1 execution checklist, output-contract checklist, and implementation acceptance gates.
  - datetime: 2026-07-01 17:00 JST
    model: GPT-5 Codex
    note: Refocused the plan on V05-1 R2/R3 for NLanczosMode=2 Lanczos Green after PR #43/#45 landed on develop.
  - datetime: 2026-07-01 17:07 JST
    model: GPT-5 Codex
    note: Fixed the mode2 C-reference staging policy and fixture selection.
  - datetime: 2026-07-01 17:31 JST
    model: GPT-5 Codex
    note: Reflected the independent review of the mode2 C-reference plan: absolute staging path, metadata gate, Tier B baseline policy, and empty-output handling.
---

# Julia-mVMC Full Lanczos implementation plan

## Goal

BackFlow を除いた C-mVMC v1.2 parity に向けて、Julia-mVMC の
`NLanczosMode > 0` rejection を段階的に外す。

最初の到達点は C v1.2 の `SpinChainLanczos` / `HubbardChainLanczos` 相当を
Julia の C-reference gate へ載せること。実装順は `NLanczosMode = 1`
energy/output path を先に通し、その後 `NLanczosMode = 2` の Lanczos Green を通す。

2026-07-01 時点で、R0/R1 は PR #43 として `develop` へ merge 済み。
現在の V05-1 対象は **R2/R3: `NLanczosMode = 2` Lanczos Green list /
accumulators / `zvo_ls_cisajs*` output** に移る。

## v0.4.3 P0 scope update

v0.4.3 P0 は R0/R1 に限定した。これは PR #43 で完了済み。

- R0: C v1.2 の `SpinChainLanczos` / `HubbardChainLanczos` inputs and outputs、
  unsupported combinations、`zvo_ls*` output contract を監査する。
- R1: serial `NSplitSize = 1` の `NLanczosMode = 1` energy/output path を実装し、
  `zvo_ls.dat` と `zvo_ls_qqqq.dat` を C-reference gate に載せる。
- R1 では `NLanczosMode = 2` を引き続き明示 reject する。
- `NLanczosMode = 2` の Lanczos Green list reconstruction / Green accumulators、
  MPI reduction、FSZ / complex / generalized orbital broader modes は R2 以降の
  separate implementation and gate とする。

## V05-1 R2/R3 scope update

R2/R3 は v0.5.0 必須の `NLanczosMode = 2` 対応として扱う。最初の実装 scope は
R1 と同じく `VMCPhysCal`, serial `NSplitSize = 1`, sz-conserved path に限定する。
`VMCParaOpt`, MPI split, FSZ/general-orbital, BackFlow は引き続き reject する。

Minimum supported path:

- `NVMCCalMode = 1`, `NLanczosMode = 2`, `NSplitSize = 1`
- R1 の `zvo_ls_out_XXX.dat` / `zvo_ls_qqqq_XXX.dat`
- mode2 の `zvo_ls_cisajs_XXX.dat`
- direct `TwoBodyG` がある場合の `zvo_ls_cisajscktalt_XXX.dat`
- factored `TwoBodyGEx` がある場合の `zvo_ls_cisajscktaltex_XXX.dat`

Reference policy:

- 既存の `hubbard_chain_lanczos/physcal_ref` と
  `spin_chain_lanczos/physcal_ref` を R2/R3 first fixtures として再利用し、
  `NLanczosMode = 2` に変更した C reference output を生成する。
- upstream C の `test/python/data/SpinChainLanczos` /
  `HubbardChainLanczos` は `StdFace.def` 上は `NLanczosMode = 1` なので、
  mode2 reference の生成手順、C commit、`OMP_NUM_THREADS=1`、変更した
  staging `modpara.def` の内容を `metadata.txt` に必ず残す。`StdFace.def`
  mode2 regeneration を cross-check に使った場合だけ、その内容も記録する。
- 既存 2 fixture は direct `TwoBodyG` の gate として使う。`TwoBodyGEx`
  (`zvo_ls_cisajscktaltex`) は `hubbard_chain_real/physcal_ref` を mode2 化して
  Tier B として追加する。Tier A が安定するまでは separate follow-up として分けてよい。
- complex path は安定した C reference fixture が見つかるまで release blocker にしない。
  まず real sz-conserved transfer / two-body path を C-reference gate に載せる。

## Mode2 C-reference staging and fixture policy

Mode2 C references are generated from staging copies only. Do not edit the
committed input fixtures in place while generating references.

### C baseline

- Use the same C baseline as the R1 references unless there is a specific
  reason to refresh all Lanczos references together:
  `mVMC @ 622166afe33c6be3402d7c926db7e9c0003a47c4`.
- Use `mVMC/build/src/mVMC/vmc.out` as the primary local binary for generation.
  If a different build is used, record the build directory and reason in the
  fixture metadata.
- Always run with `OMP_NUM_THREADS=1`, single MPI rank.
- Public metadata must not contain local absolute paths. Record C commit,
  command summary, source fixture, thread setting, and copied output filenames
  only.

### Staging layout

Use a project-local temporary staging tree outside the public Julia repository.
The path is absolute to avoid accidentally creating staging files inside
`Julia-mVMC/`:

```bash
/Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/tmp/lanczos-mode2-ref-YYYYMMDD-HHMMSS/
  <fixture>/
    inputs/        # copied from Julia-mVMC/test/integration/reference/<fixture>/physcal_ref/inputs
    zqp_opt.dat    # copied from the same physcal_ref
    c-output/      # raw C outputs copied from inputs/ after the run
    julia-output/  # optional Julia replay outputs for generation-time comparison
```

Generation steps per fixture:

1. Copy committed `physcal_ref/inputs/` and `zqp_opt.dat` into staging.
2. In the staging copy only, change `NLanczosMode` to `2`.
3. Run C from the staging `inputs/` directory:
   `OMP_NUM_THREADS=1 <vmc.out> -e namelist.def ../zqp_opt.dat`.
4. Copy raw C outputs from `inputs/` to `c-output/` for inspection, then copy only
   accepted reference outputs into
   `Julia-mVMC/test/integration/reference/<fixture>/physcal_ref/expected/`.
5. Update the fixture `metadata.txt` with source fixture, C commit,
   `NLanczosMode=2`, output filenames, and Julia-vs-C max differences.
6. Run the public local-path gate before preparing any public-facing text. The
   gate must include reference metadata, not only README/manual files:
   `rg -n "/Users/|Dropbox|Shin-mVMC|docs/reports|docs/reviews|LOG.md|benchmark" README.md docs/manual MVMCOptimizers.jl/README.md test/integration/reference`.

Standard-mode `StdFace.def` regeneration is optional cross-check only. The
committed mode2 inputs should be derived from the existing expert-mode
`physcal_ref/inputs` to keep the public fixture diff narrow and auditable.

### Fixture tiers

Tier A is the mandatory first PR scope:

| Fixture | Purpose | Expected mode2 files |
|---------|---------|----------------------|
| `hubbard_chain_lanczos/physcal_ref` | real Hubbard transfer path, existing R1 Lanczos fixture | `zvo_ls_out_001.dat`, `zvo_ls_qqqq_001.dat`, `zvo_ls_cisajs_001.dat`, `zvo_ls_cisajscktalt_001.dat` |
| `spin_chain_lanczos/physcal_ref` | real spin exchange/two-body path, existing R1 Lanczos fixture | `zvo_ls_out_001.dat`, `zvo_ls_qqqq_001.dat`, `zvo_ls_cisajs_001.dat`, `zvo_ls_cisajscktalt_001.dat` |

Tier B is the factored-Green follow-up. It can be included in the same PR only
after Tier A is stable:

| Fixture | Purpose | Expected mode2 files |
|---------|---------|----------------------|
| `hubbard_chain_real/physcal_ref` | small real non-FSZ fixture with existing `TwoBodyGEx`; low-cost check for `zvo_ls_cisajscktaltex` | existing Tier A-style files plus `zvo_ls_cisajscktaltex_001.dat` |

`hubbard_chain_real/physcal_ref` currently has non-Lanczos expected files
generated at C `66f17422968009f8cc70f1dec94b2f52e562d344`, while the Lanczos R1
references use `622166afe33c6be3402d7c926db7e9c0003a47c4`. For Tier B, do not
silently mix baselines. Either regenerate the whole Tier B fixture at `622166a`,
including existing non-Lanczos expected files, or explicitly document that only
the new `zvo_ls_*` files use `622166a` and verify the non-Lanczos outputs remain
within the existing tolerances.

Defer by default:

- `heisenberg_chain_real/physcal_ref`: useful second `TwoBodyGEx` check for
  exchange systems, but not needed for the first mode2 gate.
- `heisenberg_chain_cmp/physcal_ref`: complex mode remains deferred until a
  stable alpha / Lanczos Green reference is identified.
- FSZ/general-orbital fixtures: remain unsupported until a separate FSZ
  Lanczos plan and gate exist.

### Acceptance for generated references

- Tier A must compare `zvo_ls_out`, `zvo_ls_qqqq`, `zvo_ls_cisajs`, and
  `zvo_ls_cisajscktalt` against C.
- C opens `zvo_ls_cisajscktaltex_001.dat` for `NLanczosMode > 1` even when no
  `TwoBodyGEx` terms exist. For Tier A, treat that file as an untracked empty
  byproduct unless the test explicitly starts checking empty factored output.
- Tier B, if included, must additionally compare `zvo_ls_cisajscktaltex`.
- R1 files are regenerated under mode2 and must remain within the existing R1
  tolerances unless the metadata documents a numerical reason to adjust them.
- Staging outputs are not committed. Only curated inputs/expected/metadata
  changes are kept.

## C reference map

- `mVMC/src/mVMC/readdef.c`
  - `NLanczosMode > 1` では `CountOneBodyGForLanczos` が呼ばれ、
    `greenone.def` と factored `TwoBodyGEx` constituent から `OneBodyG` list /
    `iOneBodyGIdx` を作る。direct `TwoBodyG` はこの one-body list へ展開せず、
    `QCisAjsCktAltQDC` 側で扱う。
  - `GetInfoOneBodyG` / `GetInfoTwoBodyG` は、通常の output list とは別に
    `CisAjsIdx`, `iOneBodyGIdx`, `CisAjsCktAltIdx`,
    `CisAjsCktAltDCIdx` を作る。
  - `OrbitalGeneral` との組み合わせは C でも reject される。
- `mVMC/src/mVMC/setmemory.c`
  - `NLanczosMode > 0` で `QQQQ`, `LSLQ` を確保する。
  - `NLanczosMode > 1` で `QCisAjsQ`, `QCisAjsCktAltQ`, `LSLCisAjs` を確保する。
- `mVMC/src/mVMC/vmccal.c`
  - `NVMCCalMode == 1` で通常 Green を計算した後、
    `LSLocalQ` / `LSLocalQ_real` と `calculateQQQQ*` を呼ぶ。
  - `NLanczosMode > 1` では `LSLocalCisAjs*`, `calculateQCAQ*`,
    `calculateQCACAQ*` も呼ぶ。
- `mVMC/src/mVMC/physcal_lanczos.c`
  - `CalculateEne` で alpha と Lanczos energy を決め、`zvo_ls.dat` と
    `zvo_ls_qqqq.dat` を出す。
  - `NLanczosMode > 1` では `CalculatePhysVal_*` で
    `zvo_ls_cisajs.dat` / `zvo_ls_cisajscktalt.dat` /
    `zvo_ls_cisajscktaltex.dat` を出す。
- `mVMC/src/mVMC/initfile.c`
  - non-debug build で作られる mode2 出力は `zvo_ls_cisajs_XXX.dat`,
    `zvo_ls_cisajscktalt_XXX.dat`, `zvo_ls_cisajscktaltex_XXX.dat`。
    `_DEBUG` の `zvo_ls_qcisajsq*` / `zvo_ls_qcisajscktaltq*` は
    public output gate には含めない。
- `mVMC/src/mVMC/vmcmain.c`
  - `NVMCCalMode == 1` の output phase で `PhysCalLanczos_*` を呼ぶ。
- `mVMC/test/python/data/SpinChainLanczos`,
  `mVMC/test/python/data/HubbardChainLanczos`
  - C v1.2 の最初の parity fixtures として使う。

## Julia current map

- `Julia-mVMC/MVMCExpertModeParsers.jl/src/types/expert_types.jl`
  - `ModParaParameters.lanczos_mode` はある。
- `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
  - global validator は `NLanczosMode = 0/1/2` を受け付ける。
  - `VMCParaOpt` は `NLanczosMode > 0` を reject する。
  - `VMCPhysCal` は R1 として `NLanczosMode = 1` だけを通し、
    `NLanczosMode = 2` は reject している。
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_phys_cal.jl`
  - `NVMCCalMode = 1` 相当の measurement path はある。
  - 通常 Green output は `green_func_calc.jl` / `data_io.jl` 側に分離済み。
- `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`
  - R1 の `calculate_lanczos_h2!` / `accumulate_lanczos_qqqq!` は実装済み。
  - 既存 `build_canonical_cis_ajs_idx` は通常 PhysCal / `TwoBodyGEx`
    用の canonical one-body list を作る。mode2 では C の
    `CountOneBodyGForLanczos` と同じ ordering / de-dup semantics を明示した
    Lanczos 専用 helper へ分ける。
- `Julia-mVMC/MVMCOptimizers.jl/src/data_io.jl`
  - R1 の `zvo_ls_out_XXX.dat` / `zvo_ls_qqqq_XXX.dat` writer は実装済み。
  - mode2 の `zvo_ls_cisajs*` writer は未実装。
- `Julia-mVMC/test/integration/ctest_models.jl`
  - Standard-mode `SpinChainLanczos` / `HubbardChainLanczos` は
    `run_para_opt_from_namelist` gate からは deferred のまま。
- `Julia-mVMC/test/integration/lanczos_equivalent.jl`
  - R1 PhysCal C-reference gate として `hubbard_chain_lanczos` /
    `spin_chain_lanczos` の `zvo_ls_out` / `zvo_ls_qqqq` を比較している。

## Scope split

### R0: reference and parser audit

- C v1.2 の `SpinChainLanczos` / `HubbardChainLanczos` inputs and outputs を
  Julia integration reference layout へ追加する。
- `modpara.def` / `namelist.def` / `initial.def` の読み込み順を
  `run_phys_cal_from_namelist` に合わせて確認する。
- `NLanczosMode > 0` で C が reject する組み合わせを先に固定する。
  初期 scope では少なくとも `OrbitalGeneral` は reject のままにする。
- C output contract を整理する:
  - `NLanczosMode = 1`: `zvo_ls.dat`, `zvo_ls_qqqq.dat`
  - `NLanczosMode = 2`: 上記に加えて `zvo_ls_cisajs.dat`,
    `zvo_ls_cisajscktalt.dat`, `zvo_ls_cisajscktaltex.dat`

### R1: `NLanczosMode = 1` energy path

- `PhysicalQuantities` または専用 `LanczosQuantities` に `QQQQ` / `LSLQ`
  相当の accumulator を追加する。
- C の `NLSHam = 2` convention をそのまま固定する。
- `LSLocalQ_real` / `LSLocalQ` と `calculateQQQQ_real` / `calculateQQQQ`
  を Julia へ移植する。
- `vmc_phys_cal!` の sample loop で、通常 Green の後に Lanczos accumulator を
  更新する。
- `PhysCalLanczos_real/fcmp` の energy calculation を Julia へ移植し、
  `zvo_ls.dat` と `zvo_ls_qqqq.dat` を出す。
- `validate_supported_modpara` の reject を緩める。ただし R1 では
  `NLanczosMode = 1` の supported combination だけを通し、
  `NLanczosMode = 2` はまだ明示 reject にする。
- `SpinChainLanczos` / `HubbardChainLanczos` の C output と比較する integration
  gate を追加する。

### R2: `NLanczosMode = 2` Green list reconstruction

- C の `CountOneBodyGForLanczos`, `GetInfoOneBodyG`,
  `GetInfoTwoBodyGEx`, `GetInfoTwoBodyG` の semantics を再現する。
  既存の `build_canonical_cis_ajs_idx` は通常 Green / `TwoBodyGEx` output 用の
  helper として残し、Lanczos mode2 用の list helper は専用に分ける。
- Julia data model に以下の mapping を追加する。
  - `CisAjsLzIdx` 相当: mode2 one-body output order
  - `iOneBodyGIdx` 相当: `(site, spin) -> canonical index`
  - `CisAjsCktAltIdx` 相当: factored `TwoBodyGEx` term から two one-body indices
    への map
  - `CisAjsCktAltDCIdx` 相当: direct `TwoBodyG` output order
- normal Green output order と Lanczos Green output order が混ざらないように、
  output writer の責務を分ける。
- `NLanczosMode = 2` では通常 `zvo_cisajs*.dat` と Lanczos
  `zvo_ls_cisajs*.dat` の両方を C と同じ shape で出す。
- R2 の first unit gate は、R1 fixtures の `greenone.def` / `greentwo.def`
  を使い、`zvo_ls_cisajs` と `zvo_ls_cisajscktalt` の row order を固定する。
- R2 の second unit gate は、既存 non-FSZ `TwoBodyGEx` fixture を使い、
  `zvo_ls_cisajscktaltex` の value-only output order と constituent mapping を固定する。

### R3: `NLanczosMode = 2` Green accumulators and output

- C の `LSLocalCisAjs_real/fcmp` を移植する。row 0 は current local
  `CisAjs`、row 1 は `calHCA*` 相当の `H c†c` local value とする。
- `calculateQCAQ*` を移植し、`QCisAjsQ[NLSHam,NLSHam,NCisAjs]`
  相当を sample loop で更新する。
- `calculateQCACAQ*` を移植し、factored `TwoBodyGEx` 用の
  `QCisAjsCktAltQ[NLSHam,NLSHam,NCisAjsCktAlt]` 相当を更新する。
- `calculateQCACAQDC*` を移植し、direct `TwoBodyG` 用の
  `QCisAjsCktAltQDC[NLSHam,NLSHam,NCisAjsCktAltDC]` 相当を更新する。
  C と同じく `rq == 0` は current local direct Green、`rq == 1` は
  `H c†c c†c` local valueを使う。
- `CalculatePhysVal_real/fcmp` を移植して Lanczos Green output を作る。
  `QQQQ[3]` / `QQQQ[4]` (Julia 1-based) から H1/H2_1 を取り、R1 と同じ
  alpha を使って `(A0 + α(A01 + A10) + α^2 A11) / norm` を計算する。
- `zvo_ls_cisajs.dat`, `zvo_ls_cisajscktalt.dat`,
  `zvo_ls_cisajscktaltex.dat` を C-reference gate で比較する。
- `SpinChainLanczos` / `HubbardChainLanczos` は standard-mode
  `ctest_equivalent` へ無理に戻さず、まず `lanczos_equivalent.jl`
  の PhysCal replay gate として mode2 を固定する。

### R4: MPI / FSZ / broader modes

- R1/R2/R3 はまず serial `NSplitSize = 1` を正にする。
- MPI support は accumulator の `comm0` reduction を C の reduction order に合わせて
  別 PR で入れる。
- FSZ / complex / generalized orbital は、C の supported/rejected combination と
  fixture availability を確認してから validator を緩める。
- `BackFlow` は Full Lanczos scope から外す。C 側の newer BackFlow track が固まるまで
  Julia でも reject のままにする。

## R0/R1 execution checklist

### R0 deliverables

R0 は実装前の契約固定フェーズとする。R0 の完了物は、実装コードではなく
fixtures / output contract / validator policy / touch-point map の確定。

- C fixture audit:
  - `mVMC/test/python/data/SpinChainLanczos` と
    `mVMC/test/python/data/HubbardChainLanczos` の `namelist.def`,
    `modpara.def`, `initial.def`, Green input files, expected output files を読む。
  - 両 fixture の `NVMCCalMode`, `NLanczosMode`, `NSplitSize`, `NStore`,
    `NSRCG`, `NQPFull`, FSZ/real/complex status を表にする。
  - v0.4.3 R1 の最初の gate をどちらにするか決める。原則は小さく単純な方を
    first gate、もう一方を second gate とする。
- C output contract:
  - `zvo_ls.dat` の header 有無、列数、列意味、行数、format、正規化を
    `physcal_lanczos.c` の writer から固定する。
  - `zvo_ls_qqqq.dat` の header 有無、index order、値の並び、format、正規化を
    同じく C writer から固定する。
  - 比較 tolerance を決める。R1 は serial `NSplitSize = 1` のため、まず既存
    PhysCal Green gate と同等の tight tolerance を候補にし、C/Julia の
    Lanczos scalar 計算が BLAS/FMA 差で揺れる場合だけ明示的に緩める。
- Unsupported combination policy:
  - R1 で許可するのは `VMCPhysCal`, serial `NSplitSize = 1`,
    `NLanczosMode = 1` の supported fixture subset だけ。
  - `NLanczosMode = 2` は R1 では引き続き reject。
  - `VMCParaOpt` は R1 でも `NLanczosMode > 0` を reject する。Full Lanczos は
    `VMCPhysCal` path の機能であり、optimization entry へ接続しない。
  - MPI, FSZ, complex, `OrbitalGeneral`, BackFlow との組み合わせは、C fixture と
    C rejection behavior を確認してから R2+ で個別に緩和する。
- Reference-data handling:
  - C reference output を追加する場合は `Julia-mVMC/test/integration/reference/`
    配下の既存 README / layout に合わせる。
  - 参照データの生成コマンド、C commit/tag、threading env、比較対象ファイルを
    reference README に残す。
- R0 review gate:
  - R0 完了時点で、C function to Julia touch point の表をこの plan か別 review に残す。
  - R1 着手前に「validator policy」「output contract」「first fixture」を確定させる。

### R1 implementation checklist

- Data model and allocation:
  - `PhysicalQuantities` に直接足すか、専用 `LanczosQuantities` を持たせるかを
    実装前に決める。R1 では R2 の Green accumulator と混ざらない専用 field を優先する。
  - `QQQQ` / `LSLQ` 相当 accumulator の shape、real/complex storage、zeroing、
    sample accumulation order を C と対応づける。
- Local quantities:
  - `LSLocalQ_real` / `LSLocalQ` を Julia へ移植する。
  - `calculateQQQQ_real` / `calculateQQQQ` を Julia へ移植する。
  - R1 では `LSLocalCisAjs*`, `calculateQCAQ*`, `calculateQCACAQ*` は実装しない。
- PhysCal loop wiring:
  - `vmc_phys_cal!` の sample loop で、通常 Green accumulation の後に
    Lanczos R1 accumulator を更新する。
  - `run_phys_cal_from_namelist` 経由で `NVMCCalMode = 1`,
    `NLanczosMode = 1` fixture が同じ path を通ることを確認する。
  - `run_para_opt_from_namelist` / `vmc_para_opt!` には接続しない。
- Validator wiring:
  - global `validate_supported_modpara` から単純な `NLanczosMode > 0` 全拒否を
    entry-point specific policy へ移す。
  - `validate_supported_phys_cal_modpara` は R1 supported subset の
    `NLanczosMode = 1` だけを通し、それ以外を fail-fast する。
  - `validate_supported_para_opt_modpara` または para-opt entry 側は
    `NLanczosMode > 0` を引き続き fail-fast する。
- Output writer:
  - `PhysCalLanczos_real/fcmp` 相当の alpha / energy calculation を移植する。
  - `zvo_ls.dat` と `zvo_ls_qqqq.dat` を C の file name, row order, format に寄せる。
  - writer は rank0 / serial output convention に従う。R1 は MPI 対象外。
- Integration gate:
  - `SpinChainLanczos` / `HubbardChainLanczos` の R1 対象 fixture を
    deferred list から supported list へ移すのは、`zvo_ls*` gate が pass してからにする。
  - `ctest_equivalent` 側で `NLanczosMode = 1` の `zvo_ls.dat` と
    `zvo_ls_qqqq.dat` を比較する。
  - `NLanczosMode = 0` の既存 PhysCal / ParaOpt integration gate が regression しないことを
    確認する。

### R1 acceptance commands

R1 完了前に最低限以下を pass させる。実際の package/env に合わせ、既存 CI の
entry command が変わっている場合はそれに合わせる。

- `julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl")'`
- `julia --project=. test/integration/phys_cal_equivalent.jl`
- `julia --project=. test/integration/ctest_equivalent.jl`
- `julia --project=. test/mpi/run_mpi_smoke.jl`

MPI は R1 の対象外だが、validator 緩和が MPI / ParaOpt failure-mode を壊していないことを
見るため、既存 MPI smoke は regression gate として走らせる。

## R2/R3 execution checklist

### R2 deliverables: mode2 list / output contract

- C source audit:
  - `readdef.c` の `CountOneBodyGForLanczos`, `GetInfoOneBodyG`,
    `GetInfoTwoBodyGEx`, `GetInfoTwoBodyG` を読み、Julia helper の
    row order と de-dup rule に対応づける。
  - `initfile.c` / `physcal_lanczos.c` から mode2 output filename、
    debug-only filename、row format、末尾空行を固定する。
- Data model:
  - `PhysicalQuantities` に mode2 one-body list と lookup を追加する。
    既存 `cis_ajs_idx` を normal output 用として維持するか、
    `lanczos_cis_ajs_idx` を新設するかを実装前に決める。
  - `TwoBodyGEx` constituent mapping と direct `TwoBodyG` output order を
    mode2 用に保持する。
- Unit gates:
  - no-`TwoBodyGEx` fixture: explicit `greenone.def` orderと direct
    `greentwo.def` orderを確認する。
  - `TwoBodyGEx` fixture: C の reorder `(x0,x1,x2,x3)` と
    `(x6,x7,x4,x5)` を確認する。
  - duplicate `greenone.def` は C の practical behavior を確認してから扱う。
    不明なら初期 scope では duplicate-free fixture を前提にする。
- Reference generation:
  - `hubbard_chain_lanczos` / `spin_chain_lanczos` の PhysCal reference を
    `NLanczosMode = 2` へ変更して C を再実行し、以下を expected に追加する。
    `zvo_ls_out_001.dat`, `zvo_ls_qqqq_001.dat`,
    `zvo_ls_cisajs_001.dat`, `zvo_ls_cisajscktalt_001.dat`。
  - `TwoBodyGEx` coverage を同じ PR に含める場合は、既存 non-FSZ
    PhysCal fixture から `zvo_ls_cisajscktaltex_001.dat` reference を作る。

### R3 deliverables: mode2 accumulation / output

- Accumulators:
  - `PhysicalQuantities` と `VMCPhysAccumulator` に
    `phys_lanczos_qcisajsq`, `phys_lanczos_qcisajscktaltq`,
    `phys_lanczos_qcisajscktaltq_dc` 相当を追加する。
  - `reset_phys_quantities!`, local accumulator clear/merge/reset,
    `WeightAverageGreenFunc` 相当の normalization/reduce plumbing を更新する。
    初期 PR は serial `NSplitSize = 1` を正にし、MPI は regression smoke のみとする。
- Local values:
  - `LSLocalCisAjs*` 相当を追加し、row 0 は既存 local `CisAjs`、row 1 は
    `calHCA*` 相当を使う。
  - direct `TwoBodyG` の `HCACA` は、R1 の
    `_lanczos_apply_two_body` / `_lanczos_local_hamiltonian_from_config!`
    と同じ configuration-expansion 方針で実装し、C reference で符号を固定する。
- Sample loop:
  - `calculate_green_func!` の通常 Green accumulation 後、R1 の QQQQ 更新と同じ
    sample / weight / all_complex 判定で mode2 accumulators を更新する。
  - `NLanczosMode = 1` では mode2 accumulator を触らない。
- Output writer:
  - R1 writer の alpha / norm 計算を再利用し、
    `output_lanczos_func!` で mode2 output を追加する。
  - `zvo_ls_cisajs_XXX.dat`: 4 index + real/imag、C と同じ one-body order。
  - `zvo_ls_cisajscktalt_XXX.dat`: 8 index + real/imag、direct `TwoBodyG` order。
  - `zvo_ls_cisajscktaltex_XXX.dat`: value-only real/imag pairs、factored
    `TwoBodyGEx` order。
- Validator:
  - `validate_supported_phys_cal_modpara` は mode2 first scope だけを通す。
  - `VMCParaOpt`, `NSplitSize > 1`, FSZ/general-orbital, BackFlow は reject のまま。
  - spin-changing Transfer / InterAll の R1 guard は mode2 でも維持する。

### R2/R3 acceptance commands

R2/R3 完了前に最低限以下を pass させる。

- `julia --project=. MVMCOptimizers.jl/test/runtests.jl`
- `julia --project=. test/integration/lanczos_equivalent.jl`
- `julia --project=. test/integration/phys_cal_equivalent.jl`
- `julia --project=. test/integration/runtests.jl`
- `JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl`
- `git diff --check`
- public tree local-path gate:
  `rg -n "/Users/|Dropbox|Shin-mVMC|docs/reports|docs/reviews|LOG.md|benchmark" README.md docs/manual MVMCOptimizers.jl/README.md test/integration/reference`

## Verification plan

- R1 unit tests:
  - `CalculateEne` / alpha selectionの scalar tests。
  - `LSLocalQ*` and `calculateQQQQ*` の small-array tests。
  - `NLanczosMode = 1` は PhysCal supported subset で通り、ParaOpt では reject される
    validator tests。
- R1 integration tests:
  - `SpinChainLanczos`, `HubbardChainLanczos` for `NLanczosMode = 1`。
  - `zvo_ls.dat` and `zvo_ls_qqqq.dat` comparison against C reference。
- R2+ tests:
  - Lanczos Green list reconstruction の C-order tests。
  - `CalculatePhysVal_*` の alpha/norm formula tests。
  - `calculateQCAQ*`, `calculateQCACAQ*`, `calculateQCACAQDC*` の
    small-array tests。
  - `NLanczosMode = 2` C-reference fixtures で
    `zvo_ls_cisajs.dat`, `zvo_ls_cisajscktalt.dat`,
    必要に応じて `zvo_ls_cisajscktaltex.dat` を比較する。
  - mode2 にしても R1 `zvo_ls_out` / `zvo_ls_qqqq` が regression しないこと。
- Regression tests:
  - `NLanczosMode = 0` の existing PhysCal / optimization gates が変わらないこと。
  - unsupported combinations が sampling 前に fail-fast すること。

## Risks and decisions

- `NLanczosMode = 2` は `greenone.def` / `TwoBodyGEx` / `TwoBodyG` の list
  semantics を通常 Green output と少し変える。既存 helper と共有しすぎると
  C-order を崩すので、まず専用 helper にする。
- `PhysCalLanczos_*` output は C の formatted output に寄せる。内部値が近くても
  output order が違うと gate は壊れる。
- MPI reduction は R1/R2/R3 の後に分ける。serial parity がない段階で MPI を混ぜると、
  accumulator order と numerical difference の切り分けが難しくなる。
- Full Lanczos は `VMCPhysCal` path の機能として扱う。`run_para_opt_from_namelist`
  へ無理に接続しない。
- C の upstream standard fixtures は `NLanczosMode = 1` なので、mode2 C reference は
  local generationになる。公開側に generation provenance を書く場合は、公開 repo 内の
  input/output と C commit / command summary だけを書く。
- complex mode2 は R1 時点で stable fixture 未整備。real gate が通る前に complex を
  追加すると alpha 選択や縮退の切り分けが難しくなるため、real first とする。

## Exit criteria

### v0.4.3 R0/R1

- R0 audit で `zvo_ls.dat` / `zvo_ls_qqqq.dat` の output contract と first fixture が
  固定されている。
- `NLanczosMode = 1` fixtures が C-reference gate を pass。
- `NLanczosMode = 2` は R1 で明示 reject のまま。
- `VMCParaOpt` は `NLanczosMode > 0` を引き続き reject。
- `SpinChainLanczos` / `HubbardChainLanczos` のうち R1 gate を満たした fixture は
  deferred から supported へ移動できる。
- manual の Full Lanczos limitation を `NLanczosMode = 1` R1 scope に更新し、
  残る unsupported combinations を release note に書ける状態になる。

### R2+

- `NLanczosMode = 2` output contract and Green list reconstruction が C と一致。
- `zvo_ls_cisajs.dat` / `zvo_ls_cisajscktalt.dat` が C-reference gate を pass。
- `TwoBodyGEx` を同一 PR に含める場合は `zvo_ls_cisajscktaltex.dat` も
  C-reference gate を pass。
- `NLanczosMode = 2` でも `zvo_ls_out.dat` / `zvo_ls_qqqq.dat` が R1 reference から
  regression しない。
- `VMCPhysCal` validator は supported real sz-conserved `NLanczosMode = 2` だけを
  通し、ParaOpt / MPI split / FSZ / BackFlow は fail-fast のまま。
- MPI / FSZ / broader mode の validator 緩和は、それぞれ専用 gate ができてから行う。
